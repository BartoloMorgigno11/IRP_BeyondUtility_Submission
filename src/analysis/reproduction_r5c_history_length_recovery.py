"""R5C History-Length recovery with strict gated inference.

This phase executes only the frozen R5A HISTORY_LENGTH_RECOVERY rows. It keeps
R5B original and R5B corrected artifacts untouched, creates a separate R5C
identity, and uses the corrected exact-5 JSON index prompt/parser contract.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from src.analysis.reproduction_r5b_canddif_recovery import (
    EXPECTED_R5A_MANIFEST_SHA256,
    FAIL_STRICT_DENOMINATOR_POLICY,
    candidate_item_ids,
    canonical_json,
    git_stdout,
    hr_at_5,
    load_json,
    ndcg_at_5,
    parse_expected_count,
    positive_rank,
    protected_hash_check,
    read_csv,
    rel_path,
    sha256_file,
    sha256_text,
    write_csv,
    write_json_atomic,
)
from src.llm.canonical_llm_contract import parse_strict_index_response


PHASE = "R5C_HISTORY_LENGTH"
DEFAULT_CONFIG_PATH = Path("configs") / "reproduction_r5c_history_length_recovery.yaml"
NOTES_PATH = Path("report") / "notes.md"
R5B_CORRECTED_FREEZE_COMMIT = "83b46454866ea61f3e05eec81d555be53e4c41e8"
R5B_CORRECTED_PROTOCOL_HASH = "9596F36E1B1DBC15B68679DDF32114E6AE09914B418FB31DC112072384B708D5"
ORIGINAL_R5_NAMESPACE = "results/recovery/r5/"
R5B_CORRECTED_NAMESPACE = "results/recovery/r5b_corrected/"
R5C_PROMPT_TEMPLATE_ID = "r5c_history_length_json_index_exact5_v1"
R5C_PARSER_CONTRACT = "r5c_ranked_index_json_top5_v1"
EVIDENCE_CLASS = "PAPER_ALIGNED_ADAPTED_HISTORY_LENGTH_RECOVERY"
GLOBAL_STATUS = "CRITICAL_EXPERIMENTAL_PATH_R5C_HISTORY_LENGTH_PENDING_FREEZE"

DECISION_COMPLETE = "R5C_HISTORY_LENGTH_RECOVERY_COMPLETE"
DECISION_PREFLIGHT_FAILED = "R5C_STATIC_PREFLIGHT_FAILED"
DECISION_CONTEXT_FAILED = "R5C_CONTEXT_LENGTH_CHECK_FAILED"
DECISION_CANARY_FAILED = "R5C_CANARY_FAILED"
DECISION_EARLY_FAILED = "R5C_EARLY_RUN_FAILED"
DECISION_EXECUTION_FAILED = "R5C_EXECUTION_FAILED"
DECISION_SCIENTIFIC_FAILED = "R5C_SCIENTIFIC_VALIDITY_FAILED"
DECISION_PREFLIGHT_READY = "R5C_PREFLIGHT_READY"
DECISION_CANARY_PASS = "R5C_CANARY_PASS"
DECISION_EARLY_PASS = "R5C_EARLY_HEALTH_PASS"

TERMINAL_STATUSES = {"SUCCEEDED", "BACKEND_FAILED", "BACKEND_INCOMPLETE"}

OUTPUT_KEYS = (
    "execution_manifest",
    "request_execution_status",
    "group_validation",
    "output_validity",
    "metrics_by_condition",
    "history_length_results",
    "conditional_valid_metrics",
    "dataset_model_summary",
    "official_comparison",
    "gap_resolution",
    "runtime_provenance",
    "validation",
)

DATASET_ORDER = {"Beauty": 0, "Sports": 1, "ML-1M": 2, "LastFM": 3}
MODEL_ORDER = {"mistral:7b-instruct": 0, "qwen2:7b-instruct": 1}
CONDITION_ORDER = {
    "HISTORY_LENGTH_0": 0,
    "HISTORY_LENGTH_5": 1,
    "HISTORY_LENGTH_10": 2,
    "HISTORY_LENGTH_20": 3,
}


class R5CHistoryLengthError(RuntimeError):
    """Raised when R5C protocol identity or gated execution fails."""


@dataclass(frozen=True)
class R5COllamaResult:
    response_received: bool
    backend_complete: bool
    response_text: str
    latency_seconds: float
    raw_response: dict[str, Any] | None
    error: str
    error_category: str
    runtime_model_id: str
    done_reason: str


@dataclass(frozen=True)
class R5CBuildResult:
    decision: str
    frames: dict[str, pd.DataFrame]
    report: str
    manifest_sha256: str
    protocol_hash: str
    executed_this_run: int
    full_run_authorized: bool
    blocked_reason: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_config(path: Path | None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def normalize_rel(path: str | Path) -> str:
    return Path(path).as_posix().replace("\\", "/")


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def output_path(config: Mapping[str, Any], key: str) -> Path:
    return Path((config.get("outputs") or {})[key])


def configured_outputs(config: Mapping[str, Any]) -> dict[str, Path]:
    return {key: output_path(config, key) for key in OUTPUT_KEYS}


def diagnostics_dir(project_root: Path, config: Mapping[str, Any]) -> Path:
    return project_path(project_root, (config.get("outputs") or {})["diagnostics_dir"])


def r5c_namespace(project_root: Path, config: Mapping[str, Any]) -> Path:
    return project_path(project_root, (config.get("outputs") or {})["result_namespace"])


def expected_k(config: Mapping[str, Any]) -> int:
    return int((config.get("scope") or {}).get("expected_k", 5))


def history_length_from_condition(condition: str) -> int:
    match = re.search(r"(\d+)$", str(condition))
    if not match:
        raise R5CHistoryLengthError(f"cannot parse history length from condition: {condition}")
    return int(match.group(1))


def row_history_length(row: Mapping[str, Any]) -> int:
    value = str(row.get("history_length", "")).strip()
    if value.isdigit():
        return int(value)
    return history_length_from_condition(str(row.get("experimental_condition", "")))


def corrected_instruction(expected_count: int = 5) -> str:
    return (
        "Use only the train-history and candidate context below. "
        f"Return only valid JSON with key ranked_candidate_indices containing EXACTLY {expected_count} "
        f"unique 1-based candidate indices, ordered from rank 1 to rank {expected_count}: "
        '{"ranked_candidate_indices":[1,2,3,4,5]}. '
        f"Do not return more than {expected_count} indices. "
        f"Do not return fewer than {expected_count} indices. "
        "Use only candidate indices from the provided candidate set. "
        "Do not use item IDs in the answer. "
        "Do not include rationale, explanation, prose, Markdown, or extra keys."
    )


def render_corrected_prompt(old_prompt: str, expected_count: int = 5) -> str:
    old_sentence = (
        "Use only the train-history and candidate context below. "
        "Return JSON with key ranked_candidate_indices containing unique 1-based candidate indices."
    )
    instruction = corrected_instruction(expected_count)
    if old_sentence in old_prompt:
        return old_prompt.replace(old_sentence, instruction, 1)
    marker = "\nUser history:\n"
    if marker not in old_prompt:
        raise R5CHistoryLengthError("cannot locate User history block in frozen history-length prompt")
    head, tail = old_prompt.split(marker, 1)
    head_lines = head.rstrip().splitlines()
    if len(head_lines) < 3:
        raise R5CHistoryLengthError("unexpected frozen history-length prompt header")
    corrected_head = "\n".join(head_lines[:3] + [instruction])
    return corrected_head + marker + tail


def r5c_request_id(original_request_id: str) -> str:
    suffix = sha256_text(f"{PHASE}|{original_request_id}")[:16]
    return f"R5C-HIST-{suffix}"


def r5c_request_key(row: Mapping[str, Any]) -> str:
    fields = [
        "phase",
        "r5a_request_id",
        "recovery_family",
        "dataset",
        "model_name",
        "task",
        "experimental_condition",
        "history_length",
        "history_item_count",
        "user_id",
        "candidate_list_hash",
        "history_hash",
        "prompt_template_id",
        "parser_contract",
        "expected_output_schema",
        "prompt_hash",
    ]
    return canonical_json({field: row.get(field, "") for field in fields})


def r5c_material_context_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "phase": PHASE,
                "r5a_request_id": row.get("r5a_request_id", ""),
                "dataset": row.get("dataset", ""),
                "user_id": row.get("user_id", ""),
                "model_name": row.get("model_name", ""),
                "experimental_condition": row.get("experimental_condition", ""),
                "history_length": row.get("history_length", ""),
                "history_item_count": row.get("history_item_count", ""),
                "history_hash": row.get("history_hash", ""),
                "candidate_list_hash": row.get("candidate_list_hash", ""),
                "unordered_candidate_item_set_hash": row.get("unordered_candidate_item_set_hash", ""),
                "positive_item_id": row.get("positive_item_id", ""),
                "prompt_hash": row.get("prompt_hash", ""),
                "parser_contract": row.get("parser_contract", ""),
                "expected_output_schema": row.get("expected_output_schema", ""),
            }
        )
    )


def load_r5c_manifest(
    project_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    frozen = config.get("frozen_r5a") or {}
    path = project_path(project_root, frozen["manifest"])
    if not path.exists():
        raise FileNotFoundError(path)
    manifest_sha = sha256_file(path)
    expected_sha = str(frozen.get("manifest_sha256", EXPECTED_R5A_MANIFEST_SHA256)).upper()
    if manifest_sha != expected_sha:
        raise R5CHistoryLengthError(f"R5A manifest SHA mismatch observed={manifest_sha} expected={expected_sha}")
    manifest = read_csv(path)
    expected_rows = int(frozen.get("manifest_rows", 4800))
    if len(manifest) != expected_rows:
        raise R5CHistoryLengthError(f"R5A manifest row count changed: {len(manifest)} != {expected_rows}")

    scope = config.get("scope") or {}
    family = str(scope.get("recovery_family", "HISTORY_LENGTH_RECOVERY"))
    history = manifest.loc[manifest["recovery_family"].eq(family)].copy()
    selected_rows = int(scope.get("selected_rows", 2400))
    if len(history) != selected_rows:
        raise R5CHistoryLengthError(f"R5C history subset row count changed: {len(history)} != {selected_rows}")
    if not history["task"].eq("ranking").all():
        raise R5CHistoryLengthError("R5C history subset contains non-ranking rows")
    if history["recovery_family"].isin(set(scope.get("forbidden_recovery_families") or [])).any():
        raise R5CHistoryLengthError("R5C history subset contains forbidden recovery families")
    if history["task"].isin(set(scope.get("forbidden_tasks") or [])).any():
        raise R5CHistoryLengthError("R5C history subset contains forbidden tasks")

    expected_datasets = set(scope.get("datasets") or [])
    expected_models = set(scope.get("models") or [])
    if set(history["dataset"]) != expected_datasets:
        raise R5CHistoryLengthError(f"R5C datasets changed: {sorted(set(history['dataset']))}")
    if set(history["model_name"]) != expected_models:
        raise R5CHistoryLengthError(f"R5C models changed: {sorted(set(history['model_name']))}")

    namespace = r5c_namespace(project_root, config)
    expected_count = expected_k(config)
    original_request_ids = history["request_id"].astype(str).tolist()
    history["phase"] = PHASE
    history["r5a_request_id"] = original_request_ids
    history["original_output_destination"] = history["output_destination"].astype(str)
    history["request_id"] = [r5c_request_id(value) for value in original_request_ids]
    history["prompt_template_id"] = R5C_PROMPT_TEMPLATE_ID
    history["prompt_template_source"] = "src/analysis/reproduction_r5c_history_length_recovery.py"
    history["parser_contract"] = R5C_PARSER_CONTRACT
    history["expected_output_schema"] = canonical_json({"ranked_candidate_indices": list(range(1, expected_count + 1))})
    history["prompt_render"] = history["prompt_render"].astype(str).map(
        lambda prompt: render_corrected_prompt(prompt, expected_count)
    )
    history["prompt_hash"] = history["prompt_render"].map(sha256_text)
    history["output_destination"] = [
        normalize_rel(Path((config.get("outputs") or {})["result_namespace"]) / f"{request_id}.json")
        for request_id in history["request_id"].astype(str)
    ]
    history["request_key_sha256"] = [sha256_text(r5c_request_key(row)) for row in history.to_dict("records")]
    history["material_context_hash"] = [
        r5c_material_context_hash(row) for row in history.to_dict("records")
    ]

    if not history["request_id"].is_unique:
        raise R5CHistoryLengthError("R5C request IDs are not unique")
    if not history["output_destination"].is_unique:
        raise R5CHistoryLengthError("R5C output destinations are not unique")
    for destination in history["output_destination"].astype(str):
        normalized = normalize_rel(destination)
        resolved = project_path(project_root, destination).resolve()
        if namespace.resolve() not in [resolved, *resolved.parents]:
            raise R5CHistoryLengthError(f"R5C output path outside namespace: {destination}")
        if normalized.startswith(ORIGINAL_R5_NAMESPACE):
            raise R5CHistoryLengthError(f"R5C output path overlaps original R5 namespace: {destination}")
        if normalized.startswith(R5B_CORRECTED_NAMESPACE):
            raise R5CHistoryLengthError(f"R5C output path overlaps corrected R5B namespace: {destination}")
    return manifest, history.reset_index(drop=True), manifest_sha


def stable_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        MODEL_ORDER.get(str(row["model_name"]), 99),
        DATASET_ORDER.get(str(row["dataset"]), 99),
        CONDITION_ORDER.get(str(row["experimental_condition"]), 99),
        str(row["user_id"]),
        str(row["r5a_request_id"]),
    )


def sorted_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return sorted(frame.to_dict("records"), key=stable_sort_key)


def select_first_n_per_cell(
    frame: pd.DataFrame,
    n: int,
    *,
    skip_request_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    skipped = set(skip_request_ids)
    selected: list[dict[str, Any]] = []
    for _, group in frame.groupby(["model_name", "dataset", "experimental_condition"], sort=True):
        candidates = [row for row in sorted_records(group) if str(row["request_id"]) not in skipped]
        selected.extend(candidates[:n])
    return sorted(selected, key=stable_sort_key)


def select_all_remaining(frame: pd.DataFrame, *, skip_request_ids: Iterable[str] = ()) -> list[dict[str, Any]]:
    skipped = set(skip_request_ids)
    return [row for row in sorted_records(frame) if str(row["request_id"]) not in skipped]


def parse_output_record(row: Mapping[str, Any], response_text: str) -> dict[str, Any]:
    candidate_ids = candidate_item_ids(row)
    expected_count = parse_expected_count(row)
    parsed = parse_strict_index_response(
        response_text,
        candidate_items=candidate_ids,
        expected_count=expected_count,
        require_full_permutation=False,
    )
    rank = positive_rank(row, parsed.output_items) if parsed.parse_valid else 0
    validity_class = validity_class_from_parse(parsed.status)
    return {
        "parse_valid": bool(parsed.parse_valid),
        "parse_status": parsed.status,
        "parse_error": parsed.error,
        "validity_class": "VALID" if parsed.parse_valid else validity_class,
        "ranked_candidate_indices": list(parsed.output_indices),
        "ranked_item_ids": list(parsed.output_items),
        "duplicate_count": int(parsed.duplicate_count),
        "unknown_item_count": int(parsed.unknown_item_count),
        "output_length": int(parsed.output_length),
        "expected_output_count": expected_count,
        "positive_rank": rank,
        "hr_at_5": hr_at_5(rank) if parsed.parse_valid else 0.0,
        "ndcg_at_5": ndcg_at_5(rank) if parsed.parse_valid else 0.0,
        "metric_ready": bool(parsed.parse_valid),
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
    }


def validity_class_from_parse(status: str) -> str:
    if status == "parsed_valid":
        return "VALID"
    if status == "invalid_length":
        return "WRONG_LENGTH"
    if status in {"parse_failed", "empty_output"}:
        return "PARSE_FAILURE"
    if status == "duplicate_indices":
        return "DUPLICATE"
    if status == "unknown_indices":
        return "OUT_OF_RANGE"
    if status in {"invalid_schema", "missing_permutation_indices"}:
        return "OTHER"
    return "OTHER"


def incomplete_parse_record(row: Mapping[str, Any], response_text: str) -> dict[str, Any]:
    expected_count = parse_expected_count(row)
    return {
        "parse_valid": False,
        "parse_status": "backend_incomplete",
        "parse_error": "Ollama generation did not report done=true",
        "validity_class": "BACKEND_INCOMPLETE",
        "ranked_candidate_indices": [],
        "ranked_item_ids": [],
        "duplicate_count": 0,
        "unknown_item_count": 0,
        "output_length": 0,
        "expected_output_count": expected_count,
        "positive_rank": 0,
        "hr_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "metric_ready": False,
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
        "diagnostic_raw_response_length": len(str(response_text or "")),
    }


def ensure_local_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http":
        raise R5CHistoryLengthError(f"only local HTTP Ollama endpoints are allowed, observed {endpoint}")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise R5CHistoryLengthError(f"only localhost Ollama endpoints are allowed, observed {endpoint}")


def http_json_get(endpoint: str, timeout_seconds: int) -> dict[str, Any]:
    ensure_local_endpoint(endpoint)
    with urllib.request.urlopen(endpoint, timeout=timeout_seconds) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise R5CHistoryLengthError(f"local endpoint returned non-object JSON: {endpoint}")
    return parsed


def ollama_version(config: Mapping[str, Any]) -> str:
    endpoint = str((config.get("runtime") or {}).get("version_endpoint", "http://127.0.0.1:11434/api/version"))
    try:
        payload = http_json_get(endpoint, int((config.get("runtime") or {}).get("timeout_seconds", 120)))
    except Exception as exc:
        return f"unavailable:{exc}"
    return str(payload.get("version", "unknown"))


def ollama_tags(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    endpoint = str((config.get("runtime") or {}).get("tags_endpoint", "http://127.0.0.1:11434/api/tags"))
    payload = http_json_get(endpoint, int((config.get("runtime") or {}).get("timeout_seconds", 120)))
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise R5CHistoryLengthError("Ollama tags endpoint returned invalid models list")
    return {str(item.get("name", "")): dict(item) for item in models if isinstance(item, dict)}


def validate_runtime_available(
    config: Mapping[str, Any],
    required_models: Iterable[str],
) -> tuple[bool, str, dict[str, dict[str, Any]]]:
    try:
        tags = ollama_tags(config)
    except Exception as exc:
        return False, str(exc), {}
    missing = sorted(set(required_models).difference(tags))
    if missing:
        return False, f"missing Ollama models: {missing}", tags
    return True, "", tags


def generate_local_ollama(row: Mapping[str, Any], config: Mapping[str, Any]) -> R5COllamaResult:
    runtime_cfg = config.get("runtime") or {}
    endpoint = str(runtime_cfg.get("endpoint", "http://127.0.0.1:11434/api/generate"))
    ensure_local_endpoint(endpoint)
    options = {
        "temperature": float(row.get("temperature", runtime_cfg.get("temperature", 0)) or 0),
        "seed": int(row.get("seed", 0) or 0),
        "num_predict": int(row.get("max_new_tokens", runtime_cfg.get("max_new_tokens", 512)) or 512),
    }
    payload = {
        "model": str(row["model_name"]),
        "prompt": str(row["prompt_render"]),
        "stream": False,
        "format": str(runtime_cfg.get("format", "json")),
        "options": options,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=int(runtime_cfg.get("timeout_seconds", 120))) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return R5COllamaResult(False, False, "", time.perf_counter() - started, None, f"HTTP {exc.code}: {exc.reason}", "backend_http_error", str(row["model_name"]), "")
    except urllib.error.URLError as exc:
        return R5COllamaResult(False, False, "", time.perf_counter() - started, None, str(exc.reason), "backend_connection_error", str(row["model_name"]), "")
    except TimeoutError:
        return R5COllamaResult(False, False, "", time.perf_counter() - started, None, "timed out", "backend_timeout", str(row["model_name"]), "")
    except Exception as exc:  # pragma: no cover - local runtime state varies.
        return R5COllamaResult(False, False, "", time.perf_counter() - started, None, str(exc), "backend_error", str(row["model_name"]), "")
    latency = time.perf_counter() - started
    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        return R5COllamaResult(False, False, "", latency, None, f"invalid Ollama JSON: {exc}", "backend_invalid_json", str(row["model_name"]), "")
    if not isinstance(parsed, dict):
        return R5COllamaResult(False, False, "", latency, None, "Ollama returned non-object JSON", "backend_invalid_response", str(row["model_name"]), "")
    response_text = parsed.get("response")
    if not isinstance(response_text, str):
        return R5COllamaResult(False, False, "", latency, parsed, "Ollama response field missing", "backend_invalid_response", str(row["model_name"]), str(parsed.get("done_reason", "")))
    model_id = str(parsed.get("model") or row["model_name"])
    done = parsed.get("done")
    done_reason = str(parsed.get("done_reason", ""))
    if done is not True:
        return R5COllamaResult(True, False, response_text.strip(), latency, parsed, "Ollama done flag is not true", "backend_incomplete", model_id, done_reason)
    return R5COllamaResult(True, True, response_text.strip(), latency, parsed, "", "", model_id, done_reason)


def build_result_payload(
    row: Mapping[str, Any],
    manifest_sha: str,
    protocol_hash: str,
    runtime: R5COllamaResult,
    started_at: str,
    ended_at: str,
    gate_stage: str,
) -> dict[str, Any]:
    if not runtime.response_received:
        parse_record = incomplete_parse_record(row, "")
        parse_record["validity_class"] = "BACKEND_FAILED"
        parse_record["parse_status"] = "backend_failed"
        status = "BACKEND_FAILED"
        error_category = runtime.error_category or "backend_failed"
    elif not runtime.backend_complete:
        parse_record = incomplete_parse_record(row, runtime.response_text)
        status = "BACKEND_INCOMPLETE"
        error_category = runtime.error_category or "backend_incomplete"
    else:
        parse_record = parse_output_record(row, runtime.response_text)
        status = "SUCCEEDED"
        error_category = "" if parse_record["parse_valid"] else str(parse_record["validity_class"]).lower()

    payload: dict[str, Any] = {
        "phase": PHASE,
        "gate_stage": gate_stage,
        "request_id": str(row["request_id"]),
        "r5a_request_id": str(row["r5a_request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": protocol_hash,
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "model_label": str(row.get("model_label", "")),
        "user_id": str(row["user_id"]),
        "task": str(row["task"]),
        "recovery_family": str(row["recovery_family"]),
        "experimental_condition": str(row["experimental_condition"]),
        "history_length": str(row.get("history_length", "")),
        "history_item_count": str(row.get("history_item_count", "")),
        "history_source": str(row.get("history_source", "")),
        "prompt_template_id": str(row["prompt_template_id"]),
        "prompt_hash": str(row["prompt_hash"]),
        "candidate_list_hash": str(row["candidate_list_hash"]),
        "unordered_candidate_item_set_hash": str(row["unordered_candidate_item_set_hash"]),
        "history_hash": str(row["history_hash"]),
        "material_context_hash": str(row["material_context_hash"]),
        "request_key_sha256": str(row["request_key_sha256"]),
        "positive_item_id": str(row["positive_item_id"]),
        "positive_candidate_position": str(row["positive_candidate_position"]),
        "candidate_count": int(row["candidate_count"]),
        "parser_contract": str(row["parser_contract"]),
        "expected_output_schema": str(row["expected_output_schema"]),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "runtime_backend": "ollama",
        "runtime_model_id": runtime.runtime_model_id,
        "backend_complete": runtime.backend_complete,
        "ollama_done_reason": runtime.done_reason,
        "temperature": str(row.get("temperature", "0")),
        "seed": str(row.get("seed", "0")),
        "max_new_tokens": str(row.get("max_new_tokens", "512")),
        "retry_count": 0,
        "latency_seconds": round(runtime.latency_seconds, 6),
        "execution_status": status,
        "error_category": error_category,
        "error": runtime.error,
        "raw_response_text": runtime.response_text,
        "ollama_raw_response": runtime.raw_response or {},
        "evidence_class": EVIDENCE_CLASS,
        "global_status": GLOBAL_STATUS,
        **parse_record,
    }
    payload["metric_ready"] = bool(payload["execution_status"] == "SUCCEEDED" and payload["parse_valid"])
    payload["result_hash"] = sha256_text(canonical_json({k: v for k, v in payload.items() if k != "result_hash"}))
    return payload


def validate_existing_result(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    manifest_sha: str,
    protocol_hash: str,
    path: Path,
) -> None:
    required_equal = {
        "phase": PHASE,
        "request_id": str(row["request_id"]),
        "r5a_request_id": str(row["r5a_request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": protocol_hash,
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "user_id": str(row["user_id"]),
        "experimental_condition": str(row["experimental_condition"]),
        "history_length": str(row.get("history_length", "")),
        "prompt_hash": str(row["prompt_hash"]),
        "candidate_list_hash": str(row["candidate_list_hash"]),
        "history_hash": str(row["history_hash"]),
        "material_context_hash": str(row["material_context_hash"]),
    }
    for key, expected in required_equal.items():
        observed = str(payload.get(key, ""))
        if observed != str(expected):
            raise R5CHistoryLengthError(
                f"existing R5C output identity mismatch at {path}: {key} observed={observed} expected={expected}"
            )
    if str(payload.get("execution_status", "")) not in TERMINAL_STATUSES:
        raise R5CHistoryLengthError(f"existing R5C output is non-terminal at {path}")
    if str(payload.get("execution_status", "")) == "SUCCEEDED" and bool(payload.get("backend_complete")) is not True:
        raise R5CHistoryLengthError(f"existing R5C output has SUCCEEDED without backend_complete=true at {path}")


def execute_selected_requests(
    project_root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifest_sha: str,
    protocol_hash: str,
    *,
    allow_api_calls: bool,
    allow_local_ollama: bool,
    gate_stage: str,
) -> int:
    if not allow_api_calls:
        raise R5CHistoryLengthError("R5C execution requires --allow-api-calls")
    if not allow_local_ollama:
        raise R5CHistoryLengthError("R5C execution requires --allow-local-ollama")
    namespace = r5c_namespace(project_root, config).resolve()
    executed = 0
    for row in rows:
        destination = project_path(project_root, str(row["output_destination"]))
        rel_destination = rel_path(project_root, destination).replace("\\", "/")
        if namespace not in [destination.resolve(), *destination.resolve().parents]:
            raise R5CHistoryLengthError(f"R5C output destination outside namespace: {destination}")
        if rel_destination.startswith(ORIGINAL_R5_NAMESPACE) or rel_destination.startswith(R5B_CORRECTED_NAMESPACE):
            raise R5CHistoryLengthError(f"refusing to write non-R5C output namespace: {destination}")
        if destination.exists():
            validate_existing_result(row, load_json(destination), manifest_sha, protocol_hash, destination)
            continue
        started_at = utc_now()
        runtime = generate_local_ollama(row, config)
        ended_at = utc_now()
        payload = build_result_payload(row, manifest_sha, protocol_hash, runtime, started_at, ended_at, gate_stage)
        write_json_atomic(payload, destination)
        executed += 1
    return executed


def status_for_missing(row: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    return {
        "request_id": str(row["request_id"]),
        "r5a_request_id": str(row["r5a_request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": "",
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "user_id": str(row["user_id"]),
        "experimental_condition": str(row["experimental_condition"]),
        "history_length": str(row.get("history_length", "")),
        "history_item_count": str(row.get("history_item_count", "")),
        "output_destination": str(row["output_destination"]),
        "result_file_exists": "false",
        "execution_status": "NOT_EXECUTED",
        "terminal_accounting_status": "NOT_EXECUTED",
        "terminal_status": "false",
        "backend_complete": "false",
        "parse_valid": "false",
        "valid_output": "false",
        "metric_ready": "false",
        "parse_status": "",
        "validity_class": "NOT_EXECUTED",
        "positive_rank": 0,
        "hr_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "output_length": 0,
        "expected_output_count": parse_expected_count(row),
        "duplicate_count": 0,
        "unknown_item_count": 0,
        "error_category": "",
        "error": "",
        "retry_count": 0,
        "result_hash": "",
        "result_file_sha256": "",
        "latency_seconds": "",
        "started_at_utc": "",
        "ended_at_utc": "",
        "runtime_model_id": "",
        "gate_stage": "",
        "ollama_done_reason": "",
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
    }


def status_for_payload(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    manifest_sha: str,
    protocol_hash: str,
    path: Path,
) -> dict[str, Any]:
    validate_existing_result(row, payload, manifest_sha, protocol_hash, path)
    execution_status = str(payload.get("execution_status", ""))
    parse_valid = bool(payload.get("parse_valid", False))
    metric_ready = bool(execution_status == "SUCCEEDED" and parse_valid)
    if execution_status == "SUCCEEDED":
        terminal_accounting = "COMPLETED_VALID" if parse_valid else "COMPLETED_INVALID_OUTPUT"
    else:
        terminal_accounting = execution_status
    return {
        "request_id": str(row["request_id"]),
        "r5a_request_id": str(row["r5a_request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": str(payload.get("protocol_hash", "")),
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "user_id": str(row["user_id"]),
        "experimental_condition": str(row["experimental_condition"]),
        "history_length": str(row.get("history_length", "")),
        "history_item_count": str(row.get("history_item_count", "")),
        "output_destination": str(row["output_destination"]),
        "result_file_exists": "true",
        "execution_status": execution_status,
        "terminal_accounting_status": terminal_accounting,
        "terminal_status": "true",
        "backend_complete": bool_text(bool(payload.get("backend_complete", False))),
        "parse_valid": bool_text(parse_valid),
        "valid_output": bool_text(parse_valid),
        "metric_ready": bool_text(metric_ready),
        "parse_status": str(payload.get("parse_status", "")),
        "validity_class": str(payload.get("validity_class", "")),
        "positive_rank": int(payload.get("positive_rank", 0) or 0),
        "hr_at_5": float(payload.get("hr_at_5", 0.0) or 0.0) if metric_ready else 0.0,
        "ndcg_at_5": float(payload.get("ndcg_at_5", 0.0) or 0.0) if metric_ready else 0.0,
        "output_length": int(payload.get("output_length", 0) or 0),
        "expected_output_count": int(payload.get("expected_output_count", parse_expected_count(row)) or parse_expected_count(row)),
        "duplicate_count": int(payload.get("duplicate_count", 0) or 0),
        "unknown_item_count": int(payload.get("unknown_item_count", 0) or 0),
        "error_category": str(payload.get("error_category", "")),
        "error": str(payload.get("error", "")),
        "retry_count": int(payload.get("retry_count", 0) or 0),
        "result_hash": str(payload.get("result_hash", "")),
        "result_file_sha256": sha256_file(path),
        "latency_seconds": payload.get("latency_seconds", ""),
        "started_at_utc": str(payload.get("started_at_utc", "")),
        "ended_at_utc": str(payload.get("ended_at_utc", "")),
        "runtime_model_id": str(payload.get("runtime_model_id", "")),
        "gate_stage": str(payload.get("gate_stage", "")),
        "ollama_done_reason": str(payload.get("ollama_done_reason", "")),
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
    }


def build_status(
    project_root: Path,
    history: pd.DataFrame,
    manifest_sha: str,
    protocol_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in history.to_dict("records"):
        path = project_path(project_root, str(row["output_destination"]))
        if path.exists():
            rows.append(status_for_payload(row, load_json(path), manifest_sha, protocol_hash, path))
        else:
            rows.append(status_for_missing(row, manifest_sha))
    return pd.DataFrame(rows)


def configured_conditions_by_dataset(config: Mapping[str, Any]) -> dict[str, set[str]]:
    raw = (config.get("scope") or {}).get("conditions_by_dataset") or {}
    return {str(dataset): {str(condition) for condition in conditions} for dataset, conditions in raw.items()}


def build_group_validation(history: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    expected_by_dataset = configured_conditions_by_dataset(config)
    rows: list[dict[str, Any]] = []
    for idx, ((dataset, model_name, user_id), group) in enumerate(history.groupby(["dataset", "model_name", "user_id"], sort=True), start=1):
        conditions = set(group["experimental_condition"].astype(str))
        expected_conditions = expected_by_dataset.get(str(dataset), set())
        row_count_valid = len(group) == len(expected_conditions) == 3
        condition_valid = conditions == expected_conditions
        same_candidate_list = group["candidate_list_hash"].astype(str).nunique() == 1
        same_candidate_set = group["unordered_candidate_item_set_hash"].astype(str).nunique() == 1
        same_positive = group["positive_item_id"].astype(str).nunique() == 1
        task_valid = group["task"].astype(str).eq("ranking").all()
        family_valid = group["recovery_family"].astype(str).eq("HISTORY_LENGTH_RECOVERY").all()
        lengths = sorted(row_history_length(row) for row in group.to_dict("records"))
        valid = all([row_count_valid, condition_valid, same_candidate_list, same_candidate_set, same_positive, task_valid, family_valid])
        rows.append(
            {
                "group_id": f"R5C-HISTORY-GROUP-{idx:04d}",
                "dataset": dataset,
                "model_name": model_name,
                "user_id": user_id,
                "row_count": len(group),
                "conditions": ",".join(sorted(conditions, key=lambda value: CONDITION_ORDER.get(value, 99))),
                "expected_conditions": ",".join(sorted(expected_conditions, key=lambda value: CONDITION_ORDER.get(value, 99))),
                "history_lengths": ",".join(str(value) for value in lengths),
                "same_candidate_list_hash": bool_text(same_candidate_list),
                "same_unordered_candidate_set_hash": bool_text(same_candidate_set),
                "same_positive_item": bool_text(same_positive),
                "task_valid": bool_text(task_valid),
                "family_valid": bool_text(family_valid),
                "group_valid": bool_text(valid),
                "notes": "same candidate order/set and held-out positive across controlled history lengths" if valid else "history-length group contract failed",
            }
        )
    return pd.DataFrame(rows)


def build_execution_manifest(history: pd.DataFrame, manifest_sha: str, protocol_hash: str) -> pd.DataFrame:
    keep = [
        "request_id",
        "r5a_request_id",
        "recovery_family",
        "dataset",
        "model_name",
        "model_label",
        "task",
        "user_id",
        "experimental_condition",
        "history_length",
        "history_item_count",
        "history_hash",
        "history_source",
        "candidate_policy",
        "candidate_order_seed",
        "candidate_order_algorithm",
        "candidate_count",
        "candidate_list_hash",
        "unordered_candidate_item_set_hash",
        "positive_item_id",
        "positive_candidate_position",
        "prompt_template_id",
        "prompt_hash",
        "parser_contract",
        "expected_output_schema",
        "temperature",
        "seed",
        "max_new_tokens",
        "output_destination",
        "original_output_destination",
        "request_key_sha256",
        "material_context_hash",
        "gap_ids_served",
        "metrics_served",
    ]
    frame = history[keep].copy()
    frame.insert(0, "phase", PHASE)
    frame.insert(1, "r5a_manifest_sha256", manifest_sha)
    frame.insert(2, "protocol_hash", protocol_hash)
    frame["execution_backend"] = "local_ollama"
    frame["external_api_allowed"] = "false"
    frame["scientific_scope"] = "same_2400_frozen_r5a_history_length_rows_corrected_prompt_only"
    frame["evidence_class"] = EVIDENCE_CLASS
    return frame


def build_output_validity(status: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "request_id",
        "r5a_request_id",
        "dataset",
        "model_name",
        "user_id",
        "experimental_condition",
        "history_length",
        "history_item_count",
        "execution_status",
        "backend_complete",
        "parse_valid",
        "metric_ready",
        "parse_status",
        "validity_class",
        "output_length",
        "expected_output_count",
        "duplicate_count",
        "unknown_item_count",
        "positive_rank",
        "hr_at_5",
        "ndcg_at_5",
        "metric_denominator_policy",
        "result_hash",
    ]
    return status[keep].copy()


def build_metrics_by_condition(status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "model_name", "experimental_condition", "history_length"]
    for key, group in status.groupby(group_cols, sort=True):
        dataset, model_name, condition, history_length = key
        n_manifest = len(group)
        n_terminal = int((~group["execution_status"].eq("NOT_EXECUTED")).sum())
        n_succeeded = int(group["execution_status"].eq("SUCCEEDED").sum())
        n_valid = int(group["parse_valid"].eq("true").sum())
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "experimental_condition": condition,
                "history_length": history_length,
                "n_manifest_requests": n_manifest,
                "n_manifest": n_manifest,
                "n_terminal": n_terminal,
                "n_succeeded": n_succeeded,
                "n_valid_outputs": n_valid,
                "n_valid": n_valid,
                "n_invalid_outputs": int(n_terminal - n_valid),
                "n_invalid": int(n_terminal - n_valid),
                "n_wrong_length": int(group["validity_class"].eq("WRONG_LENGTH").sum()),
                "n_parse_failure": int(group["validity_class"].eq("PARSE_FAILURE").sum()),
                "n_backend_incomplete": int(group["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "n_backend_failed": int(group["execution_status"].eq("BACKEND_FAILED").sum()),
                "n_not_executed": int(group["execution_status"].eq("NOT_EXECUTED").sum()),
                "hr_at_5": float(group["hr_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "HR@5": float(group["hr_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "ndcg_at_5": float(group["ndcg_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "NDCG@5": float(group["ndcg_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
            }
        )
    return pd.DataFrame(rows)


def build_conditional_valid_metrics(status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in status.groupby(["dataset", "model_name", "experimental_condition", "history_length"], sort=True):
        dataset, model_name, condition, history_length = key
        valid = group.loc[group["parse_valid"].eq("true")]
        denom = len(valid)
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "experimental_condition": condition,
                "history_length": history_length,
                "n_valid_outputs": denom,
                "diagnostic_hr_at_5_valid_only": float(valid["hr_at_5"].astype(float).sum() / denom) if denom else 0.0,
                "diagnostic_ndcg_at_5_valid_only": float(valid["ndcg_at_5"].astype(float).sum() / denom) if denom else 0.0,
                "diagnostic_only": "true",
                "notes": "Conditional-valid metrics are diagnostic and separate format compliance from ranking quality.",
            }
        )
    return pd.DataFrame(rows)


def build_history_length_results(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in metrics.to_dict("records"):
        for metric_label, column in [("HR@5", "HR@5"), ("NDCG@5", "NDCG@5")]:
            rows.append(
                {
                    "dataset": record["dataset"],
                    "model_name": record["model_name"],
                    "metric": metric_label,
                    "history_length": record["history_length"],
                    "experimental_condition": record["experimental_condition"],
                    "value": float(record[column]),
                    "n_manifest": int(record["n_manifest"]),
                    "n_valid": int(record["n_valid"]),
                    "n_invalid": int(record["n_invalid"]),
                    "result_type": "level",
                    "denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
                    "evidence_class": EVIDENCE_CLASS,
                }
            )
    for (dataset, model_name), group in metrics.groupby(["dataset", "model_name"], sort=True):
        sortable = group.copy()
        sortable["_history_int"] = sortable["history_length"].astype(str).map(lambda value: int(value) if value.isdigit() else 0)
        if sortable.empty:
            continue
        short = sortable.sort_values("_history_int").iloc[0]
        long = sortable.sort_values("_history_int").iloc[-1]
        for metric_label, column in [("HR@5", "HR@5"), ("NDCG@5", "NDCG@5")]:
            rows.append(
                {
                    "dataset": dataset,
                    "model_name": model_name,
                    "metric": f"{metric_label}_delta_long_minus_short",
                    "history_length": f"{short['history_length']}->{long['history_length']}",
                    "experimental_condition": f"{short['experimental_condition']}->{long['experimental_condition']}",
                    "value": float(long[column]) - float(short[column]),
                    "n_manifest": int(short["n_manifest"]) + int(long["n_manifest"]),
                    "n_valid": int(short["n_valid"]) + int(long["n_valid"]),
                    "n_invalid": int(short["n_invalid"]) + int(long["n_invalid"]),
                    "result_type": "delta",
                    "denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
                    "evidence_class": EVIDENCE_CLASS,
                }
            )
    return pd.DataFrame(rows)


def build_dataset_model_summary(status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model_name), group in status.groupby(["dataset", "model_name"], sort=True):
        terminal = group.loc[~group["execution_status"].eq("NOT_EXECUTED")]
        valid = int(terminal["parse_valid"].eq("true").sum())
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "n_manifest_requests": len(group),
                "n_terminal": len(terminal),
                "n_valid_outputs": valid,
                "n_invalid_outputs": int(len(terminal) - valid),
                "n_wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
                "n_parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
                "n_backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "n_backend_failed": int(terminal["execution_status"].eq("BACKEND_FAILED").sum()),
                "n_not_executed": int(group["execution_status"].eq("NOT_EXECUTED").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_official_comparison(history_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in history_results.loc[history_results["result_type"].eq("level")].to_dict("records"):
        rows.append(
            {
                "dataset": record["dataset"],
                "model_name": record["model_name"],
                "metric": record["metric"],
                "history_length": record["history_length"],
                "local_adapted_value": record["value"],
                "paper_exact": "false",
                "comparison_class": "LOCAL_ADAPTED_NOT_PAPER_EXACT",
                "official_reference_file": "requested_metric_history_length_sensitivity_values.csv",
                "notes": "R5C preserves frozen local R5A scope and corrected exact-5 parser; no paper-exact numeric claim.",
            }
        )
    return pd.DataFrame(rows)


def build_gap_resolution(history: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    for gap_id, group in history.groupby("gap_ids_served", sort=True):
        status_group = status.loc[status["request_id"].isin(set(group["request_id"].astype(str)))]
        terminal_group = status_group.loc[~status_group["execution_status"].eq("NOT_EXECUTED")]
        valid = int(terminal_group["parse_valid"].eq("true").sum())
        rows.append(
            {
                "gap_id": gap_id,
                "dataset": ",".join(sorted(set(group["dataset"].astype(str)))),
                "metric": ",".join(sorted(set(group["metrics_served"].astype(str)))),
                "requests_expected": len(group),
                "requests_accounted": len(terminal_group),
                "valid": valid,
                "invalid": int(len(terminal_group) - valid),
                "wrong_length": int(terminal_group["validity_class"].eq("WRONG_LENGTH").sum()),
                "parse_failure": int(terminal_group["validity_class"].eq("PARSE_FAILURE").sum()),
                "backend_incomplete": int(terminal_group["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "gap_recovered": bool_text(len(terminal_group) == len(group) and valid == len(group)),
                "evidence_class": EVIDENCE_CLASS,
            }
        )
    if terminal.empty:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def file_hash_or_missing(path: Path) -> str:
    return sha256_file(path) if path.exists() else "MISSING"


def protected_artifact_integrity(project_root: Path) -> pd.DataFrame:
    inherited = protected_hash_check(project_root)
    if inherited.empty:
        return inherited
    frame = inherited.copy()
    hash_ok = frame["expected_sha256"].astype(str).str.upper().eq(frame["observed_sha256"].astype(str).str.upper())
    clean_status = frame["git_status"].astype(str).eq("")
    exists = frame["exists"].astype(str).eq("true")
    frame["hash_unchanged"] = [bool_text(value) for value in (exists & hash_ok & clean_status).tolist()]
    frame["protection_status"] = [
        "PROTECTED_UNCHANGED" if value else "PROTECTED_CONFLICT"
        for value in (exists & hash_ok & clean_status).tolist()
    ]
    return frame


def protocol_metadata(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    history: pd.DataFrame,
    manifest_sha: str,
    group_validation: pd.DataFrame,
    runtime_tags: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, pd.DataFrame]:
    module_path = Path(__file__).resolve()
    parser_path = project_root / "src" / "llm" / "canonical_llm_contract.py"
    manifest_builder = project_root / "src" / "analysis" / "reproduction_r5a_controlled_recovery_manifest.py"
    prompt_hashes = sorted(history["prompt_hash"].astype(str).unique())
    prompt_family_hash = sha256_text(canonical_json(prompt_hashes))
    group_hash = sha256_text(canonical_json(group_validation.fillna("").to_dict("records")))
    generation_config = {
        "endpoint": str((config.get("runtime") or {}).get("endpoint", "")),
        "format": str((config.get("runtime") or {}).get("format", "json")),
        "stream": str((config.get("runtime") or {}).get("stream", False)).lower(),
        "temperature": str((config.get("runtime") or {}).get("temperature", 0)),
        "max_retries": str((config.get("runtime") or {}).get("max_retries", 0)),
        "timeout_seconds": str((config.get("runtime") or {}).get("timeout_seconds", 120)),
        "num_predict_values": sorted(history["max_new_tokens"].astype(str).unique()),
        "seed_values": sorted(history["seed"].astype(str).unique()),
    }
    models = []
    for model in sorted(set(history["model_name"].astype(str))):
        details = (runtime_tags or {}).get(model, {})
        models.append(
            {
                "model_name": model,
                "digest": str(details.get("digest", "")),
                "details": details.get("details", {}) if isinstance(details.get("details"), dict) else {},
            }
        )
    payload = {
        "phase": PHASE,
        "manifest_sha256": manifest_sha,
        "selected_rows": len(history),
        "expected_k": expected_k(config),
        "prompt_template_id": R5C_PROMPT_TEMPLATE_ID,
        "prompt_family_hash": prompt_family_hash,
        "parser_contract": R5C_PARSER_CONTRACT,
        "parser_hash": file_hash_or_missing(parser_path),
        "runner_hash": file_hash_or_missing(module_path),
        "config_hash": file_hash_or_missing(config_path),
        "history_candidate_construction_hash": file_hash_or_missing(manifest_builder),
        "metric_hash": file_hash_or_missing(module_path),
        "group_validation_hash": group_hash,
        "generation_config": generation_config,
        "models": models,
    }
    protocol_hash = sha256_text(canonical_json(payload))
    rows = [
        {"component": "manifest", "value": manifest_sha, "sha256": manifest_sha},
        {"component": "prompt_family", "value": R5C_PROMPT_TEMPLATE_ID, "sha256": prompt_family_hash},
        {"component": "parser", "value": R5C_PARSER_CONTRACT, "sha256": payload["parser_hash"]},
        {"component": "runner", "value": normalize_rel(module_path), "sha256": payload["runner_hash"]},
        {"component": "config", "value": normalize_rel(config_path), "sha256": payload["config_hash"]},
        {"component": "history_candidate_construction", "value": "R5A frozen history-length rows", "sha256": payload["history_candidate_construction_hash"]},
        {"component": "group_validation", "value": "same candidates and positive item across history lengths", "sha256": group_hash},
        {"component": "metrics", "value": "HR@5/NDCG@5 by history length", "sha256": payload["metric_hash"]},
        {"component": "generation_config", "value": canonical_json(generation_config), "sha256": sha256_text(canonical_json(generation_config))},
        {"component": "protocol", "value": "R5C_PROTOCOL_PREFLIGHT_READY", "sha256": protocol_hash},
    ]
    for model in models:
        rows.append({"component": "model_identifier", "value": model["model_name"], "sha256": str(model.get("digest", ""))})
    return protocol_hash, pd.DataFrame(rows)


def context_length_for_model(config: Mapping[str, Any], runtime_tags: Mapping[str, Mapping[str, Any]], model: str) -> int:
    details = runtime_tags.get(model, {}).get("details", {})
    if isinstance(details, dict):
        for key in ["context_length", "num_ctx"]:
            value = details.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return int((config.get("runtime") or {}).get("assumed_context_length", 32768))


def approximate_tokens(text: str) -> int:
    return max(1, (len(str(text)) + 3) // 4)


def prompt_section_lengths(prompt: str) -> tuple[int, int]:
    text = str(prompt)
    history_part = ""
    candidate_part = ""
    if "\nUser history:\n" in text:
        after_history = text.split("\nUser history:\n", 1)[1]
        if "\n\nCandidates:\n" in after_history:
            history_part, candidate_part = after_history.split("\n\nCandidates:\n", 1)
    return len(history_part), len(candidate_part)


def build_context_preflight(
    history: pd.DataFrame,
    config: Mapping[str, Any],
    runtime_tags: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    runtime = config.get("runtime") or {}
    min_margin = int(runtime.get("min_context_margin_tokens", 1024))
    rows: list[dict[str, Any]] = []
    for key, group in history.groupby(["model_name", "dataset", "experimental_condition"], sort=True):
        model_name, dataset, condition = key
        records = group.to_dict("records")
        max_prompt = max(records, key=lambda row: len(str(row["prompt_render"])))
        history_chars, candidate_chars = prompt_section_lengths(str(max_prompt["prompt_render"]))
        input_tokens = approximate_tokens(str(max_prompt["prompt_render"]))
        output_budget = int(max_prompt.get("max_new_tokens", 512) or 512)
        context_length = context_length_for_model(config, runtime_tags, str(model_name))
        margin = context_length - input_tokens - output_budget
        rows.append(
            {
                "scope": "model_dataset_condition",
                "model_name": model_name,
                "dataset": dataset,
                "experimental_condition": condition,
                "history_length": str(max_prompt.get("history_length", "")),
                "max_prompt_chars": len(str(max_prompt["prompt_render"])),
                "max_prompt_words": len(str(max_prompt["prompt_render"]).split()),
                "history_section_chars": history_chars,
                "candidate_section_chars": candidate_chars,
                "approx_input_tokens": input_tokens,
                "output_budget_tokens": output_budget,
                "context_length": context_length,
                "margin_tokens": margin,
                "status": "pass" if margin >= min_margin else "fail",
                "worst_case_request_id": str(max_prompt["request_id"]),
                "notes": "approximate character envelope; conservative preflight guard",
            }
        )
    for key, group in history.groupby(["model_name", "dataset"], sort=True):
        model_name, dataset = key
        by_len = group.copy()
        by_len["_history_int"] = by_len["history_length"].astype(str).map(lambda value: int(value) if value.isdigit() else 0)
        short_history = int(by_len["_history_int"].min())
        long_history = int(by_len["_history_int"].max())
        short_group = by_len.loc[by_len["_history_int"].eq(short_history)]
        long_group = by_len.loc[by_len["_history_int"].eq(long_history)]
        long = long_group.loc[long_group["prompt_render"].astype(str).map(len).idxmax()]
        short_max_items = int(short_group["history_item_count"].astype(int).max())
        long_max_items = int(long_group["history_item_count"].astype(int).max())
        long_len = len(str(long["prompt_render"]))
        monotonic_items = long_history >= short_history and long_max_items >= short_max_items
        rows.append(
            {
                "scope": "long_vs_short",
                "model_name": model_name,
                "dataset": dataset,
                "experimental_condition": f"HISTORY_LENGTH_{short_history}->HISTORY_LENGTH_{long_history}",
                "history_length": f"{short_history}->{long_history}",
                "max_prompt_chars": long_len,
                "max_prompt_words": len(str(long["prompt_render"]).split()),
                "history_section_chars": prompt_section_lengths(str(long["prompt_render"]))[0],
                "candidate_section_chars": prompt_section_lengths(str(long["prompt_render"]))[1],
                "approx_input_tokens": approximate_tokens(str(long["prompt_render"])),
                "output_budget_tokens": int(long.get("max_new_tokens", 512) or 512),
                "context_length": context_length_for_model(config, runtime_tags, str(model_name)),
                "margin_tokens": context_length_for_model(config, runtime_tags, str(model_name)) - approximate_tokens(str(long["prompt_render"])) - int(long.get("max_new_tokens", 512) or 512),
                "status": "pass" if monotonic_items else "fail",
                "worst_case_request_id": str(long["request_id"]),
                "notes": "configured long condition has equal or greater history item count than shortest condition",
            }
        )
    return pd.DataFrame(rows)


def build_static_preflight(
    project_root: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    history: pd.DataFrame,
    manifest_sha: str,
    group_validation: pd.DataFrame,
    context_preflight: pd.DataFrame,
    protocol_hash: str,
) -> pd.DataFrame:
    scope = config.get("scope") or {}
    runtime = config.get("runtime") or {}
    outputs = config.get("outputs") or {}
    namespace = normalize_rel(outputs["result_namespace"]).rstrip("/") + "/"
    old_outputs = set(history["original_output_destination"].astype(str).map(normalize_rel))
    new_outputs = set(history["output_destination"].astype(str).map(normalize_rel))
    prompts = history["prompt_render"].astype(str)
    original_repo_status = git_stdout(["-C", "original_repo/EvaLLMasRecommender", "status", "--short"], project_root)
    head = git_stdout(["rev-parse", "HEAD"], project_root)
    disk_free = shutil.disk_usage(project_root).free
    protected = protected_artifact_integrity(project_root)
    r5b_freeze = project_path(project_root, ((config.get("frozen_r5b_corrected") or {}).get("freeze_manifest", "")))
    r5b_protocol = ""
    if r5b_freeze.exists():
        r5b_protocol = str(load_json(r5b_freeze).get("r5b_corrected_protocol_hash", "") or load_json(r5b_freeze).get("protocol_hash", ""))
    expected_by_dataset = configured_conditions_by_dataset(config)
    cell_counts = history.groupby(["dataset", "model_name", "experimental_condition"], sort=True).size()
    group_counts = history.groupby(["dataset", "model_name", "user_id"], sort=True).size()
    out_of_scope = manifest.loc[~manifest["recovery_family"].eq("HISTORY_LENGTH_RECOVERY")]
    checks: list[tuple[str, str, bool, str, str]] = [
        ("STATIC-001", "manifest SHA", manifest_sha == EXPECTED_R5A_MANIFEST_SHA256, manifest_sha, EXPECTED_R5A_MANIFEST_SHA256),
        ("STATIC-002", "repository HEAD is R5B corrected freeze", head == str((config.get("frozen_r5b_corrected") or {}).get("freeze_commit", R5B_CORRECTED_FREEZE_COMMIT)), head, str((config.get("frozen_r5b_corrected") or {}).get("freeze_commit", R5B_CORRECTED_FREEZE_COMMIT))),
        ("STATIC-003", "R5B corrected freeze manifest exists", r5b_freeze.exists(), normalize_rel(r5b_freeze), "exists"),
        ("STATIC-004", "R5B corrected protocol hash", r5b_protocol == R5B_CORRECTED_PROTOCOL_HASH, r5b_protocol, R5B_CORRECTED_PROTOCOL_HASH),
        ("STATIC-005", "manifest rows", len(manifest) == int((config.get("frozen_r5a") or {}).get("manifest_rows", 4800)), str(len(manifest)), "4800"),
        ("STATIC-006", "selected history rows", len(history) == int(scope.get("selected_rows", 2400)), str(len(history)), "2400"),
        ("STATIC-007", "out-of-scope rows remain unselected", len(out_of_scope) == 2400, str(len(out_of_scope)), "2400"),
        ("STATIC-008", "no CandDif rows selected", not history["recovery_family"].eq("CANDDIF_RECOVERY").any(), str(int(history["recovery_family"].eq("CANDDIF_RECOVERY").sum())), "0"),
        ("STATIC-009", "no Generation rows selected", not history["recovery_family"].eq("GENERATION_INVOLVED_RECOVERY").any(), str(int(history["recovery_family"].eq("GENERATION_INVOLVED_RECOVERY").sum())), "0"),
        ("STATIC-010", "ranking task only", history["task"].eq("ranking").all(), str(sorted(set(history["task"]))), "ranking"),
        ("STATIC-011", "datasets", set(history["dataset"]) == set(scope.get("datasets") or []), str(sorted(set(history["dataset"]))), str(sorted(scope.get("datasets") or []))),
        ("STATIC-012", "models", set(history["model_name"]) == set(scope.get("models") or []), str(sorted(set(history["model_name"]))), str(sorted(scope.get("models") or []))),
        ("STATIC-013", "expected cell count", len(cell_counts) == int(scope.get("expected_cells", 24)) and cell_counts.eq(100).all(), f"{len(cell_counts)} cells; min={int(cell_counts.min())}; max={int(cell_counts.max())}", "24 cells of 100"),
        ("STATIC-014", "expected group count", len(group_counts) == int(scope.get("expected_groups", 800)) and group_counts.eq(3).all(), f"{len(group_counts)} groups; min={int(group_counts.min())}; max={int(group_counts.max())}", "800 groups of 3"),
        ("STATIC-015", "conditions by dataset", all(set(history.loc[history["dataset"].eq(dataset), "experimental_condition"].astype(str)) == conditions for dataset, conditions in expected_by_dataset.items()), str({dataset: sorted(set(history.loc[history["dataset"].eq(dataset), "experimental_condition"].astype(str))) for dataset in expected_by_dataset}), str({k: sorted(v) for k, v in expected_by_dataset.items()})),
        ("STATIC-016", "expected K is 5", history.apply(parse_expected_count, axis=1).eq(5).all(), str(sorted(set(history.apply(parse_expected_count, axis=1)))), "[5]"),
        ("STATIC-017", "candidate count 20", history["candidate_count"].astype(int).eq(20).all(), str(sorted(set(history["candidate_count"].astype(str)))), "20"),
        ("STATIC-018", "group validation", len(group_validation) == int(scope.get("expected_groups", 800)) and group_validation["group_valid"].eq("true").all(), f"{int(group_validation['group_valid'].eq('true').sum())}/{len(group_validation)}", "800/800"),
        ("STATIC-019", "corrected prompt says EXACTLY 5", prompts.str.contains("EXACTLY 5", regex=False).all(), str(int(prompts.str.contains("EXACTLY 5", regex=False).sum())), str(len(history))),
        ("STATIC-020", "corrected prompt forbids more/fewer", prompts.str.contains("Do not return more than 5", regex=False).all() and prompts.str.contains("Do not return fewer than 5", regex=False).all(), "checked", "all prompts"),
        ("STATIC-021", "corrected prompt forbids prose", prompts.str.contains("Do not include rationale", regex=False).all(), "checked", "all prompts"),
        ("STATIC-022", "parser contract corrected", set(history["parser_contract"]) == {R5C_PARSER_CONTRACT}, str(sorted(set(history["parser_contract"]))), R5C_PARSER_CONTRACT),
        ("STATIC-023", "local endpoint only", str(runtime.get("endpoint", "")).startswith("http://127.0.0.1") or str(runtime.get("endpoint", "")).startswith("http://localhost"), str(runtime.get("endpoint", "")), "localhost Ollama"),
        ("STATIC-024", "format json stream false", str(runtime.get("format", "")) == "json" and bool(runtime.get("stream", False)) is False, f"format={runtime.get('format')};stream={runtime.get('stream')}", "format=json;stream=false"),
        ("STATIC-025", "no retry policy introduced", int(runtime.get("max_retries", 0)) == 0, str(runtime.get("max_retries", 0)), "0"),
        ("STATIC-026", "output namespace R5C", namespace == "results/recovery/r5c_history_length/", namespace, "results/recovery/r5c_history_length/"),
        ("STATIC-027", "no output path overlap with original R5", old_outputs.isdisjoint(new_outputs), str(len(old_outputs.intersection(new_outputs))), "0"),
        ("STATIC-028", "no R5C output under R5/R5B namespaces", not any(path.startswith(ORIGINAL_R5_NAMESPACE) or path.startswith(R5B_CORRECTED_NAMESPACE) for path in new_outputs), "checked", "none"),
        ("STATIC-029", "original repo clean", original_repo_status == "", original_repo_status, "empty"),
        ("STATIC-030", "protected artifacts unchanged", protected.empty or protected["protection_status"].eq("PROTECTED_UNCHANGED").all(), f"{int(protected['protection_status'].eq('PROTECTED_UNCHANGED').sum())}/{len(protected)}", f"{len(protected)}/{len(protected)}"),
        ("STATIC-031", "disk free guard", disk_free >= int(runtime.get("disk_free_min_bytes", 0)), str(disk_free), f">={runtime.get('disk_free_min_bytes', 0)}"),
        ("STATIC-032", "context envelope", context_preflight["status"].eq("pass").all(), f"{int(context_preflight['status'].eq('pass').sum())}/{len(context_preflight)}", "all pass"),
        ("STATIC-033", "protocol hash present", len(protocol_hash) == 64, protocol_hash, "64-char SHA256"),
    ]
    return pd.DataFrame(
        [
            {"check_id": check_id, "check_name": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected}
            for check_id, name, ok, observed, expected in checks
        ]
    )


def build_synthetic_tests(history: pd.DataFrame) -> pd.DataFrame:
    row = history.iloc[0].to_dict()
    empty_history_rows = history.loc[history["history_length"].astype(str).eq("0")]
    empty_row = empty_history_rows.iloc[0].to_dict() if not empty_history_rows.empty else row
    candidate_ids = candidate_item_ids(row)
    cases = [
        ("valid_json_object_indices", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5]}), True, "parsed_valid", row),
        ("valid_json_multiline", "{\n  \"ranked_candidate_indices\": [1, 2, 3, 4, 5]\n}", True, "parsed_valid", row),
        ("k_minus_1", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4]}), False, "invalid_length", row),
        ("k_plus_1", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5, 6]}), False, "invalid_length", row),
        ("numbered_ranking", "1. 342\n2. 177\n3. 901\n4. 55\n5. 20", False, "parse_failed", row),
        ("plain_json_array_candidate_ids", json.dumps(candidate_ids[:5]), False, "invalid_schema", row),
        ("leading_prose", "Here is the ranking: " + canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5]}), False, "parse_failed", row),
        ("trailing_prose", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5]}) + "\nDone.", False, "parse_failed", row),
        ("duplicate_index", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 4]}), False, "duplicate_indices", row),
        ("candidate_set_violation", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 21]}), False, "unknown_indices", row),
        ("truncated_output", '{"ranked_candidate_indices":[1,2,3', False, "parse_failed", row),
        ("empty_history_row_valid", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5]}), True, "parsed_valid", empty_row),
        ("backend_incomplete_response", '{"ranked_candidate_indices":[1,2,3', False, "backend_incomplete", row),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, text, expected_valid, expected_status, case_row in cases:
        if case_id == "backend_incomplete_response":
            parsed_record = incomplete_parse_record(case_row, text)
            observed_valid = bool(parsed_record["parse_valid"])
            observed_status = str(parsed_record["parse_status"])
            output_length = int(parsed_record["output_length"])
            duplicate_count = int(parsed_record["duplicate_count"])
            unknown_count = int(parsed_record["unknown_item_count"])
        else:
            parsed = parse_strict_index_response(
                text,
                candidate_items=candidate_item_ids(case_row),
                expected_count=5,
                require_full_permutation=False,
            )
            observed_valid = bool(parsed.parse_valid)
            observed_status = parsed.status
            output_length = parsed.output_length
            duplicate_count = parsed.duplicate_count
            unknown_count = parsed.unknown_item_count
        rows.append(
            {
                "case_id": case_id,
                "expected_valid": bool_text(expected_valid),
                "observed_valid": bool_text(observed_valid),
                "expected_status": expected_status,
                "observed_status": observed_status,
                "output_length": output_length,
                "duplicate_count": duplicate_count,
                "unknown_item_count": unknown_count,
                "pass": bool_text(observed_valid == expected_valid and observed_status == expected_status),
                "notes": "numbered lines must not be converted into candidate indices" if case_id == "numbered_ranking" else "",
            }
        )
    return pd.DataFrame(rows)


def health_summary(status: pd.DataFrame, *, label: str) -> pd.DataFrame:
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    rows: list[dict[str, Any]] = []

    def add(scope: str, key: str, group: pd.DataFrame) -> None:
        total = len(group)
        valid = int(group["parse_valid"].eq("true").sum())
        rows.append(
            {
                "health_check": label,
                "scope": scope,
                "key": key,
                "accounted": total,
                "valid": valid,
                "invalid": total - valid,
                "validity_rate": float(valid / total) if total else 0.0,
                "wrong_length": int(group["validity_class"].eq("WRONG_LENGTH").sum()),
                "parse_failure": int(group["validity_class"].eq("PARSE_FAILURE").sum()),
                "backend_incomplete": int(group["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "backend_failed": int(group["execution_status"].eq("BACKEND_FAILED").sum()),
                "duplicate": int(group["validity_class"].eq("DUPLICATE").sum()),
                "out_of_range": int(group["validity_class"].eq("OUT_OF_RANGE").sum()),
            }
        )

    add("overall", "all", terminal)
    for model, group in terminal.groupby("model_name", sort=True):
        add("model", str(model), group)
    for dataset, group in terminal.groupby("dataset", sort=True):
        add("dataset", str(dataset), group)
    for condition, group in terminal.groupby("experimental_condition", sort=True):
        add("condition", str(condition), group)
    for history_length, group in terminal.groupby("history_length", sort=True):
        add("history_length", str(history_length), group)
    for key, group in terminal.groupby(["model_name", "dataset", "experimental_condition"], sort=True):
        add("model_dataset_condition", "|".join(map(str, key)), group)
    return pd.DataFrame(rows)


def health_pass(health: pd.DataFrame, config: Mapping[str, Any]) -> tuple[bool, str]:
    gates = config.get("gates") or {}
    if health.empty:
        return False, "no terminal outputs"
    overall = health.loc[health["scope"].eq("overall")].iloc[0]
    min_overall = float(gates.get("min_overall_validity_rate", 0.75))
    min_model = float(gates.get("min_model_validity_rate", 0.65))
    max_wrong = float(gates.get("max_overall_wrong_length_rate", 0.20))
    max_cell_wrong = float(gates.get("max_cell_wrong_length_rate", 0.50))
    if float(overall["validity_rate"]) < min_overall:
        return False, f"overall validity {overall['validity_rate']} < {min_overall}"
    if int(overall["accounted"]) and int(overall["wrong_length"]) / int(overall["accounted"]) > max_wrong:
        return False, f"overall wrong_length rate exceeds {max_wrong}"
    if int(overall["backend_incomplete"]) > 0:
        return False, "backend incomplete present"
    model_rows = health.loc[health["scope"].eq("model")]
    for row in model_rows.to_dict("records"):
        if int(row["accounted"]) and float(row["validity_rate"]) < min_model:
            return False, f"model {row['key']} validity {row['validity_rate']} < {min_model}"
    cell_rows = health.loc[health["scope"].eq("model_dataset_condition")]
    for row in cell_rows.to_dict("records"):
        accounted = int(row["accounted"])
        if accounted and int(row["wrong_length"]) / accounted > max_cell_wrong:
            return False, f"cell {row['key']} wrong_length rate exceeds {max_cell_wrong}"
    return True, ""


def canary_pass(status: pd.DataFrame, canary_ids: set[str], config: Mapping[str, Any]) -> tuple[bool, str, pd.DataFrame]:
    canary = status.loc[status["request_id"].isin(canary_ids)].copy()
    table = canary[
        [
            "model_name",
            "dataset",
            "experimental_condition",
            "history_length",
            "backend_complete",
            "output_length",
            "parse_valid",
            "validity_class",
            "parse_status",
            "request_id",
            "r5a_request_id",
        ]
    ].rename(
        columns={
            "model_name": "Model",
            "dataset": "Dataset",
            "experimental_condition": "Condition",
            "history_length": "HistoryLength",
            "backend_complete": "Complete",
            "output_length": "Parsed K",
            "parse_valid": "Valid",
            "validity_class": "Failure",
        }
    )
    expected_cells = int((config.get("scope") or {}).get("expected_cells", 24))
    if len(canary) != expected_cells:
        return False, f"canary accounted {len(canary)} != {expected_cells}", table
    failed = canary.loc[~canary["parse_valid"].eq("true")]
    if not failed.empty:
        return False, f"canary invalid outputs {len(failed)}", table
    health = health_summary(canary, label="canary")
    ok, reason = health_pass(health, config)
    return ok, reason, table


def write_diagnostic_frame(project_root: Path, config: Mapping[str, Any], frame: pd.DataFrame, name: str) -> None:
    write_csv(frame, diagnostics_dir(project_root, config) / name)


def build_runtime_provenance(
    project_root: Path,
    config: Mapping[str, Any],
    status: pd.DataFrame,
    runtime_available: bool,
    runtime_error: str,
    runtime_tags: Mapping[str, Mapping[str, Any]],
    protocol_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    version = ollama_version(config) if runtime_available else "unavailable"
    required_models = sorted(set((config.get("scope") or {}).get("models") or []))
    disk_usage = shutil.disk_usage(project_root)
    for model in required_models:
        details = runtime_tags.get(model, {})
        model_details = details.get("details", {}) if isinstance(details.get("details"), dict) else {}
        model_status = status.loc[status["model_name"].eq(model)] if not status.empty else pd.DataFrame()
        rows.append(
            {
                "phase": PHASE,
                "protocol_hash": protocol_hash,
                "backend": "ollama",
                "ollama_version": version,
                "model_name": model,
                "model_available": bool_text(model in runtime_tags),
                "model_digest": str(details.get("digest", "")),
                "model_parameter_size": str(model_details.get("parameter_size", "")),
                "model_quantization_level": str(model_details.get("quantization_level", "")),
                "context_length": str(context_length_for_model(config, runtime_tags, model)),
                "runtime_available": bool_text(runtime_available),
                "runtime_error": runtime_error,
                "request_count": len(model_status),
                "succeeded": int(model_status["execution_status"].eq("SUCCEEDED").sum()) if not model_status.empty else 0,
                "backend_incomplete": int(model_status["execution_status"].eq("BACKEND_INCOMPLETE").sum()) if not model_status.empty else 0,
                "backend_failed": int(model_status["execution_status"].eq("BACKEND_FAILED").sum()) if not model_status.empty else 0,
                "not_executed": int(model_status["execution_status"].eq("NOT_EXECUTED").sum()) if not model_status.empty else 0,
                "git_head": git_stdout(["rev-parse", "HEAD"], project_root),
                "disk_free_bytes": disk_usage.free,
                "external_api_allowed": "false",
                "downloads_allowed": "false",
                "training_allowed": "false",
            }
        )
    return pd.DataFrame(rows)


def build_validation(
    static_preflight: pd.DataFrame,
    synthetic_tests: pd.DataFrame,
    context_preflight: pd.DataFrame,
    canary_table: pd.DataFrame,
    early_health: pd.DataFrame,
    status: pd.DataFrame,
    group_validation: pd.DataFrame,
    *,
    full_run_authorized: bool,
    decision: str,
) -> pd.DataFrame:
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    expected_total = 2400
    checks = [
        ("VAL-001", "static preflight", static_preflight["status"].eq("pass").all(), f"{int(static_preflight['status'].eq('pass').sum())}/{len(static_preflight)}", "all pass"),
        ("VAL-002", "synthetic tests", synthetic_tests["pass"].eq("true").all(), f"{int(synthetic_tests['pass'].eq('true').sum())}/{len(synthetic_tests)}", "all pass"),
        ("VAL-003", "context preflight", context_preflight["status"].eq("pass").all(), f"{int(context_preflight['status'].eq('pass').sum())}/{len(context_preflight)}", "all pass"),
        ("VAL-004", "canary complete when attempted", canary_table.empty or canary_table["Complete"].eq("true").all(), f"{int(canary_table.get('Complete', pd.Series(dtype=str)).eq('true').sum())}/{len(canary_table)}", "all complete"),
        ("VAL-005", "canary parsed K=5 when attempted", canary_table.empty or canary_table["Parsed K"].astype(str).eq("5").all(), f"{int(canary_table.get('Parsed K', pd.Series(dtype=str)).astype(str).eq('5').sum())}/{len(canary_table)}", "all 5"),
        ("VAL-006", "early health no backend incomplete", early_health.empty or int(early_health.loc[early_health["scope"].eq("overall"), "backend_incomplete"].iloc[0]) == 0, "checked", "0"),
        ("VAL-007", "group validation", len(group_validation) == 800 and group_validation["group_valid"].eq("true").all(), f"{int(group_validation['group_valid'].eq('true').sum())}/{len(group_validation)}", "800/800"),
        ("VAL-008", "incomplete not succeeded", not ((status["execution_status"].eq("SUCCEEDED")) & (status["backend_complete"].ne("true"))).any(), "checked", "none"),
        ("VAL-009", "full run authorization consistent", (full_run_authorized and decision in {DECISION_COMPLETE, DECISION_EARLY_PASS} and (decision != DECISION_COMPLETE or len(terminal) == expected_total)) or (not full_run_authorized and decision != DECISION_COMPLETE) or (decision == DECISION_EXECUTION_FAILED), f"authorized={full_run_authorized};decision={decision};terminal={len(terminal)}", "consistent"),
    ]
    return pd.DataFrame(
        [
            {"check_id": check_id, "check_name": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected}
            for check_id, name, ok, observed, expected in checks
        ]
    )


def render_markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    selected = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected.columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]).replace("\n", " ") for column in selected.columns) + " |"
        for row in selected.to_dict("records")
    ]
    return "\n".join([header, separator, *body])


def render_report(
    frames: Mapping[str, pd.DataFrame],
    decision: str,
    manifest_sha: str,
    protocol_hash: str,
    full_run_authorized: bool,
    blocked_reason: str,
    executed_this_run: int,
) -> str:
    status = frames["request_execution_status"]
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    valid = int(terminal["parse_valid"].eq("true").sum())
    invalid = int(len(terminal) - valid)
    report = f"""# R5C History-Length Recovery

## Scope

R5C executes only frozen R5A `HISTORY_LENGTH_RECOVERY` ranking rows. CandDif, Generation-involved, profile-generation, downloads, training, remote APIs, R5D, and `original_repo/` edits are out of scope.

## R5B corrected freeze dependency

- R5B corrected freeze commit: `{R5B_CORRECTED_FREEZE_COMMIT}`
- R5B corrected protocol hash: `{R5B_CORRECTED_PROTOCOL_HASH}`
- R5B original namespace preserved: `results/recovery/r5/`
- R5B corrected namespace preserved: `results/recovery/r5b_corrected/`

## Frozen manifest verification

Manifest SHA256: `{manifest_sha}`.

Selected R5C rows: `{len(status)}`.

## Static preflight

{render_markdown_table(frames.get("static_preflight", pd.DataFrame()), ["check_id", "check_name", "status", "observed", "expected"])}

## Synthetic tests

{render_markdown_table(frames.get("synthetic_tests", pd.DataFrame()), ["case_id", "expected_valid", "observed_valid", "expected_status", "observed_status", "pass"])}

## Context envelope

{render_markdown_table(frames.get("context_preflight", pd.DataFrame()), ["scope", "model_name", "dataset", "experimental_condition", "history_length", "approx_input_tokens", "output_budget_tokens", "context_length", "margin_tokens", "status"])}

## Protocol freeze

```text
R5C_PROTOCOL_PREFLIGHT_READY
PROTOCOL_HASH={protocol_hash}
```

{render_markdown_table(frames.get("protocol_freeze", pd.DataFrame()), ["component", "value", "sha256"])}

## Canary results

{render_markdown_table(frames.get("canary_results", pd.DataFrame()), ["Model", "Dataset", "Condition", "HistoryLength", "Complete", "Parsed K", "Valid", "Failure", "request_id", "r5a_request_id"])}

## Early health check

{render_markdown_table(frames.get("early_health", pd.DataFrame()), ["scope", "key", "accounted", "valid", "invalid", "validity_rate", "wrong_length", "parse_failure", "backend_incomplete"])}

## Full-run authorization

FULL RUN AUTHORIZED: `{"YES" if full_run_authorized else "NO"}`.

Blocked reason: `{blocked_reason}`.

## Execution accounting

- Requests expected: `2400`
- Requests accounted: `{len(terminal)}`
- Executed this invocation: `{executed_this_run}`
- Valid: `{valid}`
- Invalid: `{invalid}`
- Wrong length: `{int(terminal["validity_class"].eq("WRONG_LENGTH").sum())}`
- Parse failure: `{int(terminal["validity_class"].eq("PARSE_FAILURE").sum())}`
- Backend incomplete: `{int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum())}`

## Output validity

{render_markdown_table(frames["dataset_model_summary"], ["dataset", "model_name", "n_manifest_requests", "n_terminal", "n_valid_outputs", "n_invalid_outputs", "n_wrong_length", "n_parse_failure", "n_backend_incomplete"])}

## Metrics by history length

{render_markdown_table(frames["metrics_by_condition"], ["dataset", "model_name", "experimental_condition", "history_length", "n_manifest", "n_valid", "n_invalid", "HR@5", "NDCG@5"])}

## History-length results

{render_markdown_table(frames["history_length_results"], ["dataset", "model_name", "metric", "history_length", "experimental_condition", "value", "result_type"])}

## Official comparison

{render_markdown_table(frames["official_comparison"], ["dataset", "model_name", "metric", "history_length", "local_adapted_value", "paper_exact", "comparison_class"])}

## Gap resolution

{render_markdown_table(frames["gap_resolution"], ["gap_id", "dataset", "metric", "requests_expected", "requests_accounted", "valid", "invalid", "gap_recovered"])}

## Scientific interpretation

R5C is a local adapted history-length recovery over frozen R5A candidate/history contexts. It is not paper-exact because the local model identifiers and corrected strict JSON index protocol differ from the paper execution environment.

## Validation

{render_markdown_table(frames["validation"], ["check_id", "check_name", "status", "observed", "expected"])}

## Repository state

This runner does not commit or push.

## Final decision

`{decision}`
"""
    return report


def update_notes(project_root: Path, decision: str, manifest_sha: str, protocol_hash: str, status: pd.DataFrame) -> None:
    notes_path = project_root / NOTES_PATH
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    section = f"""
## R5C History-Length recovery

- Decision: `{decision}`.
- Manifest SHA256: `{manifest_sha}`.
- Protocol hash: `{protocol_hash}`.
- Scope: new `R5C_HISTORY_LENGTH` identity over 2400 frozen R5A `HISTORY_LENGTH_RECOVERY` rows; no CandDif, Generation, R5D, external API, download, training, push, or `original_repo/` modification.
- Outputs: `results/recovery/r5c_history_length/`, `results/diagnostics/r5c_history_length/`, `results/tables/reproduction_r5c_history_length_*.csv`, and `report/reproduction_r5c_history_length_recovery.md`.
- Request accounting: expected `2400`, accounted `{len(terminal)}`, valid `{int(terminal["parse_valid"].eq("true").sum())}`, invalid `{int(len(terminal) - terminal["parse_valid"].eq("true").sum())}`.
- Backend incomplete handling: incomplete Ollama responses are classified as `BACKEND_INCOMPLETE`, not `SUCCEEDED`.
"""
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "## R5C History-Length recovery"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n" + section
    else:
        existing = existing.rstrip() + "\n" + section
    notes_path.write_text(existing.lstrip(), encoding="utf-8")


def rebuild_frames(
    project_root: Path,
    config: Mapping[str, Any],
    history: pd.DataFrame,
    manifest_sha: str,
    protocol_hash: str,
    runtime_available: bool,
    runtime_error: str,
    runtime_tags: Mapping[str, Mapping[str, Any]],
    group_validation: pd.DataFrame,
    static_preflight: pd.DataFrame,
    synthetic_tests: pd.DataFrame,
    context_preflight: pd.DataFrame,
    protocol_freeze: pd.DataFrame,
    canary_results: pd.DataFrame | None,
    early_health: pd.DataFrame | None,
    decision: str,
    full_run_authorized: bool,
) -> dict[str, pd.DataFrame]:
    execution_manifest = build_execution_manifest(history, manifest_sha, protocol_hash)
    status = build_status(project_root, history, manifest_sha, protocol_hash)
    output_validity = build_output_validity(status)
    metrics = build_metrics_by_condition(status)
    history_results = build_history_length_results(metrics)
    conditional = build_conditional_valid_metrics(status)
    summary = build_dataset_model_summary(status)
    official = build_official_comparison(history_results)
    gaps = build_gap_resolution(history, status)
    runtime = build_runtime_provenance(project_root, config, status, runtime_available, runtime_error, runtime_tags, protocol_hash)
    canary_frame = canary_results if canary_results is not None else pd.DataFrame()
    early_frame = early_health if early_health is not None else pd.DataFrame()
    validation = build_validation(
        static_preflight,
        synthetic_tests,
        context_preflight,
        canary_frame,
        early_frame,
        status,
        group_validation,
        full_run_authorized=full_run_authorized,
        decision=decision,
    )
    return {
        "execution_manifest": execution_manifest,
        "request_execution_status": status,
        "group_validation": group_validation,
        "output_validity": output_validity,
        "metrics_by_condition": metrics,
        "history_length_results": history_results,
        "conditional_valid_metrics": conditional,
        "dataset_model_summary": summary,
        "official_comparison": official,
        "gap_resolution": gaps,
        "runtime_provenance": runtime,
        "validation": validation,
        "static_preflight": static_preflight,
        "synthetic_tests": synthetic_tests,
        "context_preflight": context_preflight,
        "protocol_freeze": protocol_freeze,
        "canary_results": canary_frame,
        "early_health": early_frame,
    }


def write_freeze_candidate(
    project_root: Path,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest_sha: str,
    protocol_hash: str,
    decision: str,
) -> None:
    status = frames["request_execution_status"]
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    namespace = r5c_namespace(project_root, config)
    raw_paths = sorted(namespace.glob("*.json")) if namespace.exists() else []
    aggregate_hash = sha256_text(canonical_json([sha256_file(path) for path in raw_paths]))
    payload = {
        "freeze_metadata_version": "r5c_history_length_freeze_candidate_v1",
        "phase": PHASE,
        "decision": decision,
        "created_utc": utc_now(),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": protocol_hash,
        "r5b_corrected_freeze_commit": R5B_CORRECTED_FREEZE_COMMIT,
        "r5b_corrected_protocol_hash": R5B_CORRECTED_PROTOCOL_HASH,
        "raw_namespace": normalize_rel((config.get("outputs") or {})["result_namespace"]),
        "raw_count": len(raw_paths),
        "raw_aggregate_sha256": aggregate_hash,
        "status_counts": {
            "requests_expected": len(status),
            "requests_accounted": len(terminal),
            "valid": int(terminal["parse_valid"].eq("true").sum()),
            "invalid": int(len(terminal) - terminal["parse_valid"].eq("true").sum()),
            "wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
            "parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
            "backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
        },
        "paper_exact": False,
        "freeze_ready": bool(decision == DECISION_COMPLETE and len(terminal) == len(status) and int(terminal["parse_valid"].eq("true").sum()) == len(status)),
    }
    destination = project_root / "results" / "manifests" / "reproduction_r5c_history_length_freeze_candidate.json"
    write_json_atomic(payload, destination)


def write_outputs(
    project_root: Path,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    report: str,
    manifest_sha: str,
    protocol_hash: str,
    decision: str,
) -> None:
    outputs = configured_outputs(config)
    for key in OUTPUT_KEYS:
        write_csv(frames[key], project_root / outputs[key])
    diag = diagnostics_dir(project_root, config)
    diag.mkdir(parents=True, exist_ok=True)
    for key in ["static_preflight", "synthetic_tests", "context_preflight", "protocol_freeze", "canary_results", "early_health"]:
        write_csv(frames.get(key, pd.DataFrame()), diag / f"{key}.csv")
    report_path = project_path(project_root, (config.get("outputs") or {})["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_freeze_candidate(project_root, config, frames, manifest_sha, protocol_hash, decision)


def build_r5c_history_length(
    project_root: Path,
    config_path: Path | None = None,
    *,
    write: bool = True,
    run_gated: bool = False,
    allow_api_calls: bool = False,
    allow_local_ollama: bool = False,
    stop_after: str = "full",
) -> R5CBuildResult:
    actual_config_path = config_path or DEFAULT_CONFIG_PATH
    config = load_config(actual_config_path)
    manifest, history, manifest_sha = load_r5c_manifest(project_root, config)
    group_validation = build_group_validation(history, config)

    runtime_available = False
    runtime_error = ""
    runtime_tags: dict[str, dict[str, Any]] = {}
    if run_gated:
        runtime_available, runtime_error, runtime_tags = validate_runtime_available(
            config, (config.get("scope") or {}).get("models", [])
        )
    context_preflight = build_context_preflight(history, config, runtime_tags)
    protocol_hash, protocol_freeze = protocol_metadata(
        project_root,
        actual_config_path,
        config,
        history,
        manifest_sha,
        group_validation,
        runtime_tags,
    )
    static_preflight = build_static_preflight(
        project_root,
        config,
        manifest,
        history,
        manifest_sha,
        group_validation,
        context_preflight,
        protocol_hash,
    )
    if run_gated and not runtime_available:
        static_preflight = pd.concat(
            [
                static_preflight,
                pd.DataFrame(
                    [
                        {
                            "check_id": "STATIC-034",
                            "check_name": "runtime models available",
                            "status": "fail",
                            "observed": runtime_error,
                            "expected": "qwen2:7b-instruct and mistral:7b-instruct available",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    synthetic_tests = build_synthetic_tests(history)

    static_pass = static_preflight["status"].eq("pass").all()
    synthetic_pass = synthetic_tests["pass"].eq("true").all()
    context_pass = context_preflight["status"].eq("pass").all()
    decision = DECISION_PREFLIGHT_FAILED
    blocked_reason = ""
    full_run_authorized = False
    executed_this_run = 0
    canary_table = pd.DataFrame()
    early_health = pd.DataFrame()
    early_ids: set[str] = set()

    if not static_pass:
        blocked_reason = "static preflight failed"
    elif not synthetic_pass:
        blocked_reason = "synthetic parser/validator tests failed"
    elif not context_pass:
        decision = DECISION_CONTEXT_FAILED
        blocked_reason = "context length preflight failed"
    elif run_gated:
        canary_rows = select_first_n_per_cell(history, int((config.get("gates") or {}).get("canary_initial_per_cell", 1)))
        canary_ids = {str(row["request_id"]) for row in canary_rows}
        executed_this_run += execute_selected_requests(
            project_root,
            config,
            canary_rows,
            manifest_sha,
            protocol_hash,
            allow_api_calls=allow_api_calls,
            allow_local_ollama=allow_local_ollama,
            gate_stage="canary",
        )
        status = build_status(project_root, history, manifest_sha, protocol_hash)
        canary_ok, canary_reason, canary_table = canary_pass(status, canary_ids, config)
        if not canary_ok:
            decision = DECISION_CANARY_FAILED
            blocked_reason = f"canary failed: {canary_reason}"
        elif stop_after == "canary":
            decision = DECISION_CANARY_PASS
            blocked_reason = "canary pass; early/full run not requested in this invocation"
        else:
            early_rows = select_first_n_per_cell(
                history,
                int((config.get("gates") or {}).get("early_additional_per_cell", 2)),
                skip_request_ids=canary_ids,
            )
            executed_this_run += execute_selected_requests(
                project_root,
                config,
                early_rows,
                manifest_sha,
                protocol_hash,
                allow_api_calls=allow_api_calls,
                allow_local_ollama=allow_local_ollama,
                gate_stage="early",
            )
            status = build_status(project_root, history, manifest_sha, protocol_hash)
            early_ids = canary_ids.union(str(row["request_id"]) for row in early_rows)
            early_health = health_summary(status.loc[status["request_id"].isin(early_ids)], label="early")
            early_ok, early_reason = health_pass(early_health, config)
            if not early_ok:
                decision = DECISION_EARLY_FAILED
                blocked_reason = f"early health failed: {early_reason}"
            else:
                full_run_authorized = True
                if stop_after == "early":
                    decision = DECISION_EARLY_PASS
                    blocked_reason = "early health pass; full run not requested in this invocation"
                else:
                    remaining = select_all_remaining(history, skip_request_ids=early_ids)
                    checkpoints = list((config.get("gates") or {}).get("periodic_accounted_checkpoints") or [])
                    next_checkpoint_idx = 0
                    batch: list[dict[str, Any]] = []
                    for row in remaining:
                        batch.append(row)
                        accounted_now = len(status.loc[~status["execution_status"].eq("NOT_EXECUTED")])
                        if next_checkpoint_idx < len(checkpoints) and accounted_now + len(batch) >= int(checkpoints[next_checkpoint_idx]):
                            checkpoint = int(checkpoints[next_checkpoint_idx])
                            executed_this_run += execute_selected_requests(
                                project_root,
                                config,
                                batch,
                                manifest_sha,
                                protocol_hash,
                                allow_api_calls=allow_api_calls,
                                allow_local_ollama=allow_local_ollama,
                                gate_stage=f"full_checkpoint_{checkpoint}",
                            )
                            batch = []
                            status = build_status(project_root, history, manifest_sha, protocol_hash)
                            checkpoint_health = health_summary(status, label=f"checkpoint_{checkpoint}")
                            ok, reason = health_pass(checkpoint_health, config)
                            write_diagnostic_frame(project_root, config, checkpoint_health, f"health_checkpoint_{checkpoint}.csv")
                            if not ok:
                                decision = DECISION_EXECUTION_FAILED
                                blocked_reason = f"CIRCUIT BREAKER TRIGGERED at {checkpoint}: {reason}"
                                full_run_authorized = False
                                break
                            next_checkpoint_idx += 1
                    if decision not in {DECISION_EXECUTION_FAILED}:
                        if batch:
                            executed_this_run += execute_selected_requests(
                                project_root,
                                config,
                                batch,
                                manifest_sha,
                                protocol_hash,
                                allow_api_calls=allow_api_calls,
                                allow_local_ollama=allow_local_ollama,
                                gate_stage="full_final",
                            )
                        status = build_status(project_root, history, manifest_sha, protocol_hash)
                        terminal_count = int((~status["execution_status"].eq("NOT_EXECUTED")).sum())
                        if terminal_count != 2400:
                            decision = DECISION_EXECUTION_FAILED
                            blocked_reason = f"full execution incomplete: {terminal_count}/2400"
                        else:
                            final_health = health_summary(status, label="full")
                            ok, reason = health_pass(final_health, config)
                            write_diagnostic_frame(project_root, config, final_health, "health_full.csv")
                            if not ok:
                                decision = DECISION_SCIENTIFIC_FAILED
                                blocked_reason = f"final scientific health failed: {reason}"
                                full_run_authorized = False
                            else:
                                decision = DECISION_COMPLETE
    elif static_pass and synthetic_pass and context_pass:
        decision = DECISION_PREFLIGHT_READY
        blocked_reason = "preflight ready but canary/full run not requested"

    frames = rebuild_frames(
        project_root,
        config,
        history,
        manifest_sha,
        protocol_hash,
        runtime_available,
        runtime_error,
        runtime_tags,
        group_validation,
        static_preflight,
        synthetic_tests,
        context_preflight,
        protocol_freeze,
        canary_table,
        early_health,
        decision,
        full_run_authorized,
    )
    report = render_report(frames, decision, manifest_sha, protocol_hash, full_run_authorized, blocked_reason, executed_this_run)
    if write:
        write_outputs(project_root, config, frames, report, manifest_sha, protocol_hash, decision)
        update_notes(project_root, decision, manifest_sha, protocol_hash, frames["request_execution_status"])
    return R5CBuildResult(
        decision=decision,
        frames=frames,
        report=report,
        manifest_sha256=manifest_sha,
        protocol_hash=protocol_hash,
        executed_this_run=executed_this_run,
        full_run_authorized=full_run_authorized,
        blocked_reason=blocked_reason,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--run-gated", action="store_true", help="Run preflight, canary, early health, and full run only if gates pass.")
    parser.add_argument("--stop-after", choices=["canary", "early", "full"], default="full")
    parser.add_argument("--allow-api-calls", action="store_true", help="Required before any local LLM/Ollama inference call.")
    parser.add_argument("--allow-local-ollama", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = Path(args.project_root or config.get("project_root", ".")).resolve()
    try:
        result = build_r5c_history_length(
            project_root,
            config_path=config_path,
            write=not args.no_write,
            run_gated=args.run_gated,
            allow_api_calls=args.allow_api_calls,
            allow_local_ollama=args.allow_local_ollama,
            stop_after=args.stop_after,
        )
    except R5CHistoryLengthError as exc:
        print(DECISION_EXECUTION_FAILED)
        print(f"blocked_reason={exc}")
        return 2
    status = result.frames["request_execution_status"]
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    print(f"FINAL_DECISION={result.decision}")
    print(f"MANIFEST_SHA_VERIFIED={bool_text(result.manifest_sha256 == EXPECTED_R5A_MANIFEST_SHA256)}")
    print(f"PROTOCOL_HASH={result.protocol_hash}")
    print(f"STATIC_PREFLIGHT={'PASS' if result.frames['static_preflight']['status'].eq('pass').all() else 'FAIL'}")
    print(f"SYNTHETIC_TESTS={'PASS' if result.frames['synthetic_tests']['pass'].eq('true').all() else 'FAIL'}")
    print(f"CONTEXT_PREFLIGHT={'PASS' if result.frames['context_preflight']['status'].eq('pass').all() else 'FAIL'}")
    print(f"FULL_RUN_AUTHORIZED={bool_text(result.full_run_authorized)}")
    print("REQUESTS_EXPECTED=2400")
    print(f"REQUESTS_ACCOUNTED={len(terminal)}")
    print(f"VALID={int(terminal['parse_valid'].eq('true').sum())}")
    print(f"INVALID={int(len(terminal) - terminal['parse_valid'].eq('true').sum())}")
    print(f"WRONG_LENGTH={int(terminal['validity_class'].eq('WRONG_LENGTH').sum())}")
    print(f"PARSE_FAILURE={int(terminal['validity_class'].eq('PARSE_FAILURE').sum())}")
    print(f"BACKEND_INCOMPLETE={int(terminal['validity_class'].eq('BACKEND_INCOMPLETE').sum())}")
    print(f"EXECUTED_THIS_RUN={result.executed_this_run}")
    if result.blocked_reason:
        print(f"blocked_reason={result.blocked_reason}")
    return 0 if result.decision in {DECISION_COMPLETE, DECISION_PREFLIGHT_READY, DECISION_CANARY_PASS, DECISION_EARLY_PASS} else 1


if __name__ == "__main__":
    raise SystemExit(main())

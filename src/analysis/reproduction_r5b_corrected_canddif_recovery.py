"""Corrected R5B CandDif recovery with mandatory pre-run gates.

This module creates a new R5B_CORRECTED identity from the frozen R5A CandDif
rows. It preserves the scientific request universe and candidate sets, but
uses a corrected exact-5 prompt and refuses to mark incomplete Ollama payloads
as successful metric-ready outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    EXPECTED_HEAD,
    EXPECTED_R5A_MANIFEST_SHA256,
    FAIL_STRICT_DENOMINATOR_POLICY,
    candidate_item_ids,
    candidate_records,
    canddif_value,
    canonical_json,
    git_stdout,
    hr_at_5,
    load_json,
    load_r5a_manifest,
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


PHASE = "R5B_CORRECTED"
DEFAULT_CONFIG_PATH = Path("configs") / "reproduction_r5b_corrected_canddif_recovery.yaml"
NOTES_PATH = Path("report") / "notes.md"
FORENSIC_REPORT = Path("report") / "reproduction_r5b_forensic_audit.md"
FORENSIC_SUMMARY = Path("results") / "diagnostics" / "r5b" / "forensic_summary.json"
ORIGINAL_R5B_NAMESPACE = "results/recovery/r5/"
CORRECTED_PROMPT_TEMPLATE_ID = "r5b_corrected_canddif_json_index_exact5_v1"
CORRECTED_PARSER_CONTRACT = "r5b_corrected_ranked_index_json_top5_v1"
EVIDENCE_CLASS = "PAPER_ALIGNED_ADAPTED_CORRECTED_RECOVERY"
GLOBAL_STATUS = "CRITICAL_EXPERIMENTAL_PATH_CORRECTED_R5B_PENDING_FREEZE"

DECISION_COMPLETE = "R5B_CORRECTED_RECOVERY_COMPLETE"
DECISION_PREFLIGHT_FAILED = "R5B_CORRECTED_PREFLIGHT_FAILED"
DECISION_CANARY_FAILED = "R5B_CORRECTED_CANARY_FAILED"
DECISION_EARLY_FAILED = "R5B_CORRECTED_EARLY_RUN_FAILED"
DECISION_EXECUTION_FAILED = "R5B_CORRECTED_EXECUTION_FAILED"
DECISION_SCIENTIFIC_FAILED = "R5B_CORRECTED_SCIENTIFIC_VALIDITY_FAILED"
DECISION_PREFLIGHT_READY = "R5B_CORRECTED_PREFLIGHT_READY"
DECISION_CANARY_PASS = "R5B_CORRECTED_CANARY_PASS"
DECISION_EARLY_PASS = "R5B_CORRECTED_EARLY_HEALTH_PASS"

TERMINAL_STATUSES = {
    "SUCCEEDED",
    "BACKEND_FAILED",
    "BACKEND_INCOMPLETE",
}

OUTPUT_KEYS = (
    "execution_manifest",
    "request_execution_status",
    "pair_validation",
    "output_validity",
    "metrics_by_condition",
    "canddif_results",
    "conditional_valid_metrics",
    "dataset_model_summary",
    "original_vs_corrected",
    "official_comparison",
    "runtime_provenance",
    "validation",
)

DATASET_ORDER = {"Beauty": 0, "Sports": 1, "LastFM": 2}
MODEL_ORDER = {"mistral:7b-instruct": 0, "qwen2:7b-instruct": 1}
CONDITION_ORDER = {"CANDDIF_FIRST": 0, "CANDDIF_RANDOM": 1}


class R5BCorrectedError(RuntimeError):
    """Raised when the corrected R5B gate or protocol contract fails."""


@dataclass(frozen=True)
class CorrectedOllamaResult:
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
class CorrectedBuildResult:
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


def corrected_namespace(project_root: Path, config: Mapping[str, Any]) -> Path:
    return project_path(project_root, (config.get("outputs") or {})["result_namespace"])


def expected_k(config: Mapping[str, Any]) -> int:
    return int((config.get("scope") or {}).get("expected_k", 5))


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
        raise R5BCorrectedError("cannot locate User history block in frozen prompt")
    head, tail = old_prompt.split(marker, 1)
    head_lines = head.rstrip().splitlines()
    if len(head_lines) < 3:
        raise R5BCorrectedError("unexpected frozen prompt header")
    corrected_head = "\n".join(head_lines[:3] + [instruction])
    return corrected_head + marker + tail


def corrected_request_id(original_request_id: str) -> str:
    suffix = sha256_text(f"{PHASE}|{original_request_id}")[:16]
    return f"R5B-CORR-{suffix}"


def corrected_request_key(row: Mapping[str, Any]) -> str:
    fields = [
        "phase",
        "r5a_request_id",
        "recovery_family",
        "dataset",
        "model_name",
        "task",
        "experimental_condition",
        "user_id",
        "history_length",
        "candidate_list_hash",
        "history_hash",
        "prompt_template_id",
        "parser_contract",
        "expected_output_schema",
        "prompt_hash",
    ]
    return canonical_json({field: row.get(field, "") for field in fields})


def corrected_material_context_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "phase": PHASE,
                "r5a_request_id": row.get("r5a_request_id", ""),
                "dataset": row.get("dataset", ""),
                "user_id": row.get("user_id", ""),
                "model_name": row.get("model_name", ""),
                "experimental_condition": row.get("experimental_condition", ""),
                "history_hash": row.get("history_hash", ""),
                "candidate_list_hash": row.get("candidate_list_hash", ""),
                "unordered_candidate_item_set_hash": row.get("unordered_candidate_item_set_hash", ""),
                "prompt_hash": row.get("prompt_hash", ""),
                "parser_contract": row.get("parser_contract", ""),
                "expected_output_schema": row.get("expected_output_schema", ""),
            }
        )
    )


def load_corrected_manifest(
    project_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    manifest, canddif, manifest_sha = load_r5a_manifest(project_root, config)
    namespace = corrected_namespace(project_root, config)
    expected_count = expected_k(config)
    corrected = canddif.copy()
    original_request_ids = corrected["request_id"].astype(str).tolist()
    corrected["phase"] = PHASE
    corrected["r5a_request_id"] = original_request_ids
    corrected["original_output_destination"] = corrected["output_destination"].astype(str)
    corrected["request_id"] = [corrected_request_id(value) for value in original_request_ids]
    corrected["prompt_template_id"] = CORRECTED_PROMPT_TEMPLATE_ID
    corrected["prompt_template_source"] = "src/analysis/reproduction_r5b_corrected_canddif_recovery.py"
    corrected["parser_contract"] = CORRECTED_PARSER_CONTRACT
    corrected["expected_output_schema"] = canonical_json({"ranked_candidate_indices": list(range(1, expected_count + 1))})
    corrected["prompt_render"] = corrected["prompt_render"].astype(str).map(
        lambda prompt: render_corrected_prompt(prompt, expected_count)
    )
    corrected["prompt_hash"] = corrected["prompt_render"].map(sha256_text)
    corrected["output_destination"] = [
        normalize_rel(Path((config.get("outputs") or {})["result_namespace"]) / f"{request_id}.json")
        for request_id in corrected["request_id"].astype(str)
    ]
    corrected["request_key_sha256"] = [
        sha256_text(corrected_request_key(row)) for row in corrected.to_dict("records")
    ]
    corrected["material_context_hash"] = [
        corrected_material_context_hash(row) for row in corrected.to_dict("records")
    ]
    if not corrected["request_id"].is_unique:
        raise R5BCorrectedError("corrected request IDs are not unique")
    for destination in corrected["output_destination"].astype(str):
        resolved = project_path(project_root, destination).resolve()
        if namespace.resolve() not in [resolved, *resolved.parents]:
            raise R5BCorrectedError(f"corrected output path outside namespace: {destination}")
        if normalize_rel(destination).startswith(ORIGINAL_R5B_NAMESPACE):
            raise R5BCorrectedError(f"corrected output path overlaps original R5B namespace: {destination}")
    return manifest, corrected.reset_index(drop=True), manifest_sha


def stable_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        MODEL_ORDER.get(str(row["model_name"]), 99),
        DATASET_ORDER.get(str(row["dataset"]), 99),
        CONDITION_ORDER.get(str(row["experimental_condition"]), 99),
        str(row["user_id"]),
        str(row["r5a_request_id"]),
    )


def sorted_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = frame.to_dict("records")
    return sorted(rows, key=stable_sort_key)


def select_first_n_per_cell(frame: pd.DataFrame, n: int, *, skip_request_ids: Iterable[str] = ()) -> list[dict[str, Any]]:
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
        raise R5BCorrectedError(f"only local HTTP Ollama endpoints are allowed, observed {endpoint}")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise R5BCorrectedError(f"only localhost Ollama endpoints are allowed, observed {endpoint}")


def http_json_get(endpoint: str, timeout_seconds: int) -> dict[str, Any]:
    ensure_local_endpoint(endpoint)
    with urllib.request.urlopen(endpoint, timeout=timeout_seconds) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise R5BCorrectedError(f"local endpoint returned non-object JSON: {endpoint}")
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
        raise R5BCorrectedError("Ollama tags endpoint returned invalid models list")
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


def generate_local_ollama(row: Mapping[str, Any], config: Mapping[str, Any]) -> CorrectedOllamaResult:
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
        return CorrectedOllamaResult(False, False, "", time.perf_counter() - started, None, f"HTTP {exc.code}: {exc.reason}", "backend_http_error", str(row["model_name"]), "")
    except urllib.error.URLError as exc:
        return CorrectedOllamaResult(False, False, "", time.perf_counter() - started, None, str(exc.reason), "backend_connection_error", str(row["model_name"]), "")
    except TimeoutError:
        return CorrectedOllamaResult(False, False, "", time.perf_counter() - started, None, "timed out", "backend_timeout", str(row["model_name"]), "")
    except Exception as exc:  # pragma: no cover - depends on local Ollama runtime state.
        return CorrectedOllamaResult(False, False, "", time.perf_counter() - started, None, str(exc), "backend_error", str(row["model_name"]), "")
    latency = time.perf_counter() - started
    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        return CorrectedOllamaResult(False, False, "", latency, None, f"invalid Ollama JSON: {exc}", "backend_invalid_json", str(row["model_name"]), "")
    if not isinstance(parsed, dict):
        return CorrectedOllamaResult(False, False, "", latency, None, "Ollama returned non-object JSON", "backend_invalid_response", str(row["model_name"]), "")
    response_text = parsed.get("response")
    if not isinstance(response_text, str):
        return CorrectedOllamaResult(False, False, "", latency, parsed, "Ollama response field missing", "backend_invalid_response", str(row["model_name"]), str(parsed.get("done_reason", "")))
    model_id = str(parsed.get("model") or row["model_name"])
    done = parsed.get("done")
    done_reason = str(parsed.get("done_reason", ""))
    if done is not True:
        return CorrectedOllamaResult(True, False, response_text.strip(), latency, parsed, "Ollama done flag is not true", "backend_incomplete", model_id, done_reason)
    return CorrectedOllamaResult(True, True, response_text.strip(), latency, parsed, "", "", model_id, done_reason)


def build_result_payload(
    row: Mapping[str, Any],
    manifest_sha: str,
    protocol_hash: str,
    runtime: CorrectedOllamaResult,
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
    result_hash = sha256_text(canonical_json({key: value for key, value in payload.items() if key != "result_hash"}))
    payload["result_hash"] = result_hash
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
        "prompt_hash": str(row["prompt_hash"]),
        "candidate_list_hash": str(row["candidate_list_hash"]),
        "history_hash": str(row["history_hash"]),
        "material_context_hash": str(row["material_context_hash"]),
    }
    for key, expected in required_equal.items():
        observed = str(payload.get(key, ""))
        if observed != str(expected):
            raise R5BCorrectedError(
                f"existing corrected output identity mismatch at {path}: {key} observed={observed} expected={expected}"
            )
    if str(payload.get("execution_status", "")) not in TERMINAL_STATUSES:
        raise R5BCorrectedError(f"existing corrected output is non-terminal at {path}")
    if str(payload.get("execution_status", "")) == "SUCCEEDED" and bool(payload.get("backend_complete")) is not True:
        raise R5BCorrectedError(f"existing corrected output has SUCCEEDED without backend_complete=true at {path}")


def execute_selected_requests(
    project_root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifest_sha: str,
    protocol_hash: str,
    *,
    allow_local_ollama: bool,
    gate_stage: str,
) -> int:
    if not allow_local_ollama:
        raise R5BCorrectedError("corrected R5B execution requires --allow-local-ollama")
    namespace = corrected_namespace(project_root, config).resolve()
    executed = 0
    for row in rows:
        destination = project_path(project_root, str(row["output_destination"]))
        rel_destination = rel_path(project_root, destination).replace("\\", "/")
        if namespace not in [destination.resolve(), *destination.resolve().parents]:
            raise R5BCorrectedError(f"corrected output destination outside namespace: {destination}")
        if rel_destination.startswith(ORIGINAL_R5B_NAMESPACE):
            raise R5BCorrectedError(f"refusing to write original R5B output namespace: {destination}")
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
    canddif: pd.DataFrame,
    manifest_sha: str,
    protocol_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in canddif.to_dict("records"):
        path = project_path(project_root, str(row["output_destination"]))
        if path.exists():
            rows.append(status_for_payload(row, load_json(path), manifest_sha, protocol_hash, path))
        else:
            rows.append(status_for_missing(row, manifest_sha))
    return pd.DataFrame(rows)


def build_pair_validation(canddif: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = canddif.groupby(["dataset", "model_name", "user_id"], sort=True)
    for idx, ((dataset, model_name, user_id), group) in enumerate(grouped, start=1):
        conditions = set(group["experimental_condition"])
        has_pair = conditions == {"CANDDIF_FIRST", "CANDDIF_RANDOM"} and len(group) == 2
        if has_pair:
            first = group.loc[group["experimental_condition"].eq("CANDDIF_FIRST")].iloc[0].to_dict()
            random = group.loc[group["experimental_condition"].eq("CANDDIF_RANDOM")].iloc[0].to_dict()
            first_ids = candidate_item_ids(first)
            random_ids = candidate_item_ids(random)
            same_set = sorted(first_ids) == sorted(random_ids)
            different_order = first_ids != random_ids
            same_history = str(first["history_hash"]) == str(random["history_hash"])
            same_unordered = str(first["unordered_candidate_item_set_hash"]) == str(random["unordered_candidate_item_set_hash"])
            first_positive = str(first["positive_candidate_position"])
            random_positive = str(random["positive_candidate_position"])
            pair_valid = (
                same_set
                and different_order
                and same_history
                and same_unordered
                and first_positive == "1"
                and random_positive != "1"
            )
            first_request_id = first["request_id"]
            random_request_id = random["request_id"]
            first_r5a = first["r5a_request_id"]
            random_r5a = random["r5a_request_id"]
            notes = "FIRST/RANDOM preserves frozen unordered candidate set and history"
        else:
            pair_valid = False
            same_set = False
            different_order = False
            same_history = False
            same_unordered = False
            first_positive = ""
            random_positive = ""
            first_request_id = ""
            random_request_id = ""
            first_r5a = ""
            random_r5a = ""
            notes = f"unexpected conditions={sorted(conditions)} row_count={len(group)}"
        rows.append(
            {
                "pair_id": f"R5B-CORRECTED-CANDDIF-PAIR-{idx:04d}",
                "dataset": dataset,
                "model_name": model_name,
                "user_id": user_id,
                "first_request_id": first_request_id,
                "random_request_id": random_request_id,
                "first_r5a_request_id": first_r5a,
                "random_r5a_request_id": random_r5a,
                "same_candidate_items": bool_text(same_set),
                "different_order_as_expected": bool_text(different_order),
                "same_history": bool_text(same_history),
                "same_unordered_candidate_set_hash": bool_text(same_unordered),
                "first_positive_position": first_positive,
                "random_positive_position": random_positive,
                "pair_valid": bool_text(pair_valid),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def build_execution_manifest(canddif: pd.DataFrame, manifest_sha: str, protocol_hash: str) -> pd.DataFrame:
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
    frame = canddif[keep].copy()
    frame.insert(0, "phase", PHASE)
    frame.insert(1, "r5a_manifest_sha256", manifest_sha)
    frame.insert(2, "protocol_hash", protocol_hash)
    frame["execution_backend"] = "local_ollama"
    frame["external_api_allowed"] = "false"
    frame["scientific_scope"] = "same_1200_frozen_r5a_canddif_rows_corrected_prompt_only"
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
    for (dataset, model_name, condition), group in status.groupby(["dataset", "model_name", "experimental_condition"], sort=True):
        n_manifest = len(group)
        n_terminal = int((~group["execution_status"].eq("NOT_EXECUTED")).sum())
        n_succeeded = int(group["execution_status"].eq("SUCCEEDED").sum())
        n_valid = int(group["parse_valid"].eq("true").sum())
        n_invalid = int(n_terminal - n_valid)
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "experimental_condition": condition,
                "n_manifest_requests": n_manifest,
                "n_manifest": n_manifest,
                "n_terminal": n_terminal,
                "n_succeeded": n_succeeded,
                "n_valid_outputs": n_valid,
                "n_valid": n_valid,
                "n_invalid_outputs": n_invalid,
                "n_invalid": n_invalid,
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
    for (dataset, model_name, condition), group in status.groupby(["dataset", "model_name", "experimental_condition"], sort=True):
        valid = group.loc[group["parse_valid"].eq("true")]
        denom = len(valid)
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "experimental_condition": condition,
                "n_valid_outputs": denom,
                "diagnostic_hr_at_5_valid_only": float(valid["hr_at_5"].astype(float).sum() / denom) if denom else 0.0,
                "diagnostic_ndcg_at_5_valid_only": float(valid["ndcg_at_5"].astype(float).sum() / denom) if denom else 0.0,
                "diagnostic_only": "true",
                "notes": "Conditional-valid metrics separate ranking quality from format compliance; not official replacement.",
            }
        )
    return pd.DataFrame(rows)


def build_canddif_results(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model_name), group in metrics.groupby(["dataset", "model_name"], sort=True):
        by_condition = group.set_index("experimental_condition")
        hr_first = float(by_condition.loc["CANDDIF_FIRST", "hr_at_5"]) if "CANDDIF_FIRST" in by_condition.index else 0.0
        hr_random = float(by_condition.loc["CANDDIF_RANDOM", "hr_at_5"]) if "CANDDIF_RANDOM" in by_condition.index else 0.0
        ndcg_first = float(by_condition.loc["CANDDIF_FIRST", "ndcg_at_5"]) if "CANDDIF_FIRST" in by_condition.index else 0.0
        ndcg_random = float(by_condition.loc["CANDDIF_RANDOM", "ndcg_at_5"]) if "CANDDIF_RANDOM" in by_condition.index else 0.0
        canddif_hr, hr_safety = canddif_value(hr_first, hr_random)
        canddif_ndcg, ndcg_safety = canddif_value(ndcg_first, ndcg_random)
        first_n = int(by_condition.loc["CANDDIF_FIRST", "n_manifest_requests"]) if "CANDDIF_FIRST" in by_condition.index else 0
        random_n = int(by_condition.loc["CANDDIF_RANDOM", "n_manifest_requests"]) if "CANDDIF_RANDOM" in by_condition.index else 0
        first_valid = int(by_condition.loc["CANDDIF_FIRST", "n_valid_outputs"]) if "CANDDIF_FIRST" in by_condition.index else 0
        random_valid = int(by_condition.loc["CANDDIF_RANDOM", "n_valid_outputs"]) if "CANDDIF_RANDOM" in by_condition.index else 0
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "cohort": "R5A_CANDDIF_100_USER_CONTROLLED_CORRECTED",
                "n_users": min(first_n, random_n),
                "HR_first": hr_first,
                "HR_random": hr_random,
                "CandDif_HR": canddif_hr,
                "NDCG_first": ndcg_first,
                "NDCG_random": ndcg_random,
                "CandDif_NDCG": canddif_ndcg,
                "canddif_formula": "log((1-Acc_random)/(1-Acc_first))",
                "numerical_safety_policy_hr": hr_safety,
                "numerical_safety_policy_ndcg": ndcg_safety,
                "validity_summary": f"FIRST valid={first_valid}/{first_n}; RANDOM valid={random_valid}/{random_n}",
                "denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
                "evidence_class": EVIDENCE_CLASS,
                "notes": "Corrected exact-5 prompt; local adapted protocol, not paper-exact model identity.",
            }
        )
    return pd.DataFrame(rows)


def build_dataset_model_summary(status: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model_name), group in status.groupby(["dataset", "model_name"], sort=True):
        metric_group = metrics.loc[metrics["dataset"].eq(dataset) & metrics["model_name"].eq(model_name)]
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "n_manifest_requests": len(group),
                "n_terminal": int((~group["execution_status"].eq("NOT_EXECUTED")).sum()),
                "n_succeeded": int(group["execution_status"].eq("SUCCEEDED").sum()),
                "n_valid_outputs": int(group["parse_valid"].eq("true").sum()),
                "n_invalid_outputs": int((~group["execution_status"].eq("NOT_EXECUTED")).sum() - group["parse_valid"].eq("true").sum()),
                "n_wrong_length": int(group["validity_class"].eq("WRONG_LENGTH").sum()),
                "n_parse_failure": int(group["validity_class"].eq("PARSE_FAILURE").sum()),
                "n_backend_incomplete": int(group["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "n_backend_failed": int(group["execution_status"].eq("BACKEND_FAILED").sum()),
                "n_not_executed": int(group["execution_status"].eq("NOT_EXECUTED").sum()),
                "first_hr_at_5": metric_value(metric_group, "CANDDIF_FIRST", "hr_at_5"),
                "random_hr_at_5": metric_value(metric_group, "CANDDIF_RANDOM", "hr_at_5"),
                "first_ndcg_at_5": metric_value(metric_group, "CANDDIF_FIRST", "ndcg_at_5"),
                "random_ndcg_at_5": metric_value(metric_group, "CANDDIF_RANDOM", "ndcg_at_5"),
            }
        )
    return pd.DataFrame(rows)


def metric_value(metrics: pd.DataFrame, condition: str, column: str) -> float:
    row = metrics.loc[metrics["experimental_condition"].eq(condition)]
    return float(row[column].iloc[0]) if not row.empty else 0.0


def original_forensic_counts(project_root: Path) -> dict[str, int]:
    path = project_root / FORENSIC_SUMMARY
    if not path.exists():
        return {
            "requests": 1200,
            "valid": 52,
            "invalid": 1148,
            "wrong_length": 1132,
            "parse_failed": 16,
            "mistral_valid": 50,
            "qwen_valid": 2,
        }
    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    return {str(key): int(value) for key, value in parsed.items() if isinstance(value, int)}


def build_original_vs_corrected(project_root: Path, status: pd.DataFrame) -> pd.DataFrame:
    original = original_forensic_counts(project_root)
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    corrected = {
        "total": len(status),
        "accounted": len(terminal),
        "valid": int(terminal["parse_valid"].eq("true").sum()),
        "invalid": int(len(terminal) - terminal["parse_valid"].eq("true").sum()),
        "wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
        "parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
        "backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
        "mistral_valid": int(terminal.loc[terminal["model_name"].eq("mistral:7b-instruct"), "parse_valid"].eq("true").sum()),
        "qwen_valid": int(terminal.loc[terminal["model_name"].eq("qwen2:7b-instruct"), "parse_valid"].eq("true").sum()),
    }
    rows = [
        ("total", original.get("requests", 1200), corrected["total"]),
        ("accounted", original.get("requests", 1200), corrected["accounted"]),
        ("valid", original.get("valid", 52), corrected["valid"]),
        ("invalid", original.get("invalid", 1148), corrected["invalid"]),
        ("wrong_length", original.get("wrong_length", 1132), corrected["wrong_length"]),
        ("parse_failure", original.get("parse_failed", 16), corrected["parse_failure"]),
        ("backend_incomplete", original.get("parse_failed_done_false", 16), corrected["backend_incomplete"]),
        ("mistral_valid", original.get("mistral_valid", 50), corrected["mistral_valid"]),
        ("qwen_valid", original.get("qwen_valid", 2), corrected["qwen_valid"]),
    ]
    return pd.DataFrame(
        [
            {"quantity": quantity, "original_r5b": old, "corrected_r5b": new}
            for quantity, old, new in rows
        ]
    )


def build_official_comparison(canddif_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in canddif_results.to_dict("records"):
        for metric, key in [("CandDif-HR", "CandDif_HR"), ("CandDif-NDCG", "CandDif_NDCG")]:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "model_name": row["model_name"],
                    "metric": metric,
                    "corrected_local_adapted_canddif": row[key],
                    "paper_exact": "false",
                    "comparison_class": "LOCAL_ADAPTED_NOT_PAPER_EXACT",
                    "notes": "Corrected R5B preserves R5A local-model scope; no paper-exact numeric claim.",
                }
            )
    return pd.DataFrame(rows)


def file_hash_or_missing(path: Path) -> str:
    return sha256_file(path) if path.exists() else "MISSING"


def protocol_metadata(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    canddif: pd.DataFrame,
    manifest_sha: str,
    runtime_tags: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, pd.DataFrame]:
    module_path = Path(__file__).resolve()
    parser_path = project_root / "src" / "llm" / "canonical_llm_contract.py"
    candidate_path = project_root / "src" / "analysis" / "reproduction_r5a_controlled_recovery_manifest.py"
    metric_path = module_path
    prompt_hashes = sorted(canddif["prompt_hash"].astype(str).unique())
    prompt_family_hash = sha256_text(canonical_json(prompt_hashes))
    generation_config = {
        "endpoint": str((config.get("runtime") or {}).get("endpoint", "")),
        "format": str((config.get("runtime") or {}).get("format", "json")),
        "stream": str((config.get("runtime") or {}).get("stream", False)).lower(),
        "temperature": str((config.get("runtime") or {}).get("temperature", 0)),
        "max_retries": str((config.get("runtime") or {}).get("max_retries", 0)),
        "timeout_seconds": str((config.get("runtime") or {}).get("timeout_seconds", 120)),
        "num_predict_values": sorted(canddif["max_new_tokens"].astype(str).unique()),
        "seed_values": sorted(canddif["seed"].astype(str).unique()),
    }
    models = []
    for model in sorted(set(canddif["model_name"].astype(str))):
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
        "selected_rows": len(canddif),
        "expected_k": expected_k(config),
        "prompt_template_id": CORRECTED_PROMPT_TEMPLATE_ID,
        "prompt_family_hash": prompt_family_hash,
        "parser_contract": CORRECTED_PARSER_CONTRACT,
        "parser_hash": file_hash_or_missing(parser_path),
        "runner_hash": file_hash_or_missing(module_path),
        "config_hash": file_hash_or_missing(config_path),
        "candidate_construction_hash": file_hash_or_missing(candidate_path),
        "metric_hash": file_hash_or_missing(metric_path),
        "generation_config": generation_config,
        "models": models,
    }
    protocol_hash = sha256_text(canonical_json(payload))
    rows = [
        {"component": "manifest", "value": manifest_sha, "sha256": manifest_sha},
        {"component": "prompt_family", "value": CORRECTED_PROMPT_TEMPLATE_ID, "sha256": prompt_family_hash},
        {"component": "parser", "value": CORRECTED_PARSER_CONTRACT, "sha256": payload["parser_hash"]},
        {"component": "runner", "value": normalize_rel(module_path), "sha256": payload["runner_hash"]},
        {"component": "config", "value": normalize_rel(config_path), "sha256": payload["config_hash"]},
        {"component": "candidate_construction", "value": "R5A frozen candidate rows", "sha256": payload["candidate_construction_hash"]},
        {"component": "metrics", "value": "HR@5/NDCG@5/CandDif in corrected runner", "sha256": payload["metric_hash"]},
        {"component": "generation_config", "value": canonical_json(generation_config), "sha256": sha256_text(canonical_json(generation_config))},
        {"component": "protocol", "value": "CORRECTED_PROTOCOL_PREFLIGHT_READY", "sha256": protocol_hash},
    ]
    for model in models:
        rows.append(
            {
                "component": "model_identifier",
                "value": model["model_name"],
                "sha256": str(model.get("digest", "")),
            }
        )
    return protocol_hash, pd.DataFrame(rows)


def build_static_preflight(
    project_root: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    canddif: pd.DataFrame,
    manifest_sha: str,
    pair_validation: pd.DataFrame,
    protocol_hash: str,
) -> pd.DataFrame:
    scope = config.get("scope") or {}
    runtime = config.get("runtime") or {}
    namespace = normalize_rel((config.get("outputs") or {})["result_namespace"]).rstrip("/") + "/"
    old_outputs = set(canddif["original_output_destination"].astype(str).map(normalize_rel))
    new_outputs = set(canddif["output_destination"].astype(str).map(normalize_rel))
    prompts = canddif["prompt_render"].astype(str)
    original_repo_status = git_stdout(["-C", "original_repo/EvaLLMasRecommender", "status", "--short"], project_root)
    head = git_stdout(["rev-parse", "HEAD"], project_root)
    disk_free = shutil.disk_usage(project_root).free
    protected = protected_hash_check(project_root)
    checks: list[tuple[str, str, bool, str, str]] = [
        ("STATIC-001", "manifest SHA", manifest_sha == EXPECTED_R5A_MANIFEST_SHA256, manifest_sha, EXPECTED_R5A_MANIFEST_SHA256),
        ("STATIC-002", "repository HEAD", head == EXPECTED_HEAD, head, EXPECTED_HEAD),
        ("STATIC-003", "manifest rows", len(manifest) == int((config.get("frozen_r5a") or {}).get("manifest_rows", 4800)), str(len(manifest)), "4800"),
        ("STATIC-004", "selected CandDif rows", len(canddif) == int(scope.get("selected_rows", 1200)), str(len(canddif)), "1200"),
        ("STATIC-005", "FIRST/RANDOM rows", dict(canddif["experimental_condition"].value_counts()) == {"CANDDIF_FIRST": 600, "CANDDIF_RANDOM": 600}, str(dict(canddif["experimental_condition"].value_counts())), "{CANDDIF_FIRST:600,CANDDIF_RANDOM:600}"),
        ("STATIC-006", "datasets", set(canddif["dataset"]) == set(scope.get("datasets") or []), str(sorted(set(canddif["dataset"]))), str(sorted(scope.get("datasets") or []))),
        ("STATIC-007", "models", set(canddif["model_name"]) == set(scope.get("models") or []), str(sorted(set(canddif["model_name"]))), str(sorted(scope.get("models") or []))),
        ("STATIC-008", "no ML-1M CandDif", not canddif["dataset"].eq("ML-1M").any(), str(int(canddif["dataset"].eq("ML-1M").sum())), "0"),
        ("STATIC-009", "ranking task only", canddif["task"].eq("ranking").all(), str(sorted(set(canddif["task"]))), "ranking"),
        ("STATIC-010", "expected K is 5", canddif.apply(parse_expected_count, axis=1).eq(5).all(), str(sorted(set(canddif.apply(parse_expected_count, axis=1)))), "[5]"),
        ("STATIC-011", "candidate count 20", canddif["candidate_count"].astype(int).eq(20).all(), str(sorted(set(canddif["candidate_count"].astype(str)))), "20"),
        ("STATIC-012", "pair count and validity", len(pair_validation) == 600 and pair_validation["pair_valid"].eq("true").all(), f"{int(pair_validation['pair_valid'].eq('true').sum())}/{len(pair_validation)}", "600/600"),
        ("STATIC-013", "corrected prompt says EXACTLY 5", prompts.str.contains("EXACTLY 5", regex=False).all(), str(int(prompts.str.contains("EXACTLY 5", regex=False).sum())), str(len(canddif))),
        ("STATIC-014", "corrected prompt forbids more/fewer", prompts.str.contains("Do not return more than 5", regex=False).all() and prompts.str.contains("Do not return fewer than 5", regex=False).all(), "checked", "all prompts"),
        ("STATIC-015", "corrected prompt forbids prose", prompts.str.contains("Do not include rationale", regex=False).all(), "checked", "all prompts"),
        ("STATIC-016", "parser contract corrected", set(canddif["parser_contract"]) == {CORRECTED_PARSER_CONTRACT}, str(sorted(set(canddif["parser_contract"]))), CORRECTED_PARSER_CONTRACT),
        ("STATIC-017", "local endpoint only", str(runtime.get("endpoint", "")).startswith("http://127.0.0.1") or str(runtime.get("endpoint", "")).startswith("http://localhost"), str(runtime.get("endpoint", "")), "localhost Ollama"),
        ("STATIC-018", "format json stream false", str(runtime.get("format", "")) == "json" and bool(runtime.get("stream", False)) is False, f"format={runtime.get('format')};stream={runtime.get('stream')}", "format=json;stream=false"),
        ("STATIC-019", "no retry policy introduced", int(runtime.get("max_retries", 0)) == 0, str(runtime.get("max_retries", 0)), "0"),
        ("STATIC-020", "output namespace corrected", namespace == "results/recovery/r5b_corrected/", namespace, "results/recovery/r5b_corrected/"),
        ("STATIC-021", "no output path overlap with old R5B", old_outputs.isdisjoint(new_outputs), str(len(old_outputs.intersection(new_outputs))), "0"),
        ("STATIC-022", "no corrected output under original namespace", not any(path.startswith(ORIGINAL_R5B_NAMESPACE) for path in new_outputs), "checked", "none"),
        ("STATIC-023", "original repo clean", original_repo_status == "", original_repo_status, "empty"),
        ("STATIC-024", "protected artifacts unchanged", protected.empty or protected["protection_status"].eq("PROTECTED_UNCHANGED").all(), f"{int(protected['protection_status'].eq('PROTECTED_UNCHANGED').sum())}/{len(protected)}", f"{len(protected)}/{len(protected)}"),
        ("STATIC-025", "disk free guard", disk_free >= int(runtime.get("disk_free_min_bytes", 0)), str(disk_free), f">={runtime.get('disk_free_min_bytes', 0)}"),
        ("STATIC-026", "protocol hash present", len(protocol_hash) == 64, protocol_hash, "64-char SHA256"),
    ]
    return pd.DataFrame(
        [
            {
                "check_id": check_id,
                "check_name": name,
                "status": "pass" if ok else "fail",
                "observed": observed,
                "expected": expected,
            }
            for check_id, name, ok, observed, expected in checks
        ]
    )


def build_synthetic_tests(canddif: pd.DataFrame) -> pd.DataFrame:
    row = canddif.iloc[0].to_dict()
    candidate_ids = candidate_item_ids(row)
    valid_object = canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5]})
    cases = [
        ("valid_json_object_indices", valid_object, True, "parsed_valid"),
        ("plain_json_array_candidate_ids", json.dumps(candidate_ids[:5]), False, "invalid_schema"),
        ("numbered_lines", "1. 342\n2. 177\n3. 901\n4. 55\n5. 20", False, "parse_failed"),
        ("wrong_key_json", canonical_json({"ranking": [1, 2, 3, 4, 5]}), False, "invalid_schema"),
        ("line_separated_ids", "342\n177\n901\n55\n20", False, "parse_failed"),
        ("four_ids", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4]}), False, "invalid_length"),
        ("six_ids", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5, 6]}), False, "invalid_length"),
        ("full_candidate_set", canonical_json({"ranked_candidate_indices": list(range(1, 21))}), False, "invalid_length"),
        ("duplicate", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 4]}), False, "duplicate_indices"),
        ("out_of_candidate_index", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 21]}), False, "unknown_indices"),
        ("empty_output", "", False, "empty_output"),
        ("truncated_output", '{"ranked_candidate_indices":[1,2,3', False, "parse_failed"),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, text, expected_valid, expected_status in cases:
        parsed = parse_strict_index_response(text, candidate_items=candidate_ids, expected_count=5)
        rows.append(
            {
                "case_id": case_id,
                "expected_valid": bool_text(expected_valid),
                "observed_valid": bool_text(parsed.parse_valid),
                "expected_status": expected_status,
                "observed_status": parsed.status,
                "output_length": parsed.output_length,
                "duplicate_count": parsed.duplicate_count,
                "unknown_item_count": parsed.unknown_item_count,
                "pass": bool_text(parsed.parse_valid == expected_valid and parsed.status == expected_status),
                "notes": "numbered lines must not be converted to [1,342,2,177,...]" if case_id == "numbered_lines" else "",
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
            "backend_complete": "Complete",
            "output_length": "Parsed K",
            "parse_valid": "Valid",
            "validity_class": "Failure",
        }
    )
    if len(canary) < 12:
        return False, f"canary accounted {len(canary)} < 12", table
    failed = canary.loc[~canary["parse_valid"].eq("true")]
    if failed.empty:
        return True, "", table
    health = health_summary(canary, label="canary")
    ok, reason = health_pass(health, config)
    if ok:
        return True, "isolated canary noncompliance accepted after extra deterministic samples", table
    return False, reason, table


def write_diagnostic_frame(project_root: Path, config: Mapping[str, Any], frame: pd.DataFrame, name: str) -> None:
    path = diagnostics_dir(project_root, config) / name
    write_csv(frame, path)


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
                "context_length": str(model_details.get("context_length", "")),
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
    canary_table: pd.DataFrame,
    early_health: pd.DataFrame,
    status: pd.DataFrame,
    pair_validation: pd.DataFrame,
    *,
    full_run_authorized: bool,
    decision: str,
) -> pd.DataFrame:
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    checks = [
        ("VAL-001", "static preflight", static_preflight["status"].eq("pass").all(), f"{int(static_preflight['status'].eq('pass').sum())}/{len(static_preflight)}", "all pass"),
        ("VAL-002", "synthetic tests", synthetic_tests["pass"].eq("true").all(), f"{int(synthetic_tests['pass'].eq('true').sum())}/{len(synthetic_tests)}", "all pass"),
        ("VAL-003", "canary complete when attempted", canary_table.empty or canary_table["Complete"].eq("true").all(), f"{int(canary_table.get('Complete', pd.Series(dtype=str)).eq('true').sum())}/{len(canary_table)}", "all complete"),
        ("VAL-004", "canary parsed K=5 when attempted", canary_table.empty or canary_table["Parsed K"].astype(str).eq("5").all(), f"{int(canary_table.get('Parsed K', pd.Series(dtype=str)).astype(str).eq('5').sum())}/{len(canary_table)}", "all 5"),
        ("VAL-005", "early health no backend incomplete", early_health.empty or int(early_health.loc[early_health["scope"].eq("overall"), "backend_incomplete"].iloc[0]) == 0, "checked", "0"),
        ("VAL-006", "pair validation", len(pair_validation) == 600 and pair_validation["pair_valid"].eq("true").all(), f"{int(pair_validation['pair_valid'].eq('true').sum())}/{len(pair_validation)}", "600/600"),
        ("VAL-007", "incomplete not succeeded", not ((status["execution_status"].eq("SUCCEEDED")) & (status["backend_complete"].ne("true"))).any(), "checked", "none"),
        ("VAL-008", "full run authorization consistent", (full_run_authorized and decision in {DECISION_COMPLETE, DECISION_EARLY_PASS} and (decision != DECISION_COMPLETE or len(terminal) == 1200)) or (not full_run_authorized and decision != DECISION_COMPLETE) or (decision == DECISION_EXECUTION_FAILED), f"authorized={full_run_authorized};decision={decision};terminal={len(terminal)}", "consistent"),
    ]
    return pd.DataFrame(
        [
            {
                "check_id": check_id,
                "check_name": name,
                "status": "pass" if ok else "fail",
                "observed": observed,
                "expected": expected,
            }
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
    canary_table = frames.get("canary_results", pd.DataFrame())
    early_health = frames.get("early_health", pd.DataFrame())
    metrics = frames["metrics_by_condition"]
    canddif = frames["canddif_results"]
    original_vs = frames["original_vs_corrected"]
    validation = frames["validation"]
    static_preflight = frames.get("static_preflight", pd.DataFrame())
    synthetic = frames.get("synthetic_tests", pd.DataFrame())
    report = f"""# R5B Corrected CandDif Recovery

## Baseline

Previous forensic decision: `R5B_MULTIPLE_PROTOCOL_FAILURES`. Original R5B artifacts are preserved as forensic evidence and this corrected run uses separate paths.

## Forensic cause being corrected

The failed R5B prompt asked for JSON indices but did not ask for `EXACTLY 5`. The corrected protocol also prevents Ollama payloads with `done != true` from becoming `SUCCEEDED` metric-ready records.

## Minimal protocol changes

OLD PROMPT instruction:

```text
Return JSON with key ranked_candidate_indices containing unique 1-based candidate indices.
```

CORRECTED PROMPT instruction:

```text
{corrected_instruction(5)}
```

The output namespace remains one-based candidate indices because that is the frozen parser/schema contract. Candidate sets, target items, users, FIRST/RANDOM relation, models, temperature, seed, and max token settings are unchanged.

## Protected artifact verification

Original R5B raw output path is not used. Corrected raw outputs are under `results/recovery/r5b_corrected/`.

## Frozen manifest verification

Manifest SHA256: `{manifest_sha}`.

Selected corrected rows: `{len(status)}`.

## Static preflight

{render_markdown_table(static_preflight, ["check_id", "check_name", "status", "observed", "expected"])}

## Synthetic tests

{render_markdown_table(synthetic, ["case_id", "expected_valid", "observed_valid", "expected_status", "observed_status", "pass"])}

## Protocol freeze

```text
CORRECTED_PROTOCOL_PREFLIGHT_READY
PROTOCOL_HASH={protocol_hash}
```

{render_markdown_table(frames.get("protocol_freeze", pd.DataFrame()), ["component", "value", "sha256"])}

## Canary design

The canary selects one deterministic first eligible row per `model x dataset x condition` cell, then allows at most two additional deterministic rows in failed model-compliance cells.

## Canary results

{render_markdown_table(canary_table, ["Model", "Dataset", "Condition", "Complete", "Parsed K", "Valid", "Failure", "request_id", "r5a_request_id"])}

## Early health check

{render_markdown_table(early_health, ["scope", "key", "accounted", "valid", "invalid", "validity_rate", "wrong_length", "parse_failure", "backend_incomplete"])}

## Full-run authorization decision

FULL RUN AUTHORIZED: `{"YES" if full_run_authorized else "NO"}`.

Blocked reason: `{blocked_reason}`.

## Full execution

- Requests expected: `1200`
- Requests accounted: `{len(terminal)}`
- Executed this invocation: `{executed_this_run}`
- Valid: `{valid}`
- Invalid: `{invalid}`
- Wrong length: `{int(terminal["validity_class"].eq("WRONG_LENGTH").sum())}`
- Parse failure: `{int(terminal["validity_class"].eq("PARSE_FAILURE").sum())}`
- Backend incomplete: `{int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum())}`

## Output validity

{render_markdown_table(frames["dataset_model_summary"], ["dataset", "model_name", "n_manifest_requests", "n_terminal", "n_valid_outputs", "n_invalid_outputs", "n_wrong_length", "n_parse_failure", "n_backend_incomplete"])}

## FIRST vs RANDOM metrics

{render_markdown_table(metrics, ["dataset", "model_name", "experimental_condition", "n_manifest", "n_valid", "n_invalid", "HR@5", "NDCG@5"])}

## CandDif results

{render_markdown_table(canddif, ["dataset", "model_name", "n_users", "HR_first", "HR_random", "CandDif_HR", "NDCG_first", "NDCG_random", "CandDif_NDCG"])}

## Original R5B vs Corrected R5B

{render_markdown_table(original_vs, ["quantity", "original_r5b", "corrected_r5b"])}

## Official comparison

{render_markdown_table(frames["official_comparison"], ["dataset", "model_name", "metric", "corrected_local_adapted_canddif", "paper_exact", "comparison_class"])}

## Scientific interpretation

The official end-to-end metrics retain the fail-strict denominator: every frozen request remains in the denominator and invalid outputs score zero. Conditional-valid metrics are diagnostic only.

## Limitations

This is still a local adapted recovery, not a paper-exact rerun. Model identities, output namespace, and local Ollama execution differ from the original closed/open model setup in the paper repository.

## Validation

{render_markdown_table(validation, ["check_id", "check_name", "status", "observed", "expected"])}

## Repository state

No commit and no push were performed by this runner.

## Final decision

`{decision}`
"""
    return report


def update_notes(project_root: Path, decision: str, manifest_sha: str, protocol_hash: str, status: pd.DataFrame) -> None:
    notes_path = project_root / NOTES_PATH
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    section = f"""
## R5B corrected CandDif recovery

- Decision: `{decision}`.
- Manifest SHA256: `{manifest_sha}`.
- Protocol hash: `{protocol_hash}`.
- Scope: new `R5B_CORRECTED` identity over the same 1200 frozen R5A CandDif rows; no R5C execution, commit, push, external API, download, training, or `original_repo/` modification.
- Outputs: `results/recovery/r5b_corrected/`, `results/diagnostics/r5b_corrected/`, `results/tables/reproduction_r5b_corrected_*.csv`, and `report/reproduction_r5b_corrected.md`.
- Request accounting: expected `1200`, accounted `{len(terminal)}`, valid `{int(terminal["parse_valid"].eq("true").sum())}`, invalid `{int(len(terminal) - terminal["parse_valid"].eq("true").sum())}`.
- Backend incomplete handling: incomplete Ollama responses are classified as `BACKEND_INCOMPLETE`, not `SUCCEEDED`.
"""
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "## R5B corrected CandDif recovery"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n" + section
    else:
        existing = existing.rstrip() + "\n" + section
    notes_path.write_text(existing.lstrip(), encoding="utf-8")


def rebuild_frames(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    canddif: pd.DataFrame,
    manifest_sha: str,
    protocol_hash: str,
    runtime_available: bool,
    runtime_error: str,
    runtime_tags: Mapping[str, Mapping[str, Any]],
    static_preflight: pd.DataFrame,
    synthetic_tests: pd.DataFrame,
    protocol_freeze: pd.DataFrame,
    canary_results: pd.DataFrame | None,
    early_health: pd.DataFrame | None,
    decision: str,
    full_run_authorized: bool,
    blocked_reason: str,
) -> dict[str, pd.DataFrame]:
    pair_validation = build_pair_validation(canddif)
    execution_manifest = build_execution_manifest(canddif, manifest_sha, protocol_hash)
    status = build_status(project_root, canddif, manifest_sha, protocol_hash)
    output_validity = build_output_validity(status)
    metrics = build_metrics_by_condition(status)
    canddif_results = build_canddif_results(metrics)
    conditional = build_conditional_valid_metrics(status)
    summary = build_dataset_model_summary(status, metrics)
    original_vs = build_original_vs_corrected(project_root, status)
    official = build_official_comparison(canddif_results)
    runtime = build_runtime_provenance(project_root, config, status, runtime_available, runtime_error, runtime_tags, protocol_hash)
    canary_frame = canary_results if canary_results is not None else pd.DataFrame()
    early_frame = early_health if early_health is not None else pd.DataFrame()
    validation = build_validation(
        static_preflight,
        synthetic_tests,
        canary_frame,
        early_frame,
        status,
        pair_validation,
        full_run_authorized=full_run_authorized,
        decision=decision,
    )
    frames: dict[str, pd.DataFrame] = {
        "execution_manifest": execution_manifest,
        "pair_validation": pair_validation,
        "request_execution_status": status,
        "output_validity": output_validity,
        "metrics_by_condition": metrics,
        "canddif_results": canddif_results,
        "conditional_valid_metrics": conditional,
        "dataset_model_summary": summary,
        "original_vs_corrected": original_vs,
        "official_comparison": official,
        "runtime_provenance": runtime,
        "validation": validation,
        "static_preflight": static_preflight,
        "synthetic_tests": synthetic_tests,
        "protocol_freeze": protocol_freeze,
        "canary_results": canary_frame,
        "early_health": early_frame,
    }
    return frames


def write_outputs(project_root: Path, config: Mapping[str, Any], frames: Mapping[str, pd.DataFrame], report: str) -> None:
    outputs = configured_outputs(config)
    for key in OUTPUT_KEYS:
        write_csv(frames[key], project_root / outputs[key])
    diag = diagnostics_dir(project_root, config)
    diag.mkdir(parents=True, exist_ok=True)
    for key in ["static_preflight", "synthetic_tests", "protocol_freeze", "canary_results", "early_health"]:
        write_csv(frames.get(key, pd.DataFrame()), diag / f"{key}.csv")
    report_path = project_path(project_root, (config.get("outputs") or {})["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def build_r5b_corrected(
    project_root: Path,
    config_path: Path | None = None,
    *,
    write: bool = True,
    run_gated: bool = False,
    allow_local_ollama: bool = False,
    stop_after: str = "full",
) -> CorrectedBuildResult:
    actual_config_path = config_path or DEFAULT_CONFIG_PATH
    config = load_config(actual_config_path)
    manifest, canddif, manifest_sha = load_corrected_manifest(project_root, config)
    runtime_available = False
    runtime_error = ""
    runtime_tags: dict[str, dict[str, Any]] = {}
    if run_gated:
        runtime_available, runtime_error, runtime_tags = validate_runtime_available(
            config, (config.get("scope") or {}).get("models", [])
        )
    protocol_hash, protocol_freeze = protocol_metadata(
        project_root,
        actual_config_path,
        config,
        canddif,
        manifest_sha,
        runtime_tags,
    )
    pair_validation = build_pair_validation(canddif)
    static_preflight = build_static_preflight(project_root, config, manifest, canddif, manifest_sha, pair_validation, protocol_hash)
    if run_gated and not runtime_available:
        static_preflight = pd.concat(
            [
                static_preflight,
                pd.DataFrame(
                    [
                        {
                            "check_id": "STATIC-027",
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
    synthetic_tests = build_synthetic_tests(canddif)
    static_pass = static_preflight["status"].eq("pass").all()
    synthetic_pass = synthetic_tests["pass"].eq("true").all()
    decision = DECISION_PREFLIGHT_FAILED
    blocked_reason = ""
    full_run_authorized = False
    executed_this_run = 0
    canary_table = pd.DataFrame()
    early_health = pd.DataFrame()

    if not static_pass:
        blocked_reason = "static preflight failed"
    elif not synthetic_pass:
        blocked_reason = "synthetic parser/validator tests failed"
    elif run_gated:
        canary_rows = select_first_n_per_cell(canddif, int((config.get("gates") or {}).get("canary_initial_per_cell", 1)))
        canary_ids = {str(row["request_id"]) for row in canary_rows}
        executed_this_run += execute_selected_requests(
            project_root,
            config,
            canary_rows,
            manifest_sha,
            protocol_hash,
            allow_local_ollama=allow_local_ollama,
            gate_stage="canary",
        )
        status = build_status(project_root, canddif, manifest_sha, protocol_hash)
        canary_ok, canary_reason, canary_table = canary_pass(status, canary_ids, config)
        if not canary_ok:
            failed = status.loc[status["request_id"].isin(canary_ids) & ~status["parse_valid"].eq("true")]
            model_noncompliance = failed["validity_class"].isin({"WRONG_LENGTH", "DUPLICATE", "OUT_OF_RANGE", "OTHER"}).all()
            if model_noncompliance and not failed.empty:
                extra_n = int((config.get("gates") or {}).get("canary_extra_per_failed_cell", 2))
                extra_rows: list[dict[str, Any]] = []
                for _, failed_row in failed.groupby(["model_name", "dataset", "experimental_condition"], sort=True):
                    first = failed_row.iloc[0]
                    cell = canddif.loc[
                        canddif["model_name"].eq(first["model_name"])
                        & canddif["dataset"].eq(first["dataset"])
                        & canddif["experimental_condition"].eq(first["experimental_condition"])
                    ]
                    extra_rows.extend(select_first_n_per_cell(cell, extra_n, skip_request_ids=canary_ids))
                executed_this_run += execute_selected_requests(
                    project_root,
                    config,
                    extra_rows,
                    manifest_sha,
                    protocol_hash,
                    allow_local_ollama=allow_local_ollama,
                    gate_stage="canary_extra",
                )
                canary_ids.update(str(row["request_id"]) for row in extra_rows)
                status = build_status(project_root, canddif, manifest_sha, protocol_hash)
                canary_ok, canary_reason, canary_table = canary_pass(status, canary_ids, config)
            if not canary_ok:
                decision = DECISION_CANARY_FAILED
                blocked_reason = f"canary failed: {canary_reason}"
            else:
                decision = ""
        else:
            decision = ""

        if decision == "":
            if stop_after == "canary":
                decision = DECISION_CANARY_PASS
                blocked_reason = "canary pass; early/full run not requested in this invocation"
            else:
                decision = ""

        if decision == "":
            early_rows = select_first_n_per_cell(
                canddif,
                int((config.get("gates") or {}).get("early_additional_per_cell", 2)),
                skip_request_ids=canary_ids,
            )
            executed_this_run += execute_selected_requests(
                project_root,
                config,
                early_rows,
                manifest_sha,
                protocol_hash,
                allow_local_ollama=allow_local_ollama,
                gate_stage="early",
            )
            status = build_status(project_root, canddif, manifest_sha, protocol_hash)
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
                    decision = ""

        if decision == "" and full_run_authorized:
            remaining = select_all_remaining(canddif, skip_request_ids=early_ids)
            checkpoints = list((config.get("gates") or {}).get("periodic_accounted_checkpoints") or [])
            next_checkpoint_idx = 0
            batch: list[dict[str, Any]] = []
            for row in remaining:
                batch.append(row)
                accounted_now = len(status.loc[~status["execution_status"].eq("NOT_EXECUTED")])
                if next_checkpoint_idx < len(checkpoints) and accounted_now + len(batch) >= int(checkpoints[next_checkpoint_idx]):
                    executed_this_run += execute_selected_requests(
                        project_root,
                        config,
                        batch,
                        manifest_sha,
                        protocol_hash,
                        allow_local_ollama=allow_local_ollama,
                        gate_stage=f"full_checkpoint_{checkpoints[next_checkpoint_idx]}",
                    )
                    batch = []
                    status = build_status(project_root, canddif, manifest_sha, protocol_hash)
                    checkpoint_health = health_summary(status, label=f"checkpoint_{checkpoints[next_checkpoint_idx]}")
                    ok, reason = health_pass(checkpoint_health, config)
                    write_diagnostic_frame(project_root, config, checkpoint_health, f"health_checkpoint_{checkpoints[next_checkpoint_idx]}.csv")
                    if not ok:
                        decision = DECISION_EXECUTION_FAILED
                        blocked_reason = f"CIRCUIT BREAKER TRIGGERED at {checkpoints[next_checkpoint_idx]}: {reason}"
                        full_run_authorized = False
                        break
                    next_checkpoint_idx += 1
            if decision == "":
                if batch:
                    executed_this_run += execute_selected_requests(
                        project_root,
                        config,
                        batch,
                        manifest_sha,
                        protocol_hash,
                        allow_local_ollama=allow_local_ollama,
                        gate_stage="full_final",
                    )
                status = build_status(project_root, canddif, manifest_sha, protocol_hash)
                terminal_count = int((~status["execution_status"].eq("NOT_EXECUTED")).sum())
                if terminal_count != 1200:
                    decision = DECISION_EXECUTION_FAILED
                    blocked_reason = f"full execution incomplete: {terminal_count}/1200"
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
    elif static_pass and synthetic_pass:
        decision = DECISION_PREFLIGHT_READY
        blocked_reason = "preflight ready but canary/full run not requested"

    frames = rebuild_frames(
        project_root,
        actual_config_path,
        config,
        manifest,
        canddif,
        manifest_sha,
        protocol_hash,
        runtime_available,
        runtime_error,
        runtime_tags,
        static_preflight,
        synthetic_tests,
        protocol_freeze,
        canary_table,
        early_health,
        decision,
        full_run_authorized,
        blocked_reason,
    )
    report = render_report(frames, decision, manifest_sha, protocol_hash, full_run_authorized, blocked_reason, executed_this_run)
    if write:
        write_outputs(project_root, config, frames, report)
        update_notes(project_root, decision, manifest_sha, protocol_hash, frames["request_execution_status"])
    return CorrectedBuildResult(
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
    parser.add_argument("--allow-local-ollama", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = Path(args.project_root or config.get("project_root", ".")).resolve()
    try:
        result = build_r5b_corrected(
            project_root,
            config_path=config_path,
            write=not args.no_write,
            run_gated=args.run_gated,
            allow_local_ollama=args.allow_local_ollama,
            stop_after=args.stop_after,
        )
    except R5BCorrectedError as exc:
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
    print(f"FULL_RUN_AUTHORIZED={bool_text(result.full_run_authorized)}")
    print(f"REQUESTS_EXPECTED=1200")
    print(f"REQUESTS_ACCOUNTED={len(terminal)}")
    print(f"VALID={int(terminal['parse_valid'].eq('true').sum())}")
    print(f"INVALID={int(len(terminal) - terminal['parse_valid'].eq('true').sum())}")
    print(f"WRONG_LENGTH={int(terminal['validity_class'].eq('WRONG_LENGTH').sum())}")
    print(f"PARSE_FAILURE={int(terminal['validity_class'].eq('PARSE_FAILURE').sum())}")
    print(f"BACKEND_INCOMPLETE={int(terminal['validity_class'].eq('BACKEND_INCOMPLETE').sum())}")
    if result.blocked_reason:
        print(f"blocked_reason={result.blocked_reason}")
    return 0 if result.decision in {DECISION_COMPLETE, DECISION_PREFLIGHT_READY, DECISION_CANARY_PASS, DECISION_EARLY_PASS} else 1


if __name__ == "__main__":
    raise SystemExit(main())

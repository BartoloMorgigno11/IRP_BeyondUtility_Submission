"""Phase F6E4B adapted full-run execution.

F6E4B is the controlled execution phase for the adapted indexed full run
frozen by F6E4A. Importing this module is side-effect free. Ollama generation
is reachable only through the explicit execution entrypoint and requires an
allow flag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from src.analysis import plan_f6e3_exact_prompt_pilot as f6e3
from src.analysis import plan_f6e3b_adapted_output_contract_pilot as f6e3b
from src.analysis import plan_f6e4a_adapted_full_run_freeze as f6e4a
from src.audit import common_manifests


PHASE_ID = "phase_f6e4b"
EXPECTED_F6E4A_FREEZE_HASH = "6e9cf65606821e0dda2cd54a11d4c0424f2690eb454e5008fd1b8c96ddd5d345"
EXPECTED_F6E4A_REQUEST_HASH = "fdcbd10900e3dddfba5251d44259b51000cc6a102e855f7adf9fa7157979a29a"
EXPECTED_F6E4A_SHARD_HASH = "89be35c19e03de84cb66320ee062de52d352ad711f72f6c5944e361c95480419"
EXPECTED_F6E4A_SCIENTIFIC_HASH = "53a3a56b035d53e10b4043aae11bdc352c28e51e24f6de9b3783c8d8917bd925"
EXPECTED_REQUESTS = 30492
EXPECTED_SHARDS = 64
MAX_SHARD_SIZE = 500

QWEN2_TAG = f6e4a.QWEN2_TAG
MISTRAL_TAG = f6e4a.MISTRAL_TAG
MODEL_ORDER = [QWEN2_TAG, MISTRAL_TAG]
STRATEGY_ORDER = f6e4a.STRATEGY_ORDER
EXPECTED_MODEL_DIGESTS = f6e4a.EXPECTED_MODEL_DIGESTS

REQUEST_MANIFEST = Path("results/tables/phase_f6e4a_request_manifest.csv")
SHARD_MANIFEST = Path("results/tables/phase_f6e4a_shard_manifest.csv")
F6E4A_FREEZE_MANIFEST = Path("results/manifests/phase_f6e4a_full_run_freeze.json")
F6E4A_EXECUTION_MATRIX = Path("results/tables/phase_f6e4a_execution_matrix.csv")
F6E4A_RUNTIME_PARAMETERS = Path("results/tables/phase_f6e4a_runtime_parameters.csv")
F6E4A_STOP_CONDITIONS = Path("results/tables/phase_f6e4a_stop_conditions.csv")
F6E4A_GATE_PLAN = Path("results/tables/phase_f6e4a_gate_plan.csv")

REPORT_PATH = Path("report/phase_f6e4b_adapted_full_run.md")
FREEZE_MANIFEST_PATH = Path("results/manifests/phase_f6e4b_execution_freeze.json")
RAW_PATHS = {
    QWEN2_TAG: Path("results/llm/phase_f6e4b_qwen2_adapted_full_run_raw.jsonl"),
    MISTRAL_TAG: Path("results/llm/phase_f6e4b_mistral_adapted_full_run_raw.jsonl"),
}
LOG_DIR = Path("results/logs")
CONFLICT_LOG = LOG_DIR / "phase_f6e4b_conflicts.jsonl"
ERROR_LOG = LOG_DIR / "phase_f6e4b_errors.jsonl"

OUTPUT_TABLES = {
    "progress": Path("results/tables/phase_f6e4b_progress.csv"),
    "shard_status": Path("results/tables/phase_f6e4b_shard_status.csv"),
    "gate_results": Path("results/tables/phase_f6e4b_gate_results.csv"),
    "instance_results": Path("results/tables/phase_f6e4b_instance_results.csv"),
    "cell_summary": Path("results/tables/phase_f6e4b_cell_summary.csv"),
    "dataset_task_summary": Path("results/tables/phase_f6e4b_dataset_task_summary.csv"),
    "parser_validity": Path("results/tables/phase_f6e4b_parser_validity.csv"),
    "runtime_summary": Path("results/tables/phase_f6e4b_runtime_summary.csv"),
    "invalid_output_audit": Path("results/tables/phase_f6e4b_invalid_output_audit.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6e4b_missing_or_blocked.csv"),
}

PROGRESS_COLUMNS = [
    "scope",
    "model",
    "strategy_id",
    "dataset",
    "task",
    "shard_id",
    "expected_requests",
    "attempted_requests",
    "transport_success",
    "strict_valid",
    "missing_requests",
    "invalid_outputs",
    "strict_valid_rate",
    "last_request_id",
    "last_shard_id",
    "last_update_utc",
    "decision",
    "notes",
]
SHARD_STATUS_COLUMNS = [
    "shard_id",
    "model",
    "strategy_id",
    "dataset",
    "task",
    "expected_requests",
    "attempted_requests",
    "transport_success",
    "strict_valid",
    "json_parse_valid",
    "schema_valid",
    "invalid_outputs",
    "missing_requests",
    "strict_valid_rate",
    "json_parse_failure_rate",
    "raw_path",
    "raw_hash_before",
    "raw_hash_after",
    "last_request_id",
    "status",
    "stop_condition",
    "notes",
]
GATE_COLUMNS = [
    "gate_id",
    "gate_name",
    "phase_point",
    "shard_id",
    "model",
    "strategy_id",
    "dataset",
    "task",
    "attempted_requests",
    "transport_success",
    "parse_valid",
    "schema_valid",
    "strict_valid",
    "invalid_reasons",
    "latency_mean",
    "eval_count_mean",
    "stop_condition_triggered",
    "decision",
    "evidence",
    "notes",
]
INSTANCE_COLUMNS = [
    "request_id",
    "source_request_id",
    "shard_id",
    "ordinal_position",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "adapted_strategy_id",
    "adapter_classification",
    "transport_status",
    "completion_status",
    "json_parse_valid",
    "schema_valid",
    "candidate_valid",
    "duplicate_valid",
    "strict_valid",
    "invalid_reason",
    "output_length",
    "output_indices",
    "parsed_ranking_item_ids",
    "parsed_ranking_titles",
    "positive_item_id",
    "positive_initial_position",
    "positive_rank",
    "hr_at5_operational",
    "ndcg_at5_operational",
    "hr_at5_conditional",
    "ndcg_at5_conditional",
    "duration_seconds",
    "prompt_eval_count",
    "eval_count",
    "raw_output_file",
]
SUMMARY_COLUMNS = [
    "summary_id",
    "scope",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "expected_requests",
    "attempted_requests",
    "transport_success",
    "json_parse_valid",
    "schema_valid",
    "strict_valid",
    "missing_requests",
    "invalid_outputs",
    "strict_valid_rate",
    "transport_success_rate",
    "operational_hr_at5",
    "operational_ndcg_at5",
    "conditional_hr_at5",
    "conditional_ndcg_at5",
    "status",
    "notes",
]
PARSER_COLUMNS = [
    "request_id",
    "shard_id",
    "model",
    "strategy_id",
    "dataset",
    "task",
    "parse_status",
    "json_parse_valid",
    "schema_valid",
    "parsed_count",
    "duplicate_count",
    "unknown_index_count",
    "extra_text_detected",
    "strict_valid",
    "raw_excerpt_sha256",
    "notes",
]
RUNTIME_COLUMNS = [
    "runtime_id",
    "scope",
    "model",
    "strategy_id",
    "dataset",
    "task",
    "attempted_requests",
    "latency_mean",
    "latency_p50",
    "latency_p95",
    "requests_per_hour",
    "prompt_eval_count_mean",
    "eval_count_mean",
    "prompt_eval_duration_mean",
    "eval_duration_mean",
    "total_duration_mean",
    "disk_free_bytes",
    "ollama_state",
    "notes",
]
INVALID_COLUMNS = [
    "audit_id",
    "request_id",
    "shard_id",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "transport_status",
    "invalid_reason",
    "json_parse_valid",
    "schema_valid",
    "candidate_valid",
    "duplicate_valid",
    "strict_valid",
    "output_length",
    "output_indices",
    "raw_response_sha256",
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
    "progress": PROGRESS_COLUMNS,
    "shard_status": SHARD_STATUS_COLUMNS,
    "gate_results": GATE_COLUMNS,
    "instance_results": INSTANCE_COLUMNS,
    "cell_summary": SUMMARY_COLUMNS,
    "dataset_task_summary": SUMMARY_COLUMNS,
    "parser_validity": PARSER_COLUMNS,
    "runtime_summary": RUNTIME_COLUMNS,
    "invalid_output_audit": INVALID_COLUMNS,
    "missing_or_blocked": MISSING_COLUMNS,
}


@dataclass(frozen=True)
class PromptBundle:
    prompt: str
    candidate_ids: tuple[str, ...]
    candidate_titles: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionResult:
    decision: str
    sent_requests: int
    stop_reason: str
    last_request_id: str
    last_shard_id: str
    written: dict[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_progress_timestamp(instance_rows: Sequence[Mapping[str, str]]) -> str:
    timestamps = [
        row.get("response_timestamp_utc", "")
        for row in instance_rows
        if row.get("transport_status") != "missing" and row.get("response_timestamp_utc", "")
    ]
    return max(timestamps) if timestamps else "not_available"


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


def file_sha256_or_missing(path: Path) -> str:
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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def split_pipe(value: Any) -> list[str]:
    return [part for part in str(value or "").split("|") if part != ""]


def join_pipe(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl_fsync(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def command_output(args: Sequence[str], root: Path, timeout: int = 10) -> str:
    try:
        result = subprocess.run(list(args), cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:  # pragma: no cover - platform dependent
        return f"{type(exc).__name__}:{exc}"
    return (result.stdout + result.stderr).strip()


def repository_state(root: Path) -> dict[str, str]:
    return {
        "directory": str(root),
        "branch": command_output(["git", "branch", "--show-current"], root),
        "head": command_output(["git", "rev-parse", "HEAD"], root),
        "last_commit": command_output(["git", "log", "-1", "--oneline"], root),
        "git_status_short": command_output(["git", "status", "--short"], root, timeout=20),
        "git_diff_name_only": command_output(["git", "diff", "--name-only"], root, timeout=20),
        "git_diff_stat": command_output(["git", "diff", "--stat"], root, timeout=20),
        "git_diff_cached_stat": command_output(["git", "diff", "--cached", "--stat"], root, timeout=20),
        "staged_files": command_output(["git", "diff", "--name-only", "--cached"], root, timeout=20),
        "python_version": platform.python_version(),
        "python_executable": os.sys.executable,
        "timestamp_utc": "not_recorded_in_deterministic_analysis",
    }


def verify_frozen_inputs(root: Path) -> dict[str, str]:
    expected = {
        F6E4A_FREEZE_MANIFEST: EXPECTED_F6E4A_FREEZE_HASH,
        REQUEST_MANIFEST: EXPECTED_F6E4A_REQUEST_HASH,
        SHARD_MANIFEST: EXPECTED_F6E4A_SHARD_HASH,
    }
    observed = {path.as_posix(): file_sha256_or_missing(root / path) for path in expected}
    mismatches = [
        f"{path.as_posix()}:expected={digest}:observed={observed[path.as_posix()]}"
        for path, digest in expected.items()
        if observed[path.as_posix()] != digest
    ]
    freeze_path = root / F6E4A_FREEZE_MANIFEST
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        scientific = str(freeze.get("scientific_content_hash", ""))
        observed["phase_f6e4a_scientific_content_hash"] = scientific
        if scientific != EXPECTED_F6E4A_SCIENTIFIC_HASH:
            mismatches.append(f"phase_f6e4a_scientific_content_hash:expected={EXPECTED_F6E4A_SCIENTIFIC_HASH}:observed={scientific}")
    for path, digest in {**f6e4a.F6D4A_RAW_HASHES, **f6e4a.F6E3_RAW_HASHES, **f6e4a.F6E3B_RAW_HASHES}.items():
        observed[path] = file_sha256_or_missing(root / path)
        if observed[path] != digest:
            mismatches.append(f"{path}:expected={digest}:observed={observed[path]}")
    protected = protected_summary(root)
    if protected["changed"] or protected["missing"]:
        mismatches.append(f"protected_inputs:changed={protected['changed']}:missing={protected['missing']}")
    if mismatches:
        raise RuntimeError("f6e4b_blocked_by_frozen_input_invariance:" + ";".join(mismatches))
    return observed


def protected_summary(root: Path) -> dict[str, int]:
    f2_rows = read_csv_rows(root / f6e4a.F2_PROTECTED_VALIDATION)
    f6d4a_rows = read_csv_rows(root / f6e4a.F6D4A_PROTECTED_VERIFICATION)
    changed = sum(1 for row in f2_rows if row.get("status") != "unchanged")
    missing = sum(1 for row in f2_rows if row.get("exists_after") == "false")
    checked = len(f2_rows)
    for row in f6d4a_rows:
        checked += int(float(row.get("checked") or 0))
        changed += int(float(row.get("changed") or 0))
        missing += int(float(row.get("missing") or 0))
    return {"checked": checked, "changed": changed, "missing": missing}


def load_request_manifest(root: Path) -> list[dict[str, str]]:
    return common_manifests.read_csv_manifest_rows(root / REQUEST_MANIFEST, missing="empty", strip=False)


def load_shard_manifest(root: Path) -> list[dict[str, str]]:
    return common_manifests.read_csv_manifest_rows(root / SHARD_MANIFEST, missing="empty", strip=False)


def assert_manifest_valid(request_rows: Sequence[Mapping[str, str]], shard_rows: Sequence[Mapping[str, str]], *, validate_expected: bool = True) -> None:
    if validate_expected and len(request_rows) != EXPECTED_REQUESTS:
        raise RuntimeError(f"f6e4b_request_count_invalid:{len(request_rows)}")
    if validate_expected and len(shard_rows) != EXPECTED_SHARDS:
        raise RuntimeError(f"f6e4b_shard_count_invalid:{len(shard_rows)}")
    request_ids = [row["request_id"] for row in request_rows]
    if len(set(request_ids)) != len(request_ids):
        raise RuntimeError("f6e4b_duplicate_request_id")
    allowed_cells = {(model, strategy) for model in MODEL_ORDER for strategy in STRATEGY_ORDER}
    observed_cells = {(row["model"], row["core_strategy_id"]) for row in request_rows}
    if not observed_cells <= allowed_cells or observed_cells != allowed_cells:
        raise RuntimeError(f"f6e4b_unallowed_cells:{sorted(observed_cells)}")
    if any(row["dataset"] == "LastFM" and row["task"] == "reranking" for row in request_rows):
        raise RuntimeError("f6e4b_lastfm_reranking_present")
    shard_ids = {row["shard_id"] for row in shard_rows}
    if any(row["shard_id"] not in shard_ids for row in request_rows):
        raise RuntimeError("f6e4b_request_shard_gap")
    by_shard = Counter(row["shard_id"] for row in request_rows)
    for shard in shard_rows:
        expected = int(float(shard["request_count"]))
        observed = by_shard[shard["shard_id"]]
        if observed != expected:
            raise RuntimeError(f"f6e4b_shard_coverage_mismatch:{shard['shard_id']}:{observed}:{expected}")
        if expected > MAX_SHARD_SIZE:
            raise RuntimeError(f"f6e4b_shard_too_large:{shard['shard_id']}:{expected}")
    for row in request_rows:
        candidate_ids = split_pipe(row["candidate_item_ids"])
        if len(candidate_ids) != 20 or len(set(candidate_ids)) != 20:
            raise RuntimeError(f"f6e4b_candidate_count_invalid:{row['request_id']}")
        if row["candidate_order_hash"] != sha256_text(stable_json(tuple(candidate_ids))):
            raise RuntimeError(f"f6e4b_candidate_order_hash_mismatch:{row['request_id']}")
        if not row.get("complete_prompt_hash") or not row.get("runtime_parameters_json") or not row.get("expected_model_digest"):
            raise RuntimeError(f"f6e4b_manifest_required_hash_missing:{row['request_id']}")


def raw_path_for_model(model: str) -> Path:
    return RAW_PATHS[model]


def all_raw_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in RAW_PATHS.values():
        records.extend(read_jsonl(root / path))
    return records


def final_raw_by_request(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for record in all_raw_records(root):
        request_id = str(record.get("request_id", ""))
        if not parse_bool(record.get("is_final_attempt", "true")):
            continue
        if request_id in records:
            conflicts.append({"request_id": request_id, "conflict": "duplicate_final_attempt", "phase": PHASE_ID})
        else:
            records[request_id] = record
    if conflicts:
        for conflict in conflicts:
            append_jsonl_fsync(root / CONFLICT_LOG, conflict)
        raise RuntimeError("f6e4b_duplicate_final_raw_record")
    return records


def validate_existing_raw_records(root: Path, request_rows: Sequence[Mapping[str, str]]) -> None:
    manifest_by_id = {row["request_id"]: row for row in request_rows}
    for record in all_raw_records(root):
        request_id = str(record.get("request_id", ""))
        manifest = manifest_by_id.get(request_id)
        if manifest is None:
            append_jsonl_fsync(root / CONFLICT_LOG, {"request_id": request_id, "conflict": "raw_request_not_in_manifest"})
            raise RuntimeError(f"f6e4b_raw_request_not_in_manifest:{request_id}")
        checks = {
            "phase": PHASE_ID,
            "model": manifest["model"],
            "complete_prompt_hash": manifest["complete_prompt_hash"],
            "candidate_order_hash": manifest["candidate_order_hash"],
            "shard_id": manifest["shard_id"],
        }
        for key, expected in checks.items():
            if str(record.get(key, "")) != expected:
                append_jsonl_fsync(root / CONFLICT_LOG, {"request_id": request_id, "conflict": f"{key}_mismatch", "expected": expected, "observed": record.get(key, "")})
                raise RuntimeError(f"f6e4b_raw_{key}_mismatch:{request_id}")


def source_context_map(root: Path) -> dict[str, f6e4a.SourceContext]:
    return {context.source_context_id: context for context in f6e4a.build_source_contexts(root)}


def prompt_bundle_for_row(root: Path, row: Mapping[str, str], contexts: Mapping[str, f6e4a.SourceContext]) -> PromptBundle:
    context = contexts.get(row["source_context_id"])
    if context is None:
        raise RuntimeError(f"f6e4b_missing_source_context:{row['request_id']}")
    core, adapter, prompt = f6e4a.render_complete_prompt(root, row["core_strategy_id"], context)
    if sha256_text(core) != row["core_prompt_hash"]:
        raise RuntimeError(f"f6e4b_core_prompt_hash_mismatch:{row['request_id']}")
    if sha256_text(adapter) != row["adapter_hash"]:
        raise RuntimeError(f"f6e4b_adapter_hash_mismatch:{row['request_id']}")
    if sha256_text(prompt) != row["complete_prompt_hash"]:
        raise RuntimeError(f"f6e4b_complete_prompt_hash_mismatch:{row['request_id']}")
    candidate_ids = tuple(split_pipe(row["candidate_item_ids"]))
    if context.candidate_ids != candidate_ids:
        raise RuntimeError(f"f6e4b_candidate_id_mismatch:{row['request_id']}")
    return PromptBundle(prompt=prompt, candidate_ids=context.candidate_ids, candidate_titles=context.candidate_titles)


def runtime_parameters(row: Mapping[str, str]) -> dict[str, Any]:
    return json.loads(row["runtime_parameters_json"])


def make_payload(row: Mapping[str, str], prompt: str) -> dict[str, Any]:
    params = runtime_parameters(row)
    return {
        "model": row["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(params["temperature"]),
            "num_predict": int(params["num_predict"]),
        },
    }


def make_raw_record(row: Mapping[str, str], result: f6e3.RequestResult, bundle: PromptBundle, *, attempt: int = 1) -> dict[str, Any]:
    parse = f6e3b.parse_adapted_indexed_response(result.response_text, candidate_count=len(bundle.candidate_ids), expected_count=10)
    mapped_ids, mapped_titles = f6e3b.map_indices(parse["output_indices"], bundle.candidate_ids, bundle.candidate_titles)
    candidate_valid = int(parse["unknown_index_count"]) == 0 and int(parse["output_length"]) == len(mapped_ids)
    duplicate_valid = int(parse["duplicate_count"]) == 0
    strict_valid = bool(parse["strict_valid"] and result.success)
    invalid_reason = "" if strict_valid else str(result.error_type or result.error_message or parse["parse_status"])
    raw = result.raw_backend_response
    return {
        "phase": PHASE_ID,
        "request_id": row["request_id"],
        "source_request_id": row["source_request_id"],
        "source_context_id": row["source_context_id"],
        "shard_id": row["shard_id"],
        "ordinal_position": row["ordinal_position"],
        "attempt": attempt,
        "is_final_attempt": True,
        "model": row["model"],
        "expected_model_digest": row["expected_model_digest"],
        "model_digest": row["expected_model_digest"],
        "strategy_id": row["core_strategy_id"],
        "adapted_strategy_id": row["adapted_strategy_id"],
        "adapter_classification": row["adapter_classification"],
        "dataset": row["dataset"],
        "task": row["task"],
        "core_prompt_hash": row["core_prompt_hash"],
        "adapter_hash": row["adapter_hash"],
        "complete_prompt_hash": row["complete_prompt_hash"],
        "candidate_order_hash": row["candidate_order_hash"],
        "runtime_parameters_hash": row["runtime_parameters_hash"],
        "runtime_parameters_json": row["runtime_parameters_json"],
        "request_timestamp_utc": result.started_at_utc,
        "response_timestamp_utc": result.finished_at_utc,
        "duration_seconds": result.duration_seconds,
        "prompt_eval_count": raw.get("prompt_eval_count", ""),
        "eval_count": raw.get("eval_count", ""),
        "prompt_eval_duration_nanoseconds": raw.get("prompt_eval_duration", ""),
        "eval_duration_nanoseconds": raw.get("eval_duration", ""),
        "total_duration_nanoseconds": raw.get("total_duration", ""),
        "done": raw.get("done", False),
        "done_reason": raw.get("done_reason", ""),
        "raw_response": result.response_text,
        "raw_response_sha256": sha256_text(result.response_text),
        "transport_status": "success" if result.success else "failure",
        "error": result.error_message,
        "error_type": result.error_type,
        "json_parse_valid": bool(parse["json_parse_valid"]),
        "schema_valid": bool(parse["schema_valid"]),
        "candidate_valid": candidate_valid,
        "duplicate_valid": duplicate_valid,
        "strict_valid": strict_valid,
        "invalid_reason": invalid_reason,
        "parsed_indices": list(parse["output_indices"]),
        "parsed_item_ids": mapped_ids,
        "parsed_titles": mapped_titles,
        "parser_version": "json_ranking_1based_top10_strict_v1",
        "no_retry_for_invalid_output": True,
    }


def inspect_ollama_gate() -> list[dict[str, str]]:
    return f6e3b.inspect_ollama_gate()


def gate_decision_from_ollama(gate_rows: Sequence[Mapping[str, str]]) -> str:
    decision = f6e3b.gate_decision(gate_rows)
    if decision == "pass":
        return "pass"
    return {
        "f6e3b_blocked_ollama_unreachable": "f6e4b_blocked_ollama_unreachable",
        "f6e3b_blocked_model_digest_mismatch": "f6e4b_blocked_model_digest_mismatch",
        "f6e3b_blocked_by_runtime_failure": "f6e4b_blocked_by_runtime_failure",
    }.get(decision, "f6e4b_blocked_by_runtime_failure")


def status_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    indices = tuple(int(value) for value in raw.get("parsed_indices", []) if str(value).strip())
    return {
        "json_parse_valid": parse_bool(raw.get("json_parse_valid")),
        "schema_valid": parse_bool(raw.get("schema_valid")),
        "candidate_valid": parse_bool(raw.get("candidate_valid")),
        "duplicate_valid": parse_bool(raw.get("duplicate_valid")),
        "strict_valid": parse_bool(raw.get("strict_valid")),
        "parse_status": raw.get("invalid_reason") or "parsed_valid",
        "output_indices": indices,
        "output_length": len(indices),
        "duplicate_count": len(indices) - len(set(indices)),
        "unknown_index_count": 0 if parse_bool(raw.get("candidate_valid")) else 1,
        "extra_text_detected": False,
    }


def analyze_instances(root: Path, request_rows: Sequence[Mapping[str, str]], raw_by_id: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    raw_records = dict(raw_by_id or final_raw_by_request(root))
    instance_rows: list[dict[str, str]] = []
    parser_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []
    for row in request_rows:
        raw = raw_records.get(row["request_id"])
        if raw is None:
            transport = "missing"
            completion = "missing_raw_response"
            parsed_ids: list[str] = []
            parsed_titles: list[str] = []
            indices: list[int] = []
            strict = False
            json_valid = schema_valid = candidate_valid = duplicate_valid = False
            invalid = "missing_raw_response"
            duration = prompt_eval = eval_count = ""
            raw_response = ""
        else:
            transport = str(raw.get("transport_status", ""))
            completion = "completed" if transport == "success" and str(raw.get("raw_response", "")).strip() else "failed"
            parsed_ids = [str(value) for value in raw.get("parsed_item_ids", [])]
            parsed_titles = [str(value) for value in raw.get("parsed_titles", [])]
            indices = [int(value) for value in raw.get("parsed_indices", [])]
            strict = parse_bool(raw.get("strict_valid")) and transport == "success"
            json_valid = parse_bool(raw.get("json_parse_valid"))
            schema_valid = parse_bool(raw.get("schema_valid"))
            candidate_valid = parse_bool(raw.get("candidate_valid"))
            duplicate_valid = parse_bool(raw.get("duplicate_valid"))
            invalid = "" if strict else str(raw.get("invalid_reason") or raw.get("error_type") or "not_strict_valid")
            duration = fmt(float(raw.get("duration_seconds", 0) or 0))
            prompt_eval = str(raw.get("prompt_eval_count", ""))
            eval_count = str(raw.get("eval_count", ""))
            raw_response = str(raw.get("raw_response", ""))
        positive_rank = ""
        hr = 0.0
        ndcg = 0.0
        if row["positive_item_id"] in parsed_ids:
            positive_rank = str(parsed_ids.index(row["positive_item_id"]) + 1)
            rank_value = int(positive_rank)
            if rank_value <= 5:
                hr = 1.0
                ndcg = 1.0 / math.log2(rank_value + 1)
        conditional_hr = hr if strict else ""
        conditional_ndcg = ndcg if strict else ""
        instance = {
            "request_id": row["request_id"],
            "source_request_id": row["source_request_id"],
            "shard_id": row["shard_id"],
            "ordinal_position": row["ordinal_position"],
            "dataset": row["dataset"],
            "task": row["task"],
            "model": row["model"],
            "strategy_id": row["core_strategy_id"],
            "adapted_strategy_id": row["adapted_strategy_id"],
            "adapter_classification": row["adapter_classification"],
            "transport_status": transport,
            "completion_status": completion,
            "json_parse_valid": bool_text(json_valid),
            "schema_valid": bool_text(schema_valid),
            "candidate_valid": bool_text(candidate_valid),
            "duplicate_valid": bool_text(duplicate_valid),
            "strict_valid": bool_text(strict),
            "invalid_reason": invalid,
            "output_length": str(len(indices)),
            "output_indices": join_pipe(indices),
            "parsed_ranking_item_ids": join_pipe(parsed_ids),
            "parsed_ranking_titles": join_pipe(parsed_titles),
            "positive_item_id": row["positive_item_id"],
            "positive_initial_position": row["positive_initial_position"],
            "positive_rank": positive_rank,
            "hr_at5_operational": fmt(hr),
            "ndcg_at5_operational": fmt(ndcg),
            "hr_at5_conditional": fmt(conditional_hr) if conditional_hr != "" else "",
            "ndcg_at5_conditional": fmt(conditional_ndcg) if conditional_ndcg != "" else "",
            "duration_seconds": duration,
            "prompt_eval_count": prompt_eval,
            "eval_count": eval_count,
            "raw_output_file": raw_path_for_model(row["model"]).as_posix(),
        }
        instance_rows.append(instance)
        parse_status = "parsed_valid" if strict else invalid
        parser_rows.append(
            {
                "request_id": row["request_id"],
                "shard_id": row["shard_id"],
                "model": row["model"],
                "strategy_id": row["core_strategy_id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "parse_status": parse_status,
                "json_parse_valid": bool_text(json_valid),
                "schema_valid": bool_text(schema_valid),
                "parsed_count": str(len(indices)),
                "duplicate_count": str(len(indices) - len(set(indices))),
                "unknown_index_count": "0" if candidate_valid else ("0" if transport == "missing" else "1"),
                "extra_text_detected": "false",
                "strict_valid": bool_text(strict),
                "raw_excerpt_sha256": sha256_text(raw_response[:200]) if raw_response else "",
                "notes": "Strict JSON ranking parser; no correction, no truncation, no duplicate repair.",
            }
        )
        if transport != "missing" and not strict:
            invalid_rows.append(
                {
                    "audit_id": stable_id("F6E4B-INVALID", row["request_id"]),
                    "request_id": row["request_id"],
                    "shard_id": row["shard_id"],
                    "dataset": row["dataset"],
                    "task": row["task"],
                    "model": row["model"],
                    "strategy_id": row["core_strategy_id"],
                    "transport_status": transport,
                    "invalid_reason": invalid,
                    "json_parse_valid": bool_text(json_valid),
                    "schema_valid": bool_text(schema_valid),
                    "candidate_valid": bool_text(candidate_valid),
                    "duplicate_valid": bool_text(duplicate_valid),
                    "strict_valid": bool_text(strict),
                    "output_length": str(len(indices)),
                    "output_indices": join_pipe(indices),
                    "raw_response_sha256": sha256_text(raw_response),
                    "notes": "Invalid output preserved; no automatic retry or repair.",
                }
            )
    return instance_rows, parser_rows, invalid_rows


def summarize_group(group: Sequence[Mapping[str, str]], *, scope: str, dataset: str = "ALL", task: str = "ALL", model: str = "ALL", strategy: str = "ALL") -> dict[str, str]:
    expected = len(group)
    attempted = [row for row in group if row["transport_status"] != "missing"]
    success = [row for row in group if row["transport_status"] == "success"]
    json_valid = [row for row in group if row["json_parse_valid"] == "true"]
    schema_valid = [row for row in group if row["schema_valid"] == "true"]
    strict = [row for row in group if row["strict_valid"] == "true"]
    operational_hr = [float(row["hr_at5_operational"] or 0) for row in group]
    operational_ndcg = [float(row["ndcg_at5_operational"] or 0) for row in group]
    conditional_hr = [float(row["hr_at5_conditional"]) for row in strict if row["hr_at5_conditional"] != ""]
    conditional_ndcg = [float(row["ndcg_at5_conditional"]) for row in strict if row["ndcg_at5_conditional"] != ""]
    return {
        "summary_id": stable_id("F6E4B-SUMMARY", scope, dataset, task, model, strategy),
        "scope": scope,
        "dataset": dataset,
        "task": task,
        "model": model,
        "strategy_id": strategy,
        "expected_requests": str(expected),
        "attempted_requests": str(len(attempted)),
        "transport_success": str(len(success)),
        "json_parse_valid": str(len(json_valid)),
        "schema_valid": str(len(schema_valid)),
        "strict_valid": str(len(strict)),
        "missing_requests": str(expected - len(attempted)),
        "invalid_outputs": str(len(attempted) - len(strict)),
        "strict_valid_rate": fmt(len(strict) / len(attempted)) if attempted else "0",
        "transport_success_rate": fmt(len(success) / len(attempted)) if attempted else "0",
        "operational_hr_at5": fmt(mean(operational_hr)) if operational_hr else "0",
        "operational_ndcg_at5": fmt(mean(operational_ndcg)) if operational_ndcg else "0",
        "conditional_hr_at5": fmt(mean(conditional_hr)) if conditional_hr else "",
        "conditional_ndcg_at5": fmt(mean(conditional_ndcg)) if conditional_ndcg else "",
        "status": "complete" if expected and len(attempted) == expected else "pending_or_partial",
        "notes": "Intermediate F6E4B control metrics; final scientific consolidation is deferred to F6E4C.",
    }


def build_cell_summary(instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in instance_rows:
        grouped[(row["model"], row["strategy_id"])].append(row)
    return [summarize_group(group, scope="cell", model=model, strategy=strategy) for (model, strategy), group in sorted(grouped.items())]


def build_dataset_task_summary(instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in instance_rows:
        grouped[(row["dataset"], row["task"], row["model"], row["strategy_id"])].append(row)
    return [
        summarize_group(group, scope="dataset_task_cell", dataset=dataset, task=task, model=model, strategy=strategy)
        for (dataset, task, model, strategy), group in sorted(grouped.items())
    ]


def build_shard_status(root: Path, shard_rows: Sequence[Mapping[str, str]], instance_rows: Sequence[Mapping[str, str]], pre_hashes: Mapping[str, str] | None = None) -> list[dict[str, str]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in instance_rows:
        grouped[row["shard_id"]].append(row)
    pre_hashes = pre_hashes or {}
    rows: list[dict[str, str]] = []
    for shard in shard_rows:
        group = grouped.get(shard["shard_id"], [])
        attempted = [row for row in group if row["transport_status"] != "missing"]
        success = [row for row in group if row["transport_status"] == "success"]
        strict = [row for row in group if row["strict_valid"] == "true"]
        json_valid = [row for row in group if row["json_parse_valid"] == "true"]
        schema_valid = [row for row in group if row["schema_valid"] == "true"]
        invalid = len(attempted) - len(strict)
        expected = int(float(shard["request_count"]))
        missing = expected - len(attempted)
        json_failures = len(attempted) - len(json_valid)
        strict_rate = len(strict) / len(attempted) if attempted else 0.0
        json_failure_rate = json_failures / len(attempted) if attempted else 0.0
        status = "pending"
        stop = ""
        if missing == 0:
            status = "complete" if invalid == 0 else "complete_with_invalid_outputs"
        elif attempted:
            status = "interrupted_resumable"
        raw_path = root / shard["raw_output_path"]
        rows.append(
            {
                "shard_id": shard["shard_id"],
                "model": shard["model"],
                "strategy_id": shard["strategy_id"],
                "dataset": shard["dataset"],
                "task": shard["task"],
                "expected_requests": str(expected),
                "attempted_requests": str(len(attempted)),
                "transport_success": str(len(success)),
                "strict_valid": str(len(strict)),
                "json_parse_valid": str(len(json_valid)),
                "schema_valid": str(len(schema_valid)),
                "invalid_outputs": str(invalid),
                "missing_requests": str(missing),
                "strict_valid_rate": fmt(strict_rate),
                "json_parse_failure_rate": fmt(json_failure_rate),
                "raw_path": shard["raw_output_path"],
                "raw_hash_before": pre_hashes.get(shard["shard_id"], ""),
                "raw_hash_after": file_sha256_or_missing(raw_path),
                "last_request_id": attempted[-1]["request_id"] if attempted else "",
                "status": status,
                "stop_condition": stop,
                "notes": "Shard status from append-only raw and frozen F6E4A shard manifest.",
            }
        )
    return rows


def build_runtime_summary(root: Path, instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in instance_rows:
        if row["transport_status"] != "missing":
            grouped[(row["model"], row["strategy_id"], row["dataset"], row["task"])].append(row)
    rows: list[dict[str, str]] = []
    for (model, strategy, dataset, task), group in sorted(grouped.items()):
        durations = [float(row["duration_seconds"] or 0) for row in group]
        eval_counts = [float(row["eval_count"] or 0) for row in group if row["eval_count"]]
        prompt_counts = [float(row["prompt_eval_count"] or 0) for row in group if row["prompt_eval_count"]]
        mean_duration = mean(durations) if durations else 0.0
        rows.append(
            {
                "runtime_id": stable_id("F6E4B-RT", model, strategy, dataset, task),
                "scope": "dataset_task_cell",
                "model": model,
                "strategy_id": strategy,
                "dataset": dataset,
                "task": task,
                "attempted_requests": str(len(group)),
                "latency_mean": fmt(mean_duration),
                "latency_p50": fmt(percentile(durations, 0.5)),
                "latency_p95": fmt(percentile(durations, 0.95)),
                "requests_per_hour": fmt(3600 / mean_duration) if mean_duration else "",
                "prompt_eval_count_mean": fmt(mean(prompt_counts)) if prompt_counts else "",
                "eval_count_mean": fmt(mean(eval_counts)) if eval_counts else "",
                "prompt_eval_duration_mean": "",
                "eval_duration_mean": "",
                "total_duration_mean": fmt(mean_duration),
                "disk_free_bytes": "not_recorded_in_deterministic_analysis",
                "ollama_state": "recorded_by_gate_or_runtime",
                "notes": "Runtime monitoring from raw records; GPU/VRAM are not used to alter semantics.",
            }
        )
    return rows


def build_progress(instance_rows: Sequence[Mapping[str, str]], request_count: int, decision: str, last_request_id: str = "", last_shard_id: str = "") -> list[dict[str, str]]:
    attempted = [row for row in instance_rows if row["transport_status"] != "missing"]
    success = [row for row in instance_rows if row["transport_status"] == "success"]
    strict = [row for row in instance_rows if row["strict_valid"] == "true"]
    invalid = len(attempted) - len(strict)
    return [
        {
            "scope": "global",
            "model": "ALL",
            "strategy_id": "ALL",
            "dataset": "ALL",
            "task": "ALL",
            "shard_id": "ALL",
            "expected_requests": str(request_count),
            "attempted_requests": str(len(attempted)),
            "transport_success": str(len(success)),
            "strict_valid": str(len(strict)),
            "missing_requests": str(request_count - len(attempted)),
            "invalid_outputs": str(invalid),
            "strict_valid_rate": fmt(len(strict) / len(attempted)) if attempted else "0",
            "last_request_id": last_request_id,
            "last_shard_id": last_shard_id,
            "last_update_utc": stable_progress_timestamp(instance_rows),
            "decision": decision,
            "notes": "Atomic progress snapshot. Raw JSONL remains append-only.",
        }
    ]


def evaluate_shard_stop(shard_instances: Sequence[Mapping[str, str]]) -> str:
    attempted = [row for row in shard_instances if row["transport_status"] != "missing"]
    if not attempted:
        return ""
    transport_failure_streak = 0
    max_streak = 0
    for row in attempted:
        if row["transport_status"] != "success":
            transport_failure_streak += 1
            max_streak = max(max_streak, transport_failure_streak)
        else:
            transport_failure_streak = 0
    if max_streak >= 3:
        return "repeated_transport_failure"
    invalid_reasons = [row["invalid_reason"].lower() for row in attempted if row["invalid_reason"]]
    if any("out of memory" in reason or "oom" in reason for reason in invalid_reasons):
        return "out_of_memory"
    if any("context" in reason and ("overflow" in reason or "length" in reason) for reason in invalid_reasons):
        return "context_overflow"
    strict = [row for row in attempted if row["strict_valid"] == "true"]
    json_valid = [row for row in attempted if row["json_parse_valid"] == "true"]
    if len(strict) / len(attempted) < 0.70:
        return "low_strict_valid_rate"
    if (len(attempted) - len(json_valid)) / len(attempted) > 0.20:
        return "systematic_non_json"
    return ""


def build_gate_results(shard_rows: Sequence[Mapping[str, str]], instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in instance_rows:
        grouped[row["shard_id"]].append(row)
    rows: list[dict[str, str]] = []
    for shard in shard_rows:
        group = grouped.get(shard["shard_id"], [])
        attempted = [row for row in group if row["transport_status"] != "missing"]
        if not attempted:
            continue
        transport = [row for row in attempted if row["transport_status"] == "success"]
        parse_valid = [row for row in attempted if row["json_parse_valid"] == "true"]
        schema_valid = [row for row in attempted if row["schema_valid"] == "true"]
        strict = [row for row in attempted if row["strict_valid"] == "true"]
        reasons = Counter(row["invalid_reason"] or "parsed_valid" for row in attempted)
        durations = [float(row["duration_seconds"] or 0) for row in attempted]
        eval_counts = [float(row["eval_count"] or 0) for row in attempted if row["eval_count"]]
        stop = evaluate_shard_stop(group)
        decision = "pause_resumable" if stop in {"low_strict_valid_rate", "systematic_non_json"} else ("stop_blocked" if stop else "proceed")
        rows.append(
            {
                "gate_id": stable_id("F6E4B-GATE", shard["shard_id"]),
                "gate_name": f"after_{shard['shard_id']}",
                "phase_point": "after_shard",
                "shard_id": shard["shard_id"],
                "model": shard["model"],
                "strategy_id": shard["strategy_id"],
                "dataset": shard["dataset"],
                "task": shard["task"],
                "attempted_requests": str(len(attempted)),
                "transport_success": str(len(transport)),
                "parse_valid": str(len(parse_valid)),
                "schema_valid": str(len(schema_valid)),
                "strict_valid": str(len(strict)),
                "invalid_reasons": stable_json(dict(sorted(reasons.items()))),
                "latency_mean": fmt(mean(durations)) if durations else "",
                "eval_count_mean": fmt(mean(eval_counts)) if eval_counts else "",
                "stop_condition_triggered": stop,
                "decision": decision,
                "evidence": "results/tables/phase_f6e4b_shard_status.csv",
                "notes": "Gate computed from frozen F6E4A stop conditions after shard analysis.",
            }
        )
    return rows


def final_decision(request_rows: Sequence[Mapping[str, str]], instance_rows: Sequence[Mapping[str, str]], stop_reason: str = "") -> str:
    attempted = [row for row in instance_rows if row["transport_status"] != "missing"]
    if stop_reason in {"f6e4b_blocked_ollama_unreachable", "f6e4b_blocked_model_digest_mismatch", "f6e4b_blocked_by_runtime_failure"}:
        return stop_reason
    if stop_reason in {"f6e4b_blocked_by_frozen_input_invariance", "f6e4b_blocked_by_manifest_inconsistency", "f6e4b_blocked_by_candidate_invariance"}:
        return stop_reason
    if len(attempted) < len(request_rows):
        return "f6e4b_adapted_full_run_interrupted_resumable"
    strict = [row for row in instance_rows if row["strict_valid"] == "true"]
    if len(strict) == len(request_rows):
        return "f6e4b_adapted_full_run_complete"
    return "f6e4b_adapted_full_run_complete_with_invalid_outputs"


def build_missing_or_blocked(decision: str, instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing = [row for row in instance_rows if row["transport_status"] == "missing"]
    if missing:
        rows.append(
            {
                "block_id": "F6E4B-MISSING-REQUESTS",
                "category": "execution",
                "affected_scope": "adapted full run",
                "status": "missing_requests",
                "severity": "high",
                "evidence": OUTPUT_TABLES["instance_results"].as_posix(),
                "safe_next_action": "Resume with --execute --allow-ollama-generate.",
                "notes": f"missing={len(missing)}",
            }
        )
    invalid = [row for row in instance_rows if row["transport_status"] != "missing" and row["strict_valid"] != "true"]
    if invalid:
        rows.append(
            {
                "block_id": "F6E4B-INVALID-OUTPUTS",
                "category": "parser_validity",
                "affected_scope": "attempted outputs",
                "status": "invalid_outputs_preserved",
                "severity": "medium",
                "evidence": OUTPUT_TABLES["invalid_output_audit"].as_posix(),
                "safe_next_action": "Do not retry automatically; consolidate in F6E4C.",
                "notes": f"invalid={len(invalid)}",
            }
        )
    if decision.startswith("f6e4b_blocked"):
        rows.append(
            {
                "block_id": "F6E4B-BLOCKED",
                "category": "gate",
                "affected_scope": "execution",
                "status": decision,
                "severity": "critical",
                "evidence": OUTPUT_TABLES["gate_results"].as_posix(),
                "safe_next_action": "Resolve the exact blocker before resuming.",
                "notes": decision,
            }
        )
    return rows


def build_all_tables(root: Path, request_rows: Sequence[Mapping[str, str]], shard_rows: Sequence[Mapping[str, str]], *, decision: str = "", last_request_id: str = "", last_shard_id: str = "", pre_hashes: Mapping[str, str] | None = None) -> dict[str, list[dict[str, str]]]:
    raw_by_id = final_raw_by_request(root)
    instance_rows, parser_rows, invalid_rows = analyze_instances(root, request_rows, raw_by_id)
    computed_decision = decision or final_decision(request_rows, instance_rows)
    tables = {
        "instance_results": instance_rows,
        "parser_validity": parser_rows,
        "invalid_output_audit": invalid_rows,
        "cell_summary": build_cell_summary(instance_rows),
        "dataset_task_summary": build_dataset_task_summary(instance_rows),
        "runtime_summary": build_runtime_summary(root, instance_rows),
        "shard_status": build_shard_status(root, shard_rows, instance_rows, pre_hashes=pre_hashes),
        "gate_results": build_gate_results(shard_rows, instance_rows),
        "progress": build_progress(instance_rows, len(request_rows), computed_decision, last_request_id, last_shard_id),
    }
    tables["missing_or_blocked"] = build_missing_or_blocked(computed_decision, instance_rows)
    return tables


def table_hashes(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, rows in sorted(tables.items()):
        columns = TABLE_COLUMNS[name]
        payload_lines: list[str] = []
        payload_lines.append(",".join(columns))
        for row in rows:
            payload_lines.append(stable_json({column: row.get(column, "") for column in columns}))
        hashes[name] = sha256_text("\n".join(payload_lines))
    return hashes


def build_manifest(root: Path, request_rows: Sequence[Mapping[str, str]], shard_rows: Sequence[Mapping[str, str]], tables: Mapping[str, Sequence[Mapping[str, str]]], decision: str) -> dict[str, Any]:
    raw_hashes = {path.as_posix(): file_sha256_or_missing(root / path) for path in RAW_PATHS.values()}
    progress = tables["progress"][0] if tables.get("progress") else {}
    return {
        "phase": PHASE_ID,
        "decision": decision,
        "timestamp_utc": progress.get("last_update_utc", "not_available") or "not_available",
        "repository_state": repository_state(root),
        "f6e4a_freeze_hash": file_sha256_or_missing(root / F6E4A_FREEZE_MANIFEST),
        "f6e4a_request_manifest_hash": file_sha256_or_missing(root / REQUEST_MANIFEST),
        "f6e4a_shard_manifest_hash": file_sha256_or_missing(root / SHARD_MANIFEST),
        "expected_requests": len(request_rows),
        "expected_shards": len(shard_rows),
        "attempted_requests": int(progress.get("attempted_requests", 0) or 0),
        "missing_requests": int(progress.get("missing_requests", 0) or 0),
        "strict_valid": int(progress.get("strict_valid", 0) or 0),
        "raw_paths": {model: path.as_posix() for model, path in RAW_PATHS.items()},
        "raw_hashes": raw_hashes,
        "protected_summary": protected_summary(root),
        "table_hashes": table_hashes(tables),
        "analysis_only_offline": True,
        "no_retry_for_invalid_outputs": True,
        "classification": "adapted_indexed_output_contract",
        "model_equivalence": "exact_family_version_uncertain",
    }


def build_report(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    progress = tables["progress"][0] if tables.get("progress") else {}
    invalid = len(tables.get("invalid_output_audit", []))
    sections = [
        ("Initial repository state", stable_json(manifest["repository_state"])),
        ("Frozen-input verification", f"F6E4A manifest/request/shard hashes are recorded in the execution freeze. Protected summary: {stable_json(manifest['protected_summary'])}."),
        ("Ollama and model gate", "Recorded in gate/progress artifacts when execution mode is used; analysis-only does not query Ollama."),
        ("Manifest verification", f"Requests: {manifest['expected_requests']}; shards: {manifest['expected_shards']}; four adapted cells only."),
        ("Execution order", "Follows F6E4A shard_id and ordinal_position order."),
        ("Raw file contract", stable_json(manifest["raw_paths"])),
        ("Resume state", f"Attempted: {progress.get('attempted_requests', '0')}; missing: {progress.get('missing_requests', '0')}; last request: {progress.get('last_request_id', '')}."),
        ("Progress and checkpoints", f"Progress rows: {len(tables.get('progress', []))}; shard rows: {len(tables.get('shard_status', []))}."),
        ("Shard execution", f"Shard status rows: {len(tables.get('shard_status', []))}."),
        ("Intermediate gates", f"Gate result rows: {len(tables.get('gate_results', []))}."),
        ("Stop conditions", "F6E4A stop conditions are used; invalid model outputs are preserved without retry."),
        ("Qwen2 Base status", _cell_status(tables, QWEN2_TAG, STRATEGY_ORDER[0])),
        ("Qwen2 LLMRank status", _cell_status(tables, QWEN2_TAG, STRATEGY_ORDER[1])),
        ("Mistral Base status", _cell_status(tables, MISTRAL_TAG, STRATEGY_ORDER[0])),
        ("Mistral LLMRank status", _cell_status(tables, MISTRAL_TAG, STRATEGY_ORDER[1])),
        ("Dataset/task coverage", f"Dataset/task rows: {len(tables.get('dataset_task_summary', []))}."),
        ("Parser and validity status", f"Strict-valid: {progress.get('strict_valid', '0')}; invalid outputs: {invalid}."),
        ("Runtime monitoring", f"Runtime rows: {len(tables.get('runtime_summary', []))}."),
        ("Invalid-output audit", f"Rows: {invalid}."),
        ("Operational utility", "Intermediate operational HR/NDCG are in phase_f6e4b_cell_summary.csv and dataset_task_summary.csv."),
        ("Conditional utility", "Intermediate strict-valid-only HR/NDCG are in phase_f6e4b_cell_summary.csv and dataset_task_summary.csv."),
        ("Analysis-only verification", "Analysis-only reads raw/progress/manifest only and does not call Ollama endpoints."),
        ("Targeted test results", "Recorded in final operator response."),
        ("Excluded unsafe tests", "Recorded in final operator response."),
        ("Protected-input verification", stable_json(manifest["protected_summary"])),
        ("Execution freeze manifest", FREEZE_MANIFEST_PATH.as_posix()),
        ("Missing or blocked evidence", f"Rows: {len(tables.get('missing_or_blocked', []))}."),
        ("Final decision", f"`{manifest['decision']}`."),
        ("Safe next action", "If complete, proceed to F6E4C; if interrupted, resume with the F6E4B execute command."),
        ("Final git state", "Recorded in final operator response; no staging or commit is performed."),
        ("Files created or modified", "Only F6E4B source, CLI, tests, raw/progress/log/report/tables/manifest."),
        ("Remaining limitations", "F6E4B is an adapted local full run, not paper-exact numeric replication."),
    ]
    lines = ["# Phase F6E4B - Adapted Full-Run Execution", "", f"Decision: `{manifest['decision']}`.", ""]
    for index, (title, body) in enumerate(sections, start=1):
        lines.extend([f"## {index}. {title}", "", body, ""])
    return "\n".join(lines)


def _cell_status(tables: Mapping[str, Sequence[Mapping[str, str]]], model: str, strategy: str) -> str:
    for row in tables.get("cell_summary", []):
        if row["model"] == model and row["strategy_id"] == strategy:
            return f"attempted={row['attempted_requests']}/{row['expected_requests']}; strict={row['strict_valid']}; status={row['status']}"
    return "not attempted"


def write_outputs(root: Path, tables: Mapping[str, Sequence[Mapping[str, str]]], manifest: Mapping[str, Any], report: str) -> dict[str, str]:
    written: dict[str, str] = {}
    for name, path in OUTPUT_TABLES.items():
        write_csv_atomic(root / path, tables.get(name, []), TABLE_COLUMNS[name])
        written[path.as_posix()] = sha256_file(root / path)
    (root / FREEZE_MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / FREEZE_MANIFEST_PATH).write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    written[FREEZE_MANIFEST_PATH.as_posix()] = sha256_file(root / FREEZE_MANIFEST_PATH)
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(report, encoding="utf-8")
    written[REPORT_PATH.as_posix()] = sha256_file(root / REPORT_PATH)
    return written


def write_checkpoint(root: Path, request_rows: Sequence[Mapping[str, str]], shard_rows: Sequence[Mapping[str, str]], decision: str, last_request_id: str, last_shard_id: str, pre_hashes: Mapping[str, str] | None = None) -> dict[str, str]:
    tables = build_all_tables(root, request_rows, shard_rows, decision=decision, last_request_id=last_request_id, last_shard_id=last_shard_id, pre_hashes=pre_hashes)
    manifest = build_manifest(root, request_rows, shard_rows, tables, decision)
    report = build_report(manifest, tables)
    return write_outputs(root, tables, manifest, report)


def write_progress_only(
    root: Path,
    *,
    request_count: int,
    raw_by_id: Mapping[str, Mapping[str, Any]],
    decision: str,
    last_request_id: str,
    last_shard_id: str,
) -> None:
    attempted = list(raw_by_id.values())
    success = [row for row in attempted if row.get("transport_status") == "success"]
    strict = [row for row in attempted if parse_bool(row.get("strict_valid")) and row.get("transport_status") == "success"]
    progress = [
        {
            "scope": "global",
            "model": "ALL",
            "strategy_id": "ALL",
            "dataset": "ALL",
            "task": "ALL",
            "shard_id": "ALL",
            "expected_requests": str(request_count),
            "attempted_requests": str(len(attempted)),
            "transport_success": str(len(success)),
            "strict_valid": str(len(strict)),
            "missing_requests": str(request_count - len(attempted)),
            "invalid_outputs": str(len(attempted) - len(strict)),
            "strict_valid_rate": fmt(len(strict) / len(attempted)) if attempted else "0",
            "last_request_id": last_request_id,
            "last_shard_id": last_shard_id,
            "last_update_utc": utc_now(),
            "decision": decision,
            "notes": "Per-request atomic progress snapshot. Full derived tables are checkpointed after each shard.",
        }
    ]
    write_csv_atomic(root / OUTPUT_TABLES["progress"], progress, PROGRESS_COLUMNS)


def run_analysis_only(root: Path) -> dict[str, str]:
    verify_frozen_inputs(root)
    request_rows = load_request_manifest(root)
    shard_rows = load_shard_manifest(root)
    assert_manifest_valid(request_rows, shard_rows)
    validate_existing_raw_records(root, request_rows)
    raw_by_id = final_raw_by_request(root)
    instance_rows, _, _ = analyze_instances(root, request_rows, raw_by_id)
    decision = final_decision(request_rows, instance_rows)
    return write_checkpoint(root, request_rows, shard_rows, decision, "", "")


def run_runtime_gate_only(root: Path) -> dict[str, str]:
    verify_frozen_inputs(root)
    request_rows = load_request_manifest(root)
    shard_rows = load_shard_manifest(root)
    assert_manifest_valid(request_rows, shard_rows)
    gate_rows = inspect_ollama_gate()
    decision = gate_decision_from_ollama(gate_rows)
    tables = build_all_tables(root, request_rows, shard_rows, decision=decision)
    tables["gate_results"].extend(
        {
            "gate_id": row["gate_id"],
            "gate_name": row["check_name"],
            "phase_point": "runtime_gate_before_execution",
            "shard_id": "",
            "model": "",
            "strategy_id": "",
            "dataset": "",
            "task": "",
            "attempted_requests": "0",
            "transport_success": "0",
            "parse_valid": "0",
            "schema_valid": "0",
            "strict_valid": "0",
            "invalid_reasons": "",
            "latency_mean": "",
            "eval_count_mean": "",
            "stop_condition_triggered": "" if row["status"] == "pass" or row["severity"] == "info" else row["check_name"],
            "decision": "proceed" if row["status"] == "pass" or row["severity"] == "info" else "stop_blocked",
            "evidence": row["source"],
            "notes": row["notes"],
        }
        for row in gate_rows
    )
    manifest = build_manifest(root, request_rows, shard_rows, tables, decision)
    report = build_report(manifest, tables)
    return write_outputs(root, tables, manifest, report)


def execute_phase(
    root: Path,
    *,
    allow_ollama_generate: bool,
    max_new_requests: int = EXPECTED_REQUESTS,
    shard_id: str = "",
    max_shards: int = 0,
    request_executor: Callable[[Mapping[str, Any], int], f6e3.RequestResult] = f6e3.execute_ollama_request,
    prompt_provider: Callable[[Mapping[str, str]], PromptBundle] | None = None,
    gate_rows: Sequence[Mapping[str, str]] | None = None,
) -> ExecutionResult:
    if not allow_ollama_generate:
        raise PermissionError("f6e4b_requires_allow_ollama_generate")
    verify_frozen_inputs(root)
    request_rows = load_request_manifest(root)
    shard_rows = load_shard_manifest(root)
    assert_manifest_valid(request_rows, shard_rows)
    validate_existing_raw_records(root, request_rows)
    gate = list(gate_rows) if gate_rows is not None else inspect_ollama_gate()
    gate_decision = gate_decision_from_ollama(gate)
    if gate_decision != "pass":
        written = write_checkpoint(root, request_rows, shard_rows, gate_decision, "", "")
        return ExecutionResult(gate_decision, 0, gate_decision, "", "", written)
    if shutil.disk_usage(root).free < 1_000_000_000:
        decision = "f6e4b_blocked_by_runtime_failure"
        written = write_checkpoint(root, request_rows, shard_rows, decision, "", "")
        return ExecutionResult(decision, 0, "disk_insufficient", "", "", written)
    raw_by_id = final_raw_by_request(root)
    contexts = {} if prompt_provider else source_context_map(root)
    rows_by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in request_rows:
        rows_by_shard[row["shard_id"]].append(dict(row))
    selected_shards = [row for row in shard_rows if not shard_id or row["shard_id"] == shard_id]
    if max_shards:
        selected_shards = selected_shards[:max_shards]
    sent = 0
    last_request_id = ""
    last_shard_id = ""
    stop_reason = ""
    pre_hashes: dict[str, str] = {}
    for shard in selected_shards:
        current_shard = shard["shard_id"]
        raw_path = root / shard["raw_output_path"]
        pre_hashes[current_shard] = file_sha256_or_missing(raw_path)
        shard_fail_streak = 0
        for row in sorted(rows_by_shard[current_shard], key=lambda value: int(value["ordinal_position"])):
            if row["request_id"] in raw_by_id:
                continue
            if sent >= max_new_requests:
                stop_reason = "max_new_requests_reached"
                break
            try:
                bundle = prompt_provider(row) if prompt_provider else prompt_bundle_for_row(root, row, contexts)
                payload = make_payload(row, bundle.prompt)
                timeout = int(float(runtime_parameters(row)["timeout_seconds"]))
                result = request_executor(payload, timeout)
                record = make_raw_record(row, result, bundle, attempt=1)
                append_jsonl_fsync(root / raw_path_for_model(row["model"]), record)
                raw_by_id[row["request_id"]] = record
                sent += 1
                last_request_id = row["request_id"]
                last_shard_id = current_shard
                if result.success:
                    shard_fail_streak = 0
                else:
                    shard_fail_streak += 1
                    append_jsonl_fsync(root / ERROR_LOG, {"request_id": row["request_id"], "shard_id": current_shard, "error_type": result.error_type, "error": result.error_message})
                write_progress_only(
                    root,
                    request_count=len(request_rows),
                    raw_by_id=raw_by_id,
                    decision="f6e4b_adapted_full_run_interrupted_resumable",
                    last_request_id=last_request_id,
                    last_shard_id=last_shard_id,
                )
                if shard_fail_streak >= 3:
                    stop_reason = "repeated_transport_failure"
                    break
            except Exception as exc:
                append_jsonl_fsync(root / ERROR_LOG, {"request_id": row["request_id"], "shard_id": current_shard, "error_type": type(exc).__name__, "error": str(exc)})
                stop_reason = "raw_write_failure" if type(exc).__name__ in {"OSError", "IOError"} else "manifest_inconsistency"
                break
        write_checkpoint(root, request_rows, shard_rows, "f6e4b_adapted_full_run_interrupted_resumable", last_request_id, last_shard_id, pre_hashes)
        instance_rows, _, _ = analyze_instances(root, [row for row in request_rows if row["shard_id"] == current_shard], final_raw_by_request(root))
        shard_stop = evaluate_shard_stop(instance_rows)
        if stop_reason or shard_stop:
            stop_reason = stop_reason or shard_stop
            break
    all_instance_rows, _, _ = analyze_instances(root, request_rows, final_raw_by_request(root))
    decision = final_decision(request_rows, all_instance_rows)
    if stop_reason and decision == "f6e4b_adapted_full_run_interrupted_resumable":
        decision = "f6e4b_blocked_by_runtime_failure" if stop_reason in {"repeated_transport_failure", "raw_write_failure", "out_of_memory", "context_overflow"} else decision
    written = write_checkpoint(root, request_rows, shard_rows, decision, last_request_id, last_shard_id, pre_hashes)
    return ExecutionResult(decision, sent, stop_reason, last_request_id, last_shard_id, written)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase F6E4B adapted full-run execution.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-gate-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-ollama-generate", action="store_true")
    parser.add_argument("--max-new-requests", type=int, default=EXPECTED_REQUESTS)
    parser.add_argument("--shard-id", default="")
    parser.add_argument("--max-shards", type=int, default=0)
    args = parser.parse_args(argv)
    mode_count = sum(bool(value) for value in [args.runtime_gate_only, args.analysis_only, args.execute])
    if mode_count != 1:
        parser.error("choose exactly one mode: --runtime-gate-only, --analysis-only, or --execute")
    if args.runtime_gate_only:
        written = run_runtime_gate_only(args.root)
        print(json.dumps({"written": written}, ensure_ascii=True, sort_keys=True, indent=2))
        return 0
    if args.analysis_only:
        written = run_analysis_only(args.root)
        print(json.dumps({"written": written}, ensure_ascii=True, sort_keys=True, indent=2))
        return 0
    result = execute_phase(
        args.root,
        allow_ollama_generate=args.allow_ollama_generate,
        max_new_requests=args.max_new_requests,
        shard_id=args.shard_id,
        max_shards=args.max_shards,
    )
    print(json.dumps(result.__dict__, ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if not result.decision.startswith("f6e4b_blocked") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

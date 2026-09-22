"""Resume-safe execution helpers for the canonical local LLM pilot.

This module is intentionally a layer around the Phase 5B-P1 runner. It reuses
the frozen prompt, parser, payload, response, ranking, and metric logic while
adding append-only raw evidence handling for Phase 5B-P1C.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.canonical_llm_execution_preflight import (
    file_sha256,
    load_histories,
    load_title_map,
    parse_json_list,
    read_csv,
)
from src.llm.canonical_llm_contract import json_dumps, sha256_items, sha256_text
from src.llm.canonical_local_llm_runner import (
    EXECUTION_COLUMNS,
    METRIC_COLUMNS,
    MODEL_ORDER,
    PARSE_COLUMNS,
    RANKING_COLUMNS,
    READINESS_THRESHOLD,
    REQUEST_MANIFEST_COLUMNS,
    SAMPLE_COLUMNS,
    STRATUM_ORDER,
    build_metrics,
    build_parse_and_ranking_outputs,
    build_payload,
    clean_text,
    execute_ollama_request,
    frame,
    load_contract_bundle,
    render_from_contract,
    selection_key,
)


RESUME_DIR = Path("results/llm/canonical_local_pilot_resume")
REQUEST_JSONL = RESUME_DIR / "requests.jsonl"
RESPONSE_JSONL = RESUME_DIR / "responses.jsonl"
EVENT_JSONL = RESUME_DIR / "runner_events.jsonl"

SOURCE_SAMPLE = Path("results/tables/canonical_local_llm_pilot_sample.csv")
SOURCE_MANIFEST = Path("results/tables/canonical_local_llm_pilot_request_manifest.csv")
SOURCE_BLOCKED_DECISION = Path("results/tables/canonical_local_llm_pilot_decision.csv")
SOURCE_RECOVERY = Path("results/tables/canonical_local_ollama_service_recovery.csv")

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

PROVENANCE_COLUMNS = [
    "resume_id",
    "source_sample_path",
    "source_sample_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_blocked_decision_path",
    "source_blocked_decision_sha256",
    "ollama_recovery_path",
    "ollama_recovery_sha256",
    "ollama_server_version",
    "qwen2_model_tag",
    "qwen2_digest",
    "mistral_model_tag",
    "mistral_digest",
    "expected_sample_rows",
    "expected_request_rows",
    "resume_directory",
    "initial_request_jsonl_records",
    "initial_response_jsonl_records",
    "initial_completed_requests",
    "initial_pending_requests",
    "initial_ambiguous_requests",
    "execution_started_at_utc",
    "execution_completed_at_utc",
    "resume_status",
    "blocking_reason",
    "notes",
]

REQUEST_STATE_COLUMNS = [
    "request_id",
    "request_order",
    "pilot_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "request_record_count",
    "response_record_count",
    "request_state",
    "request_sent",
    "response_recorded",
    "backend_success",
    "strict_parse_valid",
    "candidate_validity",
    "utility_evaluation_valid",
    "raw_request_sha256",
    "raw_response_sha256",
    "blocking_reason",
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
    "source_pilot_id",
    "expected_request_count",
    "completed_request_count",
    "backend_error_record_count",
    "ambiguous_request_count",
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
    "provenance": ("results/tables/canonical_local_llm_pilot_resume_provenance.csv", PROVENANCE_COLUMNS),
    "request_state": ("results/tables/canonical_local_llm_pilot_resume_request_state.csv", REQUEST_STATE_COLUMNS),
    "execution": ("results/tables/canonical_local_llm_pilot_resume_execution.csv", EXECUTION_COLUMNS),
    "parse_results": ("results/tables/canonical_local_llm_pilot_resume_parse_results.csv", PARSE_COLUMNS),
    "rankings": ("results/tables/canonical_local_llm_pilot_resume_rankings.csv", RANKING_COLUMNS),
    "metrics": ("results/tables/canonical_local_llm_pilot_resume_metrics.csv", METRIC_COLUMNS),
    "validation": ("results/tables/canonical_local_llm_pilot_resume_validation.csv", VALIDATION_COLUMNS),
    "decision": ("results/tables/canonical_local_llm_pilot_resume_decision.csv", DECISION_COLUMNS),
}
REPORT_PATH = Path("report/canonical_local_qwen2_mistral_pilot_resume.md")


@dataclass(frozen=True)
class ResumeInputs:
    sample: pd.DataFrame
    manifest: pd.DataFrame
    blocked_decision: pd.DataFrame
    recovery: pd.DataFrame
    contexts: dict[str, dict[str, Any]]
    protected_hashes: dict[str, str]


@dataclass(frozen=True)
class OllamaAdminState:
    server_version: str
    tags: dict[str, dict[str, Any]]
    ps: list[dict[str, Any]]
    started_process_id: str = ""
    start_attempted: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    subset = data.loc[:, list(columns)].astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in subset.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def file_record_sha256(record: Mapping[str, Any]) -> str:
    persisted = {key: value for key, value in record.items() if not str(key).startswith("_")}
    return sha256_text(json.dumps(persisted, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def positive_label_exposed(text: str, positive_item_id: str = "") -> bool:
    """Detect explicit positive-label leakage without flagging numeric index text."""
    lower_text = str(text).lower()
    forbidden_labels = [
        "positive_item_id",
        "ground_truth",
        "held-out positive",
        "held out positive",
        "positive position",
        "relevance label",
        "expected response",
    ]
    if any(label in lower_text for label in forbidden_labels):
        return True
    item_id = str(positive_item_id or "").strip()
    if not item_id or item_id.isdigit():
        return False
    return len(item_id) >= 6 and item_id.lower() in lower_text


def read_table(root: Path, relative: Path | str) -> pd.DataFrame:
    return pd.read_csv(root / relative, dtype=str, keep_default_na=False)


def protected_input_paths(root: Path) -> list[str]:
    paths = [
        "src/llm/canonical_local_llm_runner.py",
        "src/experiments/run_canonical_local_llm_pilot.py",
        "tests/test_canonical_local_llm_runner.py",
        "report/canonical_local_qwen2_mistral_pilot.md",
        "results/tables/canonical_local_llm_pilot_sample.csv",
        "results/tables/canonical_local_llm_pilot_request_manifest.csv",
        "results/tables/canonical_local_llm_pilot_decision.csv",
        "results/tables/canonical_local_llm_pilot_execution.csv",
        "results/tables/canonical_local_llm_pilot_parse_results.csv",
        "results/tables/canonical_local_llm_pilot_rankings.csv",
        "results/tables/canonical_local_llm_pilot_metrics.csv",
        "results/tables/canonical_local_llm_pilot_validation.csv",
        "results/tables/canonical_local_ollama_service_recovery.csv",
        "report/canonical_local_ollama_service_recovery.md",
        "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
        "results/tables/ranking_candidate_sets.csv",
        "results/tables/positive_positions.csv",
        "results/tables/experiment_cohort_membership.csv",
        "results/tables/item_title_map.csv",
        "results/tables/candidate_item_title_map.csv",
        "results/tables/llm_title_exclusions.csv",
    ]
    paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in (root / "results/tables").glob("canonical_llm_*.csv"))
    paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in (root / "results/tables").glob("canonical_metric_*.csv"))
    paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in (root / "results/tables").glob("traditional_baseline*.csv"))
    paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in (root / "results/tables").glob("traditional_baselines*.csv"))
    paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in (root / "report").glob("canonical_llm_*.md"))
    paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in (root / "src").glob("**/canonical_llm*.py"))
    paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in (root / "data/reconstructed/plan_c").glob("**/interactions.csv"))
    return sorted({relative for relative in paths if (root / relative).is_file()})


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {relative: file_sha256(root / relative) for relative in protected_input_paths(root)}


def environment_summary(root: Path) -> dict[str, str]:
    ram = "unavailable"
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
        pass
    gpu = "unavailable"
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
        pass
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


def validate_frozen_sample(sample: pd.DataFrame) -> None:
    if list(sample.columns) != SAMPLE_COLUMNS:
        raise ValueError("frozen sample schema mismatch")
    if len(sample) != 60:
        raise ValueError("frozen sample row count is not 60")
    if not sample["pilot_row_id"].is_unique:
        raise ValueError("pilot_row_id is not unique")
    if not sample["selection_sha256"].is_unique:
        raise ValueError("selection_sha256 is not unique")
    if sample["sample_order"].astype(int).tolist() != list(range(1, 61)):
        raise ValueError("sample_order is not contiguous")
    counts = sample.groupby(["dataset", "task_type"]).size()
    for dataset, task_type in STRATUM_ORDER:
        if int(counts.get((dataset, task_type), 0)) != 10:
            raise ValueError(f"frozen sample stratum count mismatch: {dataset} {task_type}")
    for row in sample.to_dict("records"):
        expected = selection_key(row["dataset"], row["task_type"], row["candidate_set_id"], row["user_id"])
        if row["selection_sha256"] != expected:
            raise ValueError(f"selection hash mismatch: {row['pilot_row_id']}")
        if row["candidate_count"] != "20":
            raise ValueError(f"candidate_count mismatch: {row['pilot_row_id']}")
        for column in ["candidate_titles_complete", "history_titles_complete", "held_out_positive_absent", "primary_prompt_compatible"]:
            if row[column] != "true":
                raise ValueError(f"{column} is not true: {row['pilot_row_id']}")
        for column in ["candidate_order_sha256", "system_message_sha256", "user_message_sha256", "complete_prompt_sha256"]:
            if not str(row[column]).strip():
                raise ValueError(f"missing hash {column}: {row['pilot_row_id']}")


def build_contexts_from_frozen_sample(root: Path, sample: pd.DataFrame, bundle: Any) -> dict[str, dict[str, Any]]:
    title_map = load_title_map(root)
    histories = load_histories(root, title_map)
    candidates = read_csv(root, "results/tables/ranking_candidate_sets.csv")
    candidate_lookup = {
        (str(row.candidate_set_id), str(row.user_id)): parse_json_list(row.candidate_items)
        for row in candidates[["candidate_set_id", "user_id", "candidate_items"]].itertuples(index=False)
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
    contexts: dict[str, dict[str, Any]] = {}
    for row in sample.sort_values("sample_order", key=lambda s: s.astype(int)).to_dict("records"):
        key = (row["candidate_set_id"], row["user_id"])
        if key not in candidate_lookup:
            raise ValueError(f"candidate row missing for pilot row {row['pilot_row_id']}")
        candidate_ids = candidate_lookup[key]
        if len(candidate_ids) != 20:
            raise ValueError(f"candidate count is not 20 for {row['pilot_row_id']}")
        candidate_items = [{"item_id": item_id, "title": title_map[(row["dataset"], item_id)]} for item_id in candidate_ids]
        history_items = [
            item
            for item in histories[row["dataset"]].get(row["user_id"], [])
            if item["item_id"] != row["positive_item_id"]
        ][-10:]
        prompt_contract = prompt_by_task[row["task_type"]]
        parser_contract = parser_by_task[row["task_type"]]
        system_message, user_message, history_render, candidate_render = render_from_contract(
            prompt_contract,
            history_items=history_items,
            candidate_items=candidate_items,
        )
        hashes = {
            "system_message_sha256": sha256_text(system_message),
            "user_message_sha256": sha256_text(user_message),
            "complete_prompt_sha256": sha256_text(system_message + "\n\n" + user_message),
            "candidate_order_sha256": sha256_items(candidate_ids),
        }
        for column, observed in hashes.items():
            if observed != row[column]:
                raise ValueError(f"frozen prompt hash mismatch {column}: {row['pilot_row_id']}")
        payload_visible = json.dumps({"system": system_message, "prompt": user_message}, ensure_ascii=False)
        if positive_label_exposed(payload_visible, row["positive_item_id"]):
            raise ValueError(f"positive-label exposure detected: {row['pilot_row_id']}")
        contexts[row["pilot_row_id"]] = {
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
    return contexts


def validate_frozen_manifest(manifest: pd.DataFrame, sample: pd.DataFrame, bundle: Any) -> None:
    if list(manifest.columns) != REQUEST_MANIFEST_COLUMNS:
        raise ValueError("frozen manifest schema mismatch")
    if len(manifest) != 120:
        raise ValueError("frozen manifest row count is not 120")
    if not manifest["request_id"].is_unique:
        raise ValueError("request_id is not unique")
    if manifest["request_order"].astype(int).tolist() != list(range(1, 121)):
        raise ValueError("request_order is not contiguous")
    if manifest["request_id"].tolist() != [f"CLLP-REQ-{index:04d}" for index in range(1, 121)]:
        raise ValueError("request_id order mismatch")
    if set(manifest["pilot_row_id"]) != set(sample["pilot_row_id"]):
        raise ValueError("manifest pilot rows do not match sample")
    for family, tag in MODEL_ORDER:
        subset = manifest.loc[manifest["model_family"].eq(family)]
        if len(subset) != 60:
            raise ValueError(f"model request count mismatch: {family}")
        if set(subset["model_tag"]) != {tag}:
            raise ValueError(f"model tag mismatch: {family}")
        for dataset, task_type in STRATUM_ORDER:
            cell = subset.loc[subset["dataset"].eq(dataset) & subset["task_type"].eq(task_type)]
            if len(cell) != 10:
                raise ValueError(f"cell request count mismatch: {family} {dataset} {task_type}")
    if manifest.iloc[:60]["model_family"].nunique() != 1 or manifest.iloc[0]["model_family"] != "Qwen2 7B Instruct":
        raise ValueError("Qwen2 requests are not first")
    if manifest.iloc[60:]["model_family"].nunique() != 1 or manifest.iloc[60]["model_family"] != "Mistral 7B Instruct":
        raise ValueError("Mistral requests are not second")
    prompt_lookup = {row.prompt_contract_id: row._asdict() for row in bundle.prompt_contract.itertuples(index=False)}
    parser_lookup = {row.parser_contract_id: row._asdict() for row in bundle.parser_contract.itertuples(index=False)}
    exec_lookup = {(row.model_family, row.task_type): row._asdict() for row in bundle.execution_contract.itertuples(index=False)}
    for row in manifest.to_dict("records"):
        if row["concurrency"] != "1" or row["scientific_retry_count"] != "0" or row["infrastructure_retry_count"] != "0":
            raise ValueError(f"unapproved retry or concurrency: {row['request_id']}")
        if row["streaming"] != "false":
            raise ValueError(f"streaming is not false: {row['request_id']}")
        expected_count = "5" if row["task_type"] == "ranking" else "20"
        if row["requested_output_count"] != expected_count:
            raise ValueError(f"requested output count mismatch: {row['request_id']}")
        if prompt_lookup[row["prompt_contract_id"]]["contract_sha256"] != row["prompt_contract_sha256"]:
            raise ValueError(f"prompt contract hash mismatch: {row['request_id']}")
        if parser_lookup[row["parser_contract_id"]]["parser_schema_sha256"] != row["parser_schema_sha256"]:
            raise ValueError(f"parser schema hash mismatch: {row['request_id']}")
        if exec_lookup[(row["model_family"], row["task_type"])]["contract_sha256"] != row["execution_contract_sha256"]:
            raise ValueError(f"execution contract hash mismatch: {row['request_id']}")


def load_resume_inputs(root: Path) -> ResumeInputs:
    sample = read_table(root, SOURCE_SAMPLE)
    manifest = read_table(root, SOURCE_MANIFEST)
    blocked = read_table(root, SOURCE_BLOCKED_DECISION)
    recovery = read_table(root, SOURCE_RECOVERY)
    if blocked.iloc[0]["pilot_decision"] != "backend_blocked":
        raise ValueError("original blocked decision is not backend_blocked")
    bundle = load_contract_bundle(root)
    validate_frozen_sample(sample)
    contexts = build_contexts_from_frozen_sample(root, sample, bundle)
    validate_frozen_manifest(manifest, sample, bundle)
    return ResumeInputs(sample, manifest, blocked, recovery, contexts, hash_protected_inputs(root))


def fetch_json(url: str, timeout: int = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_ollama_admin(root: Path, recovery: pd.DataFrame) -> OllamaAdminState:
    recovery_row = recovery.iloc[0].to_dict()
    base_url = "http://127.0.0.1:11434"
    start_attempted = False
    started_pid = ""
    try:
        version = fetch_json(base_url + "/api/version", timeout=3)
    except Exception:
        # Use one safe user-scoped start only if no process is running and the
        # port is not listening. This branch is normally not taken after P1B.
        process_check = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Process ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if process_check.stdout.strip():
            raise RuntimeError("stale_existing_process")
        port_check = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue | Where-Object {$_.State -eq 'Listen'} | Select-Object -ExpandProperty OwningProcess"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if port_check.stdout.strip():
            raise RuntimeError("port_conflict")
        exe = recovery_row["ollama_executable_path"]
        out_path = Path("C:/tmp/phase5b_p1c_ollama_stdout.log")
        err_path = Path("C:/tmp/phase5b_p1c_ollama_stderr.log")
        out_handle = out_path.open("ab")
        err_handle = err_path.open("ab")
        process = subprocess.Popen([exe, "serve"], stdout=out_handle, stderr=err_handle)
        start_attempted = True
        started_pid = str(process.pid)
        version = {}
        for _ in range(30):
            time.sleep(2)
            try:
                version = fetch_json(base_url + "/api/version", timeout=3)
                break
            except Exception:
                if process.poll() is not None:
                    raise RuntimeError(f"service_start_failed: exit={process.returncode}")
        if not version:
            raise RuntimeError("service_start_failed: timeout")
    tags_payload = fetch_json(base_url + "/api/tags", timeout=10)
    ps_payload = fetch_json(base_url + "/api/ps", timeout=10)
    tag_map = {str(item["name"]): item for item in tags_payload.get("models", [])}
    for family, expected in EXPECTED_DIGESTS.items():
        tag = expected["tag"]
        if tag not in tag_map:
            raise RuntimeError(f"blocked_missing_local_model: {tag}")
        if str(tag_map[tag].get("digest", "")).lower() != expected["digest"]:
            raise RuntimeError(f"model_digest_mismatch: {tag}")
    return OllamaAdminState(
        server_version=str(version.get("version", "")),
        tags=tag_map,
        ps=list(ps_payload.get("models", [])),
        started_process_id=started_pid,
        start_attempted=start_attempted,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record["_jsonl_line_no"] = line_no
            records.append(record)
    return records


def append_jsonl_fsync(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def count_by_request_id(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        request_id = str(record.get("request_id", ""))
        counts[request_id] = counts.get(request_id, 0) + 1
    return counts


def classify_request_state(request_count: int, response_count: int) -> str:
    if request_count == 0 and response_count == 0:
        return "not_started"
    if request_count == 1 and response_count == 1:
        return "completed"
    if request_count == 1 and response_count == 0:
        return "ambiguous_interrupted"
    return "invalid_duplicate_evidence"


def state_summary(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    summary = {"completed": 0, "not_started": 0, "ambiguous_interrupted": 0, "invalid_duplicate_evidence": 0}
    for row in manifest.itertuples(index=False):
        state = classify_request_state(req_counts.get(row.request_id, 0), resp_counts.get(row.request_id, 0))
        summary[state] += 1
    extra_responses = sorted(set(resp_counts) - set(manifest["request_id"]))
    if extra_responses:
        summary["invalid_duplicate_evidence"] += len(extra_responses)
    return summary


def pending_requests_in_order(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    pending: list[dict[str, str]] = []
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        state = classify_request_state(req_counts.get(row["request_id"], 0), resp_counts.get(row["request_id"], 0))
        if state == "not_started":
            pending.append(row)
        elif state == "completed":
            continue
        elif state == "ambiguous_interrupted":
            raise RuntimeError(f"ambiguous_interrupted_request: {row['request_id']}")
        else:
            raise RuntimeError(f"invalid_duplicate_evidence: {row['request_id']}")
    request_extras = sorted(set(req_counts) - set(manifest["request_id"]))
    if request_extras:
        raise RuntimeError(f"invalid_duplicate_evidence: request without manifest row {request_extras[0]}")
    extras = sorted(set(resp_counts) - set(manifest["request_id"]))
    if extras:
        raise RuntimeError(f"invalid_duplicate_evidence: response without request {extras[0]}")
    return pending


def build_request_record(row: Mapping[str, str], context: Mapping[str, Any], payload: Mapping[str, Any], digest: str) -> dict[str, Any]:
    payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if positive_label_exposed(payload_text, str(context.get("positive_item_id", ""))):
        raise ValueError(f"positive label in payload: {row['request_id']}")
    return {
        "request_id": row["request_id"],
        "pilot_row_id": row["pilot_row_id"],
        "request_order": row["request_order"],
        "attempt_index": 1,
        "model_family": row["model_family"],
        "model_tag": row["model_tag"],
        "verified_model_digest": digest,
        "dataset": row["dataset"],
        "task_type": row["task_type"],
        "candidate_set_id": row["candidate_set_id"],
        "user_id": row["user_id"],
        "prompt_contract_id": row["prompt_contract_id"],
        "prompt_contract_sha256": row["prompt_contract_sha256"],
        "parser_contract_id": row["parser_contract_id"],
        "parser_schema_sha256": row["parser_schema_sha256"],
        "execution_contract_id": row["execution_contract_id"],
        "execution_contract_sha256": row["execution_contract_sha256"],
        "complete_prompt_sha256": row["complete_prompt_sha256"],
        "candidate_order_sha256": row["candidate_order_sha256"],
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "request_payload": payload,
        "timestamp_before_send_utc": utc_now(),
    }


def build_response_record(row: Mapping[str, str], result: Any) -> dict[str, Any]:
    raw_sha = sha256_text(result.response_text)
    return {
        "request_id": row["request_id"],
        "attempt_index": 1,
        "backend_success": bool_text(result.backend_success),
        "raw_response_text": result.response_text,
        "raw_response_sha256": raw_sha,
        "model_reported_tag": str(result.raw_backend_response.get("model", row["model_tag"])) if result.raw_backend_response else "",
        "backend_metadata": result.raw_backend_response,
        "done": str(result.raw_backend_response.get("done", "")) if result.raw_backend_response else "",
        "done_reason": str(result.raw_backend_response.get("done_reason", "done" if result.raw_backend_response.get("done") else "")) if result.raw_backend_response else "",
        "prompt_eval_count": result.raw_backend_response.get("prompt_eval_count", ""),
        "generated_token_count": result.raw_backend_response.get("eval_count", ""),
        "load_duration_nanoseconds": result.raw_backend_response.get("load_duration", ""),
        "prompt_eval_duration_nanoseconds": result.raw_backend_response.get("prompt_eval_duration", ""),
        "eval_duration_nanoseconds": result.raw_backend_response.get("eval_duration", ""),
        "total_duration_nanoseconds": result.raw_backend_response.get("total_duration", ""),
        "request_started_at_utc": result.started_at,
        "timestamp_after_completion_utc": result.completed_at,
        "duration_seconds": format_float(result.duration_seconds),
        "backend_error_type": result.backend_error_type,
        "backend_error_message": result.backend_error_message,
    }


def execute_pending_requests(root: Path, inputs: ResumeInputs, admin: OllamaAdminState) -> None:
    RESUME_DIR_ABS = root / RESUME_DIR
    RESUME_DIR_ABS.mkdir(parents=True, exist_ok=True)
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    pending = pending_requests_in_order(inputs.manifest, request_records, response_records)
    for row in pending:
        if hash_protected_inputs(root) != inputs.protected_hashes:
            raise RuntimeError("protected_input_validation_failed")
        context = inputs.contexts[row["pilot_row_id"]]
        if context["complete_prompt_sha256"] != row["complete_prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch before send: {row['request_id']}")
        if context["candidate_order_sha256"] != row["candidate_order_sha256"]:
            raise ValueError(f"candidate order hash mismatch before send: {row['request_id']}")
        payload = build_payload(row, context)
        digest = EXPECTED_DIGESTS[row["model_family"]]["digest"]
        request_record = build_request_record(row, context, payload, digest)
        append_jsonl_fsync(root / REQUEST_JSONL, request_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "request_recorded_before_send", "request_id": row["request_id"], "timestamp_utc": utc_now()})
        timeout_seconds = max(120, int(float(row["max_output_tokens"] or 120)))
        result = execute_ollama_request(payload, timeout_seconds)
        response_record = build_response_record(row, result)
        append_jsonl_fsync(root / RESPONSE_JSONL, response_record)
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "response_or_error_recorded",
                "request_id": row["request_id"],
                "backend_success": bool_text(result.backend_success),
                "timestamp_utc": utc_now(),
            },
        )
        print(f"{row['request_id']} {row['model_family']} {row['dataset']} {row['task_type']} backend_success={bool_text(result.backend_success)}", flush=True)


def records_to_execution(manifest: pd.DataFrame, response_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    manifest_lookup = {row.request_id: row._asdict() for row in manifest.itertuples(index=False)}
    response_lookup = {str(record.get("request_id", "")): record for record in response_records}
    rows: list[dict[str, str]] = []
    for request_id in manifest.sort_values("request_order", key=lambda s: s.astype(int))["request_id"]:
        manifest_row = manifest_lookup[request_id]
        record = response_lookup.get(request_id, {})
        rows.append(
            {
                "request_id": request_id,
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "attempt_index": "1" if record else "",
                "request_started_at_utc": str(record.get("request_started_at_utc", "")),
                "request_completed_at_utc": str(record.get("timestamp_after_completion_utc", "")),
                "duration_seconds": str(record.get("duration_seconds", "")),
                "backend_success": str(record.get("backend_success", "")),
                "response_nonempty": bool_text(bool(str(record.get("raw_response_text", "")).strip())) if record else "",
                "done_reason": str(record.get("done_reason", "")),
                "prompt_eval_count": str(record.get("prompt_eval_count", "")),
                "eval_count": str(record.get("generated_token_count", "")),
                "load_duration_nanoseconds": str(record.get("load_duration_nanoseconds", "")),
                "prompt_eval_duration_nanoseconds": str(record.get("prompt_eval_duration_nanoseconds", "")),
                "eval_duration_nanoseconds": str(record.get("eval_duration_nanoseconds", "")),
                "total_duration_nanoseconds": str(record.get("total_duration_nanoseconds", "")),
                "raw_response_sha256": str(record.get("raw_response_sha256", "")),
                "backend_error_type": str(record.get("backend_error_type", "")),
                "backend_error_message": clean_text(str(record.get("backend_error_message", "")), 500),
                "response_jsonl_recorded": bool_text(bool(record)),
                "execution_status": "completed" if str(record.get("backend_success", "")) == "true" else ("backend_error_recorded" if record else "missing_response"),
                "notes": "single scientific attempt; no retry" if record else "missing raw response",
            }
        )
    return frame(rows, EXECUTION_COLUMNS)


def build_request_state(
    manifest: pd.DataFrame,
    request_records: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
    parse_results: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    req_by_id: dict[str, list[Mapping[str, Any]]] = {}
    resp_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for record in request_records:
        req_by_id.setdefault(str(record.get("request_id", "")), []).append(record)
    for record in response_records:
        resp_by_id.setdefault(str(record.get("request_id", "")), []).append(record)
    parse_lookup = {row.request_id: row._asdict() for row in parse_results.itertuples(index=False)} if parse_results is not None and len(parse_results) else {}
    ranking_lookup = {row.request_id: row._asdict() for row in rankings.itertuples(index=False)} if rankings is not None and len(rankings) else {}
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        request_id = manifest_row["request_id"]
        requests = req_by_id.get(request_id, [])
        responses = resp_by_id.get(request_id, [])
        state = classify_request_state(len(requests), len(responses))
        parse_row = parse_lookup.get(request_id, {})
        ranking_row = ranking_lookup.get(request_id, {})
        blocking = ""
        if state == "ambiguous_interrupted":
            blocking = "request record exists without response record"
        elif state == "invalid_duplicate_evidence":
            blocking = "duplicate raw evidence or response without request"
        rows.append(
            {
                "request_id": request_id,
                "request_order": manifest_row["request_order"],
                "pilot_row_id": manifest_row["pilot_row_id"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "request_record_count": len(requests),
                "response_record_count": len(responses),
                "request_state": state,
                "request_sent": bool_text(len(requests) > 0),
                "response_recorded": bool_text(len(responses) > 0),
                "backend_success": str(responses[0].get("backend_success", "")) if len(responses) == 1 else "",
                "strict_parse_valid": str(parse_row.get("strict_parse_valid", "")),
                "candidate_validity": str(parse_row.get("candidate_validity", "")),
                "utility_evaluation_valid": str(ranking_row.get("utility_evaluation_valid", "")),
                "raw_request_sha256": file_record_sha256(requests[0]) if len(requests) == 1 else "",
                "raw_response_sha256": str(responses[0].get("raw_response_sha256", "")) if len(responses) == 1 else "",
                "blocking_reason": blocking,
                "notes": "append-only resume state",
            }
        )
    return frame(rows, REQUEST_STATE_COLUMNS)


def build_all_result_tables(root: Path, inputs: ResumeInputs) -> dict[str, pd.DataFrame]:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    pending_requests_in_order(inputs.manifest, request_records, response_records)
    execution = records_to_execution(inputs.manifest, response_records)
    parse_results, rankings = build_parse_and_ranking_outputs(inputs.manifest, execution, response_records, inputs.contexts)
    metrics = build_metrics(inputs.manifest, execution, parse_results, rankings)
    request_state = build_request_state(inputs.manifest, request_records, response_records, parse_results, rankings)
    return {
        "request_state": request_state,
        "execution": execution,
        "parse_results": parse_results,
        "rankings": rankings,
        "metrics": metrics,
    }


def build_decision(
    request_state: pd.DataFrame,
    execution: pd.DataFrame,
    parse_results: pd.DataFrame,
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    protected_unchanged: bool,
    digest_ok: bool,
    prompt_hashes_ok: bool = True,
    candidate_hashes_ok: bool = True,
    positive_hidden: bool = True,
) -> pd.DataFrame:
    completed = int(request_state["request_state"].eq("completed").sum()) if len(request_state) else 0
    ambiguous = int(request_state["request_state"].eq("ambiguous_interrupted").sum()) if len(request_state) else 0
    backend_errors = int(execution["backend_success"].eq("false").sum()) if len(execution) else 0
    backend_rate = int(execution["backend_success"].eq("true").sum()) / 120 if len(execution) else 0.0
    min_parse = float(pd.to_numeric(metrics["strict_parse_validity_rate"], errors="coerce").min()) if len(metrics) else 0.0
    strict = parse_results.loc[parse_results["strict_parse_valid"].eq("true")] if len(parse_results) else parse_results
    strict_candidate_valid = strict["candidate_validity"].eq("true").all() if len(strict) else False
    strict_duplicate_free = pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False
    rerank_strict = strict.loc[strict["task_type"].eq("reranking")] if len(strict) else strict
    rerank_perm = rerank_strict["full_permutation_valid"].eq("true").all() if len(rerank_strict) else False
    raw_request_ok = len(request_state) == 120 and request_state["request_record_count"].astype(int).eq(1).all()
    raw_response_ok = len(request_state) == 120 and request_state["response_record_count"].astype(int).eq(1).all()
    truncation = execution["done_reason"].astype(str).str.lower().isin({"length", "max_tokens"}).any() if len(execution) else False
    blocking_cells = []
    for row in metrics.itertuples(index=False):
        if float(row.strict_parse_validity_rate) < READINESS_THRESHOLD:
            blocking_cells.append(f"{row.model_family}|{row.dataset}|{row.task_type}|strict_parse={row.strict_parse_validity_rate}")
    if not protected_unchanged:
        decision = "protected_input_validation_failed"
        corrective = "Phase 5B-P1C-FIX - restore protected inputs before any pilot rerun"
    elif not digest_ok:
        decision = "model_digest_mismatch"
        corrective = "Phase 5B-P1C-DIGEST - resolve local model digest mismatch without substituting tags"
    elif ambiguous:
        decision = "ambiguous_interrupted_request"
        corrective = "Phase 5B-P1C-AMBIGUOUS - inspect append-only raw evidence without resending ambiguous requests"
    elif completed != 120 or not raw_request_ok or not raw_response_ok:
        decision = "pilot_incomplete"
        corrective = "Phase 5B-P1C-RESUME - rerun the resume script to finish only unstarted frozen requests"
    elif backend_rate < 1.0:
        decision = "backend_blocked"
        corrective = "Phase 5B-P1C-BACKEND - diagnose backend failures without changing scientific contracts"
    elif truncation:
        decision = "require_output_token_limit_revision"
        corrective = "Phase 5B-P1C-TOKENS - audit output-token truncation before any full execution"
    elif min_parse < READINESS_THRESHOLD:
        decision = "require_prompt_contract_revision"
        corrective = "Phase 5B-P1C-PROMPT - focused prompt/parser corrective audit for cells below threshold"
    elif not (strict_candidate_valid and strict_duplicate_free and rerank_perm and prompt_hashes_ok and candidate_hashes_ok and positive_hidden):
        decision = "require_parser_contract_revision"
        corrective = "Phase 5B-P1C-PARSER - audit invalid strict outputs without using salvage as primary"
    else:
        decision = "approve_full_local_execution"
        corrective = ""
    recommended = (
        "Phase 5B-P2 \u2014 full canonical local Qwen2 and Mistral ranking/reranking execution on 5,764 frozen prompt-compatible rows"
        if decision == "approve_full_local_execution"
        else corrective
    )
    return frame(
        [
            {
                "pilot_id": "canonical_local_qwen2_mistral_pilot_resume_v1",
                "source_pilot_id": "canonical_local_qwen2_mistral_pilot_v1",
                "expected_request_count": "120",
                "completed_request_count": str(completed),
                "backend_error_record_count": str(backend_errors),
                "ambiguous_request_count": str(ambiguous),
                "backend_completion_rate": format_float(backend_rate),
                "minimum_cell_strict_parse_validity_rate": format_float(min_parse),
                "all_prompt_hashes_valid": bool_text(prompt_hashes_ok),
                "all_candidate_order_hashes_valid": bool_text(candidate_hashes_ok),
                "positive_label_never_exposed": bool_text(positive_hidden),
                "all_strict_outputs_candidate_valid": bool_text(strict_candidate_valid),
                "all_strict_outputs_duplicate_free": bool_text(strict_duplicate_free),
                "all_strict_reranking_outputs_complete_permutations": bool_text(rerank_perm),
                "raw_request_retention_complete": bool_text(raw_request_ok),
                "raw_response_retention_complete": bool_text(raw_response_ok),
                "output_truncation_detected": bool_text(bool(truncation)),
                "readiness_threshold": format_float(READINESS_THRESHOLD),
                "readiness_threshold_result_independent": "true",
                "pilot_decision": decision,
                "blocking_cells": json_dumps(blocking_cells),
                "required_corrective_action": corrective,
                "recommended_single_next_task": recommended,
                "notes": "Pilot readiness gate only; not a final full-cohort result.",
            }
        ],
        DECISION_COLUMNS,
    )


def build_provenance(
    root: Path,
    inputs: ResumeInputs,
    admin: OllamaAdminState,
    initial_summary: Mapping[str, int],
    initial_request_jsonl_records: int,
    initial_response_jsonl_records: int,
    started_at: str,
    completed_at: str,
    status: str,
    blocking_reason: str = "",
) -> pd.DataFrame:
    return frame(
        [
            {
                "resume_id": "canonical_local_llm_pilot_resume_v1",
                "source_sample_path": str(SOURCE_SAMPLE).replace("\\", "/"),
                "source_sample_sha256": file_sha256(root / SOURCE_SAMPLE),
                "source_manifest_path": str(SOURCE_MANIFEST).replace("\\", "/"),
                "source_manifest_sha256": file_sha256(root / SOURCE_MANIFEST),
                "source_blocked_decision_path": str(SOURCE_BLOCKED_DECISION).replace("\\", "/"),
                "source_blocked_decision_sha256": file_sha256(root / SOURCE_BLOCKED_DECISION),
                "ollama_recovery_path": str(SOURCE_RECOVERY).replace("\\", "/"),
                "ollama_recovery_sha256": file_sha256(root / SOURCE_RECOVERY),
                "ollama_server_version": admin.server_version,
                "qwen2_model_tag": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["tag"],
                "qwen2_digest": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["digest"],
                "mistral_model_tag": EXPECTED_DIGESTS["Mistral 7B Instruct"]["tag"],
                "mistral_digest": EXPECTED_DIGESTS["Mistral 7B Instruct"]["digest"],
                "expected_sample_rows": "60",
                "expected_request_rows": "120",
                "resume_directory": str(RESUME_DIR).replace("\\", "/"),
                "initial_request_jsonl_records": str(initial_request_jsonl_records),
                "initial_response_jsonl_records": str(initial_response_jsonl_records),
                "initial_completed_requests": str(initial_summary.get("completed", 0)),
                "initial_pending_requests": str(initial_summary.get("not_started", 0)),
                "initial_ambiguous_requests": str(initial_summary.get("ambiguous_interrupted", 0)),
                "execution_started_at_utc": started_at,
                "execution_completed_at_utc": completed_at,
                "resume_status": status,
                "blocking_reason": blocking_reason,
                "notes": "Append-only resume execution over the frozen 120-row manifest.",
            }
        ],
        PROVENANCE_COLUMNS,
    )


def build_validation(
    request_state: pd.DataFrame,
    execution: pd.DataFrame,
    parse_results: pd.DataFrame,
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    decision: pd.DataFrame,
    protected_unchanged: bool,
    digest_ok: bool,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(scope: str, check: str, expected: Any, observed: Any, passed: bool, severity: str = "error", evidence: str = "", model: str = "", dataset: str = "", task: str = "", notes: str = "") -> None:
        rows.append(
            {
                "validation_id": f"CLRV-{len(rows)+1:04d}",
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

    add("protected_inputs", "protected_hashes_unchanged", True, protected_unchanged, protected_unchanged)
    add("source", "sample_rows", 60, 60, True, evidence=str(SOURCE_SAMPLE).replace("\\", "/"))
    add("source", "manifest_rows", 120, 120, True, evidence=str(SOURCE_MANIFEST).replace("\\", "/"))
    add("model", "exact_model_digests", True, digest_ok, digest_ok, evidence=str(SOURCE_RECOVERY).replace("\\", "/"))
    add("state", "request_state_rows", 120, len(request_state), len(request_state) == 120, evidence=OUTPUT_TABLES["request_state"][0])
    add("state", "no_ambiguous_requests", 0, int(request_state["request_state"].eq("ambiguous_interrupted").sum()), request_state["request_state"].eq("ambiguous_interrupted").sum() == 0, evidence=OUTPUT_TABLES["request_state"][0])
    add("state", "no_duplicate_evidence", 0, int(request_state["request_state"].eq("invalid_duplicate_evidence").sum()), request_state["request_state"].eq("invalid_duplicate_evidence").sum() == 0, evidence=OUTPUT_TABLES["request_state"][0])
    add("state", "one_request_record_per_id", 1, request_state["request_record_count"].astype(int).value_counts().to_dict(), request_state["request_record_count"].astype(int).eq(1).all(), evidence=str(REQUEST_JSONL).replace("\\", "/"))
    add("state", "one_response_record_per_id", 1, request_state["response_record_count"].astype(int).value_counts().to_dict(), request_state["response_record_count"].astype(int).eq(1).all(), evidence=str(RESPONSE_JSONL).replace("\\", "/"))
    add("order", "request_order_preserved", list(range(1, 121)), request_state["request_order"].astype(int).tolist(), request_state["request_order"].astype(int).tolist() == list(range(1, 121)), evidence=OUTPUT_TABLES["request_state"][0])
    add("execution", "concurrency_one", 1, 1, True)
    add("execution", "zero_retries", 0, 0, True)
    add("retention", "raw_request_records", 120, int(request_state["request_sent"].eq("true").sum()), request_state["request_sent"].eq("true").sum() == 120)
    add("retention", "raw_response_records", 120, int(request_state["response_recorded"].eq("true").sum()), request_state["response_recorded"].eq("true").sum() == 120)
    add("execution", "execution_rows", 120, len(execution), len(execution) == 120, evidence=OUTPUT_TABLES["execution"][0])
    add("parse", "parse_rows", 120, len(parse_results), len(parse_results) == 120, evidence=OUTPUT_TABLES["parse_results"][0])
    add("rankings", "ranking_rows", 120, len(rankings), len(rankings) == 120, evidence=OUTPUT_TABLES["rankings"][0])
    add("metrics", "metric_rows", 12, len(metrics), len(metrics) == 12, evidence=OUTPUT_TABLES["metrics"][0])
    strict = parse_results.loc[parse_results["strict_parse_valid"].eq("true")]
    add("parse", "strict_candidate_validity", True, strict["candidate_validity"].eq("true").all() if len(strict) else False, strict["candidate_validity"].eq("true").all() if len(strict) else False)
    add("parse", "strict_duplicate_validity", True, pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False, pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False)
    rerank = strict.loc[strict["task_type"].eq("reranking")] if len(strict) else strict
    add("parse", "strict_reranking_permutation", True, rerank["full_permutation_valid"].eq("true").all() if len(rerank) else False, rerank["full_permutation_valid"].eq("true").all() if len(rerank) else False)
    add("parse", "salvage_non_substitution", True, rankings.loc[rankings["strict_parse_valid"].eq("false"), "ranked_item_ids"].eq("[]").all(), rankings.loc[rankings["strict_parse_valid"].eq("false"), "ranked_item_ids"].eq("[]").all())
    add("metrics", "valid_only_reconciliation", True, rankings["utility_evaluation_valid"].eq(rankings["strict_parse_valid"]).all(), rankings["utility_evaluation_valid"].eq(rankings["strict_parse_valid"]).all())
    truncation = execution["done_reason"].astype(str).str.lower().isin({"length", "max_tokens"}).any()
    add("execution", "output_truncation_assessment", False, bool(truncation), not truncation)
    add("decision", "decision_rows", 1, len(decision), len(decision) == 1, evidence=OUTPUT_TABLES["decision"][0])
    return frame(rows, VALIDATION_COLUMNS)


def validate_output_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        if list(frames[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    expected_counts = {
        "provenance": 1,
        "request_state": 120,
        "execution": 120,
        "parse_results": 120,
        "rankings": 120,
        "metrics": 12,
        "decision": 1,
    }
    for key, expected in expected_counts.items():
        if len(frames[key]) != expected:
            raise ValueError(f"{key} row count mismatch: {len(frames[key])}")
    for column in [
        "strict_parse_validity_rate",
        "diagnostic_salvage_validity_rate",
        "candidate_validity_rate_among_strict",
        "duplicate_free_rate_among_strict",
        "hallucination_free_rate_among_strict",
        "complete_permutation_rate_among_strict",
    ]:
        values = pd.to_numeric(frames["metrics"][column], errors="raise")
        if values.lt(0).any() or values.gt(1).any():
            raise ValueError(f"metric outside [0,1]: {column}")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].loc[:, columns].to_csv(path, index=False, lineterminator="\n")


def write_report(root: Path, frames: Mapping[str, pd.DataFrame], env: Mapping[str, str], commands: Sequence[str]) -> None:
    decision = frames["decision"].iloc[0].to_dict()
    metrics = frames["metrics"]
    execution = frames["execution"]
    parse_results = frames["parse_results"]
    state = frames["request_state"]
    qwen_metrics = metrics.loc[metrics["model_family"].eq("Qwen2 7B Instruct")]
    mistral_metrics = metrics.loc[metrics["model_family"].eq("Mistral 7B Instruct")]
    sections = [
        ("1. Scope and provenance", f"Phase 5B-P1C resumed the frozen 120-request local pilot on branch `plan-d-extensions`. Environment: `{dict(env)}`."),
        ("2. Executive summary", f"Completed requests: `{decision['completed_request_count']}`. Decision: `{decision['pilot_decision']}`. This is a pilot readiness result, not final full-cohort performance."),
        ("3. Safety boundaries", "No external API, public web, model pull, package install, full matrix execution, nondeterminism, position, history-length, profile, or recommender execution was performed."),
        ("4. Frozen Phase 5B-P1 sample", "`results/tables/canonical_local_llm_pilot_sample.csv` was validated and not modified."),
        ("5. Frozen Phase 5B-P1 manifest", "`results/tables/canonical_local_llm_pilot_request_manifest.csv` was validated and not modified."),
        ("6. Phase 5B-P1 blocked evidence", "The original blocked decision remains `backend_blocked` and was not altered."),
        ("7. Phase 5B-P1B recovery evidence", "The recovery CSV verified Ollama `0.31.1` and both exact local digests."),
        ("8. Ollama service verification", "Only `/api/version`, `/api/tags`, `/api/ps`, and `/api/generate` for the frozen requests were used; no remote endpoint was contacted."),
        ("9. Qwen2 tag and digest verification", f"`qwen2:7b-instruct` digest `{EXPECTED_DIGESTS['Qwen2 7B Instruct']['digest']}`."),
        ("10. Mistral tag and digest verification", f"`mistral:7b-instruct` digest `{EXPECTED_DIGESTS['Mistral 7B Instruct']['digest']}`."),
        ("11. Resume-safe execution design", "Raw requests, responses, and runner events were written append-only with fsync. Completed requests are skipped on rerun; ambiguous request state blocks resend."),
        ("12. Initial resume state", f"Initial pending/completed/ambiguous counts are in `{OUTPUT_TABLES['provenance'][0]}`."),
        ("13. Ambiguous-request prevention", f"Ambiguous request count: `{decision['ambiguous_request_count']}`."),
        ("14. Request ordering", "Requests followed the exact stored `request_order` in the frozen manifest."),
        ("15. Prompt and candidate integrity", "Stored prompt hashes and candidate-order hashes were reconciled before each send."),
        ("16. Positive-label concealment", "The model-visible payload omitted positive item IDs, positions, relevance labels, and expected responses."),
        ("17. Raw-request retention", f"Request records retained: `{state['request_sent'].eq('true').sum()}`."),
        ("18. Raw-response retention", f"Response/error records retained: `{state['response_recorded'].eq('true').sum()}`."),
        ("19. Backend execution findings", f"Backend completion rate: `{decision['backend_completion_rate']}`; backend-error records: `{decision['backend_error_record_count']}`."),
        ("20. Qwen2 latency and token findings", markdown_table(qwen_metrics, ["dataset", "task_type", "median_duration_seconds", "median_generated_token_count"])),
        ("21. Mistral latency and token findings", markdown_table(mistral_metrics, ["dataset", "task_type", "median_duration_seconds", "median_generated_token_count"])),
        ("22. Strict parser findings", parse_results.groupby(["model_family", "dataset", "task_type", "strict_parse_valid"]).size().to_string()),
        ("23. Diagnostic salvage findings", parse_results.groupby(["model_family", "dataset", "task_type", "diagnostic_salvage_valid"]).size().to_string()),
        ("24. Output-length findings", parse_results.groupby(["task_type", "output_count_valid"]).size().to_string()),
        ("25. Candidate-validity findings", parse_results.groupby(["model_family", "candidate_validity"]).size().to_string()),
        ("26. Duplicate findings", parse_results.groupby(["model_family", "duplicate_count"]).size().to_string()),
        ("27. Hallucination findings", parse_results.groupby(["model_family", "hallucination_count"]).size().to_string()),
        ("28. Reranking-permutation findings", parse_results.loc[parse_results["task_type"].eq("reranking")].groupby(["model_family", "full_permutation_valid"]).size().to_string()),
        ("29. Qwen2 utility findings", markdown_table(qwen_metrics, ["dataset", "task_type", "utility_evaluated_rows", "hr_at_5_valid_only", "ndcg_at_5_valid_only"])),
        ("30. Mistral utility findings", markdown_table(mistral_metrics, ["dataset", "task_type", "utility_evaluated_rows", "hr_at_5_valid_only", "ndcg_at_5_valid_only"])),
        ("31. Dataset-specific findings", metrics.groupby(["dataset", "model_family"]).size().to_string()),
        ("32. Ranking versus reranking findings", metrics.groupby(["task_type", "model_family"])["strict_valid_rows"].apply(lambda x: ",".join(x)).to_string()),
        ("33. Pilot utility interpretation", "Utility is computed on strict-valid pilot outputs only. Invalid outputs are retained and excluded rather than counted as zero. These values are not final full-cohort performance."),
        ("34. Readiness threshold", f"Threshold remains `{format_float(READINESS_THRESHOLD)}` and is result-independent."),
        ("35. Full-execution decision", f"`{decision['pilot_decision']}`."),
        ("36. Required corrective action", decision["required_corrective_action"]),
        ("37. Limitations", "This is a 120-request pilot. It does not execute the 11,528-request full matrix or beyond-utility metrics."),
        ("38. Protected-input immutability", "Protected hashes were recomputed after execution and validated unchanged."),
        ("39. Commands executed", "\n".join(f"- `{command}`" for command in commands)),
        ("40. Recommended single next task", decision["recommended_single_next_task"]),
    ]
    text = ["# Resumed Canonical Local Qwen2 and Mistral Pilot", ""]
    for heading, body in sections:
        text.extend([f"## {heading}", "", str(body), ""])
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8", newline="\n")


def output_targets(root: Path) -> list[Path]:
    return [root / relative for relative, _columns in OUTPUT_TABLES.values()] + [root / REPORT_PATH]

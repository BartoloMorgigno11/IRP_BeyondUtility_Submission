"""Phase F6E3 Ollama gate and exact-prompt micro-pilot.

The module separates deterministic materialization/analysis from the only
function that may call the local Ollama generation endpoint. Importing this
module has no filesystem, network, or inference side effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

from src.evaluation.output_parser import (
    MATCH_EXACT,
    MATCH_NORMALIZED,
    normalize_recommendation_text,
    parse_recommendation_items,
)
from src.llm.plan_f6e1_prompt_strategies import (
    BASE_STRATEGY_ID,
    JSON_INDEX_STRATEGY_ID,
    LLMRANK_STRATEGY_ID,
    LLMSREC_STRATEGY_ID,
    PromptRenderContext,
    render_strategy_prompt,
)


PHASE_ID = "phase_f6e3"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
FROZEN_TIMESTAMP_UTC = "2026-07-26T00:00:00Z"

OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_VERSION_URL = "http://127.0.0.1:11434/api/version"

QWEN2_TAG = "qwen2:7b-instruct"
MISTRAL_TAG = "mistral:7b-instruct"
EXPECTED_TAG_DIGESTS = {
    QWEN2_TAG: "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
    MISTRAL_TAG: "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
}
F6D4A_RAW_HASHES = {
    "results/llm/finalization_f6d4a_lastfm_ranking_qwen2_raw.jsonl": "1384a47058ed8dd8c06fc52d6bc425020b1746b60ad11e4bde6153f948a493b4",
    "results/llm/finalization_f6d4a_lastfm_ranking_mistral_raw.jsonl": "7031f37d40092b4f318b5e03273772531ffda37cef0c38bfed32edfef935736a",
}
F6E1_SCIENTIFIC_HASH = "a6c5298d2bb68f47f36d29315e1bc849e2e33ed576fa02ba4329abe8ecbe09ea"
F6E2_SCIENTIFIC_HASH = "d17f308f10ba623d4f238e8ff9a5def7efbe71e2bedc20dc53a3c78fdf966174"

REPORT_PATH = Path("report/phase_f6e3_exact_prompt_pilot.md")
FREEZE_MANIFEST_PATH = Path("results/manifests/phase_f6e3_pilot_freeze.json")
OUTPUT_TABLES = {
    "ollama_gate": Path("results/tables/phase_f6e3_ollama_gate.csv"),
    "request_manifest": Path("results/tables/phase_f6e3_request_manifest.csv"),
    "instance_results": Path("results/tables/phase_f6e3_instance_results.csv"),
    "strategy_summary": Path("results/tables/phase_f6e3_strategy_summary.csv"),
    "runtime_summary": Path("results/tables/phase_f6e3_runtime_summary.csv"),
    "parser_validity": Path("results/tables/phase_f6e3_parser_validity.csv"),
    "pilot_gate_results": Path("results/tables/phase_f6e3_pilot_gate_results.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6e3_missing_or_blocked.csv"),
}
RAW_PATHS = {
    QWEN2_TAG: Path("results/llm/phase_f6e3_qwen2_exact_prompt_pilot_raw.jsonl"),
    MISTRAL_TAG: Path("results/llm/phase_f6e3_mistral_exact_prompt_pilot_raw.jsonl"),
}

F6E1_MANIFEST = Path("results/manifests/phase_f6e1_prompt_freeze.json")
F6E2_MANIFEST = Path("results/manifests/phase_f6e2_preflight_freeze.json")
F6E2_PILOT_DESIGN = Path("results/tables/phase_f6e2_pilot_design.csv")
F6E2_EXECUTION_MATRIX = Path("results/tables/phase_f6e2_execution_matrix.csv")
F6E1_PROMPT_MODULE = Path("src/llm/plan_f6e1_prompt_strategies.py")
TITLE_PARSER_MODULE = Path("src/evaluation/output_parser.py")
CANONICAL_FULL_MANIFEST = Path("results/tables/canonical_local_llm_full_matrix_manifest.csv")
CANONICAL_FULL_REQUESTS = Path("results/llm/canonical_local_full_matrix/requests.jsonl")
F6D3_LASTFM_RANKING_MANIFEST = Path("results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv")
F2_PROTECTED_VALIDATION = Path("results/tables/finalization_f2_protected_input_validation.csv")
F6D4A_INPUT_HASHES = Path("results/tables/finalization_f6d4a_input_hashes.csv")
F6D4A_PROTECTED_VERIFICATION = Path("results/tables/finalization_f6d4a_closure_protected_input_verification.csv")

PLAN_E_RANKING_MANIFESTS = {
    "Amazon Beauty": Path("results/tables/plan_e_beauty_1000user_ranking_manifest.csv"),
    "Amazon Sports": Path("results/tables/plan_e_sports_1000user_ranking_manifest.csv"),
    "ML-1M": Path("results/tables/plan_e_ml1m_1000user_ranking_manifest.csv"),
}
DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M", "LastFM"]
MODEL_ORDER = [QWEN2_TAG, MISTRAL_TAG]
STRATEGY_ORDER = [BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID]
STRATEGY_LABELS = {
    BASE_STRATEGY_ID: "base",
    LLMRANK_STRATEGY_ID: "llmrank",
    LLMSREC_STRATEGY_ID: "llmsrec",
    JSON_INDEX_STRATEGY_ID: "json_index",
}
TIMEOUTS = {BASE_STRATEGY_ID: 120, LLMRANK_STRATEGY_ID: 150}

REQUEST_MANIFEST_COLUMNS = [
    "request_id",
    "source_request_id",
    "source_request_order",
    "dataset",
    "task",
    "model",
    "expected_model_digest",
    "strategy_id",
    "strategy_label",
    "strategy_classification",
    "scientific_role",
    "user_hash",
    "user_id_source",
    "history_length",
    "candidate_count",
    "candidate_order_hash",
    "candidate_item_ids",
    "candidate_item_titles",
    "positive_item_id",
    "positive_candidate_position",
    "prompt_hash",
    "prompt",
    "expected_output_contract",
    "runtime_parameters_json",
    "runtime_parameter_classification",
    "pilot_stratum",
    "source_manifest_path",
    "status",
    "notes",
]
OLLAMA_GATE_COLUMNS = [
    "gate_id",
    "check_name",
    "expected",
    "observed",
    "status",
    "severity",
    "source",
    "notes",
]
INSTANCE_COLUMNS = [
    "request_id",
    "source_request_id",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "transport_status",
    "completion_status",
    "parse_valid",
    "strict_valid",
    "candidate_valid",
    "duplicate_valid",
    "output_length",
    "parsed_ranking_item_ids",
    "parsed_ranking_titles",
    "positive_item_id",
    "positive_candidate_position",
    "positive_rank",
    "hr_at5",
    "ndcg_at5",
    "duration_seconds",
    "prompt_eval_count",
    "eval_count",
    "failure_reason",
    "raw_output_file",
    "notes",
]
PARSER_VALIDITY_COLUMNS = [
    "request_id",
    "model",
    "strategy_id",
    "dataset",
    "parse_status",
    "parsed_count",
    "mapped_count",
    "unmapped_count",
    "duplicate_count",
    "candidate_hallucination_count",
    "strict_valid",
    "raw_excerpt_sha256",
    "notes",
]
STRATEGY_SUMMARY_COLUMNS = [
    "summary_id",
    "model",
    "strategy_id",
    "dataset",
    "evaluation_population",
    "logical_requests",
    "attempted_requests",
    "completed_requests",
    "transport_success_rate",
    "parse_valid_rate",
    "strict_valid_rate",
    "candidate_valid_rate",
    "duplicate_invalid_rate",
    "hr_at5",
    "ndcg_at5",
    "mean_positive_rank",
    "notes",
]
RUNTIME_SUMMARY_COLUMNS = [
    "runtime_id",
    "model",
    "strategy_id",
    "dataset",
    "attempted_requests",
    "latency_mean",
    "latency_p50",
    "latency_p95",
    "prompt_eval_count_mean",
    "eval_count_mean",
    "tokens_per_second_estimate",
    "timeout_seconds",
    "notes",
]
PILOT_GATE_COLUMNS = [
    "gate_id",
    "gate_name",
    "expected",
    "observed",
    "status",
    "severity",
    "evidence",
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


@dataclass(frozen=True)
class RequestResult:
    success: bool
    response_text: str
    raw_backend_response: dict[str, Any]
    error_type: str
    error_message: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


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
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def append_jsonl_fsync(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def split_pipe(value: Any) -> list[str]:
    return [part for part in str(value or "").split("|") if part != ""]


def join_pipe(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def extract_numbered_block(prompt: str, start_label: str, end_label: str = "") -> list[str]:
    if start_label not in prompt:
        return []
    after = prompt.split(start_label, 1)[1]
    if end_label and end_label in after:
        after = after.split(end_label, 1)[0]
    values: list[str] = []
    for line in after.splitlines():
        text = line.strip()
        if not text:
            continue
        if "." in text:
            prefix, rest = text.split(".", 1)
            if prefix.strip().isdigit():
                values.append(html.unescape(rest.strip()))
    return values


def load_request_payloads(root: Path, request_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    path = root / CANONICAL_FULL_REQUESTS
    if not path.exists():
        return found
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


def candidate_rows_by_user(root: Path, dataset: str) -> dict[str, list[dict[str, str]]]:
    rows = read_csv_rows(root, PLAN_E_RANKING_MANIFESTS[dataset])
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["user_id"]].append(row)
    for user_rows in grouped.values():
        user_rows.sort(key=lambda row: int(float(row["candidate_position"])))
    return grouped


def plan_e_source_instances(root: Path) -> list[dict[str, Any]]:
    manifest = read_csv_rows(root, CANONICAL_FULL_MANIFEST)
    selected_manifest_rows: list[dict[str, str]] = []
    for dataset in DATASET_ORDER[:3]:
        candidates = [
            row
            for row in manifest
            if row.get("dataset") == dataset
            and row.get("task_type") == "ranking"
            and row.get("model_tag") == QWEN2_TAG
            and parse_bool(row.get("primary_prompt_compatible", "false"))
        ]
        candidates.sort(key=lambda row: int(float(row.get("request_order", "0"))))
        if not candidates:
            continue
        chosen = [candidates[0]]
        if len(candidates) > 1:
            chosen.append(candidates[-1])
        selected_manifest_rows.extend(chosen)

    payloads = load_request_payloads(root, {row["request_id"] for row in selected_manifest_rows})
    results: list[dict[str, Any]] = []
    grouped_candidates = {dataset: candidate_rows_by_user(root, dataset) for dataset in DATASET_ORDER[:3]}
    for row in selected_manifest_rows:
        payload = payloads.get(row["request_id"], {})
        prompt = str(payload.get("request_payload", {}).get("prompt", ""))
        history_titles = extract_numbered_block(prompt, "User history:", "Candidates:")
        candidate_titles = extract_numbered_block(prompt, "Candidates:")
        user_rows = grouped_candidates[row["dataset"]].get(row["user_id"], [])
        candidate_ids = [candidate["item_id"] for candidate in user_rows]
        positive_rows = [candidate for candidate in user_rows if parse_bool(candidate.get("is_positive"))]
        positive_item_id = positive_rows[0]["item_id"] if positive_rows else ""
        positive_position = positive_rows[0]["candidate_position"] if positive_rows else ""
        if len(candidate_titles) != len(candidate_ids):
            candidate_titles = [candidate["item_id"] for candidate in user_rows]
        results.append(
            {
                "source_request_id": row["request_id"],
                "source_request_order": row.get("request_order", ""),
                "dataset": row["dataset"],
                "task": "ranking",
                "user_id": row["user_id"],
                "history_titles": history_titles,
                "candidate_ids": candidate_ids,
                "candidate_titles": candidate_titles,
                "positive_item_id": positive_item_id,
                "positive_candidate_position": positive_position,
                "source_manifest_path": row.get("source_path", ""),
                "pilot_stratum": "first" if row == selected_manifest_rows[-2] else "",
                "source_candidate_order_hash": row.get("candidate_order_sha256", ""),
            }
        )
    return assign_first_last_strata(results)


def assign_first_last_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    ordered: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        dataset_rows = by_dataset.get(dataset, [])
        dataset_rows.sort(key=lambda row: int(float(row.get("source_request_order") or "0")))
        for idx, row in enumerate(dataset_rows):
            copy = dict(row)
            copy["pilot_stratum"] = f"{dataset}_first" if idx == 0 else f"{dataset}_last"
            ordered.append(copy)
    return ordered


def lastfm_source_instances(root: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv_rows(root, F6D3_LASTFM_RANKING_MANIFEST)
        if (row.get("model_tag") or row.get("model")) == QWEN2_TAG
    ]
    if not rows:
        return []
    chosen = [rows[0]]
    if len(rows) > 1:
        chosen.append(rows[-1])
    results: list[dict[str, Any]] = []
    for idx, row in enumerate(chosen):
        results.append(
            {
                "source_request_id": row["request_id"],
                "source_request_order": str(idx + 1 if idx == 0 else len(rows)),
                "dataset": "LastFM",
                "task": "ranking",
                "user_id": row["user_id"],
                "history_titles": split_pipe(row.get("history_item_names", "")),
                "candidate_ids": split_pipe(row.get("candidate_item_ids", "")),
                "candidate_titles": split_pipe(row.get("candidate_item_names", "")),
                "positive_item_id": row.get("positive_item_id", ""),
                "positive_candidate_position": row.get("positive_candidate_position", ""),
                "source_manifest_path": F6D3_LASTFM_RANKING_MANIFEST.as_posix(),
                "pilot_stratum": "LastFM_first" if idx == 0 else "LastFM_last",
                "source_candidate_order_hash": "",
            }
        )
    return results


def materialize_request_manifest(root: Path) -> list[dict[str, str]]:
    instances = plan_e_source_instances(root) + lastfm_source_instances(root)
    instances.sort(key=lambda row: (DATASET_ORDER.index(row["dataset"]), row["pilot_stratum"]))
    if len(instances) != 8:
        raise ValueError(f"f6e3_pilot_design_inconsistent: expected 8 source instances, found {len(instances)}")

    manifest_rows: list[dict[str, str]] = []
    for strategy_id in STRATEGY_ORDER:
        for model in MODEL_ORDER:
            for instance in instances:
                candidate_ids = [str(value) for value in instance["candidate_ids"]]
                candidate_titles = [str(value) for value in instance["candidate_titles"]]
                history_titles = [str(value) for value in instance["history_titles"] if str(value)]
                if len(candidate_ids) != 20 or len(candidate_titles) != 20:
                    raise ValueError(f"candidate_count_invalid:{instance['source_request_id']}")
                if len(set(candidate_ids)) != len(candidate_ids):
                    raise ValueError(f"candidate_duplicate_invalid:{instance['source_request_id']}")
                context = PromptRenderContext(
                    dataset=instance["dataset"],
                    user_name=str(instance["user_id"]),
                    user_history_titles=tuple(history_titles),
                    candidate_titles=tuple(candidate_titles),
                )
                prompt = render_strategy_prompt(strategy_id, context, root=root)
                runtime = {
                    "temperature": 0.0,
                    "num_predict": 256,
                    "stream": False,
                    "timeout_seconds": TIMEOUTS[strategy_id],
                    "top_p": "",
                    "top_k": "",
                    "seed": "",
                    "context_window": "",
                    "stop_sequences": "",
                    "keep_alive": "",
                }
                request_id = stable_id(
                    "F6E3-REQ",
                    PHASE_ID,
                    instance["source_request_id"],
                    instance["dataset"],
                    instance["task"],
                    model,
                    strategy_id,
                )
                manifest_rows.append(
                    {
                        "request_id": request_id,
                        "source_request_id": instance["source_request_id"],
                        "source_request_order": str(instance["source_request_order"]),
                        "dataset": instance["dataset"],
                        "task": instance["task"],
                        "model": model,
                        "expected_model_digest": EXPECTED_TAG_DIGESTS[model],
                        "strategy_id": strategy_id,
                        "strategy_label": STRATEGY_LABELS[strategy_id],
                        "strategy_classification": "exact_parametric",
                        "scientific_role": "paper_aligned_core" if strategy_id == BASE_STRATEGY_ID else "cross_model_extension",
                        "user_hash": sha256_text(str(instance["user_id"]))[:16],
                        "user_id_source": str(instance["user_id"]),
                        "history_length": str(len(history_titles)),
                        "candidate_count": "20",
                        "candidate_order_hash": sha256_text(stable_json(candidate_ids)),
                        "candidate_item_ids": join_pipe(candidate_ids),
                        "candidate_item_titles": join_pipe(candidate_titles),
                        "positive_item_id": str(instance["positive_item_id"]),
                        "positive_candidate_position": str(instance["positive_candidate_position"]),
                        "prompt_hash": sha256_text(prompt),
                        "prompt": prompt,
                        "expected_output_contract": "official_title_list_top10_parser",
                        "runtime_parameters_json": stable_json(runtime),
                        "runtime_parameter_classification": "adapted_runtime",
                        "pilot_stratum": instance["pilot_stratum"],
                        "source_manifest_path": str(instance["source_manifest_path"]),
                        "status": "pending",
                        "notes": "F6E3 corrective materialization from frozen F6E2 descriptive pilot design.",
                    }
                )
    return manifest_rows


def assert_manifest_valid(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != 32:
        raise ValueError(f"f6e3_request_count_invalid:{len(rows)}")
    ids = [row["request_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("f6e3_duplicate_request_id")
    for row in rows:
        if row["model"] not in MODEL_ORDER:
            raise ValueError(f"f6e3_unallowed_model:{row['model']}")
        if row["strategy_id"] not in STRATEGY_ORDER:
            raise ValueError(f"f6e3_unallowed_strategy:{row['strategy_id']}")
        if int(row["candidate_count"]) != 20:
            raise ValueError(f"f6e3_candidate_count_invalid:{row['request_id']}")
        if len(split_pipe(row["candidate_item_ids"])) != len(set(split_pipe(row["candidate_item_ids"]))):
            raise ValueError(f"f6e3_candidate_order_duplicate:{row['request_id']}")
        if not row["prompt"] or not row["prompt_hash"]:
            raise ValueError(f"f6e3_empty_prompt:{row['request_id']}")


def ollama_process_snapshot() -> str:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-Process ollama -ErrorAction SilentlyContinue | Select-Object Id,StartTime,Path | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except Exception as exc:  # pragma: no cover - platform dependent
        return f"process_snapshot_failed:{type(exc).__name__}:{exc}"
    return result.stdout.strip() or "no_process_visible"


def ollama_client_version() -> str:
    executable = shutil.which("ollama")
    if executable is None:
        candidate = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
        executable = str(candidate) if candidate.exists() else ""
    if not executable:
        return "ollama_client_not_found"
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:  # pragma: no cover - platform dependent
        return f"ollama_client_version_failed:{type(exc).__name__}:{exc}"
    return (result.stdout + result.stderr).strip()


def http_get_json(url: str, timeout_seconds: int = 5) -> tuple[bool, dict[str, Any], str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
        return True, json.loads(body), ""
    except urllib.error.URLError as exc:
        return False, {}, str(exc.reason)
    except Exception as exc:  # pragma: no cover - backend dependent
        return False, {}, f"{type(exc).__name__}:{exc}"


def inspect_ollama_gate() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    client_version = ollama_client_version()
    rows.append(
        gate_row(
            "client_version",
            "ollama client installed",
            client_version,
            "pass" if "not_found" not in client_version else "fail",
            "high",
            "ollama --version",
            "Client check does not call generation.",
        )
    )
    rows.append(
        gate_row(
            "process_snapshot",
            "process metadata if visible",
            ollama_process_snapshot(),
            "info",
            "info",
            "Get-Process ollama",
            "Read-only process inspection.",
        )
    )
    version_ok, version, version_error = http_get_json(OLLAMA_VERSION_URL)
    rows.append(
        gate_row(
            "api_version",
            "reachable",
            stable_json(version) if version_ok else version_error,
            "pass" if version_ok else "fail",
            "critical",
            OLLAMA_VERSION_URL,
            "No generation endpoint is called.",
        )
    )
    tags_ok, tags, tags_error = http_get_json(OLLAMA_TAGS_URL)
    rows.append(
        gate_row(
            "api_tags",
            "reachable",
            stable_json(tags) if tags_ok else tags_error,
            "pass" if tags_ok else "fail",
            "critical",
            OLLAMA_TAGS_URL,
            "Model inventory endpoint only.",
        )
    )
    tag_digest = {}
    if tags_ok:
        for item in tags.get("models", []):
            name = str(item.get("name") or item.get("model") or "")
            tag_digest[name] = str(item.get("digest", ""))
    for model, expected in EXPECTED_TAG_DIGESTS.items():
        observed = tag_digest.get(model, "")
        rows.append(
            gate_row(
                f"digest_{safe_name(model)}",
                expected,
                observed,
                "pass" if observed == expected else "fail",
                "critical",
                OLLAMA_TAGS_URL,
                "Digest must match F6E2/F6D4A frozen expected tag digest.",
            )
        )
    return rows


def gate_row(check_name: str, expected: str, observed: str, status: str, severity: str, source: str, notes: str) -> dict[str, str]:
    return {
        "gate_id": stable_id("F6E3-GATE", check_name),
        "check_name": check_name,
        "expected": expected,
        "observed": observed,
        "status": status,
        "severity": severity,
        "source": source,
        "notes": notes,
    }


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def gate_decision(gate_rows: Sequence[Mapping[str, str]]) -> str:
    failures = [row for row in gate_rows if row["severity"] == "critical" and row["status"] != "pass"]
    if not failures:
        return "pass"
    if any(row["check_name"] in {"api_version", "api_tags"} for row in failures):
        return "f6e3_blocked_ollama_unreachable"
    if any(row["check_name"].startswith("digest_") for row in failures):
        return "f6e3_blocked_model_digest_mismatch"
    return "f6e3_blocked_by_runtime_failure"


def runtime_parameters(row: Mapping[str, str]) -> dict[str, Any]:
    return json.loads(row["runtime_parameters_json"])


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
    except Exception as exc:  # pragma: no cover - local backend dependent
        return RequestResult(False, "", {}, type(exc).__name__, str(exc), started, utc_now(), time.perf_counter() - start_clock)


def raw_path_for_model(model: str) -> Path:
    return RAW_PATHS[model]


def final_raw_by_request(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in RAW_PATHS.values():
        for record in read_jsonl(root / path):
            if parse_bool(record.get("is_final_attempt")):
                records[str(record.get("request_id", ""))] = record
    return records


def execute_pilot(
    root: Path,
    request_rows: Sequence[Mapping[str, str]],
    gate_rows: Sequence[Mapping[str, str]],
    *,
    allow_ollama_generate: bool,
    max_new_requests: int,
    request_executor: Callable[[Mapping[str, Any], int], RequestResult] = execute_ollama_request,
) -> int:
    if not allow_ollama_generate:
        raise PermissionError("f6e3_requires_allow_ollama_generate")
    decision = gate_decision(gate_rows)
    if decision != "pass":
        raise RuntimeError(decision)
    assert_manifest_valid(request_rows)
    if max_new_requests > 32:
        raise ValueError("f6e3_max_new_requests_exceeds_micro_pilot_cap")
    existing = final_raw_by_request(root)
    sent = 0
    for row in request_rows:
        if row["request_id"] in existing:
            continue
        if sent >= max_new_requests:
            break
        params = runtime_parameters(row)
        payload = {
            "model": row["model"],
            "prompt": row["prompt"],
            "stream": False,
            "options": {
                "temperature": float(params["temperature"]),
                "num_predict": int(params["num_predict"]),
            },
        }
        result = request_executor(payload, int(params["timeout_seconds"]))
        record = make_raw_record(row, result)
        append_jsonl_fsync(root / raw_path_for_model(row["model"]), record)
        sent += 1
        if not result.success or not result.response_text.strip():
            break
    return sent


def make_raw_record(row: Mapping[str, str], result: RequestResult) -> dict[str, Any]:
    raw = result.raw_backend_response
    response_text = result.response_text
    return {
        "phase": PHASE_ID,
        "request_id": row["request_id"],
        "source_request_id": row["source_request_id"],
        "attempt": 1,
        "is_final_attempt": True,
        "model": row["model"],
        "expected_model_digest": row["expected_model_digest"],
        "strategy_id": row["strategy_id"],
        "prompt_hash": row["prompt_hash"],
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
        "raw_response": response_text,
        "response_sha256": sha256_text(response_text),
        "transport_status": "success" if result.success else "failure",
        "error": result.error_message,
        "error_type": result.error_type,
        "parser_version": "official_title_list_top10_parser_local_strict_v1",
    }


def title_index(candidate_ids: Sequence[str], candidate_titles: Sequence[str]) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    exact: dict[str, tuple[str, str]] = {}
    normalized: dict[str, tuple[str, str]] = {}
    for item_id, title in zip(candidate_ids, candidate_titles):
        exact[str(title)] = (str(item_id), str(title))
        normalized[normalize_recommendation_text(str(title))] = (str(item_id), str(title))
    return exact, normalized


def parse_title_list_response(response_text: str, candidate_ids: Sequence[str], candidate_titles: Sequence[str]) -> dict[str, Any]:
    parsed_titles = extract_title_list_items(response_text, max_items=10)
    exact, normalized = title_index(candidate_ids, candidate_titles)
    mapped_ids: list[str] = []
    mapped_titles: list[str] = []
    match_types: list[str] = []
    unmapped = 0
    for title in parsed_titles:
        cleaned = html.unescape(str(title)).strip()
        if cleaned in exact:
            item_id, canonical_title = exact[cleaned]
            match_type = MATCH_EXACT
        else:
            item = normalized.get(normalize_recommendation_text(cleaned))
            if item is None:
                unmapped += 1
                continue
            item_id, canonical_title = item
            match_type = MATCH_NORMALIZED
        mapped_ids.append(item_id)
        mapped_titles.append(canonical_title)
        match_types.append(match_type)
    duplicate_count = len(mapped_ids) - len(set(mapped_ids))
    hallucination_count = unmapped
    strict_valid = len(parsed_titles) == 10 and len(mapped_ids) == 10 and duplicate_count == 0 and hallucination_count == 0
    parse_valid = len(parsed_titles) > 0
    return {
        "parse_valid": parse_valid,
        "strict_valid": strict_valid,
        "parsed_titles": parsed_titles,
        "mapped_ids": mapped_ids,
        "mapped_titles": mapped_titles,
        "match_types": match_types,
        "output_length": len(parsed_titles),
        "mapped_count": len(mapped_ids),
        "unmapped_count": unmapped,
        "duplicate_count": duplicate_count,
        "candidate_hallucination_count": hallucination_count,
        "parse_status": "strict_valid" if strict_valid else ("parsed_not_strict" if parse_valid else "parse_failed"),
    }


def extract_title_list_items(response_text: str, *, max_items: int = 10) -> list[str]:
    numbered: list[str] = []
    for line in str(response_text or "").splitlines():
        match = re.match(r"^\s*(?:[-*]\s*)?(?:rank\s*)?\d+\s*[\.\)\]:-]\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip().strip("\"'")
            if value:
                numbered.append(value)
                if len(numbered) >= max_items:
                    break
    if numbered:
        return numbered
    return parse_recommendation_items(response_text, max_items=max_items)


def analyze_instances(root: Path, request_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw_records = final_raw_by_request(root)
    instance_rows: list[dict[str, str]] = []
    parser_rows: list[dict[str, str]] = []
    for row in request_rows:
        raw = raw_records.get(row["request_id"])
        candidate_ids = split_pipe(row["candidate_item_ids"])
        candidate_titles = split_pipe(row["candidate_item_titles"])
        if raw is None:
            parse = {
                "parse_valid": False,
                "strict_valid": False,
                "mapped_ids": [],
                "mapped_titles": [],
                "output_length": 0,
                "mapped_count": 0,
                "unmapped_count": 0,
                "duplicate_count": 0,
                "candidate_hallucination_count": 0,
                "parse_status": "missing_raw_response",
            }
            transport = "missing"
            completion = "missing_raw_response"
            response_text = ""
            failure = "missing_raw_response"
            duration = prompt_eval = eval_count = ""
            raw_file = raw_path_for_model(row["model"]).as_posix()
        else:
            response_text = str(raw.get("raw_response", ""))
            parse = parse_title_list_response(response_text, candidate_ids, candidate_titles)
            transport = str(raw.get("transport_status", ""))
            completion = "completed" if transport == "success" and response_text.strip() else "failed"
            failure = "" if parse["strict_valid"] and transport == "success" else (str(raw.get("error_type") or raw.get("error") or parse["parse_status"]))
            duration = fmt(float(raw.get("duration_seconds", 0) or 0))
            prompt_eval = str(raw.get("prompt_eval_count", ""))
            eval_count = str(raw.get("eval_count", ""))
            raw_file = raw_path_for_model(row["model"]).as_posix()
        positive_rank = ""
        hr = 0.0
        ndcg = 0.0
        if row["positive_item_id"] in parse["mapped_ids"]:
            positive_rank = str(parse["mapped_ids"].index(row["positive_item_id"]) + 1)
            rank_value = int(positive_rank)
            if rank_value <= 5:
                hr = 1.0
                ndcg = 1.0 / math.log2(rank_value + 1)
        instance_rows.append(
            {
                "request_id": row["request_id"],
                "source_request_id": row["source_request_id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "model": row["model"],
                "strategy_id": row["strategy_id"],
                "transport_status": transport,
                "completion_status": completion,
                "parse_valid": bool_text(bool(parse["parse_valid"])),
                "strict_valid": bool_text(bool(parse["strict_valid"] and transport == "success")),
                "candidate_valid": bool_text(int(parse["candidate_hallucination_count"]) == 0 and int(parse["mapped_count"]) == int(parse["output_length"])),
                "duplicate_valid": bool_text(int(parse["duplicate_count"]) == 0),
                "output_length": str(parse["output_length"]),
                "parsed_ranking_item_ids": join_pipe(parse["mapped_ids"]),
                "parsed_ranking_titles": join_pipe(parse["mapped_titles"]),
                "positive_item_id": row["positive_item_id"],
                "positive_candidate_position": row["positive_candidate_position"],
                "positive_rank": positive_rank,
                "hr_at5": fmt(hr),
                "ndcg_at5": fmt(ndcg),
                "duration_seconds": duration,
                "prompt_eval_count": prompt_eval,
                "eval_count": eval_count,
                "failure_reason": failure,
                "raw_output_file": raw_file,
                "notes": "Pilot metric only; not a full-matrix result.",
            }
        )
        parser_rows.append(
            {
                "request_id": row["request_id"],
                "model": row["model"],
                "strategy_id": row["strategy_id"],
                "dataset": row["dataset"],
                "parse_status": str(parse["parse_status"]),
                "parsed_count": str(parse["output_length"]),
                "mapped_count": str(parse["mapped_count"]),
                "unmapped_count": str(parse["unmapped_count"]),
                "duplicate_count": str(parse["duplicate_count"]),
                "candidate_hallucination_count": str(parse["candidate_hallucination_count"]),
                "strict_valid": bool_text(bool(parse["strict_valid"] and transport == "success")),
                "raw_excerpt_sha256": sha256_text(response_text[:200]) if response_text else "",
                "notes": "Strict title-list candidate parser without fuzzy matching.",
            }
        )
    return instance_rows, parser_rows


def ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0"
    return fmt(numerator / denominator)


def build_strategy_summary(instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in instance_rows:
        groups[(row["model"], row["strategy_id"], row["dataset"])].append(row)
    rows: list[dict[str, str]] = []
    for (model, strategy, dataset), group in sorted(groups.items()):
        attempted = [row for row in group if row["transport_status"] != "missing"]
        completed = [row for row in group if row["transport_status"] == "success"]
        parse_valid = [row for row in group if row["parse_valid"] == "true"]
        strict = [row for row in group if row["strict_valid"] == "true"]
        candidate_valid = [row for row in group if row["candidate_valid"] == "true"]
        duplicate_invalid = [row for row in group if row["duplicate_valid"] != "true"]
        for population_name, population in [("all_request", group), ("strict_valid", strict)]:
            hr_values = [float(row["hr_at5"] or 0) for row in population]
            ndcg_values = [float(row["ndcg_at5"] or 0) for row in population]
            ranks = [float(row["positive_rank"]) for row in population if row["positive_rank"]]
            rows.append(
                {
                    "summary_id": stable_id("F6E3-SUM", model, strategy, dataset, population_name),
                    "model": model,
                    "strategy_id": strategy,
                    "dataset": dataset,
                    "evaluation_population": population_name,
                    "logical_requests": str(len(group)),
                    "attempted_requests": str(len(attempted)),
                    "completed_requests": str(len(completed)),
                    "transport_success_rate": ratio(len(completed), len(group)),
                    "parse_valid_rate": ratio(len(parse_valid), len(group)),
                    "strict_valid_rate": ratio(len(strict), len(group)),
                    "candidate_valid_rate": ratio(len(candidate_valid), len(group)),
                    "duplicate_invalid_rate": ratio(len(duplicate_invalid), len(group)),
                    "hr_at5": fmt(mean(hr_values) if hr_values else 0.0),
                    "ndcg_at5": fmt(mean(ndcg_values) if ndcg_values else 0.0),
                    "mean_positive_rank": fmt(mean(ranks)) if ranks else "",
                    "notes": "Pilot-only aggregate; do not cite as full-run result.",
                }
            )
    return rows


def build_runtime_summary(instance_rows: Sequence[Mapping[str, str]], request_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    request_lookup = {row["request_id"]: row for row in request_rows}
    groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in instance_rows:
        groups[(row["model"], row["strategy_id"], row["dataset"])].append(row)
    rows: list[dict[str, str]] = []
    for (model, strategy, dataset), group in sorted(groups.items()):
        attempted = [row for row in group if row["transport_status"] != "missing"]
        durations = [float(row["duration_seconds"]) for row in attempted if row["duration_seconds"]]
        prompt_counts = [float(row["prompt_eval_count"]) for row in attempted if row["prompt_eval_count"]]
        eval_counts = [float(row["eval_count"]) for row in attempted if row["eval_count"]]
        timeout = ""
        for row in group:
            source = request_lookup.get(row["request_id"])
            if source:
                timeout = str(json.loads(source["runtime_parameters_json"])["timeout_seconds"])
                break
        eval_sum = sum(eval_counts)
        duration_sum = sum(durations)
        rows.append(
            {
                "runtime_id": stable_id("F6E3-RT", model, strategy, dataset),
                "model": model,
                "strategy_id": strategy,
                "dataset": dataset,
                "attempted_requests": str(len(attempted)),
                "latency_mean": fmt(mean(durations)) if durations else "",
                "latency_p50": fmt(median(durations)) if durations else "",
                "latency_p95": fmt(percentile(durations, 0.95)) if durations else "",
                "prompt_eval_count_mean": fmt(mean(prompt_counts)) if prompt_counts else "",
                "eval_count_mean": fmt(mean(eval_counts)) if eval_counts else "",
                "tokens_per_second_estimate": fmt(eval_sum / duration_sum) if duration_sum else "",
                "timeout_seconds": timeout,
                "notes": "Runtime summary from F6E3 raw records only.",
            }
        )
    return rows


def protected_summary(root: Path) -> dict[str, Any]:
    f2_rows = read_csv_rows(root, F2_PROTECTED_VALIDATION)
    unchanged = sum(1 for row in f2_rows if row.get("status") == "unchanged")
    changed = sum(1 for row in f2_rows if row.get("status") and row.get("status") != "unchanged")
    f6d4a_rows = read_csv_rows(root, F6D4A_PROTECTED_VERIFICATION)
    f6d4a_changed = 0
    f6d4a_checked = 0
    for row in f6d4a_rows:
        if row.get("verification_id") == "F6D4A-CLOSURE-PROT-F6D4A":
            f6d4a_checked = int(float(row.get("checked", "0")))
            f6d4a_changed = int(float(row.get("changed", "0")))
    raw_changed = 0
    raw_hashes: dict[str, str] = {}
    for path, expected in F6D4A_RAW_HASHES.items():
        observed = file_sha256_or_missing(root, Path(path))
        raw_hashes[path] = observed
        raw_changed += int(observed != expected)
    return {
        "f2_checked": unchanged + changed,
        "f2_changed": changed,
        "f6d4a_checked": f6d4a_checked,
        "f6d4a_changed": f6d4a_changed,
        "f6d4a_raw_changed": raw_changed,
        "f6d4a_raw_hashes": raw_hashes,
    }


def build_pilot_gate_results(
    request_rows: Sequence[Mapping[str, str]],
    gate_rows: Sequence[Mapping[str, str]],
    instance_rows: Sequence[Mapping[str, str]],
    protected: Mapping[str, Any],
    max_new_requests: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(name: str, expected: str, observed: str, passed: bool, severity: str, evidence: str, notes: str = "") -> None:
        rows.append(
            {
                "gate_id": stable_id("F6E3-PGATE", name),
                "gate_name": name,
                "expected": expected,
                "observed": observed,
                "status": "pass" if passed else "fail",
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    add("request_count", "32", str(len(request_rows)), len(request_rows) == 32, "critical", OUTPUT_TABLES["request_manifest"].as_posix())
    add("request_id_unique", "32 unique", str(len({row["request_id"] for row in request_rows})), len({row["request_id"] for row in request_rows}) == len(request_rows), "critical", OUTPUT_TABLES["request_manifest"].as_posix())
    add("no_llama3", "0", str(sum(1 for row in request_rows if "llama" in row["model"].lower())), all("llama" not in row["model"].lower() for row in request_rows), "critical", OUTPUT_TABLES["request_manifest"].as_posix())
    add("max_new_requests_cap", "<=32", str(max_new_requests), max_new_requests <= 32, "critical", "CLI")
    add("ollama_gate", "pass", gate_decision(gate_rows), gate_decision(gate_rows) == "pass", "critical", OUTPUT_TABLES["ollama_gate"].as_posix())
    add("protected_inputs", "changed=0", f"f2={protected['f2_changed']};f6d4a={protected['f6d4a_changed']};raw={protected['f6d4a_raw_changed']}", protected["f2_changed"] == 0 and protected["f6d4a_changed"] == 0 and protected["f6d4a_raw_changed"] == 0, "critical", "protected input tables")
    attempted = sum(1 for row in instance_rows if row["transport_status"] != "missing")
    add("no_extra_requests", "<=32", str(attempted), attempted <= 32, "critical", OUTPUT_TABLES["instance_results"].as_posix())
    strict_invalid = sum(1 for row in instance_rows if row["transport_status"] == "success" and row["strict_valid"] != "true")
    add("strict_validity", "strict_invalid=0", str(strict_invalid), strict_invalid == 0, "high", OUTPUT_TABLES["parser_validity"].as_posix(), "Pilot parser gate; if raw are absent this remains pass until execution.")
    return rows


def build_missing_or_blocked(gate_rows: Sequence[Mapping[str, str]], request_rows: Sequence[Mapping[str, str]], instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "block_id": "F6E3-BLOCK-JSON-INDEX-NOT-SELECTED",
            "category": "scope",
            "affected_scope": "JSON/index local ablation",
            "status": "not_executed_selected_scope",
            "severity": "info",
            "evidence": "User-selected F6E3 scope is base + LLMRank only.",
            "safe_next_action": "Run a separate plumbing-control phase if needed.",
            "notes": "JSON/index remains local_ablation.",
        },
        {
            "block_id": "F6E3-BLOCK-LLMSREC-NOT-SELECTED",
            "category": "scope",
            "affected_scope": "LLMSRec reconstructed",
            "status": "not_executed_selected_scope",
            "severity": "info",
            "evidence": "F6E1 classifies LLMSRec as reconstructed.",
            "safe_next_action": "Do not promote LLMSRec to exact; design separate reconstructed pilot if needed.",
            "notes": "No LLMSRec generation in F6E3.",
        },
        {
            "block_id": "F6E3-BLOCK-LLAMA3-70B",
            "category": "hardware_model",
            "affected_scope": "Llama 3 70B",
            "status": "blocked_hardware",
            "severity": "critical",
            "evidence": "F6E2 marked local model missing and hardware infeasible.",
            "safe_next_action": "Keep blocked; no local generation.",
            "notes": "No Llama 3 rows appear in F6E3 request manifest.",
        },
    ]
    decision = gate_decision(gate_rows)
    if decision != "pass":
        rows.append(
            {
                "block_id": "F6E3-BLOCK-OLLAMA-GATE",
                "category": "runtime",
                "affected_scope": "all F6E3 generation requests",
                "status": decision,
                "severity": "critical",
                "evidence": OUTPUT_TABLES["ollama_gate"].as_posix(),
                "safe_next_action": "Fix local Ollama availability/digest mismatch, then rerun runtime gate.",
                "notes": "No generation should be sent while this block is active.",
            }
        )
    failures = [row for row in instance_rows if row["transport_status"] == "failure"]
    if failures:
        rows.append(
            {
                "block_id": "F6E3-BLOCK-TRANSPORT",
                "category": "runtime",
                "affected_scope": "pilot execution",
                "status": "f6e3_blocked_by_runtime_failure",
                "severity": "critical",
                "evidence": OUTPUT_TABLES["instance_results"].as_posix(),
                "safe_next_action": "Inspect first failure; do not retry automatically.",
                "notes": f"transport_failures={len(failures)}",
            }
        )
    parser_failures = [row for row in instance_rows if row["transport_status"] == "success" and row["strict_valid"] != "true"]
    if parser_failures:
        rows.append(
            {
                "block_id": "F6E3-BLOCK-PARSER",
                "category": "parser",
                "affected_scope": "pilot strict-valid gate",
                "status": "f6e3_blocked_by_parser_validity",
                "severity": "high",
                "evidence": OUTPUT_TABLES["parser_validity"].as_posix(),
                "safe_next_action": "Review raw without manual correction; revise parser/prompt only in later phase.",
                "notes": f"strict_invalid={len(parser_failures)}",
            }
        )
    return rows


def final_decision(gate_rows: Sequence[Mapping[str, str]], instance_rows: Sequence[Mapping[str, str]], protected: Mapping[str, Any]) -> str:
    if protected["f2_changed"] or protected["f6d4a_changed"] or protected["f6d4a_raw_changed"]:
        return "f6e3_blocked_by_protected_input_invariance"
    gate = gate_decision(gate_rows)
    if gate != "pass":
        return gate
    attempted = [row for row in instance_rows if row["transport_status"] != "missing"]
    if not attempted:
        return "f6e3_runtime_gate_passed_pilot_not_executed"
    if any(row["transport_status"] == "failure" for row in instance_rows):
        return "f6e3_blocked_by_runtime_failure"
    if any(row["transport_status"] == "success" and row["strict_valid"] != "true" for row in instance_rows):
        return "f6e3_blocked_by_parser_validity"
    if len(attempted) == 32:
        return "f6e3_exact_prompt_pilot_complete_with_documented_blocks"
    return "f6e3_incomplete_missing_evidence"


def table_hashes(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    columns = {
        "ollama_gate": OLLAMA_GATE_COLUMNS,
        "request_manifest": REQUEST_MANIFEST_COLUMNS,
        "instance_results": INSTANCE_COLUMNS,
        "strategy_summary": STRATEGY_SUMMARY_COLUMNS,
        "runtime_summary": RUNTIME_SUMMARY_COLUMNS,
        "parser_validity": PARSER_VALIDITY_COLUMNS,
        "pilot_gate_results": PILOT_GATE_COLUMNS,
        "missing_or_blocked": MISSING_COLUMNS,
    }
    return {name: sha256_text(csv_text(rows, columns[name])) for name, rows in sorted(tables.items())}


def input_hashes(root: Path) -> dict[str, str]:
    paths = [
        F6E1_MANIFEST,
        F6E2_MANIFEST,
        F6E2_PILOT_DESIGN,
        F6E2_EXECUTION_MATRIX,
        F6E1_PROMPT_MODULE,
        TITLE_PARSER_MODULE,
        CANONICAL_FULL_MANIFEST,
        CANONICAL_FULL_REQUESTS,
        F6D3_LASTFM_RANKING_MANIFEST,
        F2_PROTECTED_VALIDATION,
        F6D4A_INPUT_HASHES,
        F6D4A_PROTECTED_VERIFICATION,
    ]
    paths.extend(PLAN_E_RANKING_MANIFESTS.values())
    paths.extend(Path(path) for path in F6D4A_RAW_HASHES)
    return {path.as_posix(): file_sha256_or_missing(root, path) for path in paths}


def raw_hashes(root: Path) -> dict[str, str]:
    return {path.as_posix(): file_sha256_or_missing(root, path) for path in RAW_PATHS.values()}


def build_freeze_manifest(root: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]], decision: str, protected: Mapping[str, Any]) -> dict[str, Any]:
    hashes = table_hashes(tables)
    raw = raw_hashes(root)
    content = {
        "phase": PHASE_ID,
        "decision": decision,
        "input_hashes": input_hashes(root),
        "raw_hashes": raw,
        "table_hashes": hashes,
        "prompt_freeze_hash": F6E1_SCIENTIFIC_HASH,
        "preflight_hash": F6E2_SCIENTIFIC_HASH,
        "protected_summary": protected,
    }
    return {
        "phase": PHASE_ID,
        "decision": decision,
        "frozen_timestamp_utc": FROZEN_TIMESTAMP_UTC,
        "input_hashes": content["input_hashes"],
        "prompt_freeze_hash": F6E1_SCIENTIFIC_HASH,
        "preflight_hash": F6E2_SCIENTIFIC_HASH,
        "model_tags": EXPECTED_TAG_DIGESTS,
        "runtime_version": gate_observed_value(tables["ollama_gate"], "api_version"),
        "request_manifest_hash": hashes.get("request_manifest", ""),
        "raw_hashes": raw,
        "result_table_hashes": hashes,
        "scientific_content_hash": sha256_text(stable_json(content)),
        "runtime_metadata": {
            "analysis_only_uses_no_ollama": True,
            "generation_endpoint_requires_allow_flag": True,
            "max_generation_calls": 32,
        },
        "pilot_decision": decision,
        "protected_summary": protected,
    }


def gate_observed_value(rows: Sequence[Mapping[str, str]], check_name: str) -> str:
    for row in rows:
        if row.get("check_name") == check_name:
            return row.get("observed", "")
    return ""


def build_report(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    decision = manifest["decision"]
    instance_rows = tables["instance_results"]
    attempted = sum(1 for row in instance_rows if row["transport_status"] != "missing")
    completed = sum(1 for row in instance_rows if row["transport_status"] == "success")
    strict = sum(1 for row in instance_rows if row["strict_valid"] == "true")
    lines = [
        "# Phase F6E3 - Ollama Runtime Gate And Exact-Prompt Micro-Pilot",
        "",
        f"Decision: `{decision}`.",
        "",
        "F6E3 materializes a corrective micro-pilot manifest from the descriptive F6E2 pilot design. It executes only base and LLMRank prompts on local Qwen2/Mistral when the Ollama gate passes.",
        "",
        "## Scope",
        "",
        "- Included: Qwen2 base, Mistral base, Qwen2 LLMRank, Mistral LLMRank.",
        "- Excluded: JSON/index control, LLMSRec reconstructed, Llama 3 70B, GPT/Claude, full run.",
        "",
        "## Runtime Gate",
        "",
    ]
    for row in tables["ollama_gate"]:
        lines.append(f"- `{row['check_name']}`: `{row['status']}` ({row['observed'][:160]}).")
    lines.extend(
        [
            "",
            "## Pilot Execution",
            "",
            f"- Request manifest rows: `{len(tables['request_manifest'])}`.",
            f"- Attempted requests: `{attempted}`.",
            f"- Completed requests: `{completed}`.",
            f"- Strict-valid responses: `{strict}`.",
            "- Raw paths are F6E3-specific and append-only.",
            "",
            "## Hashes",
            "",
            f"- Request manifest hash: `{manifest['request_manifest_hash']}`.",
            f"- Scientific content hash: `{manifest['scientific_content_hash']}`.",
            f"- Qwen2 F6E3 raw hash: `{manifest['raw_hashes'].get(RAW_PATHS[QWEN2_TAG].as_posix(), 'missing')}`.",
            f"- Mistral F6E3 raw hash: `{manifest['raw_hashes'].get(RAW_PATHS[MISTRAL_TAG].as_posix(), 'missing')}`.",
            "",
            "## Missing Or Blocked",
            "",
        ]
    )
    for row in tables["missing_or_blocked"]:
        lines.append(f"- `{row['block_id']}`: `{row['status']}`.")
    lines.extend(
        [
            "",
            "## Full-Run Recommendation",
            "",
            full_run_recommendation(decision),
            "",
        ]
    )
    return "\n".join(lines)


def full_run_recommendation(decision: str) -> str:
    if decision == "f6e3_exact_prompt_pilot_complete_with_documented_blocks":
        return "`proceed_to_full_run` only after explicit authorization; keep Llama3/LLMSRec/JSON-index outside this exact-prompt full run."
    if decision in {"f6e3_blocked_ollama_unreachable", "f6e3_blocked_model_digest_mismatch", "f6e3_blocked_by_runtime_failure"}:
        return "`revise_runtime_parameters` / fix local runtime gate before any full run."
    if decision == "f6e3_blocked_by_parser_validity":
        return "`revise_parser_contract`; do not full-run until title-list parsing passes."
    return "`repeat_pilot_required`; no full run is authorized from this state."


def analyze_only(root: Path, gate_rows: Sequence[Mapping[str, str]] | None = None, max_new_requests: int = 32) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any], str]:
    request_rows = materialize_request_manifest(root)
    assert_manifest_valid(request_rows)
    if gate_rows is None:
        gate_rows = read_csv_rows(root, OUTPUT_TABLES["ollama_gate"])
    if not gate_rows:
        gate_rows = [gate_row("analysis_only_runtime_gate_not_queried", "not_applicable", "not_queried", "info", "info", "analysis-only", "Analysis-only does not query Ollama.")]
    instance_rows, parser_rows = analyze_instances(root, request_rows)
    protected = protected_summary(root)
    strategy_summary = build_strategy_summary(instance_rows)
    runtime_summary = build_runtime_summary(instance_rows, request_rows)
    pilot_gates = build_pilot_gate_results(request_rows, gate_rows, instance_rows, protected, max_new_requests)
    missing = build_missing_or_blocked(gate_rows, request_rows, instance_rows)
    decision = final_decision(gate_rows, instance_rows, protected)
    tables = {
        "ollama_gate": [dict(row) for row in gate_rows],
        "request_manifest": request_rows,
        "instance_results": instance_rows,
        "strategy_summary": strategy_summary,
        "runtime_summary": runtime_summary,
        "parser_validity": parser_rows,
        "pilot_gate_results": pilot_gates,
        "missing_or_blocked": missing,
    }
    manifest = build_freeze_manifest(root, tables, decision, protected)
    report = build_report(manifest, tables)
    return tables, manifest, report


def write_all(root: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]], manifest: Mapping[str, Any], report: str) -> dict[str, str]:
    columns = {
        "ollama_gate": OLLAMA_GATE_COLUMNS,
        "request_manifest": REQUEST_MANIFEST_COLUMNS,
        "instance_results": INSTANCE_COLUMNS,
        "strategy_summary": STRATEGY_SUMMARY_COLUMNS,
        "runtime_summary": RUNTIME_SUMMARY_COLUMNS,
        "parser_validity": PARSER_VALIDITY_COLUMNS,
        "pilot_gate_results": PILOT_GATE_COLUMNS,
        "missing_or_blocked": MISSING_COLUMNS,
    }
    written: dict[str, str] = {}
    for name, path in OUTPUT_TABLES.items():
        write_csv(root / path, tables[name], columns[name])
        written[path.as_posix()] = sha256_file(root / path)
    (root / FREEZE_MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / FREEZE_MANIFEST_PATH).write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    written[FREEZE_MANIFEST_PATH.as_posix()] = sha256_file(root / FREEZE_MANIFEST_PATH)
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(report, encoding="utf-8")
    written[REPORT_PATH.as_posix()] = sha256_file(root / REPORT_PATH)
    return written


def run_materialize_only(root: Path) -> dict[str, str]:
    tables, manifest, report = analyze_only(root, gate_rows=[], max_new_requests=0)
    manifest = dict(manifest)
    manifest["decision"] = "f6e3_materialized_pilot_not_executed"
    manifest["pilot_decision"] = "f6e3_materialized_pilot_not_executed"
    report = build_report(manifest, tables)
    return write_all(root, tables, manifest, report)


def run_runtime_gate_only(root: Path) -> dict[str, str]:
    gate_rows = inspect_ollama_gate()
    tables, manifest, report = analyze_only(root, gate_rows=gate_rows, max_new_requests=0)
    return write_all(root, tables, manifest, report)


def run_execute_pilot(root: Path, *, allow_ollama_generate: bool, max_new_requests: int) -> dict[str, str]:
    request_rows = materialize_request_manifest(root)
    gate_rows = inspect_ollama_gate()
    execute_pilot(root, request_rows, gate_rows, allow_ollama_generate=allow_ollama_generate, max_new_requests=max_new_requests)
    tables, manifest, report = analyze_only(root, gate_rows=gate_rows, max_new_requests=max_new_requests)
    return write_all(root, tables, manifest, report)


def run_analysis_only(root: Path) -> dict[str, str]:
    tables, manifest, report = analyze_only(root)
    return write_all(root, tables, manifest, report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase F6E3 exact-prompt micro-pilot.")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--runtime-gate-only", action="store_true")
    parser.add_argument("--execute-pilot", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--allow-ollama-generate", action="store_true")
    parser.add_argument("--max-new-requests", type=int, default=32)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    mode_count = sum(bool(value) for value in [args.materialize_only, args.runtime_gate_only, args.execute_pilot, args.analysis_only])
    if mode_count != 1:
        parser.error("choose exactly one mode")
    if args.materialize_only:
        written = run_materialize_only(args.root)
    elif args.runtime_gate_only:
        written = run_runtime_gate_only(args.root)
    elif args.execute_pilot:
        written = run_execute_pilot(args.root, allow_ollama_generate=args.allow_ollama_generate, max_new_requests=args.max_new_requests)
    else:
        written = run_analysis_only(args.root)
    print(json.dumps({"written": written}, ensure_ascii=True, sort_keys=True, indent=2))
    return 0

"""Phase F6E3B adapted indexed output-contract micro-pilot.

The phase reuses the frozen F6E3 logical instances and prompt-strategy core,
but asks local models to answer with a strict JSON object containing one-based
candidate indices. Importing this module has no filesystem, network, or
inference side effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

from src.analysis import plan_f6e3_exact_prompt_pilot as f6e3
from src.analysis import plan_f6e3a_parser_recovery_audit as f6e3a
from src.llm.plan_f6e1_prompt_strategies import BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID


PHASE_ID = "phase_f6e3b"
FROZEN_TIMESTAMP_UTC = "2026-07-26T00:00:00Z"

QWEN2_TAG = f6e3.QWEN2_TAG
MISTRAL_TAG = f6e3.MISTRAL_TAG
MODEL_ORDER = [QWEN2_TAG, MISTRAL_TAG]
STRATEGY_ORDER = [BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID]
STRATEGY_LABELS = {
    BASE_STRATEGY_ID: "base",
    LLMRANK_STRATEGY_ID: "llmrank",
}
DATASET_ORDER = f6e3.DATASET_ORDER
EXPECTED_TAG_DIGESTS = f6e3.EXPECTED_TAG_DIGESTS

EXPECTED_F6E3_RAW_HASHES = {
    "results/llm/phase_f6e3_qwen2_exact_prompt_pilot_raw.jsonl": "03d5a43d3d2f10c73e50c5086f1e3f6767051d2baff072ac777b3822bcc4305d",
    "results/llm/phase_f6e3_mistral_exact_prompt_pilot_raw.jsonl": "c8e57b9dfbde8ed3471ff45e1c272fbf898a48c7364402d40f49df3b60322e49",
}
EXPECTED_F6E3A_SCIENTIFIC_HASH = "a060787682876c73a036b760fefa744ef81d5004d1ae67e63b54b0bac0f40c58"

F6E3_REQUEST_MANIFEST = Path("results/tables/phase_f6e3_request_manifest.csv")
F6E3_INSTANCE_RESULTS = Path("results/tables/phase_f6e3_instance_results.csv")
F6E3_PARSER_VALIDITY = Path("results/tables/phase_f6e3_parser_validity.csv")
F6E3_FREEZE = Path("results/manifests/phase_f6e3_pilot_freeze.json")
F6E3A_FREEZE = Path("results/manifests/phase_f6e3a_parser_recovery_freeze.json")
F6E1_FREEZE = Path("results/manifests/phase_f6e1_prompt_freeze.json")
F6E2_FREEZE = Path("results/manifests/phase_f6e2_preflight_freeze.json")
F6E2_PILOT_DESIGN = Path("results/tables/phase_f6e2_pilot_design.csv")
F6E1_PROMPT_MODULE = Path("src/llm/plan_f6e1_prompt_strategies.py")
F6E3_MODULE = Path("src/analysis/plan_f6e3_exact_prompt_pilot.py")
F6E3A_MODULE = Path("src/analysis/plan_f6e3a_parser_recovery_audit.py")

REPORT_PATH = Path("report/phase_f6e3b_adapted_output_contract_pilot.md")
FREEZE_MANIFEST_PATH = Path("results/manifests/phase_f6e3b_pilot_freeze.json")
OUTPUT_TABLES = {
    "ollama_gate": Path("results/tables/phase_f6e3b_ollama_gate.csv"),
    "request_manifest": Path("results/tables/phase_f6e3b_request_manifest.csv"),
    "instance_results": Path("results/tables/phase_f6e3b_instance_results.csv"),
    "cell_summary": Path("results/tables/phase_f6e3b_cell_summary.csv"),
    "parser_validity": Path("results/tables/phase_f6e3b_parser_validity.csv"),
    "runtime_summary": Path("results/tables/phase_f6e3b_runtime_summary.csv"),
    "exact_vs_adapted": Path("results/tables/phase_f6e3b_exact_vs_adapted.csv"),
    "gate_results": Path("results/tables/phase_f6e3b_gate_results.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6e3b_missing_or_blocked.csv"),
}
RAW_PATHS = {
    QWEN2_TAG: Path("results/llm/phase_f6e3b_qwen2_adapted_indexed_pilot_raw.jsonl"),
    MISTRAL_TAG: Path("results/llm/phase_f6e3b_mistral_adapted_indexed_pilot_raw.jsonl"),
}

REQUEST_MANIFEST_COLUMNS = [
    "request_id",
    "f6e3_request_id",
    "source_request_id",
    "source_request_order",
    "dataset",
    "task",
    "model",
    "expected_model_digest",
    "strategy_id",
    "strategy_label",
    "adapted_strategy_id",
    "strategy_classification",
    "scientific_role",
    "output_contract_id",
    "candidate_count",
    "candidate_order_hash",
    "candidate_item_ids",
    "candidate_item_titles",
    "positive_item_id",
    "positive_candidate_position",
    "core_prompt_hash",
    "candidate_serialization_hash",
    "output_adapter_hash",
    "prompt_hash",
    "prompt",
    "expected_output_contract",
    "runtime_parameters_json",
    "runtime_parameter_classification",
    "pilot_stratum",
    "source_f6e3_manifest_path",
    "source_f6e3_raw_path",
    "status",
    "notes",
]
OLLAMA_GATE_COLUMNS = f6e3.OLLAMA_GATE_COLUMNS
INSTANCE_COLUMNS = [
    "request_id",
    "f6e3_request_id",
    "source_request_id",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "adapted_strategy_id",
    "transport_status",
    "completion_status",
    "json_parse_valid",
    "schema_valid",
    "strict_valid",
    "candidate_valid",
    "duplicate_valid",
    "output_length",
    "output_indices",
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
    "f6e3_request_id",
    "model",
    "strategy_id",
    "dataset",
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
CELL_SUMMARY_COLUMNS = [
    "cell_id",
    "model",
    "strategy_id",
    "adapted_strategy_id",
    "logical_requests",
    "attempted_requests",
    "completed_requests",
    "strict_valid_requests",
    "strict_valid_rate",
    "json_parse_valid_rate",
    "schema_valid_rate",
    "transport_success_rate",
    "parse_status_distribution",
    "hr_at5_all_request",
    "ndcg_at5_all_request",
    "hr_at5_strict_valid",
    "ndcg_at5_strict_valid",
    "gate_status",
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
EXACT_VS_ADAPTED_COLUMNS = [
    "comparison_id",
    "f6e3_request_id",
    "f6e3b_request_id",
    "dataset",
    "model",
    "strategy_id",
    "f6e3_strict_valid",
    "f6e3_parse_status",
    "f6e3_duration_seconds",
    "f6e3b_strict_valid",
    "f6e3b_parse_status",
    "f6e3b_duration_seconds",
    "strict_valid_delta",
    "metric_comparability",
    "notes",
]
GATE_COLUMNS = [
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
TABLE_COLUMNS = {
    "ollama_gate": OLLAMA_GATE_COLUMNS,
    "request_manifest": REQUEST_MANIFEST_COLUMNS,
    "instance_results": INSTANCE_COLUMNS,
    "cell_summary": CELL_SUMMARY_COLUMNS,
    "parser_validity": PARSER_VALIDITY_COLUMNS,
    "runtime_summary": RUNTIME_SUMMARY_COLUMNS,
    "exact_vs_adapted": EXACT_VS_ADAPTED_COLUMNS,
    "gate_results": GATE_COLUMNS,
    "missing_or_blocked": MISSING_COLUMNS,
}


def stable_json(value: Any) -> str:
    return f6e3.stable_json(value)


def sha256_text(value: str) -> str:
    return f6e3.sha256_text(value)


def sha256_file(path: Path) -> str:
    return f6e3.sha256_file(path)


def file_sha256_or_missing(root: Path, relative_path: Path) -> str:
    return f6e3.file_sha256_or_missing(root, relative_path)


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f6e3.stable_id(prefix, *parts, length=length)


def bool_text(value: bool) -> str:
    return f6e3.bool_text(value)


def fmt(value: float | int | str) -> str:
    return f6e3.fmt(value)


def split_pipe(value: Any) -> list[str]:
    return f6e3.split_pipe(value)


def join_pipe(values: Sequence[Any]) -> str:
    return f6e3.join_pipe(values)


def read_csv_rows(root: Path, relative_path: Path) -> list[dict[str, str]]:
    return f6e3.read_csv_rows(root, relative_path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return f6e3.read_jsonl(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    f6e3.write_csv(path, rows, columns)


def csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return f6e3.csv_text(rows, columns)


def ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0"
    return fmt(numerator / denominator)


def raw_path_for_model(model: str) -> Path:
    return RAW_PATHS[model]


def verify_frozen_f6e3_inputs(root: Path) -> dict[str, Any]:
    raw_hashes = {
        path: file_sha256_or_missing(root, Path(path))
        for path in sorted(EXPECTED_F6E3_RAW_HASHES)
    }
    mismatches = [
        f"{path}:expected={EXPECTED_F6E3_RAW_HASHES[path]}:observed={observed}"
        for path, observed in raw_hashes.items()
        if observed != EXPECTED_F6E3_RAW_HASHES[path]
    ]
    freeze_path = root / F6E3A_FREEZE
    if not freeze_path.exists():
        mismatches.append(f"{F6E3A_FREEZE.as_posix()}:missing")
        f6e3a_hash = "missing"
        f6e3a_decision = "missing"
    else:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        f6e3a_hash = str(freeze.get("scientific_content_hash", ""))
        f6e3a_decision = str(freeze.get("decision", ""))
        if f6e3a_hash != EXPECTED_F6E3A_SCIENTIFIC_HASH:
            mismatches.append(
                f"{F6E3A_FREEZE.as_posix()}:expected_scientific={EXPECTED_F6E3A_SCIENTIFIC_HASH}:observed={f6e3a_hash}"
            )
    if mismatches:
        raise RuntimeError("f6e3b_frozen_input_mismatch:" + ";".join(mismatches))
    return {
        "f6e3_raw_hashes": raw_hashes,
        "f6e3a_scientific_content_hash": f6e3a_hash,
        "f6e3a_decision": f6e3a_decision,
    }


def candidate_titles_for_f6e3_row(row: Mapping[str, str]) -> list[str]:
    prompt_titles = f6e3a.parse_candidate_titles_from_prompt(row.get("prompt", ""))
    if len(prompt_titles) == int(row.get("candidate_count", "0") or 0):
        return prompt_titles
    serialized = split_pipe(row.get("candidate_item_titles", ""))
    if len(serialized) == int(row.get("candidate_count", "0") or 0):
        return serialized
    raise ValueError(f"f6e3b_candidate_title_recovery_failed:{row.get('request_id', '')}")


def output_adapter_text(candidate_titles: Sequence[str]) -> str:
    candidate_render = "\n".join(f"{index}. {title}" for index, title in enumerate(candidate_titles, start=1))
    return (
        "\n\nAdapted indexed output contract for Phase F6E3B.\n"
        "The semantic recommendation task above is unchanged, but the answer format is changed for this local audit.\n"
        "Use the one-based candidate index map below; it preserves the exact candidate order from the prompt above.\n\n"
        f"Candidate index map:\n{candidate_render}\n\n"
        "Return only valid JSON with exactly one key named \"ranking\".\n"
        "The value of \"ranking\" must be a list of exactly 10 unique integers.\n"
        f"Every integer must be between 1 and {len(candidate_titles)} and must refer to the candidate index map.\n"
        "Do not include item titles, item IDs, Markdown, code fences, explanations, or extra keys.\n"
        "Required schema example: {\"ranking\":[1,2,3,4,5,6,7,8,9,10]}"
    )


def materialize_request_manifest(root: Path) -> list[dict[str, str]]:
    verify_frozen_f6e3_inputs(root)
    f6e3_rows = read_csv_rows(root, F6E3_REQUEST_MANIFEST)
    f6e3.assert_manifest_valid(f6e3_rows)
    rows: list[dict[str, str]] = []
    for source in f6e3_rows:
        candidate_ids = split_pipe(source["candidate_item_ids"])
        candidate_titles = candidate_titles_for_f6e3_row(source)
        candidate_order_hash = sha256_text(stable_json(candidate_ids))
        if candidate_order_hash != source["candidate_order_hash"]:
            raise RuntimeError(f"f6e3b_candidate_order_hash_mismatch:{source['request_id']}")
        if len(candidate_titles) != len(candidate_ids):
            raise RuntimeError(f"f6e3b_candidate_title_count_mismatch:{source['request_id']}")
        adapter = output_adapter_text(candidate_titles)
        prompt = source["prompt"].rstrip() + adapter
        prompt_hash = sha256_text(prompt)
        strategy_label = STRATEGY_LABELS[source["strategy_id"]]
        adapted_strategy_id = f"{strategy_label}_adapted_indexed_output_contract_v1"
        request_id = stable_id(
            "F6E3B-REQ",
            PHASE_ID,
            source["request_id"],
            source["source_request_id"],
            source["dataset"],
            source["task"],
            source["model"],
            source["strategy_id"],
            prompt_hash,
        )
        rows.append(
            {
                "request_id": request_id,
                "f6e3_request_id": source["request_id"],
                "source_request_id": source["source_request_id"],
                "source_request_order": source["source_request_order"],
                "dataset": source["dataset"],
                "task": source["task"],
                "model": source["model"],
                "expected_model_digest": EXPECTED_TAG_DIGESTS[source["model"]],
                "strategy_id": source["strategy_id"],
                "strategy_label": strategy_label,
                "adapted_strategy_id": adapted_strategy_id,
                "strategy_classification": "adapted_indexed_output_contract",
                "scientific_role": "adapted_output_contract_micro_pilot",
                "output_contract_id": "json_ranking_1based_top10_strict_v1",
                "candidate_count": source["candidate_count"],
                "candidate_order_hash": candidate_order_hash,
                "candidate_item_ids": join_pipe(candidate_ids),
                "candidate_item_titles": join_pipe(candidate_titles),
                "positive_item_id": source["positive_item_id"],
                "positive_candidate_position": source["positive_candidate_position"],
                "core_prompt_hash": source["prompt_hash"],
                "candidate_serialization_hash": sha256_text(stable_json(candidate_titles)),
                "output_adapter_hash": sha256_text(adapter),
                "prompt_hash": prompt_hash,
                "prompt": prompt,
                "expected_output_contract": '{"ranking":[1,2,3,4,5,6,7,8,9,10]}',
                "runtime_parameters_json": source["runtime_parameters_json"],
                "runtime_parameter_classification": "adapted_runtime",
                "pilot_stratum": source["pilot_stratum"],
                "source_f6e3_manifest_path": F6E3_REQUEST_MANIFEST.as_posix(),
                "source_f6e3_raw_path": f6e3.raw_path_for_model(source["model"]).as_posix(),
                "status": "pending",
                "notes": "F6E3B adapted indexed output contract; same frozen F6E3 logical instance.",
            }
        )
    assert_manifest_valid(rows)
    return rows


def assert_manifest_valid(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != 32:
        raise ValueError(f"f6e3b_request_count_invalid:{len(rows)}")
    request_ids = [row["request_id"] for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("f6e3b_duplicate_request_id")
    source_ids = [row["f6e3_request_id"] for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("f6e3b_duplicate_f6e3_request_id")
    for row in rows:
        if row["model"] not in MODEL_ORDER:
            raise ValueError(f"f6e3b_unallowed_model:{row['model']}")
        if row["strategy_id"] not in STRATEGY_ORDER:
            raise ValueError(f"f6e3b_unallowed_strategy:{row['strategy_id']}")
        if row["strategy_classification"] != "adapted_indexed_output_contract":
            raise ValueError(f"f6e3b_unallowed_classification:{row['request_id']}")
        candidate_ids = split_pipe(row["candidate_item_ids"])
        if int(row["candidate_count"]) != 20 or len(candidate_ids) != 20:
            raise ValueError(f"f6e3b_candidate_count_invalid:{row['request_id']}")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"f6e3b_candidate_duplicate_invalid:{row['request_id']}")
        if row["candidate_order_hash"] != sha256_text(stable_json(candidate_ids)):
            raise RuntimeError(f"f6e3b_candidate_order_hash_mismatch:{row['request_id']}")
        if '"ranking"' not in row["prompt"]:
            raise ValueError(f"f6e3b_missing_ranking_contract:{row['request_id']}")


def gate_row(check_name: str, expected: str, observed: str, status: str, severity: str, source: str, notes: str) -> dict[str, str]:
    return {
        "gate_id": stable_id("F6E3B-GATE", check_name),
        "check_name": check_name,
        "expected": expected,
        "observed": observed,
        "status": status,
        "severity": severity,
        "source": source,
        "notes": notes,
    }


def inspect_ollama_gate() -> list[dict[str, str]]:
    rows = []
    for row in f6e3.inspect_ollama_gate():
        rows.append(gate_row(row["check_name"], row["expected"], row["observed"], row["status"], row["severity"], row["source"], row["notes"]))
    return rows


def gate_decision(gate_rows: Sequence[Mapping[str, str]]) -> str:
    if not gate_rows:
        return "not_queried"
    f6e3_decision = f6e3.gate_decision(gate_rows)
    return {
        "pass": "pass",
        "f6e3_blocked_ollama_unreachable": "f6e3b_blocked_ollama_unreachable",
        "f6e3_blocked_model_digest_mismatch": "f6e3b_blocked_model_digest_mismatch",
        "f6e3_blocked_by_runtime_failure": "f6e3b_blocked_by_runtime_failure",
    }.get(f6e3_decision, "f6e3b_blocked_by_runtime_failure")


def runtime_parameters(row: Mapping[str, str]) -> dict[str, Any]:
    return json.loads(row["runtime_parameters_json"])


def all_raw_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in RAW_PATHS.values():
        records.extend(read_jsonl(root / path))
    return records


def final_raw_by_request(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in all_raw_records(root):
        if f6e3.parse_bool(record.get("is_final_attempt")):
            request_id = str(record.get("request_id", ""))
            if request_id in records:
                raise RuntimeError(f"f6e3b_duplicate_raw_record:{request_id}")
            records[request_id] = record
    return records


def validate_existing_raw_records(root: Path, request_rows: Sequence[Mapping[str, str]]) -> None:
    manifest_by_id = {row["request_id"]: row for row in request_rows}
    for record in all_raw_records(root):
        request_id = str(record.get("request_id", ""))
        if request_id not in manifest_by_id:
            raise RuntimeError(f"f6e3b_raw_request_not_in_manifest:{request_id}")
        manifest = manifest_by_id[request_id]
        if str(record.get("phase", "")) != PHASE_ID:
            raise RuntimeError(f"f6e3b_raw_phase_mismatch:{request_id}")
        if str(record.get("prompt_hash", "")) != manifest["prompt_hash"]:
            raise RuntimeError(f"f6e3b_raw_prompt_hash_mismatch:{request_id}")
        if str(record.get("model", "")) != manifest["model"]:
            raise RuntimeError(f"f6e3b_raw_model_mismatch:{request_id}")


def execute_pilot(
    root: Path,
    request_rows: Sequence[Mapping[str, str]],
    gate_rows: Sequence[Mapping[str, str]],
    *,
    allow_ollama_generate: bool,
    max_new_requests: int,
    request_executor: Callable[[Mapping[str, Any], int], f6e3.RequestResult] = f6e3.execute_ollama_request,
) -> int:
    if not allow_ollama_generate:
        raise PermissionError("f6e3b_requires_allow_ollama_generate")
    if max_new_requests > 32:
        raise ValueError("f6e3b_max_new_requests_exceeds_micro_pilot_cap")
    decision = gate_decision(gate_rows)
    if decision != "pass":
        raise RuntimeError(decision)
    verify_frozen_f6e3_inputs(root)
    assert_manifest_valid(request_rows)
    validate_existing_raw_records(root, request_rows)
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
        f6e3.append_jsonl_fsync(root / raw_path_for_model(row["model"]), make_raw_record(row, result))
        sent += 1
        if not result.success or not result.response_text.strip():
            break
    return sent


def make_raw_record(row: Mapping[str, str], result: f6e3.RequestResult) -> dict[str, Any]:
    raw = result.raw_backend_response
    response_text = result.response_text
    return {
        "phase": PHASE_ID,
        "request_id": row["request_id"],
        "f6e3_request_id": row["f6e3_request_id"],
        "source_request_id": row["source_request_id"],
        "attempt": 1,
        "is_final_attempt": True,
        "model": row["model"],
        "expected_model_digest": row["expected_model_digest"],
        "strategy_id": row["strategy_id"],
        "adapted_strategy_id": row["adapted_strategy_id"],
        "core_prompt_hash": row["core_prompt_hash"],
        "prompt_hash": row["prompt_hash"],
        "candidate_order_hash": row["candidate_order_hash"],
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
        "parser_version": "json_ranking_1based_top10_strict_v1",
    }


def parse_adapted_indexed_response(response_text: str, *, candidate_count: int = 20, expected_count: int = 10) -> dict[str, Any]:
    text = str(response_text or "").strip()
    if not text:
        return parse_status(False, False, False, (), "empty_response", 0, 0, False)
    extra_text = not (text.startswith("{") and text.endswith("}")) or "```" in text
    if extra_text:
        return parse_status(False, False, False, (), "extra_text_or_code_fence", 0, 0, True)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return parse_status(False, False, False, (), "json_parse_failed", 0, 0, extra_text)
    if not isinstance(parsed, dict):
        return parse_status(True, False, False, (), "invalid_schema", 0, 0, extra_text)
    keys = set(parsed)
    if keys != {"ranking"}:
        status = "missing_ranking_key" if "ranking" not in keys else "extra_keys"
        return parse_status(True, False, False, (), status, 0, 0, extra_text)
    values = parsed.get("ranking")
    if not isinstance(values, list) or not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return parse_status(True, False, False, (), "invalid_schema", 0, 0, extra_text)
    indices = tuple(int(value) for value in values)
    duplicate_count = len(indices) - len(set(indices))
    unknown_count = sum(1 for value in indices if value < 1 or value > candidate_count)
    if len(indices) != expected_count:
        return parse_status(True, True, False, indices, "invalid_length", duplicate_count, unknown_count, extra_text)
    if duplicate_count:
        return parse_status(True, True, False, indices, "duplicate_indices", duplicate_count, unknown_count, extra_text)
    if unknown_count:
        return parse_status(True, True, False, indices, "out_of_range_indices", duplicate_count, unknown_count, extra_text)
    return parse_status(True, True, True, indices, "parsed_valid", 0, 0, extra_text)


def parse_status(
    json_parse_valid: bool,
    schema_valid: bool,
    strict_valid: bool,
    indices: Sequence[int],
    status: str,
    duplicate_count: int,
    unknown_count: int,
    extra_text: bool,
) -> dict[str, Any]:
    return {
        "json_parse_valid": json_parse_valid,
        "schema_valid": schema_valid,
        "strict_valid": strict_valid,
        "output_indices": tuple(indices),
        "output_length": len(indices),
        "duplicate_count": duplicate_count,
        "unknown_index_count": unknown_count,
        "parse_status": status,
        "extra_text_detected": extra_text,
    }


def map_indices(indices: Sequence[int], candidate_ids: Sequence[str], candidate_titles: Sequence[str]) -> tuple[list[str], list[str]]:
    mapped_ids: list[str] = []
    mapped_titles: list[str] = []
    for index in indices:
        if 1 <= int(index) <= len(candidate_ids):
            mapped_ids.append(str(candidate_ids[int(index) - 1]))
            mapped_titles.append(str(candidate_titles[int(index) - 1]))
    return mapped_ids, mapped_titles


def analyze_instances(root: Path, request_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw_records = final_raw_by_request(root)
    instance_rows: list[dict[str, str]] = []
    parser_rows: list[dict[str, str]] = []
    for row in request_rows:
        raw = raw_records.get(row["request_id"])
        candidate_ids = split_pipe(row["candidate_item_ids"])
        candidate_titles = candidate_titles_for_f6e3_row(row) if row.get("prompt") else split_pipe(row["candidate_item_titles"])
        if raw is None:
            parse = parse_status(False, False, False, (), "missing_raw_response", 0, 0, False)
            transport = "missing"
            completion = "missing_raw_response"
            response_text = ""
            failure = "missing_raw_response"
            duration = prompt_eval = eval_count = ""
            raw_file = raw_path_for_model(row["model"]).as_posix()
        else:
            response_text = str(raw.get("raw_response", ""))
            parse = parse_adapted_indexed_response(response_text, candidate_count=len(candidate_ids), expected_count=10)
            transport = str(raw.get("transport_status", ""))
            completion = "completed" if transport == "success" and response_text.strip() else "failed"
            strict_success = bool(parse["strict_valid"] and transport == "success")
            failure = "" if strict_success else str(raw.get("error_type") or raw.get("error") or parse["parse_status"])
            duration = fmt(float(raw.get("duration_seconds", 0) or 0))
            prompt_eval = str(raw.get("prompt_eval_count", ""))
            eval_count = str(raw.get("eval_count", ""))
            raw_file = raw_path_for_model(row["model"]).as_posix()
        mapped_ids, mapped_titles = map_indices(parse["output_indices"], candidate_ids, candidate_titles)
        strict_valid = bool(parse["strict_valid"] and transport == "success")
        positive_rank = ""
        hr = 0.0
        ndcg = 0.0
        if row["positive_item_id"] in mapped_ids:
            positive_rank = str(mapped_ids.index(row["positive_item_id"]) + 1)
            rank_value = int(positive_rank)
            if rank_value <= 5:
                hr = 1.0
                ndcg = 1.0 / math.log2(rank_value + 1)
        instance_rows.append(
            {
                "request_id": row["request_id"],
                "f6e3_request_id": row["f6e3_request_id"],
                "source_request_id": row["source_request_id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "model": row["model"],
                "strategy_id": row["strategy_id"],
                "adapted_strategy_id": row["adapted_strategy_id"],
                "transport_status": transport,
                "completion_status": completion,
                "json_parse_valid": bool_text(bool(parse["json_parse_valid"])),
                "schema_valid": bool_text(bool(parse["schema_valid"])),
                "strict_valid": bool_text(strict_valid),
                "candidate_valid": bool_text(int(parse["unknown_index_count"]) == 0 and int(parse["output_length"]) == len(mapped_ids)),
                "duplicate_valid": bool_text(int(parse["duplicate_count"]) == 0),
                "output_length": str(parse["output_length"]),
                "output_indices": join_pipe(parse["output_indices"]),
                "parsed_ranking_item_ids": join_pipe(mapped_ids),
                "parsed_ranking_titles": join_pipe(mapped_titles),
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
                "notes": "Adapted-output-contract micro-pilot metric only; not a full-run result.",
            }
        )
        parser_rows.append(
            {
                "request_id": row["request_id"],
                "f6e3_request_id": row["f6e3_request_id"],
                "model": row["model"],
                "strategy_id": row["strategy_id"],
                "dataset": row["dataset"],
                "parse_status": str(parse["parse_status"]),
                "json_parse_valid": bool_text(bool(parse["json_parse_valid"])),
                "schema_valid": bool_text(bool(parse["schema_valid"])),
                "parsed_count": str(parse["output_length"]),
                "duplicate_count": str(parse["duplicate_count"]),
                "unknown_index_count": str(parse["unknown_index_count"]),
                "extra_text_detected": bool_text(bool(parse["extra_text_detected"])),
                "strict_valid": bool_text(strict_valid),
                "raw_excerpt_sha256": sha256_text(response_text[:200]) if response_text else "",
                "notes": "Strict JSON ranking key parser; no fuzzy or title recovery.",
            }
        )
    return instance_rows, parser_rows


def build_cell_summary(instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in instance_rows:
        groups[(row["model"], row["strategy_id"])].append(row)
    rows: list[dict[str, str]] = []
    for (model, strategy), group in sorted(groups.items()):
        attempted = [row for row in group if row["transport_status"] != "missing"]
        completed = [row for row in group if row["transport_status"] == "success"]
        strict = [row for row in group if row["strict_valid"] == "true"]
        json_valid = [row for row in group if row["json_parse_valid"] == "true"]
        schema_valid = [row for row in group if row["schema_valid"] == "true"]
        parse_status = Counter(row["failure_reason"] or "parsed_valid" for row in group)
        all_hr = [float(row["hr_at5"] or 0) for row in group]
        all_ndcg = [float(row["ndcg_at5"] or 0) for row in group]
        strict_hr = [float(row["hr_at5"] or 0) for row in strict]
        strict_ndcg = [float(row["ndcg_at5"] or 0) for row in strict]
        min_pass = len(strict) >= 7
        rows.append(
            {
                "cell_id": stable_id("F6E3B-CELL", model, strategy),
                "model": model,
                "strategy_id": strategy,
                "adapted_strategy_id": f"{STRATEGY_LABELS[strategy]}_adapted_indexed_output_contract_v1",
                "logical_requests": str(len(group)),
                "attempted_requests": str(len(attempted)),
                "completed_requests": str(len(completed)),
                "strict_valid_requests": str(len(strict)),
                "strict_valid_rate": ratio(len(strict), len(group)),
                "json_parse_valid_rate": ratio(len(json_valid), len(group)),
                "schema_valid_rate": ratio(len(schema_valid), len(group)),
                "transport_success_rate": ratio(len(completed), len(group)),
                "parse_status_distribution": stable_json(dict(sorted(parse_status.items()))),
                "hr_at5_all_request": fmt(mean(all_hr) if all_hr else 0.0),
                "ndcg_at5_all_request": fmt(mean(all_ndcg) if all_ndcg else 0.0),
                "hr_at5_strict_valid": fmt(mean(strict_hr)) if strict_hr else "",
                "ndcg_at5_strict_valid": fmt(mean(strict_ndcg)) if strict_ndcg else "",
                "gate_status": "pass" if min_pass else "fail",
                "notes": "Cell gate requires at least 7/8 strict-valid for final pass; at least 6/8 for conditional.",
            }
        )
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    return f6e3.percentile(values, q)


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
                "runtime_id": stable_id("F6E3B-RT", model, strategy, dataset),
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
                "notes": "Runtime summary from F6E3B raw records only.",
            }
        )
    return rows


def build_exact_vs_adapted(root: Path, instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    f6e3_instances = {row["request_id"]: row for row in read_csv_rows(root, F6E3_INSTANCE_RESULTS)}
    f6e3_parser = {row["request_id"]: row for row in read_csv_rows(root, F6E3_PARSER_VALIDITY)}
    rows: list[dict[str, str]] = []
    for row in instance_rows:
        exact = f6e3_instances.get(row["f6e3_request_id"], {})
        exact_parser = f6e3_parser.get(row["f6e3_request_id"], {})
        exact_valid = exact.get("strict_valid", "")
        adapted_valid = row["strict_valid"]
        if exact_valid == adapted_valid:
            delta = "0"
        elif adapted_valid == "true" and exact_valid != "true":
            delta = "+1"
        else:
            delta = "-1"
        rows.append(
            {
                "comparison_id": stable_id("F6E3B-CMP", row["request_id"]),
                "f6e3_request_id": row["f6e3_request_id"],
                "f6e3b_request_id": row["request_id"],
                "dataset": row["dataset"],
                "model": row["model"],
                "strategy_id": row["strategy_id"],
                "f6e3_strict_valid": exact_valid,
                "f6e3_parse_status": exact_parser.get("parse_status", ""),
                "f6e3_duration_seconds": exact.get("duration_seconds", ""),
                "f6e3b_strict_valid": adapted_valid,
                "f6e3b_parse_status": row["failure_reason"] or "parsed_valid",
                "f6e3b_duration_seconds": row["duration_seconds"],
                "strict_valid_delta": delta,
                "metric_comparability": "compliance_only" if exact_valid != "true" or adapted_valid != "true" else "strict_valid_metric_comparable_with_adapted_contract_caveat",
                "notes": "F6E3B is adapted local evidence; it does not promote F6E3 exact-prompt failures to paper-exact replication.",
            }
        )
    return rows


def protected_summary(root: Path) -> dict[str, Any]:
    base = dict(f6e3.protected_summary(root))
    f6e3_raw_hashes = {
        path: file_sha256_or_missing(root, Path(path))
        for path in sorted(EXPECTED_F6E3_RAW_HASHES)
    }
    f6e3_raw_changed = sum(1 for path, digest in f6e3_raw_hashes.items() if digest != EXPECTED_F6E3_RAW_HASHES[path])
    f6e3a_hash = "missing"
    f6e3a_changed = 1
    freeze_path = root / F6E3A_FREEZE
    if freeze_path.exists():
        f6e3a_hash = str(json.loads(freeze_path.read_text(encoding="utf-8")).get("scientific_content_hash", ""))
        f6e3a_changed = int(f6e3a_hash != EXPECTED_F6E3A_SCIENTIFIC_HASH)
    base.update(
        {
            "f6e3_raw_changed": f6e3_raw_changed,
            "f6e3_raw_hashes": f6e3_raw_hashes,
            "f6e3a_changed": f6e3a_changed,
            "f6e3a_scientific_content_hash": f6e3a_hash,
        }
    )
    return base


def build_gate_results(
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
                "gate_id": stable_id("F6E3B-GATECHK", name),
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
    add("one_to_one_f6e3_mapping", "32 unique F6E3 request IDs", str(len({row["f6e3_request_id"] for row in request_rows})), len({row["f6e3_request_id"] for row in request_rows}) == 32, "critical", OUTPUT_TABLES["request_manifest"].as_posix())
    add("only_qwen2_mistral", "0 disallowed", str(sum(1 for row in request_rows if row["model"] not in MODEL_ORDER)), all(row["model"] in MODEL_ORDER for row in request_rows), "critical", OUTPUT_TABLES["request_manifest"].as_posix())
    add("only_base_llmrank", "0 disallowed", str(sum(1 for row in request_rows if row["strategy_id"] not in STRATEGY_ORDER)), all(row["strategy_id"] in STRATEGY_ORDER for row in request_rows), "critical", OUTPUT_TABLES["request_manifest"].as_posix())
    add("adapted_contract_classification", "all adapted_indexed_output_contract", str(sum(1 for row in request_rows if row["strategy_classification"] == "adapted_indexed_output_contract")), all(row["strategy_classification"] == "adapted_indexed_output_contract" for row in request_rows), "critical", OUTPUT_TABLES["request_manifest"].as_posix())
    add("max_new_requests_cap", "<=32", str(max_new_requests), max_new_requests <= 32, "critical", "CLI")
    add("ollama_gate", "pass", gate_decision(gate_rows), gate_decision(gate_rows) == "pass", "critical", OUTPUT_TABLES["ollama_gate"].as_posix())
    protected_changed = int(protected["f2_changed"]) + int(protected["f6d4a_changed"]) + int(protected["f6d4a_raw_changed"]) + int(protected["f6e3_raw_changed"]) + int(protected["f6e3a_changed"])
    add("protected_inputs", "changed=0", str(protected_changed), protected_changed == 0, "critical", "protected input tables and frozen F6E3/F6E3A hashes")
    candidate_mismatch = sum(1 for row in request_rows if row["candidate_order_hash"] != sha256_text(stable_json(split_pipe(row["candidate_item_ids"]))))
    add("candidate_order_invariance", "mismatch=0", str(candidate_mismatch), candidate_mismatch == 0, "critical", OUTPUT_TABLES["request_manifest"].as_posix())
    attempted = sum(1 for row in instance_rows if row["transport_status"] != "missing")
    add("no_extra_requests", "<=32", str(attempted), attempted <= 32, "critical", OUTPUT_TABLES["instance_results"].as_posix())
    strict_count = sum(1 for row in instance_rows if row["strict_valid"] == "true")
    cell_counts = Counter((row["model"], row["strategy_id"]) for row in instance_rows if row["strict_valid"] == "true")
    min_cell = min((cell_counts[(model, strategy)] for model in MODEL_ORDER for strategy in STRATEGY_ORDER), default=0)
    add("global_strict_pass_threshold", ">=31/32", str(strict_count), strict_count >= 31, "high", OUTPUT_TABLES["parser_validity"].as_posix())
    add("cell_strict_pass_threshold", ">=7/8 each", str(min_cell), min_cell >= 7, "high", OUTPUT_TABLES["cell_summary"].as_posix())
    add("global_strict_conditional_threshold", ">=28/32", str(strict_count), strict_count >= 28, "medium", OUTPUT_TABLES["parser_validity"].as_posix())
    add("cell_strict_conditional_threshold", ">=6/8 each", str(min_cell), min_cell >= 6, "medium", OUTPUT_TABLES["cell_summary"].as_posix())
    return rows


def build_missing_or_blocked(gate_rows: Sequence[Mapping[str, str]], instance_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "block_id": "F6E3B-BLOCK-JSON-INDEX-STANDALONE",
            "category": "scope",
            "affected_scope": "JSON/index standalone local ablation",
            "status": "not_executed_selected_scope",
            "severity": "info",
            "evidence": "F6E3B user-selected scope is Base/LLMRank with adapted output contract only.",
            "safe_next_action": "Keep JSON/index standalone as separate local ablation evidence.",
            "notes": "F6E3B does not modify or rerun prior JSON/index artifacts.",
        },
        {
            "block_id": "F6E3B-BLOCK-LLMSREC",
            "category": "scope",
            "affected_scope": "LLMSRec reconstructed strategy",
            "status": "not_executed_selected_scope",
            "severity": "info",
            "evidence": "F6E1 classifies LLMSRec as reconstructed.",
            "safe_next_action": "Design a separate reconstructed-strategy pilot if needed.",
            "notes": "No LLMSRec generation in F6E3B.",
        },
        {
            "block_id": "F6E3B-BLOCK-REMOTE-OR-LARGE-MODELS",
            "category": "scope",
            "affected_scope": "Llama3/GPT/Claude/full run",
            "status": "not_authorized",
            "severity": "high",
            "evidence": "F6E3B scope lock.",
            "safe_next_action": "Do not execute without a separate phase and authorization.",
            "notes": "No model pull, remote API, or full run is part of F6E3B.",
        },
    ]
    gate = gate_decision(gate_rows)
    if gate not in {"pass", "not_queried"}:
        rows.append(
            {
                "block_id": "F6E3B-BLOCK-OLLAMA-GATE",
                "category": "runtime",
                "affected_scope": "all F6E3B generation requests",
                "status": gate,
                "severity": "critical",
                "evidence": OUTPUT_TABLES["ollama_gate"].as_posix(),
                "safe_next_action": "Repair local runtime or model digest mismatch before generation.",
                "notes": "Generation must stop before /api/generate while this block is active.",
            }
        )
    failures = [row for row in instance_rows if row["transport_status"] == "failure"]
    if failures:
        rows.append(
            {
                "block_id": "F6E3B-BLOCK-TRANSPORT",
                "category": "runtime",
                "affected_scope": "pilot execution",
                "status": "f6e3b_blocked_by_runtime_failure",
                "severity": "critical",
                "evidence": OUTPUT_TABLES["instance_results"].as_posix(),
                "safe_next_action": "Inspect first failure; do not retry automatically.",
                "notes": f"transport_failures={len(failures)}",
            }
        )
    attempted = [row for row in instance_rows if row["transport_status"] != "missing"]
    strict_invalid = [row for row in attempted if row["strict_valid"] != "true"]
    if strict_invalid:
        rows.append(
            {
                "block_id": "F6E3B-BLOCK-MODEL-OUTPUTS",
                "category": "parser",
                "affected_scope": "adapted indexed strict-valid gate",
                "status": "f6e3b_adapted_output_contract_pilot_failed_model_outputs",
                "severity": "high",
                "evidence": OUTPUT_TABLES["parser_validity"].as_posix(),
                "safe_next_action": "Revise adapted contract or run a larger diagnostic pilot in a later phase.",
                "notes": f"strict_invalid={len(strict_invalid)}",
            }
        )
    if not attempted:
        rows.append(
            {
                "block_id": "F6E3B-BLOCK-NO-RAW",
                "category": "evidence",
                "affected_scope": "F6E3B raw responses",
                "status": "f6e3b_incomplete_missing_evidence",
                "severity": "medium",
                "evidence": "F6E3B raw files absent or empty.",
                "safe_next_action": "Run execute-pilot only after tests and runtime gate pass.",
                "notes": "Materialize/runtime-gate-only modes are not final pilot completion.",
            }
        )
    return rows


def final_decision(gate_rows: Sequence[Mapping[str, str]], instance_rows: Sequence[Mapping[str, str]], protected: Mapping[str, Any]) -> str:
    protected_changed = int(protected["f2_changed"]) + int(protected["f6d4a_changed"]) + int(protected["f6d4a_raw_changed"]) + int(protected["f6e3_raw_changed"]) + int(protected["f6e3a_changed"])
    if protected_changed:
        return "f6e3b_blocked_by_protected_input_invariance"
    gate = gate_decision(gate_rows)
    if gate not in {"pass", "not_queried"}:
        return gate
    attempted = [row for row in instance_rows if row["transport_status"] != "missing"]
    if not attempted or len(attempted) < 32:
        return "f6e3b_incomplete_missing_evidence"
    if any(row["transport_status"] == "failure" for row in instance_rows):
        return "f6e3b_blocked_by_runtime_failure"
    strict_count = sum(1 for row in instance_rows if row["strict_valid"] == "true")
    cell_counts = Counter((row["model"], row["strategy_id"]) for row in instance_rows if row["strict_valid"] == "true")
    min_cell = min(cell_counts[(model, strategy)] for model in MODEL_ORDER for strategy in STRATEGY_ORDER)
    if strict_count >= 31 and min_cell >= 7:
        return "f6e3b_adapted_output_contract_pilot_complete_gate_passed"
    if strict_count >= 28 and min_cell >= 6:
        return "f6e3b_adapted_output_contract_pilot_complete_conditional"
    return "f6e3b_adapted_output_contract_pilot_failed_model_outputs"


def recommendation_for_decision(decision: str) -> str:
    if decision == "f6e3b_adapted_output_contract_pilot_complete_gate_passed":
        return "proceed_to_adapted_full_run"
    if decision == "f6e3b_adapted_output_contract_pilot_complete_conditional":
        return "run_larger_adapted_pilot"
    if decision == "f6e3b_adapted_output_contract_pilot_failed_model_outputs":
        return "revise_adapted_contract"
    if decision in {"f6e3b_blocked_ollama_unreachable", "f6e3b_blocked_model_digest_mismatch", "f6e3b_blocked_by_runtime_failure"}:
        return "repair_runtime_first"
    return "keep_local_models_blocked"


def table_hashes(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    return {name: sha256_text(csv_text(rows, TABLE_COLUMNS[name])) for name, rows in sorted(tables.items())}


def input_hashes(root: Path) -> dict[str, str]:
    paths = [
        F6E3_REQUEST_MANIFEST,
        F6E3_INSTANCE_RESULTS,
        F6E3_PARSER_VALIDITY,
        F6E3_FREEZE,
        F6E3A_FREEZE,
        F6E1_FREEZE,
        F6E2_FREEZE,
        F6E2_PILOT_DESIGN,
        F6E1_PROMPT_MODULE,
        F6E3_MODULE,
        F6E3A_MODULE,
        f6e3.F2_PROTECTED_VALIDATION,
        f6e3.F6D4A_INPUT_HASHES,
        f6e3.F6D4A_PROTECTED_VERIFICATION,
    ]
    paths.extend(Path(path) for path in EXPECTED_F6E3_RAW_HASHES)
    paths.extend(Path(path) for path in f6e3.F6D4A_RAW_HASHES)
    return {path.as_posix(): file_sha256_or_missing(root, path) for path in sorted(set(paths), key=lambda value: value.as_posix())}


def raw_hashes(root: Path) -> dict[str, str]:
    return {path.as_posix(): file_sha256_or_missing(root, path) for path in RAW_PATHS.values()}


def build_freeze_manifest(
    root: Path,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    decision: str,
    protected: Mapping[str, Any],
) -> dict[str, Any]:
    hashes = table_hashes(tables)
    raw = raw_hashes(root)
    inputs = input_hashes(root)
    content = {
        "phase": PHASE_ID,
        "decision": decision,
        "input_hashes": inputs,
        "raw_hashes": raw,
        "table_hashes": hashes,
        "protected_summary": protected,
        "recommendation": recommendation_for_decision(decision),
    }
    return {
        "phase": PHASE_ID,
        "decision": decision,
        "recommendation": recommendation_for_decision(decision),
        "frozen_timestamp_utc": FROZEN_TIMESTAMP_UTC,
        "input_hashes": inputs,
        "source_f6e3_raw_hashes": {path: inputs.get(path, "missing") for path in sorted(EXPECTED_F6E3_RAW_HASHES)},
        "source_f6e3a_scientific_content_hash": protected["f6e3a_scientific_content_hash"],
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
            "raw_append_only": True,
            "no_retry": True,
        },
        "protected_summary": protected,
    }


def gate_observed_value(rows: Sequence[Mapping[str, str]], check_name: str) -> str:
    for row in rows:
        if row.get("check_name") == check_name:
            return row.get("observed", "")
    return ""


def build_report(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    instance_rows = tables["instance_results"]
    attempted = sum(1 for row in instance_rows if row["transport_status"] != "missing")
    completed = sum(1 for row in instance_rows if row["transport_status"] == "success")
    strict = sum(1 for row in instance_rows if row["strict_valid"] == "true")
    lines = [
        "# Phase F6E3B - Adapted Indexed Output-Contract Micro-Pilot",
        "",
        f"Decision: `{manifest['decision']}`.",
        f"Recommendation: `{manifest['recommendation']}`.",
        "",
        "F6E3B reuses the frozen F6E3 logical instances and changes only the output contract to strict JSON with one-based candidate indices.",
        "",
        "## Scope",
        "",
        "- Included: Qwen2 Base, Mistral Base, Qwen2 LLMRank, Mistral LLMRank.",
        "- Excluded: full run, LLMSRec, JSON/index standalone, Llama3, GPT, Claude, model pull, remote APIs.",
        "",
        "## Execution",
        "",
        f"- Manifest rows: `{len(tables['request_manifest'])}`.",
        f"- Attempted requests: `{attempted}`.",
        f"- Completed requests: `{completed}`.",
        f"- Strict-valid responses: `{strict}`.",
        "- Raw files are F6E3B-specific and append-only.",
        "",
        "## Cell Summary",
        "",
    ]
    for row in tables["cell_summary"]:
        lines.append(
            f"- `{row['model']}` `{row['strategy_id']}`: strict `{row['strict_valid_requests']}/{row['logical_requests']}`, gate `{row['gate_status']}`."
        )
    lines.extend(
        [
            "",
            "## Runtime Gate",
            "",
        ]
    )
    for row in tables["ollama_gate"]:
        lines.append(f"- `{row['check_name']}`: `{row['status']}` ({row['observed'][:160]}).")
    lines.extend(
        [
            "",
            "## Hashes",
            "",
            f"- Request manifest hash: `{manifest['request_manifest_hash']}`.",
            f"- Scientific content hash: `{manifest['scientific_content_hash']}`.",
            f"- Qwen2 F6E3B raw hash: `{manifest['raw_hashes'].get(RAW_PATHS[QWEN2_TAG].as_posix(), 'missing')}`.",
            f"- Mistral F6E3B raw hash: `{manifest['raw_hashes'].get(RAW_PATHS[MISTRAL_TAG].as_posix(), 'missing')}`.",
            "",
            "## Missing Or Blocked",
            "",
        ]
    )
    for row in tables["missing_or_blocked"]:
        lines.append(f"- `{row['block_id']}`: `{row['status']}`.")
    lines.append("")
    return "\n".join(lines)


def analyze_only(
    root: Path,
    gate_rows: Sequence[Mapping[str, str]] | None = None,
    max_new_requests: int = 32,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any], str]:
    request_rows = materialize_request_manifest(root)
    if gate_rows is None:
        gate_rows = read_csv_rows(root, OUTPUT_TABLES["ollama_gate"])
    gate_rows = [dict(row) for row in gate_rows]
    if not gate_rows:
        gate_rows = [gate_row("analysis_only_runtime_gate_not_queried", "not_applicable", "not_queried", "info", "info", "analysis-only", "Analysis-only does not query Ollama.")]
    instance_rows, parser_rows = analyze_instances(root, request_rows)
    protected = protected_summary(root)
    cell_summary = build_cell_summary(instance_rows)
    runtime_summary = build_runtime_summary(instance_rows, request_rows)
    exact_vs_adapted = build_exact_vs_adapted(root, instance_rows)
    gate_results = build_gate_results(request_rows, gate_rows, instance_rows, protected, max_new_requests)
    missing = build_missing_or_blocked(gate_rows, instance_rows)
    decision = final_decision(gate_rows, instance_rows, protected)
    tables = {
        "ollama_gate": gate_rows,
        "request_manifest": request_rows,
        "instance_results": instance_rows,
        "cell_summary": cell_summary,
        "parser_validity": parser_rows,
        "runtime_summary": runtime_summary,
        "exact_vs_adapted": exact_vs_adapted,
        "gate_results": gate_results,
        "missing_or_blocked": missing,
    }
    manifest = build_freeze_manifest(root, tables, decision, protected)
    report = build_report(manifest, tables)
    return tables, manifest, report


def write_all(root: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]], manifest: Mapping[str, Any], report: str) -> dict[str, str]:
    written: dict[str, str] = {}
    for name, path in OUTPUT_TABLES.items():
        write_csv(root / path, tables[name], TABLE_COLUMNS[name])
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


def output_hashes(root: Path) -> dict[str, str]:
    paths = [REPORT_PATH, FREEZE_MANIFEST_PATH, *OUTPUT_TABLES.values()]
    return {path.as_posix(): sha256_file(root / path) for path in paths if (root / path).exists()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase F6E3B adapted indexed output-contract micro-pilot.")
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

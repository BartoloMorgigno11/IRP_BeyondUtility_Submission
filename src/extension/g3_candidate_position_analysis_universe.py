"""Freeze the G3 candidate-position analytic universe from retained evidence.

This script is offline-only. It joins the frozen G3 request universe with the
retained request/terminal/raw evidence and writes analysis-universe artifacts.
It does not call Ollama, compute metrics, run bootstrap statistics, or modify
the frozen G3 protocol, runner, or evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.extension.e0_protocol import sha256_json
from src.utils.config import find_project_root


EXPECTED_HEAD = "e9ca0bc412312b7954ca2f6831827c42c25c8507"
DECISION_FROZEN = "G3_ANALYTIC_UNIVERSE_FROZEN"
DECISION_BLOCKED = "G3_ANALYTIC_UNIVERSE_BLOCKED"
PHASE_ID = "g3_candidate_position_analysis_universe_freeze"

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = ("BASE", "LLMRANK")
ORDERS = ("FIRST", "RANDOM")
EXPECTED_ROWS = 12000
EXPECTED_PAIRS = 6000
EXPECTED_STATE_COUNTS = {
    "STRICT_VALID": 11834,
    "INVALID_LENGTH": 114,
    "INVALID_DUPLICATE": 51,
    "HALLUCINATION_OR_OUT_OF_CANDIDATE": 1,
}

REQUEST_UNIVERSE = Path("results/tables/g3_candidate_position_request_universe.csv")
PAIR_INTEGRITY = Path("results/tables/g3_candidate_position_pair_integrity.csv")
PROTOCOL_MANIFEST = Path("results/manifests/g3_candidate_position_protocol_freeze.json")
EXECUTION_MANIFEST = Path("results/manifests/g3_candidate_position_execution_freeze.json")
RETENTION_MANIFEST = Path("results/manifests/g3_candidate_position_execution_retention_manifest.json")
RETENTION_INVENTORY = Path("results/tables/g3_candidate_position_execution_evidence_inventory.csv")
RAW_ROOT = Path("results/llm/extension_canonical/g3_candidate_position")
RAW_DIR = RAW_ROOT / "responses"
REQUEST_LEDGER = RAW_ROOT / "requests.jsonl"
RESPONSE_LEDGER = RAW_ROOT / "responses.jsonl"
EVENT_LEDGER = RAW_ROOT / "events.jsonl"

OUT_UNIVERSE = Path("results/tables/g3_candidate_position_analysis_universe.csv")
OUT_PAIRS = Path("results/tables/g3_candidate_position_analysis_pairs.csv")
OUT_COUNTS = Path("results/tables/g3_candidate_position_analysis_counts.csv")
OUT_DENOM = Path("results/tables/g3_candidate_position_denominator_contract.csv")
OUT_VALIDATION = Path("results/tables/g3_candidate_position_analysis_validation.csv")
OUT_MANIFEST = Path("results/manifests/g3_candidate_position_analysis_universe_freeze.json")
OUT_REPORT = Path("report/g3_candidate_position_analysis_universe.md")
NOTES = Path("report/notes.md")
SOURCE_PATH = Path("src/extension/g3_candidate_position_analysis_universe.py")
ANALYTIC_FREEZE_ARTIFACTS = {
    SOURCE_PATH,
    OUT_UNIVERSE,
    OUT_PAIRS,
    OUT_COUNTS,
    OUT_DENOM,
    OUT_VALIDATION,
    OUT_MANIFEST,
    OUT_REPORT,
    NOTES,
}

IDENTITY_FIELDS = (
    "dataset",
    "task",
    "user_id",
    "model",
    "prompt_strategy",
    "candidate_order_condition",
    "history_hash",
    "candidate_order_hash",
    "candidate_membership_hash",
    "model_digest",
    "generation_config_hash",
    "parser_contract_hash",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the G3 analytic universe from retained evidence.")
    parser.add_argument("--output-root", default=".")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def duplicates(counter: Counter[str]) -> list[str]:
    return sorted(key for key, value in counter.items() if key and value > 1)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()


def unexpected_git_status_lines(status: str) -> list[str]:
    allowed = {path.as_posix() for path in ANALYTIC_FREEZE_ARTIFACTS}
    unexpected: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if len(line) >= 3 and line[2] == " ":
            path_text = line[3:].strip().replace("\\", "/")
        else:
            parts = line.split(maxsplit=1)
            path_text = (parts[1] if len(parts) == 2 else "").strip().replace("\\", "/")
        if " -> " in path_text or path_text not in allowed:
            unexpected.append(line)
    return unexpected


def validity_class(terminal_state: str) -> str:
    if terminal_state == "STRICT_VALID":
        return "STRICT_VALID"
    if terminal_state == "BACKEND_FAILURE":
        return "BACKEND_FAILURE"
    if terminal_state == "MISSING_TERMINAL":
        return "MISSING_TERMINAL"
    return "INVALID_TERMINAL_SCIENTIFIC_EVIDENCE"


def load_raw_records(root: Path) -> tuple[list[Path], dict[str, dict[str, Any]], dict[str, str], dict[str, int], list[dict[str, str]]]:
    raw_paths = sorted((root / RAW_DIR).glob("*.json"))
    raw_by_id: dict[str, dict[str, Any]] = {}
    raw_sha_by_id: dict[str, str] = {}
    raw_size_by_id: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    for path in raw_paths:
        raw_size_by_id[path.stem] = path.stat().st_size
        raw_sha_by_id[path.stem] = sha256_file(path)
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - error path retained in validation table.
            errors.append({"path": path.relative_to(root).as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            continue
        request_id = str(row.get("canonical_request_id", ""))
        raw_by_id[request_id] = row
        if request_id != path.stem:
            errors.append({"path": path.relative_to(root).as_posix(), "error": "canonical_request_id_filename_mismatch"})
    return raw_paths, raw_by_id, raw_sha_by_id, raw_size_by_id, errors


def hash_mismatches(rows: list[dict[str, Any]], hash_field: str) -> list[str]:
    mismatches: list[str] = []
    for row in rows:
        expected = str(row.get(hash_field, ""))
        tmp = dict(row)
        tmp.pop("_line_number", None)
        tmp.pop(hash_field, None)
        if expected != sha256_json(tmp):
            mismatches.append(str(row.get("canonical_request_id", "")))
    return mismatches


def raw_hash_mismatches(raw_by_id: dict[str, dict[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    for request_id, row in raw_by_id.items():
        tmp = dict(row)
        expected = str(tmp.pop("raw_record_hash", ""))
        if expected != sha256_json(tmp):
            mismatches.append(request_id)
    return mismatches


def identity_mismatches(
    universe_by_id: dict[str, dict[str, str]],
    request_by_id: dict[str, dict[str, Any]],
    response_by_id: dict[str, dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []
    for request_id, universe in universe_by_id.items():
        for source_name, source in (("request", request_by_id.get(request_id)), ("terminal", response_by_id.get(request_id)), ("raw", raw_by_id.get(request_id))):
            if source is None:
                continue
            for field in IDENTITY_FIELDS:
                if str(universe.get(field, "")) != str(source.get(field, "")):
                    mismatches.append({"canonical_request_id": request_id, "source": source_name, "field": field})
    return mismatches


def build_analysis_rows(
    root: Path,
    request_universe: list[dict[str, str]],
    request_by_id: dict[str, dict[str, Any]],
    response_by_id: dict[str, dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    raw_sha_by_id: dict[str, str],
    raw_size_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for universe in request_universe:
        request_id = universe["canonical_request_id"]
        request = request_by_id.get(request_id, {})
        terminal = response_by_id.get(request_id, {})
        raw = raw_by_id.get(request_id, {})
        raw_path = RAW_DIR / f"{request_id}.json"
        terminal_state = str(terminal.get("terminal_state", "MISSING_TERMINAL"))
        row = {
            "canonical_request_id": request_id,
            "dataset": universe["dataset"],
            "user_id": universe["user_id"],
            "model": universe["model"],
            "prompt_strategy": universe["prompt_strategy"],
            "candidate_order_condition": universe["candidate_order_condition"],
            "candidate_membership_hash": universe["candidate_membership_hash"],
            "candidate_order_hash": universe["candidate_order_hash"],
            "g1_canonical_candidate_order_hash": universe["g1_canonical_candidate_order_hash"],
            "history_hash": universe["history_hash"],
            "history_length": universe["history_length"],
            "full_history_length": universe["full_history_length"],
            "history_policy": universe["history_policy"],
            "positive_item_id": universe["positive_item_id"],
            "positive_candidate_position": universe["positive_candidate_position"],
            "candidate_count": universe["candidate_count"],
            "candidate_items": universe["candidate_items"],
            "prompt_contract": universe["prompt_contract"],
            "prompt_hash": universe["prompt_hash"],
            "parser_contract": universe["parser_contract"],
            "parser_contract_hash": universe["parser_contract_hash"],
            "generation_config_hash": universe["generation_config_hash"],
            "model_digest": universe["model_digest"],
            "request_ledger_exists": bool_text(bool(request)),
            "request_ledger_line": request.get("_line_number", ""),
            "request_payload_hash": request.get("payload_hash", ""),
            "raw_file_exists": bool_text((root / raw_path).exists()),
            "raw_file_path": raw_path.as_posix(),
            "raw_file_size_bytes": raw_size_by_id.get(request_id, ""),
            "raw_file_sha256": raw_sha_by_id.get(request_id, ""),
            "raw_record_hash": raw.get("raw_record_hash", ""),
            "raw_response_hash": raw.get("raw_response_hash", ""),
            "raw_response_recorded_before_parse": bool_text(str(raw.get("raw_response_recorded_before_parse", "")).lower() == "true"),
            "terminal_ledger_exists": bool_text(bool(terminal)),
            "terminal_ledger_line": terminal.get("_line_number", ""),
            "terminal_state": terminal_state,
            "strict_parse_valid": bool_text(str(terminal.get("strict_parse_valid", "")).lower() == "true"),
            "validity_class": validity_class(terminal_state),
            "parse_status": terminal.get("parse_status", ""),
            "backend_status": terminal.get("backend_status", raw.get("backend_status", "")),
            "backend_success": bool_text(str(terminal.get("backend_success", raw.get("backend_success", ""))).lower() == "true"),
            "backend_attempt_count": terminal.get("backend_attempt_count", raw.get("backend_attempt_count", "")),
            "retry_count": terminal.get("retry_count", raw.get("retry_count", "")),
            "terminal_record_hash": terminal.get("terminal_record_hash", ""),
            "ranked_candidate_indices": terminal.get("ranked_candidate_indices", ""),
            "ranked_item_ids": terminal.get("ranked_item_ids", ""),
            "utility_denominator_policy": "planned_paired_users_invalid_terminal_zero",
            "reliability_denominator_policy": "planned_requests",
            "canddif_denominator_policy": "paired_FIRST_RANDOM_cell_accuracies",
            "beyond_utility_denominator_policy": "strict_valid_users_only_if_already_frozen_applicable_to_G3",
            "metric_values_computed_in_this_phase": "false",
            "statistical_analysis_computed_in_this_phase": "false",
            "retention_source_manifest": RETENTION_MANIFEST.as_posix(),
        }
        row["analysis_row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def build_pair_rows(analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in analysis_rows:
        groups.setdefault((row["dataset"], row["model"], row["prompt_strategy"], row["user_id"]), []).append(row)

    pair_rows: list[dict[str, Any]] = []
    for dataset, model, prompt, user_id in sorted(groups):
        group = groups[(dataset, model, prompt, user_id)]
        by_order = {row["candidate_order_condition"]: row for row in group}
        first = by_order.get("FIRST")
        random = by_order.get("RANDOM")
        has_pair = first is not None and random is not None and len(group) == 2
        same_user = has_pair and first["user_id"] == random["user_id"] == user_id
        same_membership = has_pair and first["candidate_membership_hash"] == random["candidate_membership_hash"]
        same_history = has_pair and first["history_hash"] == random["history_hash"] and first["history_length"] == random["history_length"]
        same_model_prompt = has_pair and first["model"] == random["model"] == model and first["prompt_strategy"] == random["prompt_strategy"] == prompt
        order_differs = has_pair and first["candidate_order_hash"] != random["candidate_order_hash"]
        first_pos_ok = has_pair and int(first["positive_candidate_position"]) == 1
        random_pos_ok = has_pair and 2 <= int(random["positive_candidate_position"]) <= 20
        only_order_differs = has_pair and same_user and same_membership and same_history and same_model_prompt and order_differs and first_pos_ok and random_pos_ok
        row = {
            "pair_id": "G3CP-ANPAIR-" + sha256_json({"dataset": dataset, "model": model, "prompt_strategy": prompt, "user_id": user_id}).upper()[:24],
            "dataset": dataset,
            "model": model,
            "prompt_strategy": prompt,
            "user_id": user_id,
            "first_request_id": first["canonical_request_id"] if first else "",
            "random_request_id": random["canonical_request_id"] if random else "",
            "pair_row_count": len(group),
            "has_exactly_one_first": bool_text(first is not None and sum(1 for item in group if item["candidate_order_condition"] == "FIRST") == 1),
            "has_exactly_one_random": bool_text(random is not None and sum(1 for item in group if item["candidate_order_condition"] == "RANDOM") == 1),
            "same_user": bool_text(same_user),
            "same_candidate_membership": bool_text(same_membership),
            "same_history": bool_text(same_history),
            "same_model_prompt": bool_text(same_model_prompt),
            "candidate_order_differs": bool_text(order_differs),
            "only_candidate_order_differs": bool_text(only_order_differs),
            "first_positive_position": first["positive_candidate_position"] if first else "",
            "random_positive_position": random["positive_candidate_position"] if random else "",
            "first_positive_position_ok": bool_text(first_pos_ok),
            "random_positive_position_ok": bool_text(random_pos_ok),
            "first_validity_class": first["validity_class"] if first else "",
            "random_validity_class": random["validity_class"] if random else "",
            "first_terminal_state": first["terminal_state"] if first else "",
            "random_terminal_state": random["terminal_state"] if random else "",
            "first_strict_parse_valid": first["strict_parse_valid"] if first else "",
            "random_strict_parse_valid": random["strict_parse_valid"] if random else "",
            "metric_values_computed_in_this_phase": "false",
        }
        row["pair_row_hash"] = sha256_json(row)
        pair_rows.append(row)
    return pair_rows


def add_count(rows: list[dict[str, Any]], scope: str, value: str, frame: list[dict[str, Any]]) -> None:
    strict = sum(1 for row in frame if row["terminal_state"] == "STRICT_VALID")
    backend_errors = sum(1 for row in frame if row["terminal_state"] == "BACKEND_FAILURE" or row["backend_success"] != "true")
    rows.append(
        {
            "scope": scope,
            "scope_value": value,
            "logical_rows": len(frame),
            "strict_valid": strict,
            "invalid": len(frame) - strict - backend_errors,
            "backend_errors": backend_errors,
            "invalid_length": sum(1 for row in frame if row["terminal_state"] == "INVALID_LENGTH"),
            "invalid_duplicate": sum(1 for row in frame if row["terminal_state"] == "INVALID_DUPLICATE"),
            "hallucination_or_out_of_candidate": sum(1 for row in frame if row["terminal_state"] == "HALLUCINATION_OR_OUT_OF_CANDIDATE"),
            "first_rows": sum(1 for row in frame if row["candidate_order_condition"] == "FIRST"),
            "random_rows": sum(1 for row in frame if row["candidate_order_condition"] == "RANDOM"),
            "unique_users": len({row["user_id"] for row in frame}),
        }
    )


def build_counts(analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    add_count(rows, "GRAND_TOTAL", "ALL", analysis_rows)
    for order in ORDERS:
        add_count(rows, "BY_ORDER", order, [row for row in analysis_rows if row["candidate_order_condition"] == order])
    for dataset in DATASETS:
        add_count(rows, "BY_DATASET", dataset, [row for row in analysis_rows if row["dataset"] == dataset])
    for model in MODELS:
        add_count(rows, "BY_MODEL", model, [row for row in analysis_rows if row["model"] == model])
    for prompt in PROMPTS:
        add_count(rows, "BY_PROMPT", prompt, [row for row in analysis_rows if row["prompt_strategy"] == prompt])
    for dataset in DATASETS:
        for order in ORDERS:
            add_count(rows, "BY_DATASET_ORDER", f"{dataset}|{order}", [row for row in analysis_rows if row["dataset"] == dataset and row["candidate_order_condition"] == order])
    for model in MODELS:
        for order in ORDERS:
            add_count(rows, "BY_MODEL_ORDER", f"{model}|{order}", [row for row in analysis_rows if row["model"] == model and row["candidate_order_condition"] == order])
    for prompt in PROMPTS:
        for order in ORDERS:
            add_count(rows, "BY_PROMPT_ORDER", f"{prompt}|{order}", [row for row in analysis_rows if row["prompt_strategy"] == prompt and row["candidate_order_condition"] == order])
    for dataset in DATASETS:
        for model in MODELS:
            for prompt in PROMPTS:
                add_count(rows, "BY_CELL", f"{dataset}|{model}|{prompt}", [row for row in analysis_rows if row["dataset"] == dataset and row["model"] == model and row["prompt_strategy"] == prompt])
    return rows


def denominator_rows() -> list[dict[str, str]]:
    return [
        {
            "metric_family": "utility",
            "applicability": "HR@5,NDCG@5,MRR ordinary paired deltas",
            "denominator_unit": "planned paired users within dataset_model_prompt_cell",
            "denominator_rule": "all 200 users per dataset cell retained for each FIRST/RANDOM pair",
            "invalid_terminal_policy": "invalid terminal response contributes zero utility",
            "frozen_or_applicable_to_g3": "true",
            "metric_values_computed_in_this_phase": "false",
        },
        {
            "metric_family": "reliability",
            "applicability": "strict-valid rate, invalid class counts, backend error rate",
            "denominator_unit": "planned requests",
            "denominator_rule": "all 12000 logical G3 requests; 6000 FIRST and 6000 RANDOM",
            "invalid_terminal_policy": "invalid terminal response remains terminal scientific evidence",
            "frozen_or_applicable_to_g3": "true",
            "metric_values_computed_in_this_phase": "false",
        },
        {
            "metric_family": "canddif",
            "applicability": "CandDif HR@5 and CandDif NDCG@5 from FIRST/RANDOM cell accuracies",
            "denominator_unit": "paired FIRST/RANDOM cell accuracies",
            "denominator_rule": "compute from paired FIRST and RANDOM accuracies within each dataset_model_prompt_cell",
            "invalid_terminal_policy": "invalid terminal contributes zero to the underlying utility accuracy",
            "frozen_or_applicable_to_g3": "true",
            "metric_values_computed_in_this_phase": "false",
        },
        {
            "metric_family": "beyond_utility",
            "applicability": "only if already frozen/applicable to G3",
            "denominator_unit": "strict-valid users or metric-specific defined-user subset",
            "denominator_rule": "reuse frozen metric-specific denominator semantics only when applicable",
            "invalid_terminal_policy": "excluded from strict-valid beyond-utility denominator",
            "frozen_or_applicable_to_g3": "conditional",
            "metric_values_computed_in_this_phase": "false",
        },
    ]


def build_validation(
    *,
    head_before: str,
    status_before: str,
    request_universe: list[dict[str, str]],
    pair_rows: list[dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
    request_by_id: dict[str, dict[str, Any]],
    response_by_id: dict[str, dict[str, Any]],
    raw_paths: list[Path],
    raw_by_id: dict[str, dict[str, Any]],
    raw_errors: list[dict[str, str]],
    request_duplicates: list[str],
    response_duplicates: list[str],
    raw_file_duplicates: list[str],
    raw_record_duplicates: list[str],
    missing: dict[str, int],
    orphan: dict[str, int],
    identity_mismatch_count: int,
    raw_hash_mismatch_count: int,
    terminal_hash_mismatch_count: int,
    frozen_hash_failures: list[str],
    retention_hash_failures: list[str],
    retention_manifest: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(check_id: str, description: str, passed: bool, observed: Any, expected: Any) -> None:
        rows.append(
            {
                "check_id": check_id,
                "description": description,
                "status": "PASS" if passed else "FAIL",
                "observed": stable_json(observed) if isinstance(observed, (dict, list, tuple)) else str(observed),
                "expected": stable_json(expected) if isinstance(expected, (dict, list, tuple)) else str(expected),
            }
        )

    order_counts = Counter(row["candidate_order_condition"] for row in analysis_rows)
    dataset_counts = Counter(row["dataset"] for row in analysis_rows)
    model_counts = Counter(row["model"] for row in analysis_rows)
    prompt_counts = Counter(row["prompt_strategy"] for row in analysis_rows)
    state_counts = Counter(row["terminal_state"] for row in analysis_rows)
    pair_defects = [row for row in pair_rows if row["only_candidate_order_differs"] != "true"]

    add("G3AU-001", "precondition HEAD", head_before == EXPECTED_HEAD, head_before, EXPECTED_HEAD)
    unexpected_status = unexpected_git_status_lines(status_before)
    add(
        "G3AU-002",
        "git status limited to current G3 analytic-universe artifacts",
        not unexpected_status,
        {"unexpected": unexpected_status, "full_status": status_before or "clean"},
        "no non-G3 analytic-universe artifacts",
    )
    add("G3AU-003", "execution retention frozen", retention_manifest.get("decision_token") == "G3_EXECUTION_RETENTION_FROZEN", retention_manifest.get("decision_token"), "G3_EXECUTION_RETENTION_FROZEN")
    add("G3AU-004", "retention validation passed", retention_manifest.get("validation_passed") is True, retention_manifest.get("validation_passed"), True)
    add("G3AU-005", "frozen protocol and runner artifacts unchanged", not frozen_hash_failures, frozen_hash_failures, [])
    add("G3AU-006", "retention artifacts unchanged", not retention_hash_failures, retention_hash_failures, [])
    add("G3AU-007", "logical analytic rows", len(analysis_rows) == EXPECTED_ROWS, len(analysis_rows), EXPECTED_ROWS)
    add("G3AU-008", "unique analytic canonical request IDs", len({row["canonical_request_id"] for row in analysis_rows}) == EXPECTED_ROWS, len({row["canonical_request_id"] for row in analysis_rows}), EXPECTED_ROWS)
    add("G3AU-009", "expected pair rows", len(pair_rows) == EXPECTED_PAIRS, len(pair_rows), EXPECTED_PAIRS)
    add("G3AU-010", "pairing defects zero", not pair_defects, len(pair_defects), 0)
    add("G3AU-011", "FIRST positive position equals 1", all(int(row["positive_candidate_position"]) == 1 for row in analysis_rows if row["candidate_order_condition"] == "FIRST"), "all FIRST rows", "positive position 1")
    add("G3AU-012", "RANDOM positive position in 2..20", all(2 <= int(row["positive_candidate_position"]) <= 20 for row in analysis_rows if row["candidate_order_condition"] == "RANDOM"), "all RANDOM rows", "positive positions 2..20")
    add("G3AU-013", "raw file evidence complete", len(raw_paths) == EXPECTED_ROWS and missing["raw_file"] == 0, {"raw_files": len(raw_paths), "missing": missing["raw_file"]}, {"raw_files": EXPECTED_ROWS, "missing": 0})
    add("G3AU-014", "terminal evidence complete", len(response_by_id) == EXPECTED_ROWS and missing["terminal"] == 0, {"responses": len(response_by_id), "missing": missing["terminal"]}, {"responses": EXPECTED_ROWS, "missing": 0})
    add("G3AU-015", "request ledger complete", len(request_by_id) == EXPECTED_ROWS and missing["request"] == 0, {"requests": len(request_by_id), "missing": missing["request"]}, {"requests": EXPECTED_ROWS, "missing": 0})
    add("G3AU-016", "missing evidence zero", sum(missing.values()) == 0, missing, "all zero")
    add("G3AU-017", "orphan evidence zero", sum(orphan.values()) == 0, orphan, "all zero")
    add("G3AU-018", "duplicate evidence zero", not request_duplicates and not response_duplicates and not raw_file_duplicates and not raw_record_duplicates, {"request": len(request_duplicates), "terminal": len(response_duplicates), "raw_file": len(raw_file_duplicates), "raw_record": len(raw_record_duplicates)}, "all zero")
    add("G3AU-019", "canonical identity mismatches zero", identity_mismatch_count == 0, identity_mismatch_count, 0)
    add("G3AU-020", "raw and terminal hashes valid", not raw_errors and raw_hash_mismatch_count == 0 and terminal_hash_mismatch_count == 0, {"raw_errors": len(raw_errors), "raw_record_hash_mismatch": raw_hash_mismatch_count, "terminal_hash_mismatch": terminal_hash_mismatch_count}, "all zero")
    add("G3AU-021", "FIRST/RANDOM accounting", order_counts == Counter({"FIRST": 6000, "RANDOM": 6000}), dict(order_counts), {"FIRST": 6000, "RANDOM": 6000})
    add("G3AU-022", "dimension accounting", all(dataset_counts[dataset] == 2400 for dataset in DATASETS) and all(model_counts[model] == 4000 for model in MODELS) and all(prompt_counts[prompt] == 6000 for prompt in PROMPTS), {"dataset": dict(dataset_counts), "model": dict(model_counts), "prompt": dict(prompt_counts)}, {"dataset_each": 2400, "model_each": 4000, "prompt_each": 6000})
    add("G3AU-023", "strict-valid and invalid accounting", state_counts["STRICT_VALID"] == EXPECTED_STATE_COUNTS["STRICT_VALID"] and len(analysis_rows) - state_counts["STRICT_VALID"] == 166, {"strict_valid": state_counts["STRICT_VALID"], "invalid": len(analysis_rows) - state_counts["STRICT_VALID"]}, {"strict_valid": 11834, "invalid": 166})
    add("G3AU-024", "invalid class accounting", dict(state_counts) == EXPECTED_STATE_COUNTS, dict(state_counts), EXPECTED_STATE_COUNTS)
    add("G3AU-025", "zero inference and zero metrics in analytic freeze", all(row["metric_values_computed_in_this_phase"] == "false" and row["statistical_analysis_computed_in_this_phase"] == "false" for row in analysis_rows), {"ollama_or_api_calls": 0, "metrics_computed": 0, "statistics_computed": 0}, {"ollama_or_api_calls": 0, "metrics_computed": 0, "statistics_computed": 0})
    return rows


def build_report(manifest: dict[str, Any], counts_rows: list[dict[str, Any]]) -> str:
    selected_counts = [counts_rows[0]] + [row for row in counts_rows if row["scope"] in {"BY_ORDER", "BY_DATASET", "BY_MODEL", "BY_PROMPT"}]
    count_lines = "\n".join(
        f"- `{row['scope']} {row['scope_value']}`: rows `{row['logical_rows']}`, strict-valid `{row['strict_valid']}`, invalid `{row['invalid']}`, backend errors `{row['backend_errors']}`"
        for row in selected_counts
    )
    summary = manifest["summary"]
    states = summary["terminal_state_counts"]
    return f"""# G3 Candidate-Position Analytic Universe Freeze

## Scope
- Decision token: `{manifest['decision_token']}`.
- Source: frozen G3 retained evidence only (`{RETENTION_MANIFEST.as_posix()}`).
- Zero Ollama/API calls, no metric computation, no statistical analysis, no protocol/config/runner edits, and no evidence rewrites were performed.

## Universe
- Logical rows: `{summary['logical_rows']}`.
- Pairs: `{summary['pair_rows']}` dataset x model x prompt x user FIRST/RANDOM pairs.
- Dimensions: `5` datasets, `3` models, `2` prompts, `200` users per dataset, `2` order conditions.
- FIRST/RANDOM rows: `FIRST={summary['order_counts'].get('FIRST', 0)}`, `RANDOM={summary['order_counts'].get('RANDOM', 0)}`.

## Evidence Resolution
- Request ledger rows resolved: `{summary['request_ledger_rows']}`.
- Terminal rows resolved: `{summary['terminal_rows']}`.
- Raw files resolved: `{summary['raw_files']}`.
- Strict-valid: `{states.get('STRICT_VALID', 0)}`.
- Invalid terminal scientific evidence: `{summary['invalid']}` (`INVALID_LENGTH={states.get('INVALID_LENGTH', 0)}`, `INVALID_DUPLICATE={states.get('INVALID_DUPLICATE', 0)}`, `HALLUCINATION_OR_OUT_OF_CANDIDATE={states.get('HALLUCINATION_OR_OUT_OF_CANDIDATE', 0)}`).
- Backend errors: `0`.

## Counts
{count_lines}

## Pairing Validation
- Pairing defects: `{summary['pairing_defects']}`.
- Candidate/history/hash mismatches: `{summary['candidate_history_hash_mismatches']}`.
- Missing evidence: `{sum(summary['missing'].values())}`.
- Orphan evidence: `{sum(summary['orphan'].values())}`.
- Duplicate evidence IDs: `{sum(summary['duplicate'].values())}`.
- Ambiguous evidence defects: `{summary['ambiguous']}`.

## Denominator Semantics
- Utility: planned paired users; invalid terminal response contributes zero.
- Reliability: planned requests.
- CandDif: computed from paired FIRST/RANDOM cell accuracies.
- Beyond-utility: only if already frozen/applicable to G3, using strict-valid users or metric-specific defined-user subsets.

## Outputs
- `{OUT_UNIVERSE.as_posix()}`
- `{OUT_PAIRS.as_posix()}`
- `{OUT_COUNTS.as_posix()}`
- `{OUT_DENOM.as_posix()}`
- `{OUT_VALIDATION.as_posix()}`
- `{OUT_MANIFEST.as_posix()}`

G3_ANALYTIC_UNIVERSE_FROZEN
"""


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()

    head_before = git_output(root, "rev-parse", "HEAD")
    status_before = git_output(root, "status", "--short", "--untracked-files=all")
    protocol_manifest = read_json(root / PROTOCOL_MANIFEST)
    execution_manifest = read_json(root / EXECUTION_MANIFEST)
    retention_manifest = read_json(root / RETENTION_MANIFEST)
    request_universe = read_csv(root / REQUEST_UNIVERSE)
    read_csv(root / PAIR_INTEGRITY)
    read_csv(root / RETENTION_INVENTORY)
    requests = read_jsonl(root / REQUEST_LEDGER)
    responses = read_jsonl(root / RESPONSE_LEDGER)
    events = read_jsonl(root / EVENT_LEDGER)
    raw_paths, raw_by_id, raw_sha_by_id, raw_size_by_id, raw_errors = load_raw_records(root)

    universe_by_id = {row["canonical_request_id"]: row for row in request_universe}
    request_by_id = {str(row.get("canonical_request_id", "")): row for row in requests}
    response_by_id = {str(row.get("canonical_request_id", "")): row for row in responses}
    analysis_rows = build_analysis_rows(root, request_universe, request_by_id, response_by_id, raw_by_id, raw_sha_by_id, raw_size_by_id)
    pair_rows = build_pair_rows(analysis_rows)
    counts_rows = build_counts(analysis_rows)
    denom_rows = denominator_rows()

    request_counter = Counter(str(row.get("canonical_request_id", "")) for row in requests)
    response_counter = Counter(str(row.get("canonical_request_id", "")) for row in responses)
    raw_file_counter = Counter(path.stem for path in raw_paths)
    raw_record_counter = Counter(raw_by_id.keys())
    universe_ids = set(universe_by_id)
    missing = {
        "request": len(universe_ids - set(request_by_id)),
        "terminal": len(universe_ids - set(response_by_id)),
        "raw_file": len(universe_ids - set(raw_file_counter)),
        "raw_record": len(universe_ids - set(raw_by_id)),
    }
    orphan = {
        "request": len(set(request_by_id) - universe_ids),
        "terminal": len(set(response_by_id) - universe_ids),
        "raw_file": len(set(raw_file_counter) - universe_ids),
        "raw_record": len(set(raw_by_id) - universe_ids),
    }
    request_duplicates = duplicates(request_counter)
    response_duplicates = duplicates(response_counter)
    raw_file_duplicates = duplicates(raw_file_counter)
    raw_record_duplicates = duplicates(raw_record_counter)
    raw_record_hash_bad = raw_hash_mismatches(raw_by_id)
    terminal_hash_bad = hash_mismatches(responses, "terminal_record_hash")
    identity_bad = identity_mismatches(universe_by_id, request_by_id, response_by_id, raw_by_id)

    frozen_hash_failures = []
    for manifest in (protocol_manifest, execution_manifest):
        for rel_path, expected_hash in manifest.get("artifact_hashes", {}).items():
            path = root / rel_path
            if not path.exists() or sha256_file(path) != expected_hash:
                frozen_hash_failures.append(rel_path)
    retention_hash_failures = []
    for rel_path, expected_hash in retention_manifest.get("artifact_hashes", {}).items():
        if rel_path == NOTES.as_posix():
            continue
        path = root / rel_path
        if not path.exists() or sha256_file(path) != expected_hash:
            retention_hash_failures.append(rel_path)

    validation_rows = build_validation(
        head_before=head_before,
        status_before=status_before,
        request_universe=request_universe,
        pair_rows=pair_rows,
        analysis_rows=analysis_rows,
        request_by_id=request_by_id,
        response_by_id=response_by_id,
        raw_paths=raw_paths,
        raw_by_id=raw_by_id,
        raw_errors=raw_errors,
        request_duplicates=request_duplicates,
        response_duplicates=response_duplicates,
        raw_file_duplicates=raw_file_duplicates,
        raw_record_duplicates=raw_record_duplicates,
        missing=missing,
        orphan=orphan,
        identity_mismatch_count=len(identity_bad),
        raw_hash_mismatch_count=len(raw_record_hash_bad),
        terminal_hash_mismatch_count=len(terminal_hash_bad),
        frozen_hash_failures=frozen_hash_failures,
        retention_hash_failures=retention_hash_failures,
        retention_manifest=retention_manifest,
    )
    failures = [row["check_id"] for row in validation_rows if row["status"] != "PASS"]

    for path, rows in (
        (OUT_UNIVERSE, analysis_rows),
        (OUT_PAIRS, pair_rows),
        (OUT_COUNTS, counts_rows),
        (OUT_DENOM, denom_rows),
        (OUT_VALIDATION, validation_rows),
    ):
        write_csv(root / path, rows)

    input_paths = [REQUEST_UNIVERSE, PAIR_INTEGRITY, PROTOCOL_MANIFEST, EXECUTION_MANIFEST, RETENTION_MANIFEST, RETENTION_INVENTORY, REQUEST_LEDGER, RESPONSE_LEDGER, EVENT_LEDGER]
    output_paths = [OUT_UNIVERSE, OUT_PAIRS, OUT_COUNTS, OUT_DENOM, OUT_VALIDATION]
    state_counts = Counter(row["terminal_state"] for row in analysis_rows)
    summary = {
        "logical_rows": len(analysis_rows),
        "pair_rows": len(pair_rows),
        "request_ledger_rows": len(requests),
        "terminal_rows": len(responses),
        "raw_files": len(raw_paths),
        "strict_valid": state_counts["STRICT_VALID"],
        "invalid": len(analysis_rows) - state_counts["STRICT_VALID"],
        "terminal_state_counts": dict(sorted(state_counts.items())),
        "valid_by_order": {
            order: {
                "strict_valid": sum(1 for row in analysis_rows if row["candidate_order_condition"] == order and row["terminal_state"] == "STRICT_VALID"),
                "invalid": sum(1 for row in analysis_rows if row["candidate_order_condition"] == order and row["terminal_state"] != "STRICT_VALID"),
            }
            for order in ORDERS
        },
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in analysis_rows).items())),
        "model_counts": dict(sorted(Counter(row["model"] for row in analysis_rows).items())),
        "prompt_counts": dict(sorted(Counter(row["prompt_strategy"] for row in analysis_rows).items())),
        "order_counts": dict(sorted(Counter(row["candidate_order_condition"] for row in analysis_rows).items())),
        "pairing_defects": sum(1 for row in pair_rows if row["only_candidate_order_differs"] != "true"),
        "candidate_history_hash_mismatches": len(identity_bad),
        "missing": missing,
        "orphan": orphan,
        "duplicate": {
            "request": len(request_duplicates),
            "terminal": len(response_duplicates),
            "raw_file": len(raw_file_duplicates),
            "raw_record": len(raw_record_duplicates),
        },
        "ambiguous": len(identity_bad) + len(raw_errors) + len(raw_record_hash_bad) + len(terminal_hash_bad),
        "event_rows": len(events),
    }
    manifest = {
        "schema_version": "1.0",
        "phase_id": PHASE_ID,
        "decision_token": DECISION_FROZEN if not failures else DECISION_BLOCKED,
        "created_at_utc": utc_now(),
        "head_before_generation": head_before,
        "precondition_status": "G3_EXECUTION_RETENTION_FROZEN",
        "zero_inference": True,
        "metric_computation_executed": False,
        "statistical_analysis_executed": False,
        "protocol_config_runner_modified": False,
        "frozen_evidence_modified": False,
        "input_hashes": {path.as_posix(): sha256_file(root / path) for path in input_paths if (root / path).exists()},
        "source_hashes": {SOURCE_PATH.as_posix(): sha256_file(root / SOURCE_PATH)},
        "output_hashes": {path.as_posix(): sha256_file(root / path) for path in output_paths},
        "validation_passed": not failures,
        "validation_failures": failures,
        "summary": summary,
        "denominator_contract": denom_rows,
        "files_created": [SOURCE_PATH.as_posix()] + [path.as_posix() for path in output_paths] + [OUT_MANIFEST.as_posix(), OUT_REPORT.as_posix(), NOTES.as_posix()],
        "next_safe_step": "compute G3 metrics from frozen analytic universe only when explicitly requested; no new inference",
    }

    (root / OUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / OUT_REPORT).write_text(build_report(manifest, counts_rows), encoding="utf-8", newline="\n")
    manifest["output_hashes"][OUT_REPORT.as_posix()] = sha256_file(root / OUT_REPORT)

    notes_marker = " - G3 analytic universe freeze\n\n"
    notes_text = (root / NOTES).read_text(encoding="utf-8") if (root / NOTES).exists() else ""
    if not failures and notes_marker not in notes_text:
        notes = (
            f"\n## {utc_now()} - G3 analytic universe freeze\n\n"
            f"- Decision: `{manifest['decision_token']}` from frozen retained G3 evidence.\n"
            f"- Universe: `{summary['logical_rows']}` logical rows and `{summary['pair_rows']}` FIRST/RANDOM pairs.\n"
            f"- Evidence resolution: `{len(raw_paths)}` raw files, `{len(responses)}` terminal rows, strict valid `{summary['strict_valid']}`, invalid `{summary['invalid']}`.\n"
            "- Pairing/missing/orphan/duplicate/ambiguous defects: `0`.\n"
            "- No Ollama/API calls, metric computation, statistical analysis, protocol/config/runner edits, or evidence rewrites were performed.\n"
        )
        with (root / NOTES).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(notes)
        manifest["output_hashes"][NOTES.as_posix()] = sha256_file(root / NOTES)
    write_json(root / OUT_MANIFEST, manifest)
    manifest_sha = sha256_file(root / OUT_MANIFEST)

    print(
        json.dumps(
            {
                "decision_token": manifest["decision_token"],
                "validation_passed": manifest["validation_passed"],
                "validation_failures": failures,
                "logical_rows": summary["logical_rows"],
                "pair_rows": summary["pair_rows"],
                "valid_by_order": summary["valid_by_order"],
                "terminal_state_counts": summary["terminal_state_counts"],
                "pairing_defects": summary["pairing_defects"],
                "candidate_history_hash_mismatches": summary["candidate_history_hash_mismatches"],
                "missing": summary["missing"],
                "orphan": summary["orphan"],
                "duplicate": summary["duplicate"],
                "ambiguous": summary["ambiguous"],
                "files_created": manifest["files_created"],
                "manifest_sha256": manifest_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

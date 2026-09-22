"""Freeze the G4 F1BZQ reranking analytic universe.

This phase consumes frozen retained G4 evidence only. It does not call Ollama,
does not compute metric values or statistics, and does not alter the frozen
G4 protocol, runner, F1BZQ pools, or raw/terminal evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.extension.e0_protocol import bool_text, sha256_json, stable_json
from src.utils.config import find_project_root


EXPECTED_HEAD = "4b18f0a998c694a961eee563f8c5cf18e2b81485"
PHASE_ID = "g4_f1bzq_analytic_universe_freeze"
DECISION_FROZEN = "G4_ANALYTIC_UNIVERSE_FROZEN"
DECISION_BLOCKED = "G4_ANALYTIC_UNIVERSE_BLOCKED"

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = ("BASE", "LLMRANK")
EXPECTED_USERS = {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}
EXPECTED_USER_TOTAL = 4882
EXPECTED_ROWS = 29292
EXPECTED_CELLS = 30
EXPECTED_STRICT_VALID = 28986
EXPECTED_INVALID = 306
EXPECTED_INVALID_BREAKDOWN = {"INVALID_DUPLICATE": 51, "INVALID_LENGTH": 255}
EXPECTED_BACKEND_ERRORS = 0

REQUEST_UNIVERSE = Path("results/tables/g4_f1bzq_reranking_request_universe.csv")
EXECUTION_ORDER = Path("results/tables/g4_f1bzq_reranking_execution_order.csv")
CANDIDATE_HASHES = Path("results/tables/g4_f1bzq_candidate_hashes.csv")
CANDIDATE_RECALL = Path("results/tables/g4_f1bzq_candidate_recall_at_20.csv")
MODEL_DIGESTS = Path("results/tables/g4_f1bzq_model_digest_freeze.csv")
METRIC_CONTRACT = Path("results/tables/g4_f1bzq_metric_denominator_contract.csv")
PROTOCOL_MANIFEST = Path("results/manifests/g4_f1bzq_reranking_protocol_freeze.json")
REQUEST_MANIFEST = Path("results/manifests/g4_f1bzq_request_universe_manifest.json")
RETENTION_MANIFEST = Path("results/manifests/g4_f1bzq_execution_retention_manifest.json")
RETENTION_INVENTORY = Path("results/tables/g4_f1bzq_execution_evidence_inventory.csv")

RAW_ROOT = Path("results/llm/extension_canonical/g4_f1bzq_reranking")
REQUEST_LEDGER = RAW_ROOT / "requests.jsonl"
RESPONSE_LEDGER = RAW_ROOT / "responses.jsonl"
EVENT_LEDGER = RAW_ROOT / "events.jsonl"
RESPONSE_DIR = RAW_ROOT / "responses"

OUT_UNIVERSE = Path("results/tables/g4_f1bzq_analytic_universe.csv")
OUT_CELL_COUNTS = Path("results/tables/g4_f1bzq_analytic_cell_counts.csv")
OUT_DENOMINATOR_COUNTS = Path("results/tables/g4_f1bzq_analytic_denominator_counts.csv")
OUT_DENOMINATOR_CONTRACT = Path("results/tables/g4_f1bzq_analytic_denominator_contract.csv")
OUT_VALIDATION = Path("results/tables/g4_f1bzq_analytic_validation.csv")
OUT_MANIFEST = Path("results/manifests/g4_f1bzq_analytic_universe_freeze.json")
OUT_REPORT = Path("report/g4_f1bzq_analytic_universe_freeze.md")
NOTES = Path("report/notes.md")
SOURCE_PATH = Path("src/analysis/g4_f1bzq_analytic_universe.py")

ALLOWED_DIRTY_PATHS = {
    SOURCE_PATH,
    OUT_UNIVERSE,
    OUT_CELL_COUNTS,
    OUT_DENOMINATOR_COUNTS,
    OUT_DENOMINATOR_CONTRACT,
    OUT_VALIDATION,
    OUT_MANIFEST,
    OUT_REPORT,
    NOTES,
}

G4_FROZEN_ARTIFACTS = (
    Path("configs/g4_f1bzq_reranking_execution.yaml"),
    Path("src/extension/g4_f1bzq_reranking_execution.py"),
    Path("tests/test_extension_g4_f1bzq_reranking_execution.py"),
    REQUEST_UNIVERSE,
    EXECUTION_ORDER,
    CANDIDATE_HASHES,
    CANDIDATE_RECALL,
    MODEL_DIGESTS,
    METRIC_CONTRACT,
    PROTOCOL_MANIFEST,
    REQUEST_MANIFEST,
    RETENTION_MANIFEST,
    RETENTION_INVENTORY,
)

IDENTITY_FIELDS = (
    "dataset",
    "task",
    "model",
    "model_digest",
    "prompt_strategy",
    "user_id",
    "candidate_set_id",
    "pool_id",
    "pool_hash",
    "candidate_hash",
    "prompt_hash",
    "parser_contract",
    "protocol_hash",
    "generation_config_hash",
)

REQUEST_IDENTITY_FIELDS = IDENTITY_FIELDS + ("history_contract", "history_length", "payload_hash")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the G4 F1BZQ analytic universe from retained evidence.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Validate and build in memory without writing outputs.")
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_record_hash(value: Any) -> str:
    return sha256_json(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"path": path.as_posix(), "line_number": line_number, "error": str(exc)})
                continue
            row["_line_number"] = line_number
            rows.append(row)
    return rows, errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def git_output(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def dirty_path_from_status_line(line: str) -> str:
    if len(line) >= 3 and line[2] == " ":
        path_text = line[3:].strip()
    else:
        parts = line.split(maxsplit=1)
        path_text = parts[1].strip() if len(parts) == 2 else ""
    if " -> " in path_text:
        path_text = path_text.split(" -> ", 1)[1].strip()
    return path_text.replace("\\", "/")


def unexpected_git_status_lines(status: str) -> list[str]:
    allowed = {path.as_posix() for path in ALLOWED_DIRTY_PATHS}
    unexpected: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if dirty_path_from_status_line(line) not in allowed:
            unexpected.append(line)
    return unexpected


def parse_json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    loaded = json.loads(str(value))
    if not isinstance(loaded, list):
        raise ValueError(f"expected JSON list, observed {value!r}")
    return loaded


def parse_bool_text(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def terminal_state_from_response(response: dict[str, Any]) -> str:
    if not response:
        return "MISSING_TERMINAL"
    if not parse_bool_text(response.get("backend_success", False)):
        return "BACKEND_FAILURE"
    parse_status = str(response.get("parse_status", ""))
    if parse_bool_text(response.get("strict_parse_valid", False)) and parse_status == "parsed_valid":
        return "STRICT_VALID"
    if parse_status == "invalid_length":
        return "INVALID_LENGTH"
    if parse_status == "duplicate_indices":
        return "INVALID_DUPLICATE"
    if parse_status == "unknown_indices":
        return "HALLUCINATION_OR_OUT_OF_CANDIDATE"
    if parse_status == "missing_permutation_indices":
        return "INVALID_MISSING_PERMUTATION"
    if parse_status:
        return "INVALID_" + parse_status.upper()
    return "INVALID_UNKNOWN"


def invalid_class_from_state(terminal_state: str) -> str:
    return "" if terminal_state == "STRICT_VALID" else terminal_state


def count_duplicates(values: list[str]) -> list[str]:
    counter = Counter(values)
    return sorted(value for value, count in counter.items() if value and count > 1)


def load_raw_records(root: Path) -> tuple[list[Path], dict[str, dict[str, Any]], dict[str, str], dict[str, int], list[dict[str, str]]]:
    raw_paths = sorted((root / RESPONSE_DIR).glob("*.json"), key=lambda path: path.name)
    raw_by_id: dict[str, dict[str, Any]] = {}
    raw_sha_by_id: dict[str, str] = {}
    raw_size_by_id: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    for path in raw_paths:
        rel_path = path.relative_to(root).as_posix()
        raw_sha_by_id[path.stem] = sha256_file(path)
        raw_size_by_id[path.stem] = path.stat().st_size
        try:
            row = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append({"path": rel_path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        request_id = str(row.get("canonical_request_id", ""))
        if request_id != path.stem:
            errors.append({"path": rel_path, "error": "canonical_request_id_filename_mismatch"})
        if request_id in raw_by_id:
            errors.append({"path": rel_path, "error": "duplicate_raw_record_id"})
        raw_by_id[request_id] = row
    return raw_paths, raw_by_id, raw_sha_by_id, raw_size_by_id, errors


def make_index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    counts = Counter(str(row.get("canonical_request_id") or row.get("request_id") or "") for row in rows)
    for row in rows:
        request_id = str(row.get("canonical_request_id") or row.get("request_id") or "")
        if request_id and counts[request_id] == 1:
            indexed[request_id] = row
    return indexed


def row_identity_mismatches(
    universe_by_id: dict[str, dict[str, str]],
    request_by_id: dict[str, dict[str, Any]],
    response_by_id: dict[str, dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []
    for request_id, universe in universe_by_id.items():
        for source_name, source, fields in (
            ("request_ledger", request_by_id.get(request_id), REQUEST_IDENTITY_FIELDS),
            ("terminal_ledger", response_by_id.get(request_id), IDENTITY_FIELDS),
            ("raw_file", raw_by_id.get(request_id), IDENTITY_FIELDS),
        ):
            if source is None:
                continue
            for field in fields:
                if str(universe.get(field, "")) != str(source.get(field, "")):
                    mismatches.append(
                        {
                            "canonical_request_id": request_id,
                            "source": source_name,
                            "field": field,
                            "universe": str(universe.get(field, "")),
                            "source_value": str(source.get(field, "")),
                        }
                    )
    return mismatches


def candidate_mapping_mismatches(
    request_universe: list[dict[str, str]],
    candidate_hashes: list[dict[str, str]],
    candidate_recall: list[dict[str, str]],
) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []
    candidate_by_user = {(row["dataset"], row["user_id"]): row for row in candidate_hashes}
    recall_by_dataset = {row["dataset"]: row for row in candidate_recall}
    unique_universe: dict[tuple[str, str], dict[str, str]] = {}
    for row in request_universe:
        key = (row["dataset"], row["user_id"])
        unique_universe.setdefault(key, row)
    for key, universe in unique_universe.items():
        candidate = candidate_by_user.get(key)
        if candidate is None:
            mismatches.append({"dataset": key[0], "user_id": key[1], "defect": "missing_from_candidate_hashes"})
            continue
        for field in (
            "candidate_set_id",
            "pool_id",
            "pool_hash",
            "candidate_hash",
            "candidate_items",
            "candidate_titles_hash",
            "candidate_source",
            "candidate_source_hash",
            "positive_item_id",
            "positive_present",
            "positive_position",
            "history_contract",
            "history_length",
            "full_history_length",
            "history_hash",
            "foodcom_history_class",
        ):
            if str(universe.get(field, "")) != str(candidate.get(field, "")):
                mismatches.append(
                    {
                        "dataset": key[0],
                        "user_id": key[1],
                        "defect": "candidate_hash_field_mismatch",
                        "field": field,
                        "universe": str(universe.get(field, "")),
                        "candidate_hashes": str(candidate.get(field, "")),
                    }
                )
    for key in sorted(set(candidate_by_user) - set(unique_universe)):
        mismatches.append({"dataset": key[0], "user_id": key[1], "defect": "candidate_hash_user_missing_from_universe"})
    for dataset, expected_users in EXPECTED_USERS.items():
        recall = recall_by_dataset.get(dataset, {})
        dataset_users = [row for row in unique_universe.values() if row["dataset"] == dataset]
        positive_present = sum(1 for row in dataset_users if parse_bool_text(row.get("positive_present", "")))
        positive_absent = len(dataset_users) - positive_present
        if int(recall.get("evaluation_N", -1)) != expected_users:
            mismatches.append({"dataset": dataset, "defect": "candidate_recall_evaluation_N_mismatch"})
        if int(recall.get("positive_present_N", -1)) != positive_present:
            mismatches.append({"dataset": dataset, "defect": "candidate_recall_positive_present_N_mismatch"})
        if int(recall.get("positive_absent_N", -1)) != positive_absent:
            mismatches.append({"dataset": dataset, "defect": "candidate_recall_positive_absent_N_mismatch"})
    return mismatches


def variant_consistency_defects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["user_id"]))].append(row)
    expected_variants = {(model, prompt) for model in MODELS for prompt in PROMPTS}
    stable_fields = (
        "candidate_set_id",
        "pool_id",
        "pool_hash",
        "candidate_hash",
        "candidate_items",
        "candidate_titles_hash",
        "candidate_count",
        "candidate_source",
        "candidate_source_hash",
        "positive_item_id",
        "positive_present_in_pool",
        "positive_input_position",
        "history_contract",
        "history_length",
        "full_history_length",
        "history_hash",
        "foodcom_history_class",
    )
    defects: list[dict[str, Any]] = []
    for key, group in grouped.items():
        variants = {(str(row["model"]), str(row["prompt_strategy"])) for row in group}
        if len(group) != len(expected_variants) or variants != expected_variants:
            defects.append(
                {
                    "dataset": key[0],
                    "user_id": key[1],
                    "defect": "missing_or_extra_model_prompt_variants",
                    "row_count": len(group),
                    "variant_count": len(variants),
                }
            )
        for field in stable_fields:
            observed = {str(row.get(field, "")) for row in group}
            if len(observed) != 1:
                defects.append(
                    {
                        "dataset": key[0],
                        "user_id": key[1],
                        "defect": "model_prompt_variant_field_mismatch",
                        "field": field,
                        "observed_count": len(observed),
                    }
                )
    return defects


def output_diagnostics(indices: list[Any], candidate_count: int) -> dict[str, Any]:
    int_indices = [value for value in indices if isinstance(value, int) and not isinstance(value, bool)]
    return {
        "output_length": len(indices),
        "output_unique_index_count": len(set(int_indices)),
        "duplicate_index_count": max(0, len(int_indices) - len(set(int_indices))),
        "unknown_index_count": sum(1 for value in int_indices if value < 1 or value > candidate_count),
    }


def build_analysis_rows(
    root: Path,
    request_universe: list[dict[str, str]],
    execution_order_by_id: dict[str, dict[str, str]],
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
        order = execution_order_by_id.get(request_id, {})
        raw_path = Path(universe["raw_response_path"])
        terminal_state = terminal_state_from_response(terminal)
        strict_valid = terminal_state == "STRICT_VALID"
        candidate_items = parse_json_list(universe["candidate_items"])
        ranked_indices = parse_json_list(terminal.get("ranked_candidate_indices", "[]")) if terminal else []
        ranked_items = parse_json_list(terminal.get("ranked_item_ids", "[]")) if terminal else []
        diag = output_diagnostics(ranked_indices, int(universe["candidate_count"]))
        terminal_record_hash = canonical_record_hash({k: v for k, v in terminal.items() if k != "_line_number"}) if terminal else ""
        raw_record_hash = canonical_record_hash(raw) if raw else ""
        row = {
            "canonical_request_id": request_id,
            "request_id": universe["request_id"],
            "request_order": universe["request_order"],
            "physical_execution_order": order.get("physical_execution_order", terminal.get("physical_execution_order", "")),
            "dataset": universe["dataset"],
            "user_id": universe["user_id"],
            "model": universe["model"],
            "model_family": universe["model_family"],
            "model_digest": universe["model_digest"],
            "prompt_strategy": universe["prompt_strategy"],
            "task": universe["task"],
            "scientific_status": universe["scientific_status"],
            "candidate_set_id": universe["candidate_set_id"],
            "pool_id": universe["pool_id"],
            "pool_hash": universe["pool_hash"],
            "candidate_hash": universe["candidate_hash"],
            "candidate_count": universe["candidate_count"],
            "candidate_items": universe["candidate_items"],
            "candidate_titles_hash": universe["candidate_titles_hash"],
            "candidate_source": universe["candidate_source"],
            "candidate_source_hash": universe["candidate_source_hash"],
            "candidate_source_protocol": universe["candidate_source_protocol"],
            "candidate_pool_policy": universe["candidate_pool_policy"],
            "candidate_pool_algorithm": universe["candidate_pool_algorithm"],
            "candidate_ordering": universe["candidate_ordering"],
            "candidate_order_immutable_for_llm": universe["candidate_order_immutable_for_llm"],
            "force_positive": universe["force_positive"],
            "target_identity_used_for_ranking": universe["target_identity_used_for_ranking"],
            "positive_item_id": universe["positive_item_id"],
            "positive_present_in_pool": universe["positive_present"],
            "positive_input_position": universe["positive_position"],
            "history_contract": universe["history_contract"],
            "history_length": universe["history_length"],
            "full_history_length": universe["full_history_length"],
            "history_hash": universe["history_hash"],
            "foodcom_history_class": universe["foodcom_history_class"],
            "prompt_contract": universe["prompt_contract"],
            "prompt_hash": universe["prompt_hash"],
            "parser_contract": universe["parser_contract"],
            "parser_expected_length": universe["parser_expected_length"],
            "generation_config_hash": universe["generation_config_hash"],
            "payload_hash": universe["payload_hash"],
            "request_ledger_exists": bool_text(bool(request)),
            "request_ledger_line": request.get("_line_number", ""),
            "request_payload_hash": request.get("payload_hash", ""),
            "request_timestamp_before_send_utc": request.get("timestamp_before_send_utc", ""),
            "terminal_ledger_exists": bool_text(bool(terminal)),
            "terminal_ledger_line": terminal.get("_line_number", ""),
            "terminal_record_hash": terminal_record_hash,
            "raw_file_exists": bool_text((root / raw_path).exists()),
            "raw_file_path": raw_path.as_posix(),
            "raw_file_size_bytes": raw_size_by_id.get(request_id, ""),
            "raw_file_sha256": raw_sha_by_id.get(request_id, ""),
            "raw_record_hash": raw_record_hash,
            "raw_terminal_record_hash_match": bool_text(bool(raw and terminal and raw_record_hash == terminal_record_hash)),
            "raw_response_hash": terminal.get("raw_response_hash", raw.get("raw_response_hash", "")),
            "raw_response_text_sha256_matches_record": bool_text(
                bool(terminal)
                and sha256_text(str(terminal.get("raw_response", ""))) == str(terminal.get("raw_response_hash", ""))
            ),
            "raw_response_recorded_before_parse": bool_text(parse_bool_text(terminal.get("raw_response_recorded_before_parse", False))),
            "backend_status": terminal.get("backend_status", raw.get("backend_status", "")),
            "backend_success": bool_text(parse_bool_text(terminal.get("backend_success", raw.get("backend_success", False)))),
            "backend_error_type": terminal.get("backend_error_type", raw.get("backend_error_type", "")),
            "backend_error_message": terminal.get("backend_error_message", raw.get("backend_error_message", "")),
            "backend_attempt_count": terminal.get("backend_attempt_count", raw.get("backend_attempt_count", "")),
            "retry_count": terminal.get("retry_count", raw.get("retry_count", "")),
            "request_started_at_utc": terminal.get("request_started_at_utc", raw.get("request_started_at_utc", "")),
            "request_completed_at_utc": terminal.get("request_completed_at_utc", raw.get("request_completed_at_utc", "")),
            "runtime_seconds": terminal.get("runtime_seconds", raw.get("runtime_seconds", "")),
            "strict_parse_valid": bool_text(strict_valid),
            "terminal_state": terminal_state,
            "invalid_class": invalid_class_from_state(terminal_state),
            "parse_status": terminal.get("parse_status", raw.get("parse_status", "")),
            "preserved_ranked_candidate_indices": stable_json(ranked_indices),
            "preserved_ranked_item_ids": stable_json(ranked_items),
            "parsed_top10_candidate_indices_if_strict_valid": stable_json(ranked_indices) if strict_valid else "",
            "parsed_top10_item_ids_if_strict_valid": stable_json(ranked_items) if strict_valid else "",
            "parsed_output_length": diag["output_length"],
            "parsed_unique_index_count": diag["output_unique_index_count"],
            "parsed_duplicate_index_count": diag["duplicate_index_count"],
            "parsed_unknown_index_count": diag["unknown_index_count"],
            "utility_denominator_policy": "all_planned_users_invalid_terminal_zero",
            "candidate_recall_denominator_policy": "all_users",
            "conditional_utility_denominator_policy": "positive_present_users_only_invalid_terminal_zero",
            "beyond_utility_denominator_policy": "strict_valid_users_metric_specific_defined_subsets",
            "reliability_denominator_policy": "all_planned_requests",
            "metric_values_computed_in_this_phase": "false",
            "statistical_analysis_computed_in_this_phase": "false",
            "retention_source_manifest": RETENTION_MANIFEST.as_posix(),
        }
        row["analysis_row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def build_cell_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            for prompt in PROMPTS:
                group = [row for row in rows if row["dataset"] == dataset and row["model"] == model and row["prompt_strategy"] == prompt]
                state_counts = Counter(str(row["terminal_state"]) for row in group)
                present_users = len({row["user_id"] for row in group if row["positive_present_in_pool"] == "true"})
                absent_users = len({row["user_id"] for row in group if row["positive_present_in_pool"] == "false"})
                expected_users = EXPECTED_USERS[dataset]
                cell_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "prompt_strategy": prompt,
                        "logical_rows": len(group),
                        "planned_users_denominator": expected_users,
                        "unique_users": len({row["user_id"] for row in group}),
                        "positive_present_users": present_users,
                        "positive_absent_users": absent_users,
                        "strict_valid": state_counts["STRICT_VALID"],
                        "invalid": len(group) - state_counts["STRICT_VALID"],
                        "invalid_length": state_counts["INVALID_LENGTH"],
                        "invalid_duplicate": state_counts["INVALID_DUPLICATE"],
                        "backend_errors": state_counts["BACKEND_FAILURE"],
                        "cell_complete": bool_text(len(group) == expected_users and len({row["user_id"] for row in group}) == expected_users),
                        "metric_values_computed_in_this_phase": "false",
                    }
                )
    return cell_rows


def build_denominator_counts(candidate_hashes: list[dict[str, str]], analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unique_users = {(row["dataset"], row["user_id"]): row for row in candidate_hashes}
    for dataset in DATASETS:
        user_rows = [row for (ds, _user), row in unique_users.items() if ds == dataset]
        analytic = [row for row in analysis_rows if row["dataset"] == dataset]
        planned_users = len(user_rows)
        positive_present = sum(1 for row in user_rows if parse_bool_text(row.get("positive_present", "")))
        positive_absent = planned_users - positive_present
        strict_valid = sum(1 for row in analytic if row["strict_parse_valid"] == "true")
        rows.append(
            {
                "scope": "BY_DATASET",
                "dataset": dataset,
                "planned_users": planned_users,
                "positive_present_users": positive_present,
                "positive_absent_users": positive_absent,
                "model_prompt_variants": len(MODELS) * len(PROMPTS),
                "planned_request_rows": len(analytic),
                "positive_present_request_rows": sum(1 for row in analytic if row["positive_present_in_pool"] == "true"),
                "positive_absent_request_rows": sum(1 for row in analytic if row["positive_present_in_pool"] == "false"),
                "strict_valid_request_rows": strict_valid,
                "invalid_terminal_request_rows": len(analytic) - strict_valid,
                "backend_error_request_rows": sum(1 for row in analytic if row["terminal_state"] == "BACKEND_FAILURE"),
                "source": "results/tables/g4_f1bzq_candidate_hashes.csv plus frozen terminal evidence",
                "metric_values_computed_in_this_phase": "false",
            }
        )
    totals = {
        "scope": "GRAND_TOTAL",
        "dataset": "ALL",
        "planned_users": sum(int(row["planned_users"]) for row in rows),
        "positive_present_users": sum(int(row["positive_present_users"]) for row in rows),
        "positive_absent_users": sum(int(row["positive_absent_users"]) for row in rows),
        "model_prompt_variants": len(MODELS) * len(PROMPTS),
        "planned_request_rows": len(analysis_rows),
        "positive_present_request_rows": sum(int(row["positive_present_request_rows"]) for row in rows),
        "positive_absent_request_rows": sum(int(row["positive_absent_request_rows"]) for row in rows),
        "strict_valid_request_rows": sum(int(row["strict_valid_request_rows"]) for row in rows),
        "invalid_terminal_request_rows": sum(int(row["invalid_terminal_request_rows"]) for row in rows),
        "backend_error_request_rows": sum(int(row["backend_error_request_rows"]) for row in rows),
        "source": "sum of BY_DATASET rows",
        "metric_values_computed_in_this_phase": "false",
    }
    rows.append(totals)
    return rows


def build_denominator_contract(denominator_counts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals = next(row for row in denominator_counts if row["scope"] == "GRAND_TOTAL")
    return [
        {
            "metric_group": "primary_unconditional_utility",
            "metric_ids": stable_json(["hr_at_5", "ndcg_at_5", "mrr"]),
            "denominator_unit": "planned user-request rows within each dataset_model_prompt cell",
            "denominator_scope": "all planned users",
            "global_denominator_rows": totals["planned_request_rows"],
            "per_cell_denominator_source": "planned_users by dataset in g4_f1bzq_analytic_denominator_counts.csv",
            "invalid_terminal_policy": "invalid terminal response contributes zero",
            "metric_values_computed_in_this_phase": "false",
        },
        {
            "metric_group": "candidate_recall",
            "metric_ids": stable_json(["candidate_recall_at_20"]),
            "denominator_unit": "unique dataset/user candidate pools",
            "denominator_scope": "all users",
            "global_denominator_rows": totals["planned_users"],
            "per_cell_denominator_source": "not model/prompt specific; candidate-stage denominator by dataset",
            "invalid_terminal_policy": "not applicable to terminal response validity",
            "metric_values_computed_in_this_phase": "false",
        },
        {
            "metric_group": "conditional_utility",
            "metric_ids": stable_json(["conditional_hr_at_5", "conditional_ndcg_at_5"]),
            "denominator_unit": "positive-present user-request rows within each dataset_model_prompt cell",
            "denominator_scope": "positive-present users only",
            "global_denominator_rows": totals["positive_present_request_rows"],
            "per_cell_denominator_source": "positive_present_users by dataset in g4_f1bzq_analytic_denominator_counts.csv",
            "invalid_terminal_policy": "invalid terminal response contributes zero within the positive-present subset",
            "metric_values_computed_in_this_phase": "false",
        },
        {
            "metric_group": "beyond_utility",
            "metric_ids": "reuse_only_already_implemented_frozen_g1_g2_compatible_metrics",
            "denominator_unit": "strict-valid user-request rows, with metric-specific defined subsets where required",
            "denominator_scope": "strict-valid users",
            "global_denominator_rows": totals["strict_valid_request_rows"],
            "per_cell_denominator_source": "strict_valid by dataset_model_prompt in g4_f1bzq_analytic_cell_counts.csv",
            "invalid_terminal_policy": "invalid terminal responses excluded from strict-valid beyond-utility denominator",
            "metric_values_computed_in_this_phase": "false",
        },
        {
            "metric_group": "reliability",
            "metric_ids": "strict_parse_validity,structural_invalidity_rate,invalid_reason_distribution,duplicate_rate,hallucination,backend_failure_rate,latency,throughput,token_counts,hardware_runtime_provenance",
            "denominator_unit": "planned requests",
            "denominator_scope": "all planned requests",
            "global_denominator_rows": totals["planned_request_rows"],
            "per_cell_denominator_source": "logical_rows by dataset_model_prompt in g4_f1bzq_analytic_cell_counts.csv",
            "invalid_terminal_policy": "invalid terminal response remains terminal scientific evidence and is counted",
            "metric_values_computed_in_this_phase": "false",
        },
    ]


def build_validation(
    *,
    head_before: str,
    status_before: str,
    request_universe: list[dict[str, str]],
    execution_order: list[dict[str, str]],
    candidate_hashes: list[dict[str, str]],
    candidate_recall: list[dict[str, str]],
    model_digests: list[dict[str, str]],
    metric_contract: list[dict[str, str]],
    protocol_manifest: dict[str, Any],
    request_manifest: dict[str, Any],
    retention_manifest: dict[str, Any],
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    events: list[dict[str, Any]],
    request_jsonl_errors: list[dict[str, Any]],
    response_jsonl_errors: list[dict[str, Any]],
    event_jsonl_errors: list[dict[str, Any]],
    raw_paths: list[Path],
    raw_by_id: dict[str, dict[str, Any]],
    raw_errors: list[dict[str, str]],
    analysis_rows: list[dict[str, Any]],
    cell_counts: list[dict[str, Any]],
    denominator_counts: list[dict[str, Any]],
    candidate_mismatches: list[dict[str, str]],
    identity_mismatches: list[dict[str, str]],
    variant_defects: list[dict[str, Any]],
    frozen_hash_failures: list[str],
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

    universe_ids = [row["canonical_request_id"] for row in request_universe]
    order_ids = [row["canonical_request_id"] for row in execution_order]
    request_ids = [str(row.get("canonical_request_id") or row.get("request_id") or "") for row in requests]
    response_ids = [str(row.get("canonical_request_id") or row.get("request_id") or "") for row in responses]
    raw_file_ids = [path.stem for path in raw_paths]
    raw_record_ids = list(raw_by_id)
    id_sets = {
        "request_universe": set(universe_ids),
        "execution_order": set(order_ids),
        "request_ledger": set(request_ids),
        "terminal_ledger": set(response_ids),
        "raw_file": set(raw_file_ids),
        "raw_record": set(raw_record_ids),
    }
    duplicate_counts = {
        "request_universe": len(count_duplicates(universe_ids)),
        "execution_order": len(count_duplicates(order_ids)),
        "request_ledger": len(count_duplicates(request_ids)),
        "terminal_ledger": len(count_duplicates(response_ids)),
        "raw_file": len(count_duplicates(raw_file_ids)),
        "raw_record": len(count_duplicates(raw_record_ids)),
    }
    base_ids = id_sets["request_universe"]
    missing = {
        name: len(base_ids - ids)
        for name, ids in id_sets.items()
        if name != "request_universe"
    }
    orphan = {
        name: len(ids - base_ids)
        for name, ids in id_sets.items()
        if name != "request_universe"
    }
    state_counts = Counter(row["terminal_state"] for row in analysis_rows)
    parse_counts = Counter(row["parse_status"] for row in analysis_rows)
    dataset_users = Counter(row["dataset"] for row in {(row["dataset"], row["user_id"]): row for row in analysis_rows}.values())
    row_counts = Counter(row["dataset"] for row in analysis_rows)
    model_counts = Counter(row["model"] for row in analysis_rows)
    prompt_counts = Counter(row["prompt_strategy"] for row in analysis_rows)
    cell_complete = all(row["cell_complete"] == "true" for row in cell_counts) and len(cell_counts) == EXPECTED_CELLS
    total_denom = next(row for row in denominator_counts if row["scope"] == "GRAND_TOTAL")
    expected_positive_by_dataset = {
        "beauty": {"planned_users": 990, "positive_present_users": 49, "positive_absent_users": 941},
        "sports": {"planned_users": 892, "positive_present_users": 28, "positive_absent_users": 864},
        "ml1m": {"planned_users": 1000, "positive_present_users": 193, "positive_absent_users": 807},
        "lastfm": {"planned_users": 1000, "positive_present_users": 469, "positive_absent_users": 531},
        "foodcom": {"planned_users": 1000, "positive_present_users": 4, "positive_absent_users": 996},
    }
    observed_positive_by_dataset = {
        row["dataset"]: {
            "planned_users": int(row["planned_users"]),
            "positive_present_users": int(row["positive_present_users"]),
            "positive_absent_users": int(row["positive_absent_users"]),
        }
        for row in denominator_counts
        if row["scope"] == "BY_DATASET"
    }
    model_digest_expected = {row["model"]: row["frozen_digest"] for row in model_digests}
    model_digest_observed = {
        model: sorted({row["model_digest"] for row in analysis_rows if row["model"] == model})
        for model in MODELS
    }
    positive_position_ok = all(
        (row["positive_present_in_pool"] == "true" and str(row["positive_input_position"]).isdigit() and 1 <= int(row["positive_input_position"]) <= 20)
        or (row["positive_present_in_pool"] == "false" and str(row["positive_input_position"]) == "")
        for row in analysis_rows
    )
    strict_output_ok = all(
        (
            row["strict_parse_valid"] == "true"
            and row["parsed_top10_candidate_indices_if_strict_valid"]
            and row["parsed_top10_item_ids_if_strict_valid"]
            and int(row["parsed_output_length"]) == 10
            and int(row["parsed_unique_index_count"]) == 10
            and int(row["parsed_unknown_index_count"]) == 0
        )
        or (
            row["strict_parse_valid"] == "false"
            and row["parsed_top10_candidate_indices_if_strict_valid"] == ""
            and row["parsed_top10_item_ids_if_strict_valid"] == ""
        )
        for row in analysis_rows
    )
    raw_terminal_match_count = sum(1 for row in analysis_rows if row["raw_terminal_record_hash_match"] == "true")
    raw_response_hash_bad = sum(1 for row in analysis_rows if row["raw_response_text_sha256_matches_record"] != "true")
    denominator_contract_rows = [row for row in metric_contract if row.get("included_in_g4") == "true"]
    expected_metric_denominators = {
        "hr_at_5": "planned_applicable_users_invalid_terminal_zero",
        "ndcg_at_5": "planned_applicable_users_invalid_terminal_zero",
        "mrr": "planned_applicable_users_invalid_terminal_zero",
        "candidate_recall_at_20": "all_users",
        "conditional_hr_at_5": "positive_present_users_only",
        "conditional_ndcg_at_5": "positive_present_users_only",
    }
    observed_metric_denominators = {
        row["metric_id"]: row["denominator"]
        for row in metric_contract
        if row["metric_id"] in expected_metric_denominators
    }

    add("G4AU-001", "precondition HEAD", head_before == EXPECTED_HEAD, head_before, EXPECTED_HEAD)
    unexpected_status = unexpected_git_status_lines(status_before)
    add("G4AU-002", "git status limited to current analytic-universe artifacts", not unexpected_status, {"unexpected": unexpected_status, "full_status": status_before or "clean"}, "no non-G4 analytic-universe changes")
    add("G4AU-003", "retention freeze token", retention_manifest.get("decision_token") == "G4_EXECUTION_RETENTION_FROZEN", retention_manifest.get("decision_token"), "G4_EXECUTION_RETENTION_FROZEN")
    add("G4AU-004", "retention validation passed", retention_manifest.get("validation_passed") is True, retention_manifest.get("validation_passed"), True)
    add("G4AU-005", "protocol freeze token", protocol_manifest.get("decision_token") == "G4_F1BZQ_RERANKING_PROTOCOL_FROZEN", protocol_manifest.get("decision_token"), "G4_F1BZQ_RERANKING_PROTOCOL_FROZEN")
    add("G4AU-006", "request universe freeze token", request_manifest.get("decision_token") == "G4_F1BZQ_RERANKING_PROTOCOL_FROZEN", request_manifest.get("decision_token"), "G4_F1BZQ_RERANKING_PROTOCOL_FROZEN")
    add("G4AU-007", "frozen G4 artifacts unchanged", not frozen_hash_failures, frozen_hash_failures, [])
    add("G4AU-008", "logical analytic rows", len(analysis_rows) == EXPECTED_ROWS, len(analysis_rows), EXPECTED_ROWS)
    add("G4AU-009", "unique analytic canonical request IDs", len({row["canonical_request_id"] for row in analysis_rows}) == EXPECTED_ROWS, len({row["canonical_request_id"] for row in analysis_rows}), EXPECTED_ROWS)
    add("G4AU-010", "unique users", len({(row["dataset"], row["user_id"]) for row in analysis_rows}) == EXPECTED_USER_TOTAL, len({(row["dataset"], row["user_id"]) for row in analysis_rows}), EXPECTED_USER_TOTAL)
    add("G4AU-011", "dataset user counts", dict(sorted(dataset_users.items())) == EXPECTED_USERS, dict(sorted(dataset_users.items())), EXPECTED_USERS)
    add("G4AU-012", "dataset row counts", dict(sorted(row_counts.items())) == {ds: users * 6 for ds, users in EXPECTED_USERS.items()}, dict(sorted(row_counts.items())), {ds: users * 6 for ds, users in EXPECTED_USERS.items()})
    add("G4AU-013", "model and prompt row counts", all(model_counts[model] == 9764 for model in MODELS) and all(prompt_counts[prompt] == 14646 for prompt in PROMPTS), {"model": dict(model_counts), "prompt": dict(prompt_counts)}, {"model_each": 9764, "prompt_each": 14646})
    add("G4AU-014", "30 complete cells", cell_complete, {"cells": len(cell_counts), "incomplete": [row for row in cell_counts if row["cell_complete"] != "true"]}, EXPECTED_CELLS)
    add("G4AU-015", "candidate mapping exact", not candidate_mismatches, candidate_mismatches[:20], [])
    add("G4AU-016", "same pool/history/positive status across model-prompt variants", not variant_defects, variant_defects[:20], [])
    add("G4AU-017", "positive position contract", positive_position_ok, "all rows", "present => 1..20; absent => empty")
    add("G4AU-018", "positive-present counts by dataset", observed_positive_by_dataset == expected_positive_by_dataset, observed_positive_by_dataset, expected_positive_by_dataset)
    add("G4AU-019", "canonical ID sets match across universe/order/evidence", all(ids == base_ids for ids in id_sets.values()), {name: len(ids) for name, ids in id_sets.items()}, "all equal request_universe IDs")
    add("G4AU-020", "duplicate evidence IDs zero", all(value == 0 for value in duplicate_counts.values()), duplicate_counts, "all zero")
    add("G4AU-021", "missing evidence zero", all(value == 0 for value in missing.values()), missing, "all zero")
    add("G4AU-022", "orphan evidence zero", all(value == 0 for value in orphan.values()), orphan, "all zero")
    add("G4AU-023", "JSONL/raw parse defects zero", not request_jsonl_errors and not response_jsonl_errors and not event_jsonl_errors and not raw_errors, {"request_jsonl": len(request_jsonl_errors), "response_jsonl": len(response_jsonl_errors), "event_jsonl": len(event_jsonl_errors), "raw": len(raw_errors)}, "all zero")
    add("G4AU-024", "candidate/history/hash identity mismatches zero", not identity_mismatches, identity_mismatches[:20], [])
    add("G4AU-025", "raw and terminal evidence hashes match", raw_terminal_match_count == EXPECTED_ROWS and raw_response_hash_bad == 0, {"raw_terminal_record_hash_match": raw_terminal_match_count, "raw_response_hash_bad": raw_response_hash_bad}, {"raw_terminal_record_hash_match": EXPECTED_ROWS, "raw_response_hash_bad": 0})
    add("G4AU-026", "strict-valid and invalid counts", state_counts["STRICT_VALID"] == EXPECTED_STRICT_VALID and len(analysis_rows) - state_counts["STRICT_VALID"] == EXPECTED_INVALID, {"strict_valid": state_counts["STRICT_VALID"], "invalid": len(analysis_rows) - state_counts["STRICT_VALID"]}, {"strict_valid": EXPECTED_STRICT_VALID, "invalid": EXPECTED_INVALID})
    add("G4AU-027", "invalid breakdown", {key: state_counts[key] for key in EXPECTED_INVALID_BREAKDOWN} == EXPECTED_INVALID_BREAKDOWN, {key: state_counts[key] for key in EXPECTED_INVALID_BREAKDOWN}, EXPECTED_INVALID_BREAKDOWN)
    add("G4AU-028", "backend errors zero", state_counts["BACKEND_FAILURE"] == EXPECTED_BACKEND_ERRORS, state_counts["BACKEND_FAILURE"], EXPECTED_BACKEND_ERRORS)
    add("G4AU-029", "strict-valid parsed top-10 fields populated only for strict-valid rows", strict_output_ok, {"parse_status_counts": dict(parse_counts)}, "strict-valid rows have top-10; invalid rows blank")
    add("G4AU-030", "model identities match frozen digests", all(model_digest_observed[model] == [model_digest_expected[model]] for model in MODELS), model_digest_observed, model_digest_expected)
    add("G4AU-031", "candidate-stage denominators", int(total_denom["planned_users"]) == EXPECTED_USER_TOTAL and int(total_denom["positive_present_users"]) == 743 and int(total_denom["positive_absent_users"]) == 4139, total_denom, {"planned_users": 4882, "positive_present_users": 743, "positive_absent_users": 4139})
    add("G4AU-032", "utility/candidate/conditional denominator contract frozen", observed_metric_denominators == expected_metric_denominators, observed_metric_denominators, expected_metric_denominators)
    add("G4AU-033", "beyond-utility and reliability contract available", any(row["metric_family"] == "beyond_utility_g1_g2_compatible" for row in denominator_contract_rows) and any(row["metric_family"] == "reliability_efficiency" for row in denominator_contract_rows), Counter(row["metric_family"] for row in denominator_contract_rows), "beyond_utility_g1_g2_compatible and reliability_efficiency included")
    add("G4AU-034", "zero inference, metrics, and statistics in analytic freeze", all(row["metric_values_computed_in_this_phase"] == "false" and row["statistical_analysis_computed_in_this_phase"] == "false" for row in analysis_rows), {"ollama_or_api_calls": 0, "metrics_computed": 0, "statistics_computed": 0}, {"ollama_or_api_calls": 0, "metrics_computed": 0, "statistics_computed": 0})
    add("G4AU-035", "retention accounting matches analytic universe", retention_manifest.get("accounting", {}) == {"backend_errors": 0, "g4_backend_calls": 29292, "invalid_terminal_scientific_evidence": 306, "logical_requests": 29292, "remaining_calls": 0, "response_reuse": 0, "strict_valid": 28986}, retention_manifest.get("accounting", {}), "retention final accounting")
    add("G4AU-036", "event ledger retained and parseable", len(events) == int(retention_manifest.get("canonical_evidence_completeness", {}).get("event_rows", -1)), len(events), retention_manifest.get("canonical_evidence_completeness", {}).get("event_rows", "retention event_rows"))
    return rows


def summarize(analysis_rows: list[dict[str, Any]], cell_counts: list[dict[str, Any]], denominator_counts: list[dict[str, Any]]) -> dict[str, Any]:
    states = Counter(row["terminal_state"] for row in analysis_rows)
    parse_status = Counter(row["parse_status"] for row in analysis_rows)
    by_dataset = {}
    for dataset in DATASETS:
        group = [row for row in analysis_rows if row["dataset"] == dataset]
        state_counts = Counter(row["terminal_state"] for row in group)
        denom = next(row for row in denominator_counts if row["scope"] == "BY_DATASET" and row["dataset"] == dataset)
        by_dataset[dataset] = {
            "logical_rows": len(group),
            "planned_users": int(denom["planned_users"]),
            "positive_present_users": int(denom["positive_present_users"]),
            "positive_absent_users": int(denom["positive_absent_users"]),
            "strict_valid": state_counts["STRICT_VALID"],
            "invalid": len(group) - state_counts["STRICT_VALID"],
            "invalid_length": state_counts["INVALID_LENGTH"],
            "invalid_duplicate": state_counts["INVALID_DUPLICATE"],
            "backend_errors": state_counts["BACKEND_FAILURE"],
        }
    total = next(row for row in denominator_counts if row["scope"] == "GRAND_TOTAL")
    return {
        "logical_rows": len(analysis_rows),
        "unique_users": len({(row["dataset"], row["user_id"]) for row in analysis_rows}),
        "cells": len(cell_counts),
        "strict_valid": states["STRICT_VALID"],
        "invalid": len(analysis_rows) - states["STRICT_VALID"],
        "terminal_state_counts": dict(sorted(states.items())),
        "parse_status_counts": dict(sorted(parse_status.items())),
        "dataset_counts": by_dataset,
        "model_counts": dict(sorted(Counter(row["model"] for row in analysis_rows).items())),
        "prompt_counts": dict(sorted(Counter(row["prompt_strategy"] for row in analysis_rows).items())),
        "denominator_totals": total,
    }


def build_report(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    dataset_lines = "\n".join(
        "- `{dataset}`: planned users `{planned_users}`, positive-present `{positive_present_users}`, "
        "positive-absent `{positive_absent_users}`, rows `{logical_rows}`, strict-valid `{strict_valid}`, invalid `{invalid}`".format(
            dataset=dataset,
            **values,
        )
        for dataset, values in summary["dataset_counts"].items()
    )
    files = "\n".join(f"- `{path}`" for path in manifest["files_created"])
    return f"""# G4 F1BZQ Analytic Universe Freeze

## Scope
- Decision token: `{manifest['decision_token']}`.
- Source: frozen G4 retained evidence only (`{RETENTION_MANIFEST.as_posix()}`).
- Zero Ollama/API calls, no metric computation, no statistical analysis, no protocol/config/runner/F1BZQ-pool edits, and no evidence rewrites were performed.

## Universe
- Logical rows: `{summary['logical_rows']}`.
- Unique users: `{summary['unique_users']}`.
- Cells: `{summary['cells']}` dataset x model x prompt cells.
- Dimensions: `5` datasets, `3` models, `2` prompts.

## Evidence Resolution
- Request ledger rows resolved: `29292`.
- Terminal rows resolved: `29292`.
- Raw files resolved: `29292`.
- Strict-valid: `{summary['strict_valid']}`.
- Structural invalid: `{summary['invalid']}` (`INVALID_LENGTH={summary['terminal_state_counts'].get('INVALID_LENGTH', 0)}`, `INVALID_DUPLICATE={summary['terminal_state_counts'].get('INVALID_DUPLICATE', 0)}`).
- Backend errors: `{summary['terminal_state_counts'].get('BACKEND_FAILURE', 0)}`.

## Candidate-Stage Denominators
{dataset_lines}

## Denominator Contract
- Utility (`HR@5`, `NDCG@5`, `MRR`): all planned users; invalid terminal response contributes zero.
- Candidate recall: all users.
- Conditional utility (`conditional_hr_at_5`, `conditional_ndcg_at_5`): positive-present users only; invalid terminal response contributes zero within that subset.
- Beyond-utility: strict-valid users, with metric-specific defined subsets where required.
- Reliability: all planned requests.

## Validation
- Validation passed: `{str(manifest['validation_passed']).lower()}`.
- Mismatches/defects: `{len(manifest['validation_failures'])}` failing checks.
- Candidate mapping, shared pool/history across model-prompt variants, canonical IDs, model identities, raw/terminal evidence, and denominator counts all passed when validation is true.

## Outputs
{files}

G4_ANALYTIC_UNIVERSE_FROZEN
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    head_before = git_output(root, "rev-parse", "HEAD")
    status_before = git_output(root, "status", "--short", "--untracked-files=all")

    request_universe = read_csv(root / REQUEST_UNIVERSE)
    execution_order = read_csv(root / EXECUTION_ORDER)
    candidate_hashes = read_csv(root / CANDIDATE_HASHES)
    candidate_recall = read_csv(root / CANDIDATE_RECALL)
    model_digests = read_csv(root / MODEL_DIGESTS)
    metric_contract = read_csv(root / METRIC_CONTRACT)
    protocol_manifest = read_json(root / PROTOCOL_MANIFEST)
    request_manifest = read_json(root / REQUEST_MANIFEST)
    retention_manifest = read_json(root / RETENTION_MANIFEST)
    requests, request_jsonl_errors = read_jsonl(root / REQUEST_LEDGER)
    responses, response_jsonl_errors = read_jsonl(root / RESPONSE_LEDGER)
    events, event_jsonl_errors = read_jsonl(root / EVENT_LEDGER)
    raw_paths, raw_by_id, raw_sha_by_id, raw_size_by_id, raw_errors = load_raw_records(root)

    universe_by_id = {row["canonical_request_id"]: row for row in request_universe}
    execution_order_by_id = {row["canonical_request_id"]: row for row in execution_order}
    request_by_id = make_index_by_id(requests)
    response_by_id = make_index_by_id(responses)
    analysis_rows = build_analysis_rows(
        root,
        request_universe,
        execution_order_by_id,
        request_by_id,
        response_by_id,
        raw_by_id,
        raw_sha_by_id,
        raw_size_by_id,
    )
    cell_counts = build_cell_counts(analysis_rows)
    denominator_counts = build_denominator_counts(candidate_hashes, analysis_rows)
    denominator_contract = build_denominator_contract(denominator_counts)

    candidate_mismatches = candidate_mapping_mismatches(request_universe, candidate_hashes, candidate_recall)
    identity_mismatches = row_identity_mismatches(universe_by_id, request_by_id, response_by_id, raw_by_id)
    variant_defects = variant_consistency_defects(analysis_rows)
    tracked_hashes = {}
    tracked_hashes.update(protocol_manifest.get("artifact_hashes", {}))
    tracked_hashes.update(request_manifest.get("artifact_hashes", {}))
    tracked_hashes.update(retention_manifest.get("frozen_identity_checks", {}).get("tracked_frozen_artifact_hashes", {}))
    tracked_hashes[RETENTION_INVENTORY.as_posix()] = retention_manifest.get("retention_policy", {}).get("evidence_inventory_sha256", "")
    frozen_hash_failures = []
    for path in G4_FROZEN_ARTIFACTS:
        rel_path = path.as_posix()
        expected = tracked_hashes.get(rel_path)
        if not expected:
            continue
        if not (root / path).exists() or sha256_file(root / path) != expected:
            frozen_hash_failures.append(rel_path)

    validation_rows = build_validation(
        head_before=head_before,
        status_before=status_before,
        request_universe=request_universe,
        execution_order=execution_order,
        candidate_hashes=candidate_hashes,
        candidate_recall=candidate_recall,
        model_digests=model_digests,
        metric_contract=metric_contract,
        protocol_manifest=protocol_manifest,
        request_manifest=request_manifest,
        retention_manifest=retention_manifest,
        requests=requests,
        responses=responses,
        events=events,
        request_jsonl_errors=request_jsonl_errors,
        response_jsonl_errors=response_jsonl_errors,
        event_jsonl_errors=event_jsonl_errors,
        raw_paths=raw_paths,
        raw_by_id=raw_by_id,
        raw_errors=raw_errors,
        analysis_rows=analysis_rows,
        cell_counts=cell_counts,
        denominator_counts=denominator_counts,
        candidate_mismatches=candidate_mismatches,
        identity_mismatches=identity_mismatches,
        variant_defects=variant_defects,
        frozen_hash_failures=frozen_hash_failures,
    )
    failures = [row["check_id"] for row in validation_rows if row["status"] != "PASS"]
    summary = summarize(analysis_rows, cell_counts, denominator_counts)

    output_tables = [
        (OUT_UNIVERSE, analysis_rows),
        (OUT_CELL_COUNTS, cell_counts),
        (OUT_DENOMINATOR_COUNTS, denominator_counts),
        (OUT_DENOMINATOR_CONTRACT, denominator_contract),
        (OUT_VALIDATION, validation_rows),
    ]
    if not args.no_write:
        for path, rows in output_tables:
            write_csv(root / path, rows)

    input_paths = [
        REQUEST_UNIVERSE,
        EXECUTION_ORDER,
        CANDIDATE_HASHES,
        CANDIDATE_RECALL,
        MODEL_DIGESTS,
        METRIC_CONTRACT,
        PROTOCOL_MANIFEST,
        REQUEST_MANIFEST,
        RETENTION_MANIFEST,
        RETENTION_INVENTORY,
        REQUEST_LEDGER,
        RESPONSE_LEDGER,
        EVENT_LEDGER,
    ]
    output_paths = [path for path, _rows in output_tables]
    manifest = {
        "schema_version": "1.0",
        "phase_id": PHASE_ID,
        "decision_token": DECISION_FROZEN if not failures else DECISION_BLOCKED,
        "created_at_utc": utc_now(),
        "head_before_generation": head_before,
        "precondition_status": "G4_EXECUTION_RETENTION_FROZEN",
        "source": "frozen_retained_g4_evidence_only",
        "zero_inference": True,
        "ollama_or_api_calls_executed": 0,
        "metric_computation_executed": False,
        "statistical_analysis_executed": False,
        "protocol_config_runner_modified": False,
        "f1bzq_pools_modified": False,
        "frozen_evidence_modified": False,
        "input_hashes": {path.as_posix(): sha256_file(root / path) for path in input_paths if (root / path).is_file()},
        "source_hashes": {SOURCE_PATH.as_posix(): sha256_file(root / SOURCE_PATH)},
        "output_hashes": {path.as_posix(): sha256_file(root / path) for path in output_paths if (root / path).is_file()},
        "validation_passed": not failures,
        "validation_failures": failures,
        "summary": summary,
        "denominator_contract": denominator_contract,
        "files_created": [
            SOURCE_PATH.as_posix(),
            OUT_UNIVERSE.as_posix(),
            OUT_CELL_COUNTS.as_posix(),
            OUT_DENOMINATOR_COUNTS.as_posix(),
            OUT_DENOMINATOR_CONTRACT.as_posix(),
            OUT_VALIDATION.as_posix(),
            OUT_MANIFEST.as_posix(),
            OUT_REPORT.as_posix(),
            NOTES.as_posix(),
        ],
        "next_safe_step": "compute G4 metrics from the frozen analytic universe only when explicitly requested; no new inference",
    }

    if not args.no_write:
        (root / OUT_REPORT).parent.mkdir(parents=True, exist_ok=True)
        (root / OUT_REPORT).write_text(build_report(manifest), encoding="utf-8", newline="\n")
        manifest["output_hashes"][OUT_REPORT.as_posix()] = sha256_file(root / OUT_REPORT)
        notes_text = (root / NOTES).read_text(encoding="utf-8") if (root / NOTES).exists() else ""
        notes_marker = " - G4 analytic universe freeze\n\n"
        if not failures and notes_marker not in notes_text:
            notes = (
                f"\n## {utc_now()} - G4 analytic universe freeze\n\n"
                f"- Decision: `{manifest['decision_token']}` from frozen retained G4 evidence only.\n"
                f"- Universe: `{summary['logical_rows']}` logical rows, `{summary['unique_users']}` users, and `{summary['cells']}` dataset x model x prompt cells.\n"
                f"- Evidence resolution: `29292` raw files, `29292` terminal rows, strict-valid `{summary['strict_valid']}`, invalid `{summary['invalid']}`, backend errors `{summary['terminal_state_counts'].get('BACKEND_FAILURE', 0)}`.\n"
                f"- Candidate-stage denominators: planned users `4882`, positive-present `743`, positive-absent `4139`.\n"
                "- No Ollama/API calls, metric computation, statistical analysis, protocol/config/runner edits, F1BZQ-pool edits, or evidence rewrites were performed.\n"
            )
            with (root / NOTES).open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(notes)
            manifest["output_hashes"][NOTES.as_posix()] = sha256_file(root / NOTES)
        write_json(root / OUT_MANIFEST, manifest)

    manifest_sha = sha256_file(root / OUT_MANIFEST) if (root / OUT_MANIFEST).exists() else ""
    print(
        json.dumps(
            {
                "decision_token": manifest["decision_token"],
                "validation_passed": manifest["validation_passed"],
                "validation_failures": failures,
                "logical_rows": summary["logical_rows"],
                "unique_users": summary["unique_users"],
                "cells": summary["cells"],
                "strict_valid": summary["strict_valid"],
                "invalid": summary["invalid"],
                "terminal_state_counts": summary["terminal_state_counts"],
                "denominator_totals": summary["denominator_totals"],
                "candidate_mismatches": len(candidate_mismatches),
                "identity_mismatches": len(identity_mismatches),
                "variant_defects": len(variant_defects),
                "frozen_hash_failures": frozen_hash_failures,
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

"""G2 history-length execution runner freeze.

This adapter consumes the frozen nested G2 request universe. Dry-run mode
validates request identity, G1 reuse, physical ordering, raw-evidence paths,
and accounting without calling Ollama. Future execution remains behind an
explicit --allow-api-calls flag and uses raw-first persistence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.extension import g1_primary_ranking_execution as g1
from src.extension import g2_history_length_sensitivity_protocol as g2_protocol
from src.extension.e0_parser import parse_extension_top10_response
from src.extension.e0_prompts import render_extension_prompt
from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    FUTURE_MODELS,
    METRIC_K,
    PARSER_CONTRACT_ID,
    PROMPT_LLMRANK,
    REQUESTED_OUTPUT_LENGTH,
    bool_text,
    sha256_json,
    sha256_text,
    stable_json,
)
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1bx_execution_stage1 import frame_hash, sort_key
from src.extension.g0_canonical_llm_canary import MODEL_FAMILY, RequestResult, call_ollama_generate, write_raw_response_file
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync, positive_label_exposed, read_jsonl
from src.utils.config import find_project_root


PHASE_ID = "g2_history_length_execution_freeze"
DECISION_FROZEN = "G2_EXECUTION_RUNNER_FROZEN"
DECISION_BLOCKED = "G2_EXECUTION_RUNNER_BLOCKED"
PROTOCOL_DECISION = "G2_NESTED_HISTORY_PROTOCOL_FROZEN"

CONFIG_PATH = Path("configs/g2_history_length_execution.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_PATH = Path("report/g2_history_length_execution_runner.md")
NOTES_PATH = Path("report/notes.md")
MANIFEST_PATH = MANIFEST_DIR / "g2_history_length_execution_freeze.json"

EXECUTION_ORDER_PATH = TABLE_DIR / "g2_history_length_execution_order.csv"
EXECUTION_ACCOUNTING_PATH = TABLE_DIR / "g2_history_length_execution_accounting.csv"
G1_EXECUTION_REUSE_PATH = TABLE_DIR / "g2_history_length_g1_execution_reuse.csv"

G1_REUSE_ACTION = g2_protocol.G1_REUSE_ACTION
NEW_CALL_ACTION = g2_protocol.NEW_CALL_ACTION
PRIMARY_VIEW = g2_protocol.PRIMARY_VIEW
LONG_VIEW = g2_protocol.LONG_VIEW
SECONDARY_VIEW = g2_protocol.SECONDARY_VIEW
TRUE_ZERO_VIEW = g2_protocol.TRUE_ZERO_VIEW

TERMINAL_STATES = (
    "STRICT_VALID",
    "INVALID_FORMAT",
    "INVALID_LENGTH",
    "INVALID_DUPLICATE",
    "INVALID_CANDIDATE",
    "HALLUCINATION_OR_OUT_OF_CANDIDATE",
    "BACKEND_FAILURE",
)
STRICT_STATUS_TO_TERMINAL = {
    "parsed_valid": "STRICT_VALID",
    "empty_output": "INVALID_FORMAT",
    "parse_failed": "INVALID_FORMAT",
    "invalid_schema": "INVALID_FORMAT",
    "invalid_length": "INVALID_LENGTH",
    "duplicate_indices": "INVALID_DUPLICATE",
    "unknown_indices": "HALLUCINATION_OR_OUT_OF_CANDIDATE",
    "missing_permutation_indices": "INVALID_CANDIDATE",
}


@dataclass(frozen=True)
class ExistingEvidenceState:
    request_records: list[dict[str, Any]]
    response_records: list[dict[str, Any]]
    terminal_by_id: dict[str, dict[str, Any]]
    interrupted_request_ids: set[str]
    validation_failures: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute the frozen G2 history-length request universe.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write pre-execution artifacts without inference.")
    parser.add_argument("--resume", action="store_true", help="Skip G1 reused and already-terminal G2 evidence.")
    parser.add_argument("--max-new-calls", type=int, default=None, help="Maximum newly executed Ollama calls in this invocation.")
    parser.add_argument("--allow-api-calls", action="store_true", help="Required before local Ollama metadata or generation calls.")
    parser.add_argument("--no-write", action="store_true", help="Build in memory without writing freeze artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    config = load_config(root, Path(args.config))
    validate_config(config)

    if not args.dry_run and not args.allow_api_calls:
        print("error=--allow-api-calls is required before any local Ollama execution path is enabled")
        return 1

    outputs = build_execution_freeze(root, config, validate_live_models=False)
    if args.dry_run or args.no_write:
        if not args.no_write:
            write_outputs(root, outputs)
            append_phase_notes(root, outputs, mode="dry-run")
        manifest = outputs["manifest"]
        print_summary(manifest, new_calls_this_invocation=0)
        return 0 if manifest["decision_token"] == DECISION_FROZEN else 1

    sent = execute_pending_requests(
        root,
        config,
        max_new_calls=args.max_new_calls,
        resume=bool(args.resume),
        call_fn=call_ollama_generate,
        sleep_fn=time.sleep,
    )
    print_summary(outputs["manifest"], new_calls_this_invocation=sent)
    return 0


def print_summary(manifest: Mapping[str, Any], *, new_calls_this_invocation: int) -> None:
    print(f"decision={manifest['decision_token']}")
    print(f"logical_applicable_requests={manifest['logical_applicable_requests']}")
    print(f"g1_reusable_requests={manifest['g1_reusable_requests']}")
    print(f"future_new_calls_required={manifest['future_new_calls_required']}")
    print(f"existing_g2_terminal_requests={manifest['existing_g2_terminal_requests']}")
    print(f"unresolved_new_requests={manifest['unresolved_new_requests']}")
    print(f"new_ollama_calls_in_this_invocation={new_calls_this_invocation}")


def build_execution_freeze(root: Path | None = None, config: Mapping[str, Any] | None = None, *, validate_live_models: bool = False) -> dict[str, Any]:
    project_root = root or find_project_root()
    execution_config = dict(config or load_config(project_root, CONFIG_PATH))
    validate_config(execution_config)

    protocol_manifest = read_json(project_root / str(execution_config["inputs"]["protocol_manifest"]))
    request_universe = read_csv(project_root / str(execution_config["inputs"]["request_universe"]))
    protocol_accounting = read_csv(project_root / str(execution_config["inputs"]["protocol_accounting"]))
    g1_execution_state = read_csv(project_root / str(execution_config["inputs"]["g1_execution_state"]))
    g1_strict_parsing = read_csv(project_root / str(execution_config["inputs"]["g1_strict_parsing"]))

    expected_digests = expected_model_digests(request_universe)
    model_preflight = (
        build_live_model_preflight(execution_config, expected_digests)
        if validate_live_models
        else frozen_model_preflight(expected_digests)
    )
    existing_state = inspect_existing_g2_evidence(project_root, execution_config, request_universe)
    g1_reuse = build_g1_execution_reuse(request_universe, g1_execution_state, g1_strict_parsing)
    execution_order = build_execution_order(request_universe, existing_state)
    execution_accounting = build_execution_accounting(request_universe, execution_order, g1_reuse, existing_state)
    validation_failures = validate_freeze(
        config=execution_config,
        protocol_manifest=protocol_manifest,
        request_universe=request_universe,
        protocol_accounting=protocol_accounting,
        g1_reuse=g1_reuse,
        execution_order=execution_order,
        execution_accounting=execution_accounting,
        model_preflight=model_preflight,
        existing_state=existing_state,
        require_live_model_match=validate_live_models,
    )
    tables = {
        "execution_order": execution_order,
        "execution_accounting": execution_accounting,
        "g1_execution_reuse": g1_reuse,
    }
    manifest = build_manifest(
        project_root,
        execution_config,
        protocol_manifest,
        tables,
        model_preflight,
        existing_state,
        validation_failures,
    )
    report = render_report(manifest, execution_accounting, model_preflight)
    return {"config": execution_config, "tables": tables, "manifest": manifest, "report": report}


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def validate_config(config: Mapping[str, Any]) -> None:
    if str(config["phase"]["id"]) != PHASE_ID:
        raise ValueError("unexpected G2 execution phase id")
    if str(config["phase"]["required_protocol_decision"]) != PROTOCOL_DECISION:
        raise ValueError("G2 execution runner must consume the nested protocol")
    forbidden = [
        "allow_local_ollama_inference",
        "allow_local_ollama_metadata",
        "allow_external_api_calls",
        "allow_downloads",
        "allow_network_downloads",
        "allow_training",
        "allow_tuning",
        "allow_g3",
        "allow_g4",
        "allow_g5",
    ]
    if any(bool(config["execution"].get(key, False)) for key in forbidden):
        raise ValueError("default execution config must be no-inference/no-network")
    if int(config["execution"]["scientific_retry_count"]) != 0:
        raise ValueError("structural invalidity retry count must remain zero")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def expected_model_digests(request_universe: pd.DataFrame) -> dict[str, str]:
    rows: dict[str, str] = {}
    for model, group in request_universe.groupby("model", sort=False):
        digests = sorted(set(group["model_digest"].astype(str)))
        if len(digests) != 1 or not digests[0]:
            raise ValueError(f"model digest not unique for {model}")
        rows[str(model)] = digests[0]
    return rows


def frozen_model_preflight(expected_digests: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for model in FUTURE_MODELS:
        rows.append(
            {
                "model": model,
                "model_family": MODEL_FAMILY[model],
                "expected_g1_digest": str(expected_digests[model]),
                "current_local_digest": "",
                "digest_match": "not_checked",
                "current_model_identity_status": "FROZEN_G1_DIGEST_REFERENCE_ONLY",
                "metadata_access_status": "DRY_RUN_NO_OLLAMA_METADATA_CALL",
                "llm_generation_call": "false",
                "status": "DRY_RUN_MODEL_CHECK_DEFERRED_UNTIL_EXECUTION",
            }
        )
    return pd.DataFrame(rows)


def build_live_model_preflight(config: Mapping[str, Any], expected_digests: Mapping[str, str]) -> pd.DataFrame:
    base_url = g1.normalize_loopback_url(str(config["execution"]["ollama_base_url"]), purpose="G2 Ollama endpoint")
    tags_payload = g1.local_http_json("GET", base_url + "/api/tags", None, timeout=10)
    tags_error = str(tags_payload.get("_error", ""))
    models_payload = tags_payload.get("models", [])
    if tags_error or not isinstance(models_payload, list):
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "model_family": MODEL_FAMILY[model],
                    "expected_g1_digest": str(expected_digests[model]),
                    "current_local_digest": "",
                    "digest_match": "false",
                    "current_model_identity_status": "OLLAMA_UNREACHABLE",
                    "metadata_access_status": "LOCAL_METADATA_FAILED",
                    "llm_generation_call": "false",
                    "status": "BLOCK_BEFORE_INFERENCE",
                }
                for model in FUTURE_MODELS
            ]
        )
    tags = g1.parse_ollama_tags(models_payload)
    rows = []
    for model in FUTURE_MODELS:
        tag = tags.get(model)
        expected = str(expected_digests[model])
        current = str((tag or {}).get("digest") or (tag or {}).get("id") or "")
        matched = bool(tag) and current == expected
        status = "MODEL_AVAILABLE_MATCHED" if matched else ("MODEL_NOT_INSTALLED" if tag is None else "MODEL_DIGEST_MISMATCH")
        rows.append(
            {
                "model": model,
                "model_family": MODEL_FAMILY[model],
                "expected_g1_digest": expected,
                "current_local_digest": current,
                "digest_match": bool_text(matched),
                "current_model_identity_status": status,
                "metadata_access_status": "LOCAL_METADATA_OK",
                "llm_generation_call": "false",
                "status": status if matched else "BLOCK_BEFORE_INFERENCE",
            }
        )
    return pd.DataFrame(rows)


def validate_model_preflight(model_preflight: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    if set(model_preflight["model"].astype(str)) != set(FUTURE_MODELS):
        failures.append("MODEL_PREFLIGHT_SET_MISMATCH")
    live_rows = model_preflight.loc[model_preflight["digest_match"].astype(str).isin({"true", "false"})]
    if not live_rows.empty and not live_rows["digest_match"].eq("true").all():
        failures.append("MODEL_DIGEST_MISMATCH_BLOCK_BEFORE_INFERENCE")
    return failures


def inspect_existing_g2_evidence(root: Path, config: Mapping[str, Any], request_universe: pd.DataFrame) -> ExistingEvidenceState:
    request_path = root / str(config["raw_evidence"]["request_jsonl"])
    response_path = root / str(config["raw_evidence"]["response_jsonl"])
    response_dir = root / str(config["raw_evidence"]["response_dir"])
    request_records = [dict(row) for row in read_jsonl(request_path)] if request_path.exists() else []
    response_records = [dict(row) for row in read_jsonl(response_path)] if response_path.exists() else []
    request_counts = count_by_request_id(request_records)
    response_counts = count_by_request_id(response_records)
    expected_new = set(request_universe.loc[request_universe["planned_action"].eq(NEW_CALL_ACTION), "canonical_request_id"].astype(str))
    universe_by_id = request_universe.set_index("canonical_request_id").to_dict("index")
    terminal_by_id: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    extra = sorted((set(request_counts) | set(response_counts)) - expected_new)
    if extra:
        failures.append(f"RAW_EVIDENCE_OUTSIDE_NEW_CALL_SET:{extra[0]}")
    duplicate_responses = sorted(request_id for request_id, count in response_counts.items() if count > 1)
    if duplicate_responses:
        failures.append(f"DUPLICATE_TERMINAL_LEDGER_ROWS:{duplicate_responses[0]}")

    for record in response_records:
        request_id = request_id_for_record(record)
        if request_id not in expected_new:
            continue
        row = universe_by_id[request_id]
        for key in ("dataset", "model", "prompt_strategy", "user_id", "history_length", "history_hash", "candidate_hash", "model_digest", "generation_config_hash", "parser_contract_hash"):
            if str(record.get(key, "")) != str(row.get(key, "")):
                failures.append(f"CONFLICTING_TERMINAL_IDENTITY:{request_id}:{key}")
        raw_file = root / str(row["g2_future_raw_response_path"])
        if not raw_file.exists():
            failures.append(f"TERMINAL_LEDGER_ROW_MISSING_RAW_FILE:{request_id}")
        else:
            try:
                raw_payload = json.loads(raw_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                failures.append(f"RAW_FILE_INVALID_JSON:{request_id}")
                raw_payload = {}
            if str(raw_payload.get("raw_record_hash", "")) != str(record.get("raw_record_hash", "")):
                failures.append(f"CONFLICTING_RAW_RECORD_HASH:{request_id}")
            if str(raw_payload.get("raw_response_hash", "")) != str(record.get("raw_response_hash", "")):
                failures.append(f"CONFLICTING_RAW_RESPONSE_HASH:{request_id}")
        terminal_by_id[request_id] = dict(record)

    if response_dir.exists():
        raw_files = list(response_dir.glob("*.json"))
        raw_ids = [path.stem for path in raw_files]
        raw_counts = {request_id: raw_ids.count(request_id) for request_id in set(raw_ids)}
        duplicate_raw = sorted(request_id for request_id, count in raw_counts.items() if count > 1)
        if duplicate_raw:
            failures.append(f"DUPLICATE_RAW_FILES:{duplicate_raw[0]}")
        for request_id in sorted(set(raw_ids)):
            if request_id not in response_counts:
                failures.append(f"RAW_FILE_WITHOUT_TERMINAL_LEDGER_ROW:{request_id}")

    interrupted = {request_id for request_id, count in request_counts.items() if count > 0 and response_counts.get(request_id, 0) == 0 and request_id in expected_new}
    return ExistingEvidenceState(request_records, response_records, terminal_by_id, interrupted, failures)


def request_id_for_record(record: Mapping[str, Any]) -> str:
    return str(record.get("canonical_request_id") or record.get("request_id") or "")


def count_by_request_id(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        request_id = request_id_for_record(record)
        if request_id:
            counts[request_id] = counts.get(request_id, 0) + 1
    return counts


def build_g1_execution_reuse(request_universe: pd.DataFrame, g1_execution_state: pd.DataFrame, g1_strict_parsing: pd.DataFrame) -> pd.DataFrame:
    execution_by_id = g1_execution_state.set_index("canonical_request_id").to_dict("index")
    parse_by_id = g1_strict_parsing.set_index("canonical_request_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    reusable = request_universe.loc[request_universe["planned_action"].eq(G1_REUSE_ACTION)].copy()
    for row in reusable.sort_values("request_order", key=lambda series: series.astype(int), kind="mergesort").to_dict("records"):
        g1_id = str(row["matched_g1_request_id"])
        execution = execution_by_id.get(g1_id, {})
        parsed = parse_by_id.get(g1_id, {})
        rows.append(
            {
                "g2_canonical_request_id": str(row["canonical_request_id"]),
                "g1_canonical_request_id": g1_id,
                "dataset": str(row["dataset"]),
                "user_id": str(row["user_id"]),
                "model": str(row["model"]),
                "prompt_strategy": str(row["prompt_strategy"]),
                "history_length": int(row["history_length"]),
                "history_hash": str(row["history_hash"]),
                "candidate_hash": str(row["candidate_hash"]),
                "positive_item_id": str(row["positive_item_id"]),
                "model_digest": str(row["model_digest"]),
                "generation_config_hash": str(row["generation_config_hash"]),
                "parser_contract_hash": str(row["parser_contract_hash"]),
                "analysis_memberships": str(row["reporting_view_memberships"]),
                "g1_raw_response_path": str(row["matched_g1_raw_response_path"]),
                "g1_raw_response_hash": str(row["matched_g1_raw_response_hash"]),
                "g1_evidence_origin": str(row["matched_g1_evidence_origin"]),
                "g1_terminal": str(execution.get("terminal", "")).lower(),
                "g1_terminal_state": normalize_terminal_state(str(parsed.get("parse_status", "")), str(execution.get("backend_success", "")).lower() == "true"),
                "reuse_status": "G1_EXACT_REUSE_TERMINAL_ZERO_CALL",
                "ollama_call_required": "false",
                "retention_policy": "TRACKED_G1_REUSE_INVENTORY_REFERENCES_EXTERNALIZED_G1_RAW_HASH",
            }
        )
    return pd.DataFrame(rows)


def normalize_terminal_state(parse_status: str, backend_success: bool) -> str:
    if not backend_success:
        return "BACKEND_FAILURE"
    return STRICT_STATUS_TO_TERMINAL.get(str(parse_status), "INVALID_FORMAT")


def build_execution_order(request_universe: pd.DataFrame, existing_state: ExistingEvidenceState) -> pd.DataFrame:
    frame = request_universe.copy()
    model_order = {model: index for index, model in enumerate(FUTURE_MODELS)}
    dataset_order = {dataset: index for index, dataset in enumerate(g2_protocol.DATASET_ORDER)}
    prompt_order = {prompt: index for index, prompt in enumerate(g2_protocol.PROMPT_ORDER)}
    frame["_model_order"] = frame["model"].map(model_order).astype(int)
    frame["_dataset_order"] = frame["dataset"].map(dataset_order).astype(int)
    frame["_prompt_order"] = frame["prompt_strategy"].map(prompt_order).astype(int)
    frame["_history_order"] = frame["history_length"].astype(int)
    frame["_user_sort"] = frame["user_id"].map(sort_key)
    frame = frame.sort_values(["_model_order", "_dataset_order", "_prompt_order", "_history_order", "_user_sort"], kind="mergesort").copy()
    frame["physical_execution_order"] = range(1, len(frame) + 1)

    def category(row: Mapping[str, Any]) -> str:
        request_id = str(row["canonical_request_id"])
        if str(row["planned_action"]) == G1_REUSE_ACTION:
            return "G1_REUSE_TERMINAL_ZERO_CALL"
        if request_id in existing_state.terminal_by_id:
            return "G2_TERMINAL_SKIP_ZERO_CALL"
        if request_id in existing_state.interrupted_request_ids:
            return "INTERRUPTED_NONTERMINAL_RESUMABLE"
        return "NEW_G2_CALL_PENDING"

    frame["execution_category"] = [category(row) for row in frame.to_dict("records")]
    frame["ollama_call_required"] = frame["execution_category"].eq("NEW_G2_CALL_PENDING") | frame["execution_category"].eq("INTERRUPTED_NONTERMINAL_RESUMABLE")
    frame["ollama_call_required"] = frame["ollama_call_required"].map(bool_text)
    frame["resume_action"] = frame["execution_category"].map(
        {
            "G1_REUSE_TERMINAL_ZERO_CALL": "SKIP_G1_REUSE",
            "G2_TERMINAL_SKIP_ZERO_CALL": "SKIP_ALREADY_TERMINAL_G2",
            "INTERRUPTED_NONTERMINAL_RESUMABLE": "RESUME_EXECUTE_AGAIN",
            "NEW_G2_CALL_PENDING": "EXECUTE_WHEN_ALLOWED",
        }
    )
    columns = [
        "physical_execution_order",
        "request_order",
        "canonical_request_id",
        "dataset",
        "task",
        "user_id",
        "model",
        "model_digest",
        "prompt_strategy",
        "prompt_contract",
        "prompt_hash",
        "history_length",
        "history_hash",
        "candidate_hash",
        "candidate_items",
        "positive_item_id",
        "generation_config_hash",
        "parser_contract_hash",
        "reporting_view_memberships",
        "primary_reporting_view",
        "planned_action",
        "execution_category",
        "ollama_call_required",
        "resume_action",
        "g2_future_raw_response_path",
        "request_identity_hash",
        "retention_policy",
    ]
    return frame[columns].reset_index(drop=True)


def build_execution_accounting(
    request_universe: pd.DataFrame,
    execution_order: pd.DataFrame,
    g1_reuse: pd.DataFrame,
    existing_state: ExistingEvidenceState,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_row(scope: str, dataset: str, model: str, prompt: str, history: str | int, group: pd.DataFrame, *, note: str) -> None:
        g1_count = int(group["execution_category"].eq("G1_REUSE_TERMINAL_ZERO_CALL").sum())
        g2_terminal = int(group["execution_category"].eq("G2_TERMINAL_SKIP_ZERO_CALL").sum())
        pending = int(group["execution_category"].isin(["NEW_G2_CALL_PENDING", "INTERRUPTED_NONTERMINAL_RESUMABLE"]).sum())
        rows.append(
            {
                "accounting_scope": scope,
                "dataset": dataset,
                "model": model,
                "prompt_strategy": prompt,
                "history_length": history,
                "analysis_membership": "DEDUPLICATED_EXECUTION",
                "logical_applicable_requests": len(group),
                "g1_reusable_requests": g1_count,
                "existing_g2_terminal_requests": g2_terminal,
                "new_call_budget_total": int(group["planned_action"].eq(NEW_CALL_ACTION).sum()),
                "unresolved_new_requests": pending,
                "future_ollama_calls_remaining": pending,
                "new_ollama_calls_in_this_invocation": 0,
                "retention_class": "TRACKED_SCIENTIFIC_FOREGROUND",
                "status": "RECONCILED",
                "note": note,
            }
        )

    append_row("GRAND_TOTAL", "ALL", "ALL", "ALL", "ALL", execution_order, note="Deduplicated canonical request universe.")
    for dataset, group in execution_order.groupby("dataset", sort=False):
        append_row("BY_DATASET", str(dataset), "ALL", "ALL", "ALL", group, note="Deduplicated requests by dataset.")
    for model, group in execution_order.groupby("model", sort=False):
        append_row("BY_MODEL", "ALL", str(model), "ALL", "ALL", group, note="Deduplicated requests by model.")
    for prompt, group in execution_order.groupby("prompt_strategy", sort=False):
        append_row("BY_PROMPT", "ALL", "ALL", str(prompt), "ALL", group, note="Deduplicated requests by prompt.")
    for history, group in execution_order.groupby("history_length", sort=False):
        append_row("BY_HISTORY_LENGTH", "ALL", "ALL", "ALL", int(history), group, note="Deduplicated requests by history length.")

    membership_rows: list[dict[str, Any]] = []
    for row in request_universe.to_dict("records"):
        memberships = json.loads(str(row["reporting_view_memberships"]))
        for membership in memberships:
            membership_rows.append({**row, "analysis_membership": membership})
    membership_frame = pd.DataFrame(membership_rows)
    for membership, group in membership_frame.groupby("analysis_membership", sort=False):
        rows.append(
            {
                "accounting_scope": "BY_ANALYSIS_MEMBERSHIP",
                "dataset": "ALL",
                "model": "ALL",
                "prompt_strategy": "ALL",
                "history_length": "ALL",
                "analysis_membership": str(membership),
                "logical_applicable_requests": len(group),
                "g1_reusable_requests": int(group["planned_action"].eq(G1_REUSE_ACTION).sum()),
                "existing_g2_terminal_requests": sum(1 for request_id in group["canonical_request_id"].astype(str) if request_id in existing_state.terminal_by_id),
                "new_call_budget_total": int(group["planned_action"].eq(NEW_CALL_ACTION).sum()),
                "unresolved_new_requests": sum(
                    1
                    for row_id, action in zip(group["canonical_request_id"].astype(str), group["planned_action"].astype(str))
                    if action == NEW_CALL_ACTION and row_id not in existing_state.terminal_by_id
                ),
                "future_ollama_calls_remaining": sum(
                    1
                    for row_id, action in zip(group["canonical_request_id"].astype(str), group["planned_action"].astype(str))
                    if action == NEW_CALL_ACTION and row_id not in existing_state.terminal_by_id
                ),
                "new_ollama_calls_in_this_invocation": 0,
                "retention_class": "TRACKED_SCIENTIFIC_FOREGROUND",
                "status": "REPORTING_VIEW_DENOMINATOR_ONLY",
                "note": "Analysis membership does not create additional physical inference calls.",
            }
        )
    return pd.DataFrame(rows)


def validate_freeze(
    *,
    config: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
    request_universe: pd.DataFrame,
    protocol_accounting: pd.DataFrame,
    g1_reuse: pd.DataFrame,
    execution_order: pd.DataFrame,
    execution_accounting: pd.DataFrame,
    model_preflight: pd.DataFrame,
    existing_state: ExistingEvidenceState,
    require_live_model_match: bool,
) -> list[str]:
    failures: list[str] = []
    expected = config["expected_accounting"]
    if str(protocol_manifest.get("decision_token", "")) != PROTOCOL_DECISION:
        failures.append("PROTOCOL_NOT_NESTED_FROZEN")
    if int(protocol_manifest.get("planned_applicable_requests", 0)) != int(expected["logical_applicable_requests"]):
        failures.append("PROTOCOL_REQUEST_TOTAL_MISMATCH")
    if len(request_universe) != int(expected["logical_applicable_requests"]):
        failures.append("REQUEST_UNIVERSE_TOTAL_MISMATCH")
    if request_universe["canonical_request_id"].nunique() != len(request_universe):
        failures.append("CANONICAL_REQUEST_IDS_NOT_UNIQUE")
    identity_columns = ["dataset", "user_id", "history_length", "prompt_strategy", "model", "population_label"]
    if request_universe.duplicated(identity_columns).any():
        failures.append("DUPLICATE_USER_H_PROMPT_MODEL_IDENTITY")
    if len(g1_reuse) != int(expected["exact_g1_reuse"]):
        failures.append(f"G1_REUSE_COUNT_MISMATCH:{len(g1_reuse)}")
    new_count = int(request_universe["planned_action"].eq(NEW_CALL_ACTION).sum())
    if new_count != int(expected["new_ollama_calls_required"]):
        failures.append(f"NEW_CALL_BUDGET_MISMATCH:{new_count}")
    by_dataset = {
        str(row["dataset"]): {
            "total": int(row["logical_applicable_requests"]),
            "reuse": int(row["g1_reusable_requests"]),
            "new": int(row["new_call_budget_total"]),
        }
        for row in execution_accounting.loc[execution_accounting["accounting_scope"].eq("BY_DATASET")].to_dict("records")
    }
    if by_dataset != dict(expected["by_dataset"]):
        failures.append(f"DATASET_ACCOUNTING_MISMATCH:{by_dataset}")
    model_values = {
        (
            int(row["logical_applicable_requests"]),
            int(row["g1_reusable_requests"]),
            int(row["new_call_budget_total"]),
        )
        for row in execution_accounting.loc[execution_accounting["accounting_scope"].eq("BY_MODEL")].to_dict("records")
    }
    if model_values != {(int(expected["per_model"]["total"]), int(expected["per_model"]["reuse"]), int(expected["per_model"]["new"]))}:
        failures.append(f"MODEL_ACCOUNTING_MISMATCH:{model_values}")
    prompt_values = {
        (
            int(row["logical_applicable_requests"]),
            int(row["g1_reusable_requests"]),
            int(row["new_call_budget_total"]),
        )
        for row in execution_accounting.loc[execution_accounting["accounting_scope"].eq("BY_PROMPT")].to_dict("records")
    }
    if prompt_values != {(int(expected["per_prompt"]["total"]), int(expected["per_prompt"]["reuse"]), int(expected["per_prompt"]["new"]))}:
        failures.append(f"PROMPT_ACCOUNTING_MISMATCH:{prompt_values}")
    if execution_order["physical_execution_order"].astype(int).tolist() != list(range(1, len(execution_order) + 1)):
        failures.append("PHYSICAL_EXECUTION_ORDER_NOT_CONTIGUOUS")
    h10 = request_universe.loc[
        request_universe["history_length"].astype(int).eq(10)
        & request_universe["reporting_view_memberships"].str.contains(PRIMARY_VIEW, regex=False)
        & request_universe["reporting_view_memberships"].str.contains(LONG_VIEW, regex=False)
    ]
    if h10.empty or h10["canonical_request_id"].duplicated().any():
        failures.append("B1_B2_H10_DEDUPLICATION_MISSING")
    if protocol_accounting.empty:
        failures.append("PROTOCOL_ACCOUNTING_MISSING")
    failures.extend(existing_state.validation_failures)
    if require_live_model_match:
        failures.extend(validate_model_preflight(model_preflight))
    return failures


def build_manifest(
    root: Path,
    config: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    model_preflight: pd.DataFrame,
    existing_state: ExistingEvidenceState,
    validation_failures: Sequence[str],
) -> dict[str, Any]:
    execution_order = tables["execution_order"]
    grand = tables["execution_accounting"].loc[tables["execution_accounting"]["accounting_scope"].eq("GRAND_TOTAL")].iloc[0]
    artifact_hashes = {}
    for rel in [
        CONFIG_PATH,
        Path("src/extension/g2_history_length_execution.py"),
        Path("tests/test_extension_g2_history_length_execution.py"),
        EXECUTION_ORDER_PATH,
        EXECUTION_ACCOUNTING_PATH,
        G1_EXECUTION_REUSE_PATH,
        REPORT_PATH,
    ]:
        path = root / rel
        if path.exists():
            artifact_hashes[rel.as_posix()] = sha256_file(path)
    manifest = {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_FROZEN if not validation_failures else DECISION_BLOCKED,
        "generated_at": utc_now(),
        "protocol_decision": str(protocol_manifest.get("decision_token", "")),
        "protocol_manifest": str(config["inputs"]["protocol_manifest"]),
        "protocol_request_universe_hash": str(protocol_manifest.get("request_universe_hash", "")),
        "logical_applicable_requests": int(grand["logical_applicable_requests"]),
        "g1_reusable_requests": int(grand["g1_reusable_requests"]),
        "future_new_calls_required": int(grand["new_call_budget_total"]),
        "existing_g2_terminal_requests": int(grand["existing_g2_terminal_requests"]),
        "unresolved_new_requests": int(grand["unresolved_new_requests"]),
        "interrupted_nonterminal_requests": len(existing_state.interrupted_request_ids),
        "new_ollama_calls_in_this_invocation": 0,
        "dry_run_generation_calls": 0,
        "model_preflight": model_preflight.to_dict("records"),
        "terminal_states": list(TERMINAL_STATES),
        "generation_contract": {
            "temperature": float(config["execution"]["temperature"]),
            "num_predict": int(config["execution"]["max_output_tokens"]["ranking"]),
            "stream": bool(config["execution"]["stream"]),
            "timeout_seconds": int(config["execution"]["timeout_seconds"]),
            "format": str(config["execution"]["response_format"]),
            "backend_retry_count": int(config["execution"]["backend_retry_count"]),
            "backend_retry_backoff_seconds": list(config["execution"]["backend_retry_backoff_seconds"]),
            "max_attempts": int(config["execution"]["max_attempts"]),
            "scientific_retry_count": int(config["execution"]["scientific_retry_count"]),
            "structural_invalidity_retry_count": int(config["execution"]["structural_invalidity_retry_count"]),
        },
        "parser_contract": PARSER_CONTRACT_ID,
        "parser_contract_hash": str(protocol_manifest.get("parser_contract_hash", "")),
        "physical_order": str(config["execution"]["physical_execution_order"]),
        "canonical_logical_order": str(config["execution"]["canonical_logical_order"]),
        "analysis_memberships": [PRIMARY_VIEW, LONG_VIEW, SECONDARY_VIEW, TRUE_ZERO_VIEW],
        "tables": {
            "execution_order": EXECUTION_ORDER_PATH.as_posix(),
            "execution_accounting": EXECUTION_ACCOUNTING_PATH.as_posix(),
            "g1_execution_reuse": G1_EXECUTION_REUSE_PATH.as_posix(),
        },
        "raw_evidence_paths": dict(config["raw_evidence"]),
        "retention": dict(config["retention"]),
        "table_hashes": {path.as_posix(): frame_hash(frame) for path, frame in zip([EXECUTION_ORDER_PATH, EXECUTION_ACCOUNTING_PATH, G1_EXECUTION_REUSE_PATH], tables.values())},
        "artifact_hashes": artifact_hashes,
        "protected_path_policy": {
            "original_repo_modified": False,
            "paper_modified": False,
            "data_raw_modified": False,
            "g0_raw_evidence_modified": False,
            "g1_raw_evidence_modified": False,
            "g1_scientific_foreground_modified": False,
        },
        "no_inference_guards": {
            "llm_generation_calls_executed": 0,
            "ollama_metadata_calls_executed_in_dry_run": 0,
            "external_api_calls_executed": 0,
            "downloads_executed": 0,
            "training_runs_executed": 0,
            "tuning_runs_executed": 0,
            "g3_requests_executed": 0,
            "g4_requests_executed": 0,
            "g5_requests_executed": 0,
            "metrics_generated": False,
        },
        "validation_failures": list(validation_failures),
        "validation_passed": not validation_failures,
    }
    return manifest


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for rel in [TABLE_DIR, MANIFEST_DIR, REPORT_PATH.parent]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    outputs["tables"]["execution_order"].to_csv(root / EXECUTION_ORDER_PATH, index=False, lineterminator="\n")
    outputs["tables"]["execution_accounting"].to_csv(root / EXECUTION_ACCOUNTING_PATH, index=False, lineterminator="\n")
    outputs["tables"]["g1_execution_reuse"].to_csv(root / G1_EXECUTION_REUSE_PATH, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8", newline="\n")
    manifest = dict(outputs["manifest"])
    for rel in [EXECUTION_ORDER_PATH, EXECUTION_ACCOUNTING_PATH, G1_EXECUTION_REUSE_PATH, REPORT_PATH]:
        manifest["artifact_hashes"][rel.as_posix()] = sha256_file(root / rel)
    manifest["table_hashes"] = {
        EXECUTION_ORDER_PATH.as_posix(): frame_hash(outputs["tables"]["execution_order"]),
        EXECUTION_ACCOUNTING_PATH.as_posix(): frame_hash(outputs["tables"]["execution_accounting"]),
        G1_EXECUTION_REUSE_PATH.as_posix(): frame_hash(outputs["tables"]["g1_execution_reuse"]),
    }
    (root / MANIFEST_PATH).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    outputs["manifest"].clear()
    outputs["manifest"].update(manifest)


def append_phase_notes(root: Path, outputs: Mapping[str, Any], *, mode: str) -> None:
    manifest = outputs["manifest"]
    text = (
        f"\n## {utc_now()} - G2 history-length execution runner freeze\n\n"
        f"- Decision: `{manifest['decision_token']}`.\n"
        f"- Mode: `{mode}`; no Ollama generation or metadata calls were executed.\n"
        f"- Logical applicable requests: `{manifest['logical_applicable_requests']}`; G1 reusable: `{manifest['g1_reusable_requests']}`; future new calls: `{manifest['future_new_calls_required']}`.\n"
        f"- Existing G2 terminal requests: `{manifest['existing_g2_terminal_requests']}`; unresolved new requests: `{manifest['unresolved_new_requests']}`.\n"
        "- Created pre-execution runner artifacts for physical order, execution accounting, and exact G1 execution reuse.\n"
        "- No G2B inference, G3/G4/G5, metrics, downloads, training, tuning, commit, or G0/G1 artifact edits were performed.\n"
    )
    with (root / NOTES_PATH).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def render_report(manifest: Mapping[str, Any], accounting: pd.DataFrame, model_preflight: pd.DataFrame) -> str:
    dataset = accounting.loc[accounting["accounting_scope"].eq("BY_DATASET")][
        ["dataset", "logical_applicable_requests", "g1_reusable_requests", "new_call_budget_total", "unresolved_new_requests"]
    ]
    model = accounting.loc[accounting["accounting_scope"].eq("BY_MODEL")][
        ["model", "logical_applicable_requests", "g1_reusable_requests", "new_call_budget_total", "unresolved_new_requests"]
    ]
    prompt = accounting.loc[accounting["accounting_scope"].eq("BY_PROMPT")][
        ["prompt_strategy", "logical_applicable_requests", "g1_reusable_requests", "new_call_budget_total", "unresolved_new_requests"]
    ]
    return "\n".join(
        [
            "# G2 History-Length Execution Runner",
            "",
            "## PRECONDITION",
            f"- Frozen protocol decision: `{manifest['protocol_decision']}`.",
            f"- Logical applicable requests: `{manifest['logical_applicable_requests']}`; G1 reuse: `{manifest['g1_reusable_requests']}`; future new calls: `{manifest['future_new_calls_required']}`.",
            "",
            "## RUNNER",
            "- `src/extension/g2_history_length_execution.py` is a separated execution adapter over the frozen request universe.",
            "- Dry run writes only pre-execution artifacts and performs zero generation calls.",
            "",
            "## CANONICAL IDENTITY",
            "- Identity is inherited from the frozen G2 request universe and includes dataset, user, model, prompt, H, history hash, candidate hash, target, generation contract hash, parser contract hash, and model digest.",
            "",
            "## ANALYSIS MEMBERSHIP",
            "- Requests can belong to PRIMARY_B1_GE10, LONG_HISTORY_B2_GE20, SECONDARY_FULL_CURVE_GE20, or FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC without changing canonical identity.",
            "",
            "## G1 REUSE",
            "- G1 reused requests are terminal reusable evidence and consume zero Ollama calls.",
            "",
            "## REQUEST ACCOUNTING",
            "By dataset:",
            dataset.to_csv(index=False, lineterminator="\n").strip(),
            "By model:",
            model.to_csv(index=False, lineterminator="\n").strip(),
            "By prompt:",
            prompt.to_csv(index=False, lineterminator="\n").strip(),
            "",
            "## MODEL PREFLIGHT",
            model_preflight.to_csv(index=False, lineterminator="\n").strip(),
            "",
            "## EXECUTION ORDER",
            f"- Physical order: `{manifest['physical_order']}`; logical order: `{manifest['canonical_logical_order']}`.",
            "",
            "## GENERATION CONTRACT",
            f"- `{stable_json(manifest['generation_contract'])}`",
            "",
            "## RETRY / TERMINAL CONTRACT",
            f"- Terminal states: `{';'.join(manifest['terminal_states'])}`. Backend failures retry up to the frozen backend retry contract; structural invalidity has zero scientific retry.",
            "",
            "## RAW-FIRST PERSISTENCE",
            "- Future execution records request intent, calls the backend, persists raw response JSON before parsing, then writes terminal ledger state.",
            "",
            "## RESUME",
            "- `--resume` skips G1 reuse and already-terminal G2 rows; interrupted non-terminal requests may be retried.",
            "",
            "## ORPHAN POLICY",
            "- Ambiguous terminal evidence blocks; only request-without-response is resumable without repair.",
            "",
            "## RETENTION",
            "- Foreground runner artifacts are Git-eligible. Raw responses and large per-request ledgers remain externalized evidence.",
            "",
            "## DRY RUN",
            "- Dry run constructs the full universe, reuse inventory, physical order, accounting, model reference rows, and validates zero inference calls.",
            "",
            "## TESTS",
            "- Focused tests are in `tests/test_extension_g2_history_length_execution.py`.",
            "",
            "## EXECUTION COMMAND",
            "```bash",
            "python src/extension/g2_history_length_execution.py --config configs/g2_history_length_execution.yaml --allow-api-calls --resume --max-new-calls 5000",
            "```",
            "",
            str(manifest["decision_token"]),
            "",
        ]
    )


def select_pending_execution_rows(execution_order: pd.DataFrame, *, max_new_calls: int | None, resume: bool) -> pd.DataFrame:
    if max_new_calls is not None and max_new_calls < 0:
        raise ValueError("--max-new-calls must be non-negative")
    pending = execution_order.loc[
        execution_order["execution_category"].isin(["NEW_G2_CALL_PENDING", "INTERRUPTED_NONTERMINAL_RESUMABLE"])
    ].copy()
    if not resume:
        pending = pending.loc[pending["execution_category"].eq("NEW_G2_CALL_PENDING")]
    pending = pending.sort_values("physical_execution_order", kind="mergesort")
    if max_new_calls is not None:
        pending = pending.head(max_new_calls)
    return pending


def execute_pending_requests(
    root: Path,
    config: Mapping[str, Any],
    *,
    max_new_calls: int | None,
    resume: bool,
    call_fn: Callable[[str, Mapping[str, Any], int], RequestResult],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    if max_new_calls is not None and max_new_calls < 0:
        raise ValueError("--max-new-calls must be non-negative")
    protocol_manifest = read_json(root / str(config["inputs"]["protocol_manifest"]))
    request_universe = read_csv(root / str(config["inputs"]["request_universe"]))
    expected_digests = expected_model_digests(request_universe)
    preflight = build_live_model_preflight(config, expected_digests)
    model_failures = validate_model_preflight(preflight)
    if model_failures:
        raise RuntimeError(";".join(model_failures))
    existing = inspect_existing_g2_evidence(root, config, request_universe)
    if existing.validation_failures:
        raise RuntimeError(";".join(existing.validation_failures))
    order = build_execution_order(request_universe, existing)
    pending = select_pending_execution_rows(order, max_new_calls=max_new_calls, resume=resume)
    sent = 0
    contexts = build_prompt_context_cache(root, config)
    for row in pending.to_dict("records"):
        context = build_context_for_request(row, contexts)
        payload = build_payload(config, row, context)
        request_record = build_request_record(row, payload)
        append_jsonl_fsync(root / str(config["raw_evidence"]["request_jsonl"]), request_record)
        append_jsonl_fsync(root / str(config["raw_evidence"]["event_jsonl"]), {"event": "request_recorded_before_send", "phase_id": PHASE_ID, "canonical_request_id": row["canonical_request_id"], "timestamp_utc": utc_now()})
        result, attempts = call_with_backend_retries(root, config, row, payload, call_fn=call_fn, sleep_fn=sleep_fn)
        raw_record = build_unparsed_raw_record(row, result, attempts)
        write_raw_response_file(root / str(row["g2_future_raw_response_path"]), raw_record)
        terminal_record = build_terminal_record(row, raw_record)
        append_jsonl_fsync(root / str(config["raw_evidence"]["response_jsonl"]), terminal_record)
        append_jsonl_fsync(root / str(config["raw_evidence"]["event_jsonl"]), {"event": "terminal_response_recorded_after_raw_persistence", "phase_id": PHASE_ID, "canonical_request_id": row["canonical_request_id"], "terminal_state": terminal_record["terminal_state"], "timestamp_utc": utc_now()})
        sent += 1
    if int(protocol_manifest.get("planned_applicable_requests", 0)) != len(request_universe):
        raise RuntimeError("protocol/request universe mismatch")
    return sent


def build_prompt_context_cache(root: Path, config: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    protocol_config = g2_protocol.load_config(root, Path(str(config["inputs"]["protocol_config"])))
    g1_config = g1.load_config(root, Path(protocol_config["inputs"]["g1_config"]))
    g1_inputs = g1.load_inputs(root, g1_config)
    return {"_g1_inputs": {"value": g1_inputs}, **g2_protocol.build_user_contexts(g1_inputs)}


def build_context_for_request(row: Mapping[str, Any], contexts: Mapping[Any, Any]) -> dict[str, Any]:
    g1_inputs = contexts["_g1_inputs"]["value"]
    base = contexts[(str(row["dataset"]), str(row["user_id"]))]
    history_ids = g2_protocol.visible_history_ids(base["full_history_ids"], int(row["history_length"]))
    history_items = g2_protocol.history_items_for(g1_inputs, str(row["dataset"]), history_ids)
    bundle = render_extension_prompt(
        prompt_strategy=str(row["prompt_strategy"]),
        dataset_label=str(g1_inputs.specs[str(row["dataset"])].display_name),
        candidate_items=list(base["ranking_candidates"]),
        history_items=history_items,
        history_length=int(row["history_length"]),
    )
    if bundle.prompt_hash != str(row["prompt_hash"]):
        raise RuntimeError(f"prompt hash mismatch for {row['canonical_request_id']}")
    return {
        "system_message": bundle.system_message,
        "user_message": bundle.user_message,
        "positive_item_id": str(base["positive_item_id"]),
    }


def build_payload(config: Mapping[str, Any], row: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "model": str(row["model"]),
        "system": str(context["system_message"]),
        "prompt": str(context["user_message"]),
        "stream": bool(config["execution"]["stream"]),
        "format": str(config["execution"]["response_format"]),
        "options": {
            "temperature": float(config["execution"]["temperature"]),
            "num_predict": int(config["execution"]["max_output_tokens"][str(row["task"])]),
        },
    }
    if positive_label_exposed(json.dumps(payload, ensure_ascii=False, sort_keys=True), str(context["positive_item_id"])):
        raise ValueError(f"positive label exposed in prompt payload: {row['canonical_request_id']}")
    return payload


def build_request_record(row: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "canonical_request_id": str(row["canonical_request_id"]),
        "request_id": str(row["canonical_request_id"]),
        "physical_execution_order": int(row["physical_execution_order"]),
        "dataset": str(row["dataset"]),
        "task": str(row.get("task", "ranking")),
        "user_id": str(row["user_id"]),
        "model": str(row["model"]),
        "prompt_strategy": str(row["prompt_strategy"]),
        "history_length": int(row["history_length"]),
        "history_hash": str(row["history_hash"]),
        "candidate_hash": str(row["candidate_hash"]),
        "model_digest": str(row["model_digest"]),
        "generation_config_hash": str(row["generation_config_hash"]),
        "parser_contract_hash": str(row["parser_contract_hash"]),
        "payload_hash": sha256_json(payload),
        "request_payload": payload,
        "timestamp_before_send_utc": utc_now(),
    }


def call_with_backend_retries(
    root: Path,
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    call_fn: Callable[[str, Mapping[str, Any], int], RequestResult],
    sleep_fn: Callable[[float], None],
) -> tuple[RequestResult, list[dict[str, Any]]]:
    retry_count = int(config["execution"]["backend_retry_count"])
    backoffs = [float(value) for value in config["execution"]["backend_retry_backoff_seconds"]]
    attempts: list[dict[str, Any]] = []
    result: RequestResult | None = None
    for attempt in range(1, retry_count + 2):
        result = call_fn(str(config["execution"]["ollama_base_url"]), payload, int(config["execution"]["timeout_seconds"]))
        attempts.append(
            {
                "backend_attempt_number": attempt,
                "backend_success": bool(result.backend_success),
                "backend_status": "BACKEND_SUCCESS" if result.backend_success else "BACKEND_FAILURE",
                "backend_error_type": result.backend_error_type,
                "backend_error_message": result.backend_error_message,
                "request_started_at_utc": result.started_at,
                "request_completed_at_utc": result.completed_at,
                "runtime_seconds": result.duration_seconds,
            }
        )
        append_jsonl_fsync(root / str(config["raw_evidence"]["event_jsonl"]), {"event": "backend_attempt_completed", "phase_id": PHASE_ID, "canonical_request_id": row["canonical_request_id"], **attempts[-1], "timestamp_utc": utc_now()})
        if result.backend_success:
            break
        if attempt <= retry_count:
            sleep_fn(backoffs[attempt - 1])
    if result is None:
        raise RuntimeError("backend retry loop did not execute")
    return result, attempts


def build_unparsed_raw_record(row: Mapping[str, Any], result: RequestResult, attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    record = {
        "phase_id": PHASE_ID,
        "canonical_request_id": str(row["canonical_request_id"]),
        "request_id": str(row["canonical_request_id"]),
        "dataset": str(row["dataset"]),
        "task": str(row.get("task", "ranking")),
        "user_id": str(row["user_id"]),
        "model": str(row["model"]),
        "model_digest": str(row["model_digest"]),
        "prompt_strategy": str(row["prompt_strategy"]),
        "history_length": int(row["history_length"]),
        "history_hash": str(row["history_hash"]),
        "candidate_hash": str(row["candidate_hash"]),
        "generation_config_hash": str(row["generation_config_hash"]),
        "parser_contract_hash": str(row["parser_contract_hash"]),
        "backend_success": bool(result.backend_success),
        "backend_status": "BACKEND_SUCCESS" if result.backend_success else "BACKEND_FAILURE",
        "backend_attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "attempts": list(attempts),
        "runtime_seconds": result.duration_seconds,
        "raw_response": result.response_text,
        "raw_response_hash": sha256_text(result.response_text),
        "raw_backend_response": result.raw_backend_response,
        "request_started_at_utc": result.started_at,
        "request_completed_at_utc": result.completed_at,
        "raw_response_recorded_before_parse": True,
    }
    record["raw_record_hash"] = sha256_json(record)
    return record


def build_terminal_record(row: Mapping[str, Any], raw_record: Mapping[str, Any]) -> dict[str, Any]:
    backend_success = bool(raw_record.get("backend_success", False))
    parse_status = "backend_failure_not_parsed"
    strict_valid = False
    output_length = 0
    duplicate_count = 0
    hallucination_count = 0
    parse_error = ""
    ranked_candidate_indices = []
    ranked_item_ids = []
    if backend_success:
        parsed = parse_extension_top10_response(str(raw_record.get("raw_response", "")), candidate_items=json.loads(str(row["candidate_items"])))
        parse_status = parsed.status
        strict_valid = bool(parsed.parse_valid)
        output_length = len(parsed.output_indices)
        duplicate_count = int(parsed.duplicate_count)
        hallucination_count = int(getattr(parsed, "unknown_item_count", getattr(parsed, "unknown_count", 0)))
        parse_error = parsed.error
        ranked_candidate_indices = list(parsed.output_indices)
        ranked_item_ids = list(parsed.output_items)
    terminal_state = normalize_terminal_state(parse_status, backend_success)
    record = {
        **dict(raw_record),
        "terminal": True,
        "terminal_state": terminal_state,
        "strict_parse_valid": strict_valid,
        "parse_status": parse_status,
        "output_length": output_length,
        "duplicate_count": duplicate_count,
        "hallucination_count": hallucination_count,
        "ranked_candidate_indices": stable_json(ranked_candidate_indices),
        "ranked_item_ids": stable_json(ranked_item_ids),
        "parse_error": parse_error,
        "structural_invalidity_retried": False,
        "raw_response_recorded_before_parse": True,
    }
    record["terminal_record_hash"] = sha256_json(record)
    return record


if __name__ == "__main__":
    raise SystemExit(main())

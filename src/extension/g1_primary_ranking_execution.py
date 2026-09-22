"""G1 primary-ranking execution over the frozen extension universe.

This phase is deliberately narrow: it executes only canonical primary ranking
requests, reuses validated G0 ranking evidence by exact identity, preserves raw
model output before parsing, and computes fail-strict utility plus diagnostic
beyond-utility readouts.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import numpy as np
import pandas as pd
import yaml

from src.extension.e0_parser import parse_extension_top10_response
from src.extension.e0_prompts import render_extension_prompt
from src.extension.e0_protocol import (
    ACTIVE_EXTENSION_DATASETS,
    CANDIDATE_COUNT,
    FUTURE_MODELS,
    METRIC_K,
    PARSER_CONTRACT_ID,
    PROMPT_STRATEGIES,
    REQUESTED_OUTPUT_LENGTH,
    bool_text,
    item_list_hash,
    semantic_hash_for_row,
    sha256_json,
    sha256_text,
    stable_json,
)
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1a_offline_metric_closure import (
    average_recommendation_popularity,
    calibration_at_k,
    exposure_counts,
    gini_index,
    ild_jaccard_at_k,
    jain_index,
    long_tail_rate_at_k,
    ndcg_at_k_from_rank,
    oic_at_k,
    reciprocal_rank,
    self_information_at_k,
    shannon_category_entropy_at_k,
)
from src.extension.f1arr_popreo_dpd_recovery import dpd_at_k_from_ranks, popreo_at_k_from_hits
from src.extension.f1bx_execution_stage1 import (
    BEYOND_METRICS,
    TOP_K,
    DatasetMaterial,
    build_dataset_material,
    dataset_specs,
    format_float,
    frame_hash,
    per_user_stats as baseline_per_user_stats,
    read_csv,
    read_json,
    sort_key,
)
from src.extension.f1bz_final_traditional_execution import (
    foodcom_final_training_events,
    original_four_training_events,
)
from src.extension.f1bzq_authoritative_reranking_pool_materialization import load_title_map
from src.extension.g0_canonical_llm_canary import (
    G0_RAW_PROTOCOL,
    MODEL_FAMILY,
    RequestResult,
    call_ollama_generate,
    candidate_items_from_frame,
    canonical_request_id,
    classify_state,
    count_by_id,
    git_output,
    gpu_summary,
    physical_ram_gb,
    single_record_by_id,
    title_for_item,
    utc_now,
    write_raw_response_file,
)
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync, positive_label_exposed, read_jsonl
from src.utils.config import find_project_root


PHASE_ID = "g1_primary_ranking_execution"
DECISION_VALIDATED = "G1_PRIMARY_RANKING_VALIDATED"
DECISION_BLOCKED = "G1_PRIMARY_RANKING_BLOCKED"
SCIENTIFIC_BASE_CHECKPOINT = "5dfec10537a8acd1404c586323a3935add48bc2a"
TASK = "ranking"
G1_RAW_PROTOCOL = G0_RAW_PROTOCOL

DATASET_ORDER = tuple(ACTIVE_EXTENSION_DATASETS)
MODEL_ORDER = tuple(FUTURE_MODELS)
PROMPT_ORDER = tuple(PROMPT_STRATEGIES)

CONFIG_PATH = Path("configs/g1_primary_ranking_execution.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_PATH = Path("report/g1_primary_ranking_execution.md")
NOTES_PATH = Path("report/notes.md")
RAW_DIR = Path("results/llm/extension_canonical/g1_primary_ranking")
REQUEST_JSONL = RAW_DIR / "requests.jsonl"
RESPONSE_JSONL = RAW_DIR / "responses.jsonl"
EVENT_JSONL = RAW_DIR / "events.jsonl"
RESPONSE_DIR = RAW_DIR / "responses"

OUTPUT_TABLES = {
    "model_preflight": TABLE_DIR / "g1p_local_ollama_preflight.csv",
    "request_universe": TABLE_DIR / "g1_primary_ranking_request_universe.csv",
    "physical_execution_order": TABLE_DIR / "g1_primary_ranking_physical_execution_order.csv",
    "g0_reuse_inventory": TABLE_DIR / "g1_primary_ranking_g0_reuse_inventory.csv",
    "execution_state": TABLE_DIR / "g1_primary_ranking_execution_state.csv",
    "raw_response_inventory": TABLE_DIR / "g1_primary_ranking_raw_response_inventory.csv",
    "strict_parsing": TABLE_DIR / "g1_primary_ranking_strict_parsing.csv",
    "per_request_stats": TABLE_DIR / "g1_primary_ranking_per_request_stats.csv",
    "recommendations": TABLE_DIR / "g1_primary_ranking_recommendations.csv",
    "utility": TABLE_DIR / "g1_primary_ranking_utility.csv",
    "complete_case": TABLE_DIR / "g1_primary_ranking_complete_case.csv",
    "beyond_utility": TABLE_DIR / "g1_primary_ranking_beyond_utility.csv",
    "reliability": TABLE_DIR / "g1_primary_ranking_reliability.csv",
    "foodcom_warm_cold": TABLE_DIR / "g1_primary_ranking_foodcom_warm_cold.csv",
    "runtime": TABLE_DIR / "g1_primary_ranking_runtime.csv",
    "completeness": TABLE_DIR / "g1_primary_ranking_30_cell_completeness.csv",
    "validation": TABLE_DIR / "g1_primary_ranking_validation.csv",
}

OUTPUT_MANIFESTS = {
    "request_manifest": MANIFEST_DIR / "g1_primary_ranking_request_manifest.json",
    "execution_manifest": MANIFEST_DIR / "g1_primary_ranking_execution_manifest.json",
    "model_runtime_provenance": MANIFEST_DIR / "g1_primary_ranking_model_runtime_provenance.json",
    "g0_reuse": MANIFEST_DIR / "g1_primary_ranking_g0_reuse_manifest.json",
    "resume_idempotence": MANIFEST_DIR / "g1_primary_ranking_resume_idempotence_manifest.json",
    "budget_invariance": MANIFEST_DIR / "g1_primary_ranking_budget_invariance_manifest.json",
}

REQUIRED_BUDGET = {
    "CORE_MANDATORY_NOW_REQUESTS": 29292,
    "REQUIRED_SENSITIVITY_REQUESTS": 21000,
    "MANDATORY_RAW_REQUESTS_TOTAL": 50292,
    "TRAINING_DEPENDENT_REQUESTS": 29292,
    "CONDITIONAL_REQUESTS": 9000,
    "OPTIONAL_REQUESTS": 12000,
    "CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL": 100584,
}

EXPECTED_DATASET_USERS = {
    "beauty": 990,
    "sports": 892,
    "ml1m": 1000,
    "lastfm": 1000,
    "foodcom": 1000,
}

STRICT_STATUS_MAP = {
    "parsed_valid": "STRICT_VALID",
    "invalid_length": "INVALID_LENGTH",
    "duplicate_indices": "INVALID_DUPLICATE",
    "unknown_indices": "INVALID_CANDIDATE",
    "missing_permutation_indices": "INVALID_CANDIDATE",
    "parse_failed": "INVALID_FORMAT",
    "invalid_schema": "INVALID_FORMAT",
    "empty_output": "INVALID_FORMAT",
}

LOCAL_METADATA_OK = "LOCAL_METADATA_OK"
LOCAL_METADATA_CHECK_NOT_PERMITTED = "LOCAL_METADATA_CHECK_NOT_PERMITTED"
OLLAMA_UNREACHABLE = "OLLAMA_UNREACHABLE"
MODEL_NOT_INSTALLED = "MODEL_NOT_INSTALLED"
MODEL_DIGEST_MISMATCH = "MODEL_DIGEST_MISMATCH"
MODEL_AVAILABLE_MATCHED = "MODEL_AVAILABLE_MATCHED"
G0_PROVENANCE_AVAILABLE = "G0_PROVENANCE_AVAILABLE"
G0_PROVENANCE_MISSING = "G0_PROVENANCE_MISSING"


@dataclass(frozen=True)
class G1Inputs:
    config: dict[str, Any]
    specs: dict[str, Any]
    ranking_candidates: dict[str, pd.DataFrame]
    title_maps: dict[str, dict[str, str]]
    history_by_dataset: dict[str, dict[str, list[str]]]
    f0_primary: pd.DataFrame
    f0_budget: pd.DataFrame
    f0_manifest: dict[str, Any]
    f0r_manifest: dict[str, Any]
    f0_inference_contract: dict[str, Any]
    g0_request_matrix: pd.DataFrame
    g0_execution: pd.DataFrame
    g0_parse_validation: pd.DataFrame
    g0_response_records: list[dict[str, Any]]
    g0_request_manifest: dict[str, Any]
    g0_execution_manifest: dict[str, Any]
    f1a_manifest: dict[str, Any]
    f1arr_manifest: dict[str, Any]
    f1bzq_manifest: dict[str, Any]
    baseline_recommendations: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate G1 primary ranking execution.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--resume", action="store_true", help="Skip completed canonical requests by exact identity.")
    parser.add_argument("--validate-only", action="store_true", help="Rebuild G1 readouts from existing terminal evidence.")
    parser.add_argument("--no-write", action="store_true", help="Run in memory without writing reports, manifests, tables, or notes.")
    parser.add_argument("--allow-api-calls", action="store_true", help="Required consent flag for local Ollama inference only.")
    parser.add_argument("--max-new-calls", type=int, default=None, help="Optional batch cap for new G1 Ollama calls.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    if not args.validate_only and not args.allow_api_calls:
        print("error=--allow-api-calls is required before any local Ollama inference path is enabled")
        return 1
    outputs = run_g1(
        root,
        Path(args.config),
        resume=bool(args.resume),
        validate_only=bool(args.validate_only),
        max_new_calls=args.max_new_calls,
    )
    if not args.no_write:
        write_outputs(root, outputs)
        append_phase_notes(root, outputs, validate_only=bool(args.validate_only))
    manifest = outputs["manifests"]["execution_manifest"]
    print(f"decision={manifest['decision_token']}")
    print(f"planned_requests={manifest['planned_requests']}")
    print(f"g0_reused_ranking_requests={manifest['g0_reused_ranking_requests']}")
    print(f"new_g1_terminal_requests={manifest['new_g1_terminal_requests']}")
    print(f"new_g1_ollama_calls_expected={manifest['new_g1_ollama_calls_expected']}")
    print(f"llm_inference_calls_executed_in_this_invocation={manifest['llm_inference_calls_executed_in_this_invocation']}")
    print(f"terminal_requests={manifest['terminal_requests']}")
    print(f"strict_valid={manifest['strict_valid_requests']}")
    print(f"validation_passed={str(manifest['validation_passed']).lower()}")
    return 0 if manifest["decision_token"] == DECISION_VALIDATED else 1


def run_g1(
    root: Path,
    config_path: Path,
    *,
    resume: bool,
    validate_only: bool,
    max_new_calls: int | None,
) -> dict[str, Any]:
    ensure_output_dirs(root)
    config = load_config(root, config_path)
    validate_local_endpoint(config)
    inputs = load_inputs(root, config)
    initial_git_status = git_output(root, ["status", "--short", "--untracked-files=all"])
    model_preflight = ollama_model_preflight(config, inputs)
    request_universe, contexts = build_request_universe(root, inputs, model_preflight)
    g0_reuse = build_g0_reuse_inventory(root, inputs, request_universe)
    physical_execution_order = build_physical_execution_order_table(request_universe, g0_reuse)
    protocol_defects = pre_execution_protocol_defects(root, config, inputs, request_universe, physical_execution_order, g0_reuse, model_preflight, initial_git_status)
    invocation_start = utc_now()
    initial_request_records = read_jsonl(root / REQUEST_JSONL)
    initial_response_records = read_jsonl(root / RESPONSE_JSONL)
    sent_now = 0
    if not validate_only:
        persist_pre_inference_freeze(root, request_universe, physical_execution_order, g0_reuse)
        if not protocol_defects:
            sent_now = execute_pending_requests(root, config, request_universe, physical_execution_order, contexts, g0_reuse, model_preflight, max_new_calls=max_new_calls)
        else:
            append_jsonl_fsync(
                root / EVENT_JSONL,
                {
                    "event": "pre_execution_protocol_block",
                    "phase_id": PHASE_ID,
                    "timestamp_utc": utc_now(),
                    "defects": list(protocol_defects),
                },
            )
    invocation_end = utc_now()
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    execution_state = build_execution_state(request_universe, g0_reuse, request_records, response_records, inputs.g0_response_records)
    raw_inventory = build_raw_response_inventory(request_universe, execution_state, response_records, inputs.g0_response_records)
    strict_parsing = build_strict_parsing(request_universe, contexts, execution_state)
    per_request_stats, recommendations = build_recommendations_and_stats(request_universe, strict_parsing)
    utility = build_utility_results(per_request_stats, strict_parsing)
    complete_case = build_complete_case_results(per_request_stats, strict_parsing)
    reliability = build_reliability_results(request_universe, execution_state, strict_parsing)
    beyond = build_beyond_utility_results(root, inputs, per_request_stats, execution_state)
    foodcom = build_foodcom_warm_cold(per_request_stats, strict_parsing)
    runtime = build_runtime_results(execution_state)
    completeness = build_completeness_results(request_universe, execution_state, utility, reliability)
    resume_validation = build_resume_validation(
        request_universe,
        g0_reuse,
        initial_request_records,
        initial_response_records,
        request_records,
        response_records,
        sent_now,
        resume=resume,
        invocation_start=invocation_start,
        invocation_end=invocation_end,
    )
    validation = build_validation(
        root,
        config,
        inputs,
        request_universe,
        physical_execution_order,
        g0_reuse,
        execution_state,
        raw_inventory,
        strict_parsing,
        utility,
        beyond,
        foodcom,
        runtime,
        completeness,
        resume_validation,
        model_preflight,
        protocol_defects,
        initial_git_status,
    )
    decision = DECISION_VALIDATED if bool(validation["status"].eq("PASS").all()) else DECISION_BLOCKED
    manifests = build_manifests(
        root,
        config,
        inputs,
        decision,
        request_universe,
        physical_execution_order,
        g0_reuse,
        execution_state,
        raw_inventory,
        strict_parsing,
        utility,
        beyond,
        foodcom,
        runtime,
        completeness,
        resume_validation,
        validation,
        model_preflight,
        sent_now,
    )
    report = render_report(
        decision,
        model_preflight,
        request_universe,
        g0_reuse,
        execution_state,
        strict_parsing,
        utility,
        beyond,
        foodcom,
        runtime,
        reliability,
        resume_validation,
        validation,
    )
    return {
        "request_universe": request_universe,
        "physical_execution_order": physical_execution_order,
        "g0_reuse_inventory": g0_reuse,
        "execution_state": execution_state,
        "raw_response_inventory": raw_inventory,
        "strict_parsing": strict_parsing,
        "per_request_stats": per_request_stats,
        "recommendations": recommendations,
        "utility": utility,
        "complete_case": complete_case,
        "beyond_utility": beyond,
        "reliability": reliability,
        "foodcom_warm_cold": foodcom,
        "runtime": runtime,
        "completeness": completeness,
        "validation": validation,
        "resume_validation": resume_validation,
        "model_preflight": model_preflight,
        "manifests": manifests,
        "report": report,
    }


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def validate_local_endpoint(config: Mapping[str, Any]) -> None:
    normalize_loopback_url(str(config["execution"]["ollama_base_url"]), purpose="G1 Ollama endpoint")
    if bool(config["execution"].get("allow_external_api_calls", False)):
        raise ValueError("G1 forbids external API calls")


def git_success(root: Path, args: Sequence[str]) -> bool:
    completed = subprocess.run(["git", *args], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.returncode == 0


def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    if not ancestor or not descendant:
        return False
    return git_success(root, ["merge-base", "--is-ancestor", ancestor, descendant])


def git_tracked(root: Path, rel_path: Path) -> bool:
    return git_success(root, ["ls-files", "--error-unmatch", rel_path.as_posix()])


def physical_model_switch_count(physical_execution_order: pd.DataFrame) -> int:
    if physical_execution_order.empty:
        return 0
    ordered = physical_execution_order.sort_values("physical_execution_order", kind="mergesort")["model"].astype(str).tolist()
    return sum(1 for previous, current in zip(ordered, ordered[1:]) if previous != current)


def logical_order_is_dataset_user_prompt_model(request_universe: pd.DataFrame) -> bool:
    expected = request_universe.copy()
    expected["_dataset_order"] = expected["dataset"].map({dataset: index for index, dataset in enumerate(DATASET_ORDER)})
    expected["_user_sort"] = expected["user_id"].map(sort_key)
    expected["_prompt_order"] = expected["prompt_strategy"].map({prompt: index for index, prompt in enumerate(PROMPT_ORDER)})
    expected["_model_order"] = expected["model"].map({model: index for index, model in enumerate(MODEL_ORDER)})
    expected = expected.sort_values(["_dataset_order", "_user_sort", "_prompt_order", "_model_order"], kind="mergesort")
    return expected["canonical_request_id"].astype(str).tolist() == request_universe.sort_values("request_order", kind="mergesort")["canonical_request_id"].astype(str).tolist()


def physical_order_is_model_grouped(physical_execution_order: pd.DataFrame) -> bool:
    if physical_execution_order.empty:
        return False
    ordered = physical_execution_order.sort_values("physical_execution_order", kind="mergesort")
    if ordered["physical_execution_order"].astype(int).tolist() != list(range(1, len(ordered) + 1)):
        return False
    return physical_model_switch_count(ordered) == len(MODEL_ORDER) - 1 and ordered["model"].drop_duplicates().tolist() == list(MODEL_ORDER)


def normalize_loopback_url(url: str, *, purpose: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{purpose} must be an http(s) URL on 127.0.0.1 or localhost")
    if not hostname_is_loopback(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)):
        raise ValueError(f"{purpose} must resolve to loopback; observed host={parsed.hostname}")
    base = f"{parsed.scheme}://{parsed.netloc}"
    return base.rstrip("/")


def hostname_is_loopback(hostname: str, port: int) -> bool:
    host = str(hostname).strip("[]").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False
    if host == "localhost":
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError:
            return False
        resolved = []
        for info in infos:
            address = str(info[4][0]).split("%", 1)[0]
            try:
                resolved.append(ipaddress.ip_address(address))
            except ValueError:
                return False
        return bool(resolved) and all(address.is_loopback for address in resolved)
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class LoopbackOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> urllib.request.Request | None:
        normalize_loopback_url(newurl, purpose="G1 Ollama redirect target")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def local_http_json(method: str, url: str, payload: Mapping[str, Any] | None, *, timeout: int) -> dict[str, Any]:
    normalize_loopback_url(url, purpose="G1 Ollama metadata URL")
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    opener = urllib.request.build_opener(LoopbackOnlyRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            normalize_loopback_url(final_url, purpose="G1 Ollama metadata final URL")
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}", "models": []}


def expected_model_digests_from_g0(inputs: G1Inputs) -> dict[str, str]:
    return {
        str(row["model"]): str(row["digest"])
        for row in inputs.g0_execution_manifest.get("model_preflight", [])
        if str(row.get("model")) in MODEL_ORDER
    }


def ollama_model_preflight(
    config: Mapping[str, Any],
    inputs: G1Inputs,
    *,
    http_get: Any = local_http_json,
) -> pd.DataFrame:
    expected_digests = expected_model_digests_from_g0(inputs)
    if not bool(config["execution"].get("allow_local_ollama_metadata", True)):
        return blocked_model_preflight_rows(config, expected_digests, LOCAL_METADATA_CHECK_NOT_PERMITTED, "local metadata inspection disabled by config")
    base_url = normalize_loopback_url(str(config["execution"]["ollama_base_url"]), purpose="G1 Ollama endpoint")
    version_payload = http_get("GET", base_url + "/api/version", None, timeout=5)
    tags_payload = http_get("GET", base_url + "/api/tags", None, timeout=10)
    ps_payload = http_get("GET", base_url + "/api/ps", None, timeout=5)
    tags_error = str(tags_payload.get("_error", ""))
    models_payload = tags_payload.get("models", [])
    if tags_error or not isinstance(models_payload, list):
        detail = tags_error or "invalid /api/tags schema: models is not a list"
        return blocked_model_preflight_rows(config, expected_digests, OLLAMA_UNREACHABLE, detail, base_url=base_url, version_payload=version_payload, ps_payload=ps_payload)
    tags = parse_ollama_tags(models_payload)
    gpu_info = gpu_summary()
    ram_gb = physical_ram_gb()
    loaded = ps_payload.get("models", []) if isinstance(ps_payload, dict) and isinstance(ps_payload.get("models", []), list) else []
    rows = []
    for model in config["models"]:
        model = str(model)
        tag = tags.get(model)
        expected_digest = expected_digests.get(model, "")
        current_digest = str((tag or {}).get("digest") or (tag or {}).get("id") or "")
        digest_match = bool(expected_digest) and current_digest == expected_digest
        if tag is None:
            current_status = MODEL_NOT_INSTALLED
        elif not digest_match:
            current_status = MODEL_DIGEST_MISMATCH
        else:
            current_status = MODEL_AVAILABLE_MATCHED
        details = tag.get("details", {}) if isinstance(tag, dict) and isinstance(tag.get("details", {}), dict) else {}
        rows.append(
            {
                "model": model,
                "model_tag": model,
                "model_family": MODEL_FAMILY.get(model, model),
                "available": bool_text(tag is not None),
                "digest": current_digest,
                "g0_expected_digest": expected_digest,
                "current_local_digest": current_digest,
                "digest_match": bool_text(digest_match),
                "g0_model_identity_status": G0_PROVENANCE_AVAILABLE if expected_digest else G0_PROVENANCE_MISSING,
                "current_model_identity_status": current_status,
                "metadata_access_status": LOCAL_METADATA_OK,
                "metadata_error": "",
                "ollama_base_url": base_url,
                "ollama_host_env": str(os.environ.get("OLLAMA_HOST", "")),
                "size_bytes": str((tag or {}).get("size", "")),
                "modified_at": str((tag or {}).get("modified_at", "")),
                "parameter_size": str(details.get("parameter_size", "")),
                "quantization_level": str(details.get("quantization_level", "")),
                "context_length": str(details.get("context_length", "")),
                "capabilities": stable_json((tag or {}).get("capabilities", [])),
                "ollama_version": str(version_payload.get("version", "")) if isinstance(version_payload, dict) else "",
                "ollama_version_error": str(version_payload.get("_error", "")) if isinstance(version_payload, dict) else "",
                "ollama_ps_error": str(ps_payload.get("_error", "")) if isinstance(ps_payload, dict) else "",
                "ollama_models_loaded": stable_json([str(row.get("name") or row.get("model")) for row in loaded if isinstance(row, dict)]),
                "local_metadata_request_count": 3,
                "llm_generation_call": "false",
                "gpu": gpu_info,
                "ram_gb": format_float(ram_gb),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "status": current_status,
            }
        )
    return pd.DataFrame(rows)


def blocked_model_preflight_rows(
    config: Mapping[str, Any],
    expected_digests: Mapping[str, str],
    status: str,
    detail: str,
    *,
    base_url: str = "",
    version_payload: Mapping[str, Any] | None = None,
    ps_payload: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    rows = []
    version_payload = version_payload or {}
    ps_payload = ps_payload or {}
    for model in config["models"]:
        model = str(model)
        expected_digest = str(expected_digests.get(model, ""))
        rows.append(
            {
                "model": model,
                "model_tag": model,
                "model_family": MODEL_FAMILY.get(model, model),
                "available": "false",
                "digest": "",
                "g0_expected_digest": expected_digest,
                "current_local_digest": "",
                "digest_match": "false",
                "g0_model_identity_status": G0_PROVENANCE_AVAILABLE if expected_digest else G0_PROVENANCE_MISSING,
                "current_model_identity_status": status,
                "metadata_access_status": status,
                "metadata_error": detail,
                "ollama_base_url": base_url,
                "ollama_host_env": str(os.environ.get("OLLAMA_HOST", "")),
                "size_bytes": "",
                "modified_at": "",
                "parameter_size": "",
                "quantization_level": "",
                "context_length": "",
                "capabilities": "[]",
                "ollama_version": str(version_payload.get("version", "")),
                "ollama_version_error": str(version_payload.get("_error", "")),
                "ollama_ps_error": str(ps_payload.get("_error", "")),
                "ollama_models_loaded": "[]",
                "local_metadata_request_count": 0,
                "llm_generation_call": "false",
                "gpu": gpu_summary(),
                "ram_gb": format_float(physical_ram_gb()),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def parse_ollama_tags(models_payload: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    tags: dict[str, Mapping[str, Any]] = {}
    for model in models_payload:
        if not isinstance(model, Mapping):
            continue
        for key in ("name", "model"):
            value = str(model.get(key, "")).strip()
            if value:
                tags[value] = model
    return tags


def load_inputs(root: Path, config: Mapping[str, Any]) -> G1Inputs:
    f1bx_config = load_config(root, Path(str(config["inputs"]["f1bx_config"])))
    specs = {spec.name: spec for spec in dataset_specs(f1bx_config)}
    ranking_candidates = {dataset: load_ranking_candidates(root, specs[dataset]) for dataset in DATASET_ORDER}
    title_maps: dict[str, dict[str, str]] = {}
    for dataset in DATASET_ORDER:
        title_maps[dataset], _source = load_title_map(root, specs[dataset])
    history_by_dataset = load_histories(root, specs, ranking_candidates)
    return G1Inputs(
        config=dict(config),
        specs=specs,
        ranking_candidates=ranking_candidates,
        title_maps=title_maps,
        history_by_dataset=history_by_dataset,
        f0_primary=read_csv(root / str(config["inputs"]["f0_primary_ranking_matrix"])),
        f0_budget=read_csv(root / str(config["inputs"]["f0_request_budget"])),
        f0_manifest=read_json(root / str(config["inputs"]["f0_manifest"])),
        f0r_manifest=read_json(root / str(config["inputs"]["f0r_manifest"])),
        f0_inference_contract=read_json(root / str(config["inputs"]["f0_inference_contract"])),
        g0_request_matrix=read_csv(root / str(config["inputs"]["g0_request_matrix"])),
        g0_execution=read_csv(root / str(config["inputs"]["g0_execution"])),
        g0_parse_validation=read_csv(root / str(config["inputs"]["g0_parse_validation"])),
        g0_response_records=[dict(row) for row in read_jsonl(root / str(config["inputs"]["g0_response_jsonl"]))],
        g0_request_manifest=read_json(root / str(config["inputs"]["g0_request_manifest"])),
        g0_execution_manifest=read_json(root / str(config["inputs"]["g0_execution_manifest"])),
        f1a_manifest=read_json(root / str(config["inputs"]["f1a_manifest"])),
        f1arr_manifest=read_json(root / str(config["inputs"]["f1arr_manifest"])),
        f1bzq_manifest=read_json(root / str(config["inputs"]["f1bzq_manifest"])),
        baseline_recommendations=read_csv(root / str(config["inputs"]["f1bx_baseline_recommendations"])),
    )


def load_ranking_candidates(root: Path, spec: Any) -> pd.DataFrame:
    from src.extension.f1bx_execution_stage1 import load_candidates

    return load_candidates(root, spec)


def load_histories(root: Path, specs: Mapping[str, Any], ranking_candidates: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, list[str]]]:
    histories: dict[str, dict[str, list[str]]] = {}
    for dataset in DATASET_ORDER:
        spec = specs[dataset]
        if dataset == "foodcom":
            candidates = ranking_candidates[dataset]
            positive_by_user = candidates.drop_duplicates("user_id").set_index("user_id")["positive_item_id"].astype(str).to_dict()
            users = sorted(candidates["user_id"].astype(str).unique(), key=sort_key)
            events, _definition = foodcom_final_training_events(root, spec, users, positive_by_user)
        else:
            events, _definition = original_four_training_events(root, spec)
        histories[dataset] = {
            str(user_id): group.sort_values("prefix_position", kind="mergesort")["item_id"].astype(str).tolist()
            for user_id, group in events.groupby("user_id", sort=False)
        }
    return histories


def build_offline_g1_plan(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, Path(config_path))
    inputs = load_inputs(project_root, config)
    model_preflight = expected_model_preflight_from_g0(inputs)
    request_universe, contexts = build_request_universe(project_root, inputs, model_preflight)
    g0_reuse = build_g0_reuse_inventory(project_root, inputs, request_universe)
    physical_execution_order = build_physical_execution_order_table(request_universe, g0_reuse)
    return {
        "config": config,
        "inputs": inputs,
        "model_preflight": model_preflight,
        "request_universe": request_universe,
        "physical_execution_order": physical_execution_order,
        "contexts": contexts,
        "g0_reuse_inventory": g0_reuse,
    }


def expected_model_preflight_from_g0(inputs: G1Inputs) -> pd.DataFrame:
    rows = []
    for row in inputs.g0_execution_manifest.get("model_preflight", []):
        if str(row.get("model")) in MODEL_ORDER:
            rows.append(dict(row))
    return pd.DataFrame(rows)


def build_request_universe(
    root: Path,
    inputs: G1Inputs,
    model_preflight: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    digests = model_preflight.set_index("model")["digest"].to_dict() if not model_preflight.empty and "digest" in model_preflight else {}
    canonical_counts = {
        (str(row["dataset"]), str(row["model_name"]), str(row["prompt"])): int(row["request_count"])
        for row in inputs.f0_primary.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    order = 1
    for dataset in DATASET_ORDER:
        candidate_frame = inputs.ranking_candidates[dataset]
        users = sorted(candidate_frame["user_id"].astype(str).unique(), key=sort_key)
        dataset_label = str(inputs.specs[dataset].display_name)
        for user_id in users:
            base_context = build_ranking_user_context(inputs, dataset, user_id)
            candidate_items = list(base_context["ranking_candidates"])
            candidate_hash = str(base_context["candidate_hash"])
            candidate_set_id = candidate_set_id_for(dataset, user_id, candidate_hash)
            for prompt_strategy in PROMPT_ORDER:
                bundle = render_extension_prompt(
                    prompt_strategy=prompt_strategy,
                    dataset_label=dataset_label,
                    candidate_items=candidate_items,
                    history_items=base_context["history_items"],
                    history_length=int(base_context["visible_history_length"]),
                )
                for model in MODEL_ORDER:
                    request_id = canonical_request_id(
                        dataset=dataset,
                        task=TASK,
                        model=model,
                        prompt_strategy=prompt_strategy,
                        user_id=user_id,
                        candidate_hash=candidate_hash,
                        prompt_hash=bundle.prompt_hash,
                    )
                    row = {
                        "canonical_request_id": request_id,
                        "request_id": request_id,
                        "request_order": order,
                        "dataset": dataset,
                        "dataset_label": dataset_label,
                        "task": TASK,
                        "model": model,
                        "model_family": MODEL_FAMILY[model],
                        "model_digest": str(digests.get(model, "")),
                        "prompt_strategy": prompt_strategy,
                        "prompt_contract": bundle.prompt_contract,
                        "parser_contract": PARSER_CONTRACT_ID,
                        "protocol_hash": protocol_hash_for_row(candidate_count=CANDIDATE_COUNT, requested_output_length=REQUESTED_OUTPUT_LENGTH),
                        "generation_config_id": str(inputs.f0_inference_contract.get("generation_config_id", "")),
                        "generation_config_hash": str(inputs.f0_inference_contract.get("generation_config_hash", "")),
                        "user_id": user_id,
                        "candidate_set_id": candidate_set_id,
                        "candidate_source": str(base_context["candidate_source"]),
                        "candidate_source_protocol": "F0_FROZEN_RANKING_CANDIDATES",
                        "candidate_hash": candidate_hash,
                        "candidate_count": CANDIDATE_COUNT,
                        "candidate_items": stable_json([item["item_id"] for item in candidate_items]),
                        "candidate_titles_hash": sha256_json([item["title"] for item in candidate_items]),
                        "prompt_hash": bundle.prompt_hash,
                        "history_length": int(base_context["visible_history_length"]),
                        "full_history_length": int(base_context["full_history_length"]),
                        "history_hash": sha256_json([item["item_id"] for item in base_context["history_items"]]),
                        "positive_item_id": str(base_context["positive_item_id"]),
                        "foodcom_history_class": str(base_context["foodcom_history_class"]),
                        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                        "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
                        "metric_k": METRIC_K,
                        "canonical_cell_request_count": canonical_counts[(dataset, model, prompt_strategy)],
                        "raw_response_protocol": G1_RAW_PROTOCOL,
                        "raw_response_path": (RESPONSE_DIR / f"{request_id}.json").as_posix(),
                        "manifest_status": "PLANNED_CANONICAL_PRIMARY_RANKING_REQUEST",
                    }
                    row["request_identity_hash"] = semantic_hash_for_row(
                        row,
                        exclude=("request_order", "manifest_status", "raw_response_path"),
                    )
                    rows.append(row)
                    contexts[request_id] = {
                        **base_context,
                        "system_message": bundle.system_message,
                        "user_message": bundle.user_message,
                        "prompt_bundle": bundle,
                        "candidate_items_for_task": candidate_items,
                        "candidate_set_id": candidate_set_id,
                        "candidate_hash": candidate_hash,
                    }
                    order += 1
    universe = pd.DataFrame(rows)
    if len(universe) != 29292:
        raise ValueError(f"G1 request universe expected 29292 rows, observed {len(universe)}")
    if universe["canonical_request_id"].nunique() != 29292:
        raise ValueError("G1 canonical request ids are not unique")
    return universe, contexts


def build_ranking_user_context(inputs: G1Inputs, dataset: str, user_id: str) -> dict[str, Any]:
    frame = inputs.ranking_candidates[dataset].loc[inputs.ranking_candidates[dataset]["user_id"].eq(user_id)].copy()
    ranking_items = candidate_items_from_frame(frame, inputs.title_maps[dataset], title_column="")
    candidate_ids = [item["item_id"] for item in ranking_items]
    positive_items = frame.loc[frame["is_ground_truth"].astype(bool), "positive_item_id"].astype(str).unique().tolist()
    if len(positive_items) != 1:
        raise ValueError(f"ranking candidate positive contract broken for {dataset}/{user_id}")
    positive_item = str(positive_items[0])
    history_ids = inputs.history_by_dataset[dataset].get(str(user_id), [])
    visible_history_ids = list(history_ids)[-10:]
    history_items = [{"item_id": item_id, "title": title_for_item(inputs.title_maps[dataset], item_id)} for item_id in visible_history_ids]
    if len(ranking_items) != CANDIDATE_COUNT or len(set(candidate_ids)) != CANDIDATE_COUNT:
        raise ValueError(f"ranking candidate contract broken for {dataset}/{user_id}")
    if not all(item["title"] for item in ranking_items):
        raise ValueError(f"candidate title missing for {dataset}/{user_id}")
    if not all(item["title"] for item in history_items):
        raise ValueError(f"history title missing for {dataset}/{user_id}")
    foodcom_class = ""
    if dataset == "foodcom":
        foodcom_class = "TRUE_ZERO_HISTORY_USERS" if not history_ids else "WARM_USERS"
    return {
        "dataset": dataset,
        "user_id": str(user_id),
        "positive_item_id": positive_item,
        "ranking_candidates": ranking_items,
        "history_items": history_items,
        "full_history_length": len(history_ids),
        "visible_history_length": len(history_items),
        "candidate_hash": item_list_hash(candidate_ids),
        "candidate_source": inputs.specs[dataset].candidate_source.as_posix(),
        "foodcom_history_class": foodcom_class,
    }


def candidate_set_id_for(dataset: str, user_id: str, candidate_hash: str) -> str:
    return "G1-RANK-CSET-" + sha256_json({"dataset": dataset, "user_id": str(user_id), "candidate_hash": candidate_hash}).upper()[:20]


def protocol_hash_for_row(*, candidate_count: int, requested_output_length: int) -> str:
    return sha256_json(
        {
            "raw_response_protocol": G1_RAW_PROTOCOL,
            "task": TASK,
            "parser_contract": PARSER_CONTRACT_ID,
            "candidate_count": candidate_count,
            "requested_output_length": requested_output_length,
            "parser_expected_length": requested_output_length,
            "metric_k": METRIC_K,
        }
    )


def build_g0_reuse_inventory(root: Path, inputs: G1Inputs, request_universe: pd.DataFrame) -> pd.DataFrame:
    g1_by_id = request_universe.set_index("canonical_request_id").to_dict("index")
    g0_matrix = inputs.g0_request_matrix.loc[inputs.g0_request_matrix["task"].eq(TASK)].copy()
    g0_execution = inputs.g0_execution.set_index("canonical_request_id").to_dict("index")
    g0_parse = inputs.g0_parse_validation.loc[
        inputs.g0_parse_validation["canary_request_id"].ne("AGGREGATE")
        & inputs.g0_parse_validation["task"].eq(TASK)
    ].set_index("canonical_request_id").to_dict("index")
    g0_response_by_id = single_record_by_id(inputs.g0_response_records)
    rows: list[dict[str, Any]] = []
    for g0_row in g0_matrix.sort_values("request_order", kind="mergesort").to_dict("records"):
        request_id = str(g0_row["canonical_request_id"])
        g1_row = g1_by_id.get(request_id, {})
        execution = g0_execution.get(request_id, {})
        parse = g0_parse.get(request_id, {})
        response = g0_response_by_id.get(request_id, {})
        g0_protocol_hash = protocol_hash_for_row(
            candidate_count=int(g0_row["candidate_count"]),
            requested_output_length=int(g0_row["requested_output_length"]),
        )
        field_matches = {
            "canonical_request_id": bool(g1_row),
            "dataset": str(g1_row.get("dataset", "")) == str(g0_row.get("dataset", "")),
            "user_id": str(g1_row.get("user_id", "")) == str(g0_row.get("user_id", "")),
            "model": str(g1_row.get("model", "")) == str(g0_row.get("model", "")),
            "prompt_strategy": str(g1_row.get("prompt_strategy", "")) == str(g0_row.get("prompt_strategy", "")),
            "candidate_hash": str(g1_row.get("candidate_hash", "")) == str(g0_row.get("candidate_hash", "")),
            "candidate_items": str(g1_row.get("candidate_items", "")) == str(g0_row.get("candidate_items", "")),
            "prompt_hash": str(g1_row.get("prompt_hash", "")) == str(g0_row.get("prompt_hash", "")),
            "parser_contract": str(g1_row.get("parser_contract", "")) == str(g0_row.get("parser_contract", "")),
            "protocol_hash": str(g1_row.get("protocol_hash", "")) == g0_protocol_hash,
            "raw_response_hash": str(execution.get("raw_response_hash", "")) == str(parse.get("raw_response_hash", "")) == str(response.get("raw_response_hash", "")),
        }
        reusable = all(field_matches.values()) and str(execution.get("terminal", "")).lower() == "true"
        rows.append(
            {
                "canonical_request_id": request_id,
                "dataset": str(g0_row.get("dataset", "")),
                "user_id": str(g0_row.get("user_id", "")),
                "model": str(g0_row.get("model", "")),
                "prompt_strategy": str(g0_row.get("prompt_strategy", "")),
                "task": str(g0_row.get("task", "")),
                "candidate_set_id": candidate_set_id_for(str(g0_row.get("dataset", "")), str(g0_row.get("user_id", "")), str(g0_row.get("candidate_hash", ""))),
                "candidate_hash": str(g0_row.get("candidate_hash", "")),
                "g0_candidate_hash": str(g0_row.get("candidate_hash", "")),
                "g1_candidate_hash": str(g1_row.get("candidate_hash", "")),
                "g0_candidate_items": str(g0_row.get("candidate_items", "")),
                "g1_candidate_items": str(g1_row.get("candidate_items", "")),
                "prompt_hash": str(g0_row.get("prompt_hash", "")),
                "g0_prompt_hash": str(g0_row.get("prompt_hash", "")),
                "g1_prompt_hash": str(g1_row.get("prompt_hash", "")),
                "protocol_hash": g0_protocol_hash,
                "g0_parser_contract": str(g0_row.get("parser_contract", "")),
                "g1_parser_contract": str(g1_row.get("parser_contract", "")),
                "g0_protocol_hash": g0_protocol_hash,
                "g1_protocol_hash": str(g1_row.get("protocol_hash", "")),
                "raw_response_hash": str(response.get("raw_response_hash", "")),
                "g0_raw_response_path": str(g0_row.get("raw_response_path", "")),
                "g1_raw_response_path": str(g1_row.get("raw_response_path", "")),
                "g0_parse_status": str(parse.get("parse_status", "")),
                "g0_strict_parse_valid": str(parse.get("strict_parse_valid", "")),
                "all_identity_fields_match": bool_text(all(field_matches.values())),
                "reused_in_g1": bool_text(reusable),
                "reuse_status": "REUSED_G0_RANKING" if reusable else "BLOCKED_IDENTITY_MISMATCH",
                "field_match_json": stable_json(field_matches),
                "reason": "Canonical G0 ranking response reused without rewriting." if reusable else "G0 ranking row did not exactly match G1 canonical identity.",
            }
        )
    return pd.DataFrame(rows)


def build_physical_execution_order_table(request_universe: pd.DataFrame, g0_reuse: pd.DataFrame) -> pd.DataFrame:
    g0_ids = set(g0_reuse.loc[g0_reuse["reused_in_g1"].eq("true"), "canonical_request_id"].astype(str)) if not g0_reuse.empty else set()
    rows: list[dict[str, Any]] = []
    physical_order = 1
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            dataset_frame = request_universe.loc[
                request_universe["model"].eq(model) & request_universe["dataset"].eq(dataset)
            ].copy()
            dataset_frame["_prompt_order"] = dataset_frame["prompt_strategy"].map({prompt: index for index, prompt in enumerate(PROMPT_ORDER)})
            dataset_frame["_user_sort"] = dataset_frame["user_id"].map(sort_key)
            dataset_frame = dataset_frame.sort_values(["_prompt_order", "_user_sort", "request_order"], kind="mergesort")
            for row in dataset_frame.to_dict("records"):
                rows.append(
                    {
                        "canonical_request_id": row["canonical_request_id"],
                        "canonical_request_order": int(row["request_order"]),
                        "physical_execution_order": physical_order,
                        "dataset": row["dataset"],
                        "user_id": row["user_id"],
                        "model": row["model"],
                        "prompt_strategy": row["prompt_strategy"],
                        "scheduler_policy": "model_dataset_prompt_user",
                        "execution_action": "REUSE_G0_NO_CALL" if str(row["canonical_request_id"]) in g0_ids else "G1_LOCAL_OLLAMA_CALL_WHEN_ALLOWED",
                        "identity_effect": "NONE_OPERATIONAL_ORDER_ONLY",
                    }
                )
                physical_order += 1
    frame = pd.DataFrame(rows)
    if len(frame) != len(request_universe) or frame["canonical_request_id"].nunique() != len(request_universe):
        raise ValueError("physical execution order does not cover the canonical universe exactly once")
    return frame


def pre_execution_protocol_defects(
    root: Path,
    config: Mapping[str, Any],
    inputs: G1Inputs,
    request_universe: pd.DataFrame,
    physical_execution_order: pd.DataFrame,
    g0_reuse: pd.DataFrame,
    model_preflight: pd.DataFrame,
    initial_git_status: str,
) -> list[str]:
    defects: list[str] = []
    base_checkpoint = str(config["phase"].get("expected_base_checkpoint", ""))
    current_head = git_output(root, ["rev-parse", "HEAD"])
    if base_checkpoint != SCIENTIFIC_BASE_CHECKPOINT:
        defects.append("scientific_base_checkpoint_config_mismatch")
    if bool(config["phase"].get("base_checkpoint_required_as_ancestor", True)) and not git_is_ancestor(root, base_checkpoint, current_head):
        defects.append("scientific_base_not_ancestor_of_execution_head")
    if bool(config["phase"].get("phase_start_clean_required", True)) and initial_git_status.strip():
        defects.append("precondition_working_tree_not_clean")
    expected_digests = {
        str(row["model"]): str(row["digest"])
        for row in inputs.g0_execution_manifest.get("model_preflight", [])
        if str(row.get("model")) in MODEL_ORDER
    }
    observed_digests = model_preflight.set_index("model")["digest"].to_dict() if not model_preflight.empty and "digest" in model_preflight else {}
    model_statuses = set(model_preflight["current_model_identity_status"].astype(str)) if "current_model_identity_status" in model_preflight else set()
    if LOCAL_METADATA_CHECK_NOT_PERMITTED in model_statuses:
        defects.append("local_metadata_check_not_permitted")
    if OLLAMA_UNREACHABLE in model_statuses:
        defects.append("ollama_unreachable")
    if MODEL_NOT_INSTALLED in model_statuses:
        defects.append("missing_required_local_model")
    if MODEL_DIGEST_MISMATCH in model_statuses or (expected_digests and any(str(observed_digests.get(model, "")) != expected_digests[model] for model in MODEL_ORDER)):
        defects.append("model_digest_mismatch")
    if len(request_universe) != 29292 or request_universe["canonical_request_id"].nunique() != 29292:
        defects.append("request_universe_not_29292_unique")
    if len(physical_execution_order) != len(request_universe) or physical_execution_order["canonical_request_id"].nunique() != len(request_universe):
        defects.append("physical_execution_order_not_exactly_universe")
    if physical_model_switch_count(physical_execution_order) > len(MODEL_ORDER) - 1:
        defects.append("physical_execution_not_model_grouped")
    if request_universe.groupby(["dataset", "user_id"], sort=False).size().ne(len(MODEL_ORDER) * len(PROMPT_ORDER)).any():
        defects.append("user_model_prompt_cross_not_complete")
    if not request_universe["task"].eq(TASK).all():
        defects.append("non_ranking_task_in_g1_universe")
    if "video_games" in set(request_universe["dataset"].astype(str)):
        defects.append("video_games_active")
    if len(g0_reuse) != 36 or not g0_reuse["reused_in_g1"].eq("true").all():
        defects.append("g0_reuse_not_exactly_36")
    budget = {str(row["budget_item"]): int(row["request_count"]) for row in inputs.f0_budget.to_dict("records")}
    if any(budget.get(key) != value for key, value in REQUIRED_BUDGET.items()):
        defects.append("f0_budget_changed")
    if backend_retry_count(config, inputs) != 2:
        defects.append("backend_retry_count_not_f0_frozen")
    if backend_retry_backoffs(config, inputs) != [5, 15]:
        defects.append("backend_retry_backoff_not_f0_frozen")
    if bool(config["execution"].get("allow_reranking_execution")):
        defects.append("reranking_execution_enabled")
    if bool(config["execution"].get("allow_history_grid_execution")):
        defects.append("history_grid_execution_enabled")
    if bool(config["execution"].get("allow_canddif_execution")):
        defects.append("canddif_execution_enabled")
    if bool(config["execution"].get("allow_profile_generation")):
        defects.append("profile_generation_enabled")
    if bool(config["execution"].get("allow_pool_size_sensitivity")):
        defects.append("pool_size_sensitivity_enabled")
    return defects


def backend_retry_count(config: Mapping[str, Any], inputs: G1Inputs) -> int:
    frozen = int(inputs.f0_inference_contract.get("invalidity_policy", {}).get("backend_retry_count", config["execution"]["backend_retry_count"]))
    configured = int(config["execution"]["backend_retry_count"])
    if frozen != configured:
        return -1
    return configured


def backend_retry_backoffs(config: Mapping[str, Any], inputs: G1Inputs) -> list[int]:
    frozen = [int(value) for value in inputs.f0_inference_contract.get("invalidity_policy", {}).get("backend_retry_backoff_seconds", [])]
    configured = [int(value) for value in config["execution"]["backend_retry_backoff_seconds"]]
    return configured if frozen == configured else []


def persist_pre_inference_freeze(root: Path, request_universe: pd.DataFrame, physical_execution_order: pd.DataFrame, g0_reuse: pd.DataFrame) -> None:
    ensure_output_dirs(root)
    request_universe.to_csv(root / OUTPUT_TABLES["request_universe"], index=False, lineterminator="\n")
    physical_execution_order.to_csv(root / OUTPUT_TABLES["physical_execution_order"], index=False, lineterminator="\n")
    g0_reuse.to_csv(root / OUTPUT_TABLES["g0_reuse_inventory"], index=False, lineterminator="\n")
    append_jsonl_fsync(
        root / EVENT_JSONL,
        {
            "event": "request_universe_persisted_before_inference",
            "phase_id": PHASE_ID,
            "timestamp_utc": utc_now(),
            "planned_requests": int(len(request_universe)),
            "g0_reused_ranking_requests": int(g0_reuse["reused_in_g1"].eq("true").sum()),
            "request_universe_hash": frame_hash(request_universe),
            "physical_execution_order_hash": frame_hash(physical_execution_order),
            "g0_reuse_hash": frame_hash(g0_reuse),
        },
    )


def execute_pending_requests(
    root: Path,
    config: Mapping[str, Any],
    request_universe: pd.DataFrame,
    physical_execution_order: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    g0_reuse: pd.DataFrame,
    model_preflight: pd.DataFrame,
    *,
    max_new_calls: int | None,
) -> int:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    g0_ids = set(g0_reuse.loc[g0_reuse["reused_in_g1"].eq("true"), "canonical_request_id"].astype(str))
    verify_existing_g1_raw_state(request_universe, g0_ids, request_records, response_records)
    model_available = model_preflight.set_index("model")["available"].to_dict()
    req_counts = count_by_id(request_records)
    resp_counts = count_by_id(response_records)
    physical_order = physical_execution_order.set_index("canonical_request_id")["physical_execution_order"].to_dict()
    ordered_universe = request_universe.copy()
    ordered_universe["physical_execution_order"] = ordered_universe["canonical_request_id"].map(physical_order).astype(int)
    pending = []
    for row in ordered_universe.sort_values("physical_execution_order", kind="mergesort").to_dict("records"):
        request_id = str(row["canonical_request_id"])
        if request_id in g0_ids:
            continue
        state = classify_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0))
        if state == "not_started":
            pending.append(row)
    if max_new_calls is not None:
        if max_new_calls < 0:
            raise ValueError("--max-new-calls must be non-negative")
        pending = pending[:max_new_calls]
    sent = 0
    for row in pending:
        if str(model_available.get(row["model"], "false")) != "true":
            raise RuntimeError(f"model not locally available: {row['model']}")
        context = contexts[row["canonical_request_id"]]
        payload = build_payload(config, row, context)
        request_record = build_request_record(row, context, payload)
        append_jsonl_fsync(root / REQUEST_JSONL, request_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "request_recorded_before_send", "phase_id": PHASE_ID, "canonical_request_id": row["canonical_request_id"], "timestamp_utc": utc_now()})
        result, attempt_rows = call_with_backend_retries(root, config, row, payload)
        response_record = build_response_record(row, result, attempt_rows)
        write_raw_response_file(root / row["raw_response_path"], response_record)
        append_jsonl_fsync(root / RESPONSE_JSONL, response_record)
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "response_recorded_before_parse",
                "phase_id": PHASE_ID,
                "canonical_request_id": row["canonical_request_id"],
                "backend_success": result.backend_success,
                "backend_attempt_count": len(attempt_rows),
                "timestamp_utc": utc_now(),
            },
        )
        sent += 1
        print(
            "g1_request "
            f"physical_order={row['physical_execution_order']} canonical_order={row['request_order']} dataset={row['dataset']} "
            f"prompt={row['prompt_strategy']} model={row['model']} "
            f"backend={bool_text(result.backend_success)} attempts={len(attempt_rows)} "
            f"duration={format_float(result.duration_seconds)}",
            flush=True,
        )
    return sent


def verify_existing_g1_raw_state(
    request_universe: pd.DataFrame,
    g0_reused_ids: set[str],
    request_records: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
) -> None:
    expected_new = set(request_universe["canonical_request_id"].astype(str)) - set(g0_reused_ids)
    req_counts = count_by_id(request_records)
    resp_counts = count_by_id(response_records)
    extra = sorted((set(req_counts) | set(resp_counts)) - expected_new)
    if extra:
        raise RuntimeError(f"raw evidence contains request outside G1 new-call set: {extra[0]}")
    request_by_id = request_universe.set_index("canonical_request_id").to_dict("index")
    for request_id in sorted(expected_new):
        state = classify_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0))
        if state in {"ambiguous_interrupted", "invalid_duplicate_evidence"}:
            raise RuntimeError(f"{state}: {request_id}")
    for record in response_records:
        request_id = str(record.get("canonical_request_id") or record.get("request_id") or "")
        row = request_by_id.get(request_id, {})
        for key in ("dataset", "model", "prompt_strategy", "user_id", "candidate_hash", "prompt_hash"):
            if str(record.get(key, "")) != str(row.get(key, "")):
                raise RuntimeError(f"conflicting terminal evidence for {request_id}: {key}")


def build_payload(config: Mapping[str, Any], row: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "model": row["model"],
        "system": context["system_message"],
        "prompt": context["user_message"],
        "stream": bool(config["execution"].get("stream", False)),
        "format": str(config["execution"].get("response_format", "json")),
        "options": {
            "temperature": float(config["execution"]["temperature"]),
            "num_predict": int(config["execution"]["max_output_tokens"][row["task"]]),
        },
    }
    visible_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if positive_label_exposed(visible_payload, str(context["positive_item_id"])):
        raise ValueError(f"positive label exposed in prompt payload: {row['canonical_request_id']}")
    return payload


def build_request_record(row: Mapping[str, Any], context: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "canonical_request_id": row["canonical_request_id"],
        "request_id": row["canonical_request_id"],
        "request_order": int(row["request_order"]),
        "canonical_request_order": int(row["request_order"]),
        "physical_execution_order": int(row.get("physical_execution_order", row["request_order"])),
        "dataset": row["dataset"],
        "task": row["task"],
        "model": row["model"],
        "model_digest": row["model_digest"],
        "prompt_strategy": row["prompt_strategy"],
        "prompt_contract": row["prompt_contract"],
        "parser_contract": row["parser_contract"],
        "protocol_hash": row["protocol_hash"],
        "generation_config_hash": row["generation_config_hash"],
        "user_id": row["user_id"],
        "candidate_set_id": row["candidate_set_id"],
        "candidate_hash": row["candidate_hash"],
        "prompt_hash": row["prompt_hash"],
        "history_length": int(row["history_length"]),
        "raw_response_path": row["raw_response_path"],
        "payload_hash": sha256_json(payload),
        "request_payload": payload,
        "timestamp_before_send_utc": utc_now(),
    }


def call_with_backend_retries(
    root: Path,
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[RequestResult, list[dict[str, Any]]]:
    retry_count = int(config["execution"]["backend_retry_count"])
    backoffs = [int(value) for value in config["execution"]["backend_retry_backoff_seconds"]]
    attempts: list[dict[str, Any]] = []
    result: RequestResult | None = None
    for attempt in range(1, retry_count + 2):
        result = call_ollama_generate(
            str(config["execution"]["ollama_base_url"]),
            payload,
            int(config["execution"]["timeout_seconds"]),
        )
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
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "backend_attempt_completed",
                "phase_id": PHASE_ID,
                "canonical_request_id": row["canonical_request_id"],
                **attempts[-1],
                "timestamp_utc": utc_now(),
            },
        )
        if result.backend_success:
            break
        if attempt <= retry_count:
            time.sleep(backoffs[attempt - 1])
    if result is None:
        raise RuntimeError("backend retry loop did not execute")
    return result, attempts


def build_response_record(row: Mapping[str, Any], result: RequestResult, attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "canonical_request_id": row["canonical_request_id"],
        "request_id": row["canonical_request_id"],
        "request_order": int(row["request_order"]),
        "canonical_request_order": int(row["request_order"]),
        "physical_execution_order": int(row.get("physical_execution_order", row["request_order"])),
        "dataset": row["dataset"],
        "task": row["task"],
        "model": row["model"],
        "model_digest": row["model_digest"],
        "prompt_strategy": row["prompt_strategy"],
        "user_id": row["user_id"],
        "candidate_set_id": row["candidate_set_id"],
        "candidate_hash": row["candidate_hash"],
        "prompt_hash": row["prompt_hash"],
        "parser_contract": row["parser_contract"],
        "protocol_hash": row["protocol_hash"],
        "generation_config_hash": row["generation_config_hash"],
        "backend_attempt_number": len(attempts),
        "backend_attempt_count": len(attempts),
        "backend_success": result.backend_success,
        "backend_status": "BACKEND_SUCCESS" if result.backend_success else "BACKEND_FAILURE",
        "backend_error_type": result.backend_error_type,
        "backend_error_message": result.backend_error_message,
        "request_started_at_utc": result.started_at,
        "request_completed_at_utc": result.completed_at,
        "runtime_seconds": result.duration_seconds,
        "retry_count": max(0, len(attempts) - 1),
        "attempts": list(attempts),
        "raw_response": result.response_text,
        "raw_response_hash": sha256_text(result.response_text),
        "raw_backend_response": result.raw_backend_response,
        "raw_response_recorded_before_parse": True,
    }


def build_execution_state(
    request_universe: pd.DataFrame,
    g0_reuse: pd.DataFrame,
    request_records: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
    g0_response_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    req_counts = count_by_id(request_records)
    g1_by_id = single_record_by_id(response_records)
    g0_by_id = single_record_by_id(g0_response_records)
    g0_ids = set(g0_reuse.loc[g0_reuse["reused_in_g1"].eq("true"), "canonical_request_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for row in request_universe.sort_values("request_order", kind="mergesort").to_dict("records"):
        request_id = str(row["canonical_request_id"])
        origin = "G0_REUSED" if request_id in g0_ids else "G1_NEW"
        response = g0_by_id.get(request_id, {}) if origin == "G0_REUSED" else g1_by_id.get(request_id, {})
        response_count = 1 if response else 0
        request_count = 1 if origin == "G0_REUSED" and response else req_counts.get(request_id, 0)
        state = "completed" if origin == "G0_REUSED" and response else classify_state(request_count, response_count)
        backend_success = bool(response.get("backend_success", False))
        terminal = bool(response)
        rows.append(
            {
                **{key: row[key] for key in request_universe.columns if key not in {"manifest_status"}},
                "evidence_origin": origin,
                "request_record_count": request_count,
                "response_record_count": response_count,
                "request_state": state,
                "terminal": bool_text(terminal),
                "terminal_state": ("BACKEND_SUCCESS_UNPARSED" if backend_success else "BACKEND_FAILURE") if terminal else state,
                "backend_success": bool_text(backend_success),
                "backend_status": str(response.get("backend_status", "")),
                "backend_attempt_count": int(response.get("backend_attempt_count", 1 if origin == "G0_REUSED" and response else 0) or 0),
                "retry_count": int(response.get("retry_count", 0) or 0),
                "request_started_at_utc": str(response.get("request_started_at_utc", "")),
                "request_completed_at_utc": str(response.get("request_completed_at_utc", "")),
                "runtime_seconds": format_float(float(response.get("runtime_seconds", math.nan))) if response else "",
                "raw_response": str(response.get("raw_response", "")),
                "raw_response_hash": str(response.get("raw_response_hash", "")),
                "raw_response_recorded_before_parse": bool_text(bool(response.get("raw_response_recorded_before_parse", False))),
                "backend_error_type": str(response.get("backend_error_type", "")),
                "backend_error_message": str(response.get("backend_error_message", "")),
            }
        )
    return pd.DataFrame(rows)


def build_raw_response_inventory(
    request_universe: pd.DataFrame,
    execution_state: pd.DataFrame,
    response_records: Sequence[Mapping[str, Any]],
    g0_response_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    g1_by_id = single_record_by_id(response_records)
    g0_by_id = single_record_by_id(g0_response_records)
    rows: list[dict[str, Any]] = []
    for row in execution_state.loc[execution_state["terminal"].eq("true")].sort_values("request_order", kind="mergesort").to_dict("records"):
        request_id = str(row["canonical_request_id"])
        response = g0_by_id.get(request_id, {}) if row["evidence_origin"] == "G0_REUSED" else g1_by_id.get(request_id, {})
        rows.append(
            {
                "canonical_request_id": request_id,
                "request_order": int(row["request_order"]),
                "dataset": row["dataset"],
                "user_id": row["user_id"],
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "evidence_origin": row["evidence_origin"],
                "raw_response_path": row["raw_response_path"] if row["evidence_origin"] == "G1_NEW" else str(response.get("raw_response_path", "")),
                "backend_status": row["backend_status"],
                "backend_attempt_count": int(row["backend_attempt_count"]),
                "retry_count": int(row["retry_count"]),
                "request_started_at_utc": row["request_started_at_utc"],
                "request_completed_at_utc": row["request_completed_at_utc"],
                "runtime_seconds": row["runtime_seconds"],
                "raw_response_hash": row["raw_response_hash"],
                "raw_response_recorded_before_parse": row["raw_response_recorded_before_parse"],
                "inventory_status": "TERMINAL_EVIDENCE_INDEXED",
            }
        )
    return pd.DataFrame(rows)


def build_strict_parsing(
    request_universe: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    execution_state: pd.DataFrame,
) -> pd.DataFrame:
    response_by_id = execution_state.set_index("canonical_request_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for row in request_universe.sort_values("request_order", kind="mergesort").to_dict("records"):
        request_id = str(row["canonical_request_id"])
        response = response_by_id.get(request_id, {})
        candidate_items = [item["item_id"] for item in contexts[request_id]["candidate_items_for_task"]]
        if not response or response.get("terminal") != "true":
            parse_status = "not_executed"
            parse_valid = False
            output_indices: Sequence[int] = []
            output_items: Sequence[str] = []
            duplicate_count = 0
            unknown_count = 0
            output_length = 0
            error = ""
        elif response.get("backend_success") != "true":
            parse_status = "backend_failure_not_parsed"
            parse_valid = False
            output_indices = []
            output_items = []
            duplicate_count = 0
            unknown_count = 0
            output_length = 0
            error = str(response.get("backend_error_message", ""))
        else:
            parsed = parse_extension_top10_response(str(response.get("raw_response", "")), candidate_items=candidate_items)
            parse_status = parsed.status
            parse_valid = parsed.parse_valid
            output_indices = parsed.output_indices
            output_items = parsed.output_items
            duplicate_count = parsed.duplicate_count
            unknown_count = parsed.unknown_item_count
            output_length = parsed.output_length
            error = parsed.error
        normalized = normalized_structural_status(parse_status, bool(response and response.get("backend_success") == "true"))
        rows.append(
            {
                "canonical_request_id": request_id,
                "request_order": int(row["request_order"]),
                "dataset": row["dataset"],
                "user_id": row["user_id"],
                "model": row["model"],
                "model_family": row["model_family"],
                "prompt_strategy": row["prompt_strategy"],
                "candidate_hash": row["candidate_hash"],
                "prompt_hash": row["prompt_hash"],
                "raw_response_hash": str(response.get("raw_response_hash", "")),
                "evidence_origin": str(response.get("evidence_origin", "")),
                "parser_contract": PARSER_CONTRACT_ID,
                "expected_output_length": REQUESTED_OUTPUT_LENGTH,
                "output_length": int(output_length),
                "strict_parse_valid": bool_text(parse_valid),
                "parse_status": parse_status,
                "terminal_structural_status": normalized,
                "ranked_candidate_indices": stable_json(list(output_indices)),
                "ranked_item_ids": stable_json(list(output_items)),
                "duplicate_count": int(duplicate_count),
                "invalid_candidate_reference_count": int(unknown_count),
                "hallucination_count": int(unknown_count),
                "length_violation": bool_text(bool(output_length != REQUESTED_OUTPUT_LENGTH)),
                "invalid_output_retried": "false",
                "structurally_invalid_preserved": bool_text(not parse_valid and bool(response and response.get("backend_success") == "true")),
                "parse_error": error,
                "status": "COMPUTED" if response and response.get("terminal") == "true" else "PENDING_TERMINAL_EVIDENCE",
                "reason": "Strict parser result preserved without repair or scientific retry.",
            }
        )
    return pd.DataFrame(rows)


def normalized_structural_status(parse_status: str, backend_success: bool) -> str:
    if parse_status == "not_executed":
        return "NOT_TERMINAL"
    if not backend_success or parse_status == "backend_failure_not_parsed":
        return "BACKEND_FAILURE"
    return STRICT_STATUS_MAP.get(str(parse_status), "OTHER_FROZEN_STRUCTURAL_STATUS")


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    return [str(item) for item in json.loads(text)]


def build_recommendations_and_stats(request_universe: pd.DataFrame, strict_parsing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_by_id = request_universe.set_index("canonical_request_id").to_dict("index")
    stats_rows: list[dict[str, Any]] = []
    rec_rows: list[dict[str, Any]] = []
    for parsed in strict_parsing.sort_values("request_order", kind="mergesort").to_dict("records"):
        request_id = str(parsed["canonical_request_id"])
        row = universe_by_id[request_id]
        valid = str(parsed["strict_parse_valid"]) == "true"
        output_items = parse_json_list(parsed["ranked_item_ids"]) if valid else []
        output_indices = [int(value) for value in json.loads(parsed["ranked_candidate_indices"])] if valid else []
        positive = str(row["positive_item_id"])
        positive_rank = next((index + 1 for index, item in enumerate(output_items) if str(item) == positive), None)
        rank_zero = None if positive_rank is None else positive_rank - 1
        hit = 1 if rank_zero is not None and rank_zero < METRIC_K else 0
        ndcg = ndcg_at_k_from_rank(rank_zero, METRIC_K) if rank_zero is not None else 0.0
        mrr = reciprocal_rank(rank_zero) if rank_zero is not None else 0.0
        run_id = run_id_for(row["dataset"], row["model"], row["prompt_strategy"])
        stats_rows.append(
            {
                "canonical_request_id": request_id,
                "run_id": run_id,
                "dataset": row["dataset"],
                "model": row["model"],
                "model_name": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "prompt_variant": row["prompt_strategy"],
                "user_id": row["user_id"],
                "positive_item_id": positive,
                "strict_parse_valid": bool_text(valid),
                "backend_success": "",
                "rank_one_based": "" if positive_rank is None else positive_rank,
                "hit_at_5": hit,
                "ndcg_at_5": ndcg,
                "mrr": mrr,
                "history_length": int(row["history_length"]),
                "full_history_length": int(row["full_history_length"]),
                "foodcom_history_class": row["foodcom_history_class"],
                "top_items": output_items[:TOP_K],
                "top10_items": output_items,
                "evidence_view": "PRIMARY_FAIL_STRICT",
            }
        )
        if valid:
            for rank, item_id in enumerate(output_items, start=1):
                rec_rows.append(
                    {
                        "run_id": run_id,
                        "canonical_request_id": request_id,
                        "user_id": row["user_id"],
                        "model_name": row["model"],
                        "dataset": row["dataset"],
                        "rank": rank,
                        "recommended_item_id": str(item_id),
                        "candidate_position": output_indices[rank - 1],
                        "is_ground_truth": bool(str(item_id) == positive),
                        "positive_item_id": positive,
                        "candidate_set_id": row["candidate_set_id"],
                        "candidate_hash": row["candidate_hash"],
                        "prompt_variant": row["prompt_strategy"],
                        "parser_contract": PARSER_CONTRACT_ID,
                        "parse_status": parsed["parse_status"],
                        "match_type": "strict_index_candidate_reference",
                        "source_file": parsed["evidence_origin"],
                        "candidate_ordering": "CANONICAL_FROZEN_ORDER",
                        "history_length": int(row["history_length"]),
                    }
                )
    return pd.DataFrame(stats_rows), pd.DataFrame(rec_rows)


def run_id_for(dataset: str, model: str, prompt: str) -> str:
    slug = str(model).replace(":", "_").replace("/", "_").replace("-", "_")
    return f"g1_primary_{dataset}_{slug}_{prompt.lower()}"


def build_utility_results(per_request_stats: pd.DataFrame, strict_parsing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parse_by_id = strict_parsing.set_index("canonical_request_id").to_dict("index")
    for (dataset, model, prompt), group in per_request_stats.groupby(["dataset", "model", "prompt_strategy"], sort=False):
        planned = len(group)
        terminal_group = [parse_by_id[str(request_id)] for request_id in group["canonical_request_id"]]
        strict_valid = sum(1 for row in terminal_group if row["strict_parse_valid"] == "true")
        backend_success = sum(1 for row in terminal_group if row["terminal_structural_status"] != "BACKEND_FAILURE" and row["terminal_structural_status"] != "NOT_TERMINAL")
        status = "COMPUTED" if all(row["status"] == "COMPUTED" for row in terminal_group) else "PENDING_TERMINAL_EVIDENCE"
        values = {
            "hr_at_5": float(pd.to_numeric(group["hit_at_5"]).mean()) if planned else math.nan,
            "ndcg_at_5": float(pd.to_numeric(group["ndcg_at_5"]).mean()) if planned else math.nan,
            "mrr": float(pd.to_numeric(group["mrr"]).mean()) if planned else math.nan,
        }
        for metric, value in values.items():
            rows.append(
                {
                    "result_id": "G1-UTIL-" + sha256_json({"dataset": dataset, "model": model, "prompt": prompt, "metric": metric, "view": "PRIMARY_FAIL_STRICT"}).upper()[:16],
                    "run_id": run_id_for(str(dataset), str(model), str(prompt)),
                    "dataset": dataset,
                    "task": TASK,
                    "model_name": model,
                    "prompt_variant": prompt,
                    "candidate_ordering": "CANONICAL_FROZEN_ORDER",
                    "history_length": "visible_last_10_or_available",
                    "k": METRIC_K if metric != "mrr" else "",
                    "metric": metric,
                    "value": format_float(value),
                    "evidence_view": "PRIMARY_FAIL_STRICT",
                    "planned_users": planned,
                    "strict_valid_users": strict_valid,
                    "backend_success_users": backend_success,
                    "source_file": OUTPUT_TABLES["per_request_stats"].as_posix(),
                    "config_path": CONFIG_PATH.as_posix(),
                    "timestamp": utc_now(),
                    "status": status,
                    "reason": "Fail-strict denominator includes structurally invalid and backend-failed rows as zero utility.",
                }
            )
    return pd.DataFrame(rows)


def build_complete_case_results(per_request_stats: pd.DataFrame, strict_parsing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model, prompt), group in per_request_stats.groupby(["dataset", "model", "prompt_strategy"], sort=False):
        valid = group.loc[group["strict_parse_valid"].eq("true")]
        values = {
            "hr_at_5": float(pd.to_numeric(valid["hit_at_5"]).mean()) if len(valid) else math.nan,
            "ndcg_at_5": float(pd.to_numeric(valid["ndcg_at_5"]).mean()) if len(valid) else math.nan,
            "mrr": float(pd.to_numeric(valid["mrr"]).mean()) if len(valid) else math.nan,
        }
        for metric, value in values.items():
            rows.append(
                {
                    "result_id": "G1-CC-" + sha256_json({"dataset": dataset, "model": model, "prompt": prompt, "metric": metric}).upper()[:16],
                    "run_id": run_id_for(str(dataset), str(model), str(prompt)),
                    "dataset": dataset,
                    "task": TASK,
                    "model_name": model,
                    "prompt_variant": prompt,
                    "metric": metric,
                    "k": METRIC_K if metric != "mrr" else "",
                    "value": "" if not len(valid) else format_float(value),
                    "evidence_view": "COMPLETE_CASE",
                    "planned_users": len(group),
                    "evaluated_users": len(valid),
                    "status": "COMPUTED" if len(valid) else "BLOCKED_EMPTY_COMPLETE_CASE",
                    "reason": "Complete-case diagnostic includes only strict-valid parsed outputs.",
                }
            )
    return pd.DataFrame(rows)


def build_reliability_results(request_universe: pd.DataFrame, execution_state: pd.DataFrame, strict_parsing: pd.DataFrame) -> pd.DataFrame:
    merged = execution_state.merge(
        strict_parsing[
            [
                "canonical_request_id",
                "strict_parse_valid",
                "terminal_structural_status",
                "duplicate_count",
                "length_violation",
                "invalid_candidate_reference_count",
                "hallucination_count",
                "parse_status",
            ]
        ],
        on="canonical_request_id",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for (dataset, model, prompt), group in merged.groupby(["dataset", "model", "prompt_strategy"], sort=False):
        planned = len(group)
        terminal = int(group["terminal"].eq("true").sum())
        backend_success = int(group["backend_success"].eq("true").sum())
        strict_valid = int(group["strict_parse_valid"].eq("true").sum())
        structurally_invalid = int((group["backend_success"].eq("true") & group["strict_parse_valid"].ne("true")).sum())
        rows.append(
            {
                "dataset": dataset,
                "task": TASK,
                "model_name": model,
                "prompt_variant": prompt,
                "planned_N": planned,
                "terminal_N": terminal,
                "backend_success_N": backend_success,
                "backend_failure_N": int(group["terminal"].eq("true").sum() - backend_success),
                "strict_valid_N": strict_valid,
                "structurally_invalid_N": structurally_invalid,
                "strict_valid_rate": format_float(strict_valid / planned if planned else math.nan),
                "duplicate_violation_count": int(pd.to_numeric(group["duplicate_count"], errors="coerce").fillna(0).sum()),
                "length_violation_count": int(group["length_violation"].eq("true").sum()),
                "invalid_candidate_count": int(pd.to_numeric(group["invalid_candidate_reference_count"], errors="coerce").fillna(0).sum()),
                "hallucination_out_of_candidate_count": int(pd.to_numeric(group["hallucination_count"], errors="coerce").fillna(0).sum()),
                "retry_count": int(pd.to_numeric(group["retry_count"], errors="coerce").fillna(0).sum()),
                "parse_status_counts": stable_json({str(key): int(value) for key, value in group["parse_status"].value_counts().sort_index().items()}),
                "evidence_view": "RELIABILITY",
                "status": "COMPUTED" if terminal == planned else "PENDING_TERMINAL_EVIDENCE",
            }
        )
    return pd.DataFrame(rows)


def build_beyond_utility_results(root: Path, inputs: G1Inputs, per_request_stats: pd.DataFrame, execution_state: pd.DataFrame) -> pd.DataFrame:
    if execution_state.empty or execution_state["terminal"].ne("true").any():
        return build_pending_beyond_utility_results(per_request_stats)
    specs = {spec.name: spec for spec in dataset_specs(load_config(root, Path(str(inputs.config["inputs"]["f1bx_config"]))))}
    materials = {dataset: build_dataset_material(root, specs[dataset]) for dataset in DATASET_ORDER}
    mostpop_stats = {}
    for dataset in DATASET_ORDER:
        recs = inputs.baseline_recommendations.loc[
            inputs.baseline_recommendations["dataset"].eq(dataset)
            & inputs.baseline_recommendations["method"].eq("MostPop")
        ].copy()
        mostpop_stats[dataset] = baseline_per_user_stats(recs, materials[dataset])
    rows: list[dict[str, Any]] = []
    for (dataset, model, prompt), group in per_request_stats.groupby(["dataset", "model", "prompt_strategy"], sort=False):
        material = materials[str(dataset)]
        valid = group.loc[group["strict_parse_valid"].eq("true")].copy()
        execution = {
            "run_id": run_id_for(str(dataset), str(model), str(prompt)),
            "dataset": dataset,
            "method": model,
            "cohort_id": material.spec.cohort_id,
            "recommendation_output_hash": frame_hash(valid),
        }
        if valid.empty:
            for metric in BEYOND_METRICS:
                rows.append(g1_metric_row(execution, metric, "", "BLOCKED_EMPTY_STRICT_VALID", "no strict-valid G1 rows in cell", len(group), len(group), {}, prompt))
            continue
        top_lists = [list(items) for items in valid["top_items"]]
        ranks = [int(rank) if str(rank).strip() else TOP_K + 1 for rank in valid["rank_one_based"]]
        hits = [int(value) for value in valid["hit_at_5"]]
        targets = valid["positive_item_id"].astype(str).tolist()
        histories = [material.history_items_by_user.get(str(user_id), []) for user_id in valid["user_id"]]
        mostpop_hit_by_user = mostpop_stats[str(dataset)].set_index("user_id")["hit_at_5"].astype(float).to_dict()
        values = g1_beyond_metric_values(material, valid, top_lists, ranks, hits, targets, histories, mostpop_hit_by_user)
        for metric in BEYOND_METRICS:
            value, status, reason, undefined_count, value_json = values[metric]
            rows.append(g1_metric_row(execution, metric, value, status, reason, len(group), int(undefined_count) + (len(group) - len(valid)), value_json, prompt))
    return pd.DataFrame(rows)


def build_pending_beyond_utility_results(per_request_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model, prompt), group in per_request_stats.groupby(["dataset", "model", "prompt_strategy"], sort=False):
        execution = {
            "run_id": run_id_for(str(dataset), str(model), str(prompt)),
            "dataset": dataset,
            "method": model,
            "cohort_id": "",
            "recommendation_output_hash": frame_hash(group),
        }
        for metric in BEYOND_METRICS:
            rows.append(
                g1_metric_row(
                    execution,
                    metric,
                    "",
                    "PENDING_TERMINAL_EVIDENCE",
                    "Beyond-utility diagnostic deferred until every planned G1 primary-ranking row has terminal evidence.",
                    len(group),
                    len(group),
                    {},
                    prompt,
                )
            )
    return pd.DataFrame(rows)


def g1_beyond_metric_values(
    material: DatasetMaterial,
    stats: pd.DataFrame,
    top_lists: Sequence[Sequence[str]],
    ranks: Sequence[int],
    hits: Sequence[int],
    targets: Sequence[str],
    histories: Sequence[Sequence[str]],
    mostpop_hit_by_user: Mapping[str, float],
) -> dict[str, tuple[float | str, str, str, int, Mapping[str, Any]]]:
    train_values = list(material.train_counts.values())
    long_tail_threshold = float(np.quantile(train_values, 0.8)) if train_values else math.nan
    candidate_union = set(material.candidates["candidate_item_id"].astype(str))
    catalog_union = set(material.catalog_items)
    unique_top = {item for row in top_lists for item in row}
    exposure = exposure_counts(top_lists, TOP_K)
    exposure_universe = sorted(catalog_union.union(unique_top), key=sort_key)
    exposure_vector = [float(exposure.get(item, 0)) for item in exposure_universe]

    self_info_values = [self_information_at_k(row, material.train_counts, material.total_train_interactions, TOP_K) for row in top_lists]
    shannon_values = [shannon_category_entropy_at_k(row, material.item_features, TOP_K) for row in top_lists]
    ild_values = [ild_jaccard_at_k(row, material.item_features, TOP_K) for row in top_lists]
    calibration_values = [calibration_at_k(history, row, material.item_features, TOP_K) for history, row in zip(histories, top_lists)]
    ser_values = [
        float(hit) - (float(mostpop_hit_by_user.get(str(user_id), 0.0)) * float(hit))
        for user_id, hit in zip(stats["user_id"], hits)
    ]
    popreo = popreo_at_k_from_hits(hits, targets, material.train_counts)
    dpd = dpd_at_k_from_ranks(ranks, [material.history_lengths_by_user.get(str(user_id), 0) for user_id in stats["user_id"]], k=TOP_K)
    return {
        "self_information_at_5": aggregate_optional_metric(self_info_values, "mean over strict-valid users with nonzero train-popularity support"),
        "serendipity_at_5": (float(np.mean(ser_values)), "COMPUTED", "binary relevance unexpectedness against aligned MostPop primitive", 0, {}),
        "arp_at_5": (mean_non_null([average_recommendation_popularity(row, material.train_counts, TOP_K) for row in top_lists]), "COMPUTED", "mean train interaction count over strict-valid top-5", 0, {}),
        "aplt_at_5": (mean_non_null([long_tail_rate_at_k(row, material.train_counts, long_tail_threshold, TOP_K) for row in top_lists]), "COMPUTED", "fraction of strict-valid top-5 items below train-popularity 0.8 quantile", 0, {"long_tail_threshold": long_tail_threshold}),
        "popreo_at_5": recovered_metric_tuple(popreo, "repository PopREO protocol over target popularity quintiles"),
        "item_coverage_at_5": (len(unique_top) / len(candidate_union) if candidate_union else "", "COMPUTED" if candidate_union else "BLOCKED_DATA_REQUIREMENT", "unique strict-valid top-5 items over prompt-compatible candidate union", 0, {"candidate_union_N": len(candidate_union)}),
        "catalog_coverage_at_5": (len(unique_top) / len(catalog_union) if catalog_union else "", "COMPUTED" if catalog_union else "BLOCKED_DATA_REQUIREMENT", "unique strict-valid top-5 items over catalog item universe", 0, {"catalog_N": len(catalog_union)}),
        "oic_at_5": optional_scalar_tuple(oic_at_k(top_lists, TOP_K), "mean pairwise top-5 overlap over strict-valid users"),
        "shannon_category_entropy_at_5": aggregate_optional_metric(shannon_values, "mean per-user top-5 feature entropy over strict-valid users"),
        "gini_at_5": optional_scalar_tuple(gini_index(exposure_vector), "Gini over full catalog exposure vector including zero exposure"),
        "dpd_at_5": recovered_metric_tuple(dpd, "repository DPD median-history split over strict-valid users"),
        "jain_index_at_5": optional_scalar_tuple(jain_index(exposure_vector), "Jain index over full catalog exposure vector including zero exposure"),
        "ild_at_5": aggregate_optional_metric(ild_values, "mean per-user Jaccard ILD over feature-backed strict-valid top-5 items"),
        "calibration_at_5": aggregate_optional_metric(calibration_values, "mean over strict-valid users with train-history and recommendation feature distributions"),
    }


def aggregate_optional_metric(values: Sequence[float | None], reason: str) -> tuple[float | str, str, str, int, Mapping[str, Any]]:
    defined = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    undefined = len(values) - len(defined)
    if not defined:
        return "", "BLOCKED_DATA_REQUIREMENT", reason + "; no defined rows", undefined, {"defined_user_count": 0}
    return float(np.mean(defined)), "COMPUTED", reason, undefined, {"defined_user_count": len(defined)}


def mean_non_null(values: Sequence[float | None]) -> float | str:
    defined = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(defined)) if defined else ""


def optional_scalar_tuple(value: float | None, reason: str) -> tuple[float | str, str, str, int, Mapping[str, Any]]:
    if value is None or not math.isfinite(float(value)):
        return "", "NOT_APPLICABLE", reason + "; scalar undefined", 0, {}
    return float(value), "COMPUTED", reason, 0, {}


def recovered_metric_tuple(result: Mapping[str, Any], reason: str) -> tuple[float | str, str, str, int, Mapping[str, Any]]:
    status = str(result.get("computation_status", ""))
    if status == "COMPUTED":
        return float(result["metric_value"]), "COMPUTED", reason, 0, result
    return "", "DEGENERATE_PROTOCOL_INPUT", str(result.get("reason", reason)), 0, result


def g1_metric_row(
    execution: Mapping[str, Any],
    metric: str,
    value: float | str,
    status: str,
    reason: str,
    planned_users: int,
    undefined_user_count: int,
    value_json: Mapping[str, Any],
    prompt: str,
) -> dict[str, Any]:
    value_text = "" if value == "" else format_float(float(value))
    return {
        "result_id": "G1-BEYOND-" + sha256_json({"run_id": execution["run_id"], "metric": metric, "status": status, "value": value_text}).upper()[:16],
        "run_id": execution["run_id"],
        "dataset": execution["dataset"],
        "task": TASK,
        "model_name": execution["method"],
        "prompt_variant": prompt,
        "cohort_id": execution["cohort_id"],
        "metric": metric,
        "k": TOP_K if metric.endswith("_at_5") else "",
        "value": value_text,
        "value_json": stable_json(value_json) if value_json else "",
        "status": status,
        "evidence_view": "BEYOND_UTILITY_STRICT_VALID",
        "planned_users": int(planned_users),
        "undefined_user_count": int(undefined_user_count),
        "source_recommendation_hash": execution["recommendation_output_hash"],
        "reason": reason,
    }


def build_foodcom_warm_cold(per_request_stats: pd.DataFrame, strict_parsing: pd.DataFrame) -> pd.DataFrame:
    food = per_request_stats.loc[per_request_stats["dataset"].eq("foodcom")].copy()
    rows: list[dict[str, Any]] = []
    scopes = [
        ("ALL_USERS", food),
        ("WARM_USERS", food.loc[food["foodcom_history_class"].eq("WARM_USERS")]),
        ("TRUE_ZERO_HISTORY_USERS", food.loc[food["foodcom_history_class"].eq("TRUE_ZERO_HISTORY_USERS")]),
    ]
    for scope, scope_frame in scopes:
        for (model, prompt), group in scope_frame.groupby(["model", "prompt_strategy"], sort=False):
            planned = len(group)
            strict_valid = int(group["strict_parse_valid"].eq("true").sum())
            rows.extend(
                [
                    foodcom_metric_row(scope, model, prompt, "hr_at_5", METRIC_K, group["hit_at_5"], planned, strict_valid),
                    foodcom_metric_row(scope, model, prompt, "ndcg_at_5", METRIC_K, group["ndcg_at_5"], planned, strict_valid),
                    foodcom_metric_row(scope, model, prompt, "mrr", "", group["mrr"], planned, strict_valid),
                    {
                        "scope": scope,
                        "dataset": "foodcom",
                        "model_name": model,
                        "prompt_variant": prompt,
                        "metric": "strict_valid_rate",
                        "k": "",
                        "value": format_float(strict_valid / planned if planned else math.nan),
                        "planned_users": planned,
                        "strict_valid_users": strict_valid,
                        "evidence_view": "FOODCOM_WARM_COLD_DIAGNOSTIC",
                        "status": "COMPUTED" if planned else "BLOCKED_EMPTY_SCOPE",
                    },
                ]
            )
    return pd.DataFrame(rows)


def foodcom_metric_row(scope: str, model: str, prompt: str, metric: str, k: int | str, values: pd.Series, planned: int, strict_valid: int) -> dict[str, Any]:
    value = float(pd.to_numeric(values, errors="coerce").fillna(0).mean()) if planned else math.nan
    return {
        "scope": scope,
        "dataset": "foodcom",
        "model_name": model,
        "prompt_variant": prompt,
        "metric": metric,
        "k": k,
        "value": format_float(value),
        "planned_users": planned,
        "strict_valid_users": strict_valid,
        "evidence_view": "FOODCOM_WARM_COLD_DIAGNOSTIC",
        "status": "COMPUTED" if planned else "BLOCKED_EMPTY_SCOPE",
    }


def build_runtime_results(execution_state: pd.DataFrame) -> pd.DataFrame:
    terminal = execution_state.loc[execution_state["terminal"].eq("true")].copy()
    scopes = [
        ("overall", []),
        ("by_model", ["model"]),
        ("by_dataset", ["dataset"]),
        ("by_prompt", ["prompt_strategy"]),
        ("by_dataset_model_prompt", ["dataset", "model", "prompt_strategy"]),
    ]
    rows: list[dict[str, Any]] = []
    for scope_name, group_cols in scopes:
        grouped = [((), terminal)] if not group_cols else terminal.groupby(group_cols, sort=False)
        for keys, group in grouped:
            if group_cols and not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(group_cols, keys if group_cols else ()))
            values = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna().to_numpy(dtype=float)
            total_seconds = float(values.sum()) if len(values) else math.nan
            rows.append(
                {
                    "runtime_scope": scope_name,
                    "dataset": key_map.get("dataset", "ALL"),
                    "model": key_map.get("model", "ALL"),
                    "prompt_strategy": key_map.get("prompt_strategy", "ALL"),
                    "request_count": int(len(values)),
                    "mean_seconds": format_float(float(values.mean())) if len(values) else "",
                    "median_seconds": format_float(float(np.median(values))) if len(values) else "",
                    "min_seconds": format_float(float(values.min())) if len(values) else "",
                    "max_seconds": format_float(float(values.max())) if len(values) else "",
                    "p90_seconds": format_float(float(np.percentile(values, 90))) if len(values) else "",
                    "p95_seconds": format_float(float(np.percentile(values, 95))) if len(values) else "",
                    "p99_seconds": format_float(float(np.percentile(values, 99))) if len(values) else "",
                    "throughput_per_hour": format_float((3600.0 * len(values)) / total_seconds) if len(values) and total_seconds > 0 else "",
                    "total_wall_clock_inference_seconds": format_float(total_seconds) if len(values) else "",
                    "paid_api_marginal_cost": "0",
                    "local_computational_cost": "local_ollama_runtime_only",
                    "status": "COMPUTED" if len(values) else "PENDING_TERMINAL_EVIDENCE",
                }
            )
    return pd.DataFrame(rows)


def build_completeness_results(
    request_universe: pd.DataFrame,
    execution_state: pd.DataFrame,
    utility: pd.DataFrame,
    reliability: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model, prompt), group in execution_state.groupby(["dataset", "model", "prompt_strategy"], sort=False):
        planned = len(group)
        terminal = int(group["terminal"].eq("true").sum())
        rows.append(
            {
                "dataset": dataset,
                "task": TASK,
                "model_name": model,
                "prompt_variant": prompt,
                "planned_N": planned,
                "terminal_N": terminal,
                "utility_metrics_expected": 3,
                "utility_metrics_computed": int(utility.loc[utility["dataset"].eq(dataset) & utility["model_name"].eq(model) & utility["prompt_variant"].eq(prompt)]["status"].eq("COMPUTED").sum()),
                "reliability_rows": int(len(reliability.loc[reliability["dataset"].eq(dataset) & reliability["model_name"].eq(model) & reliability["prompt_variant"].eq(prompt)])),
                "cell_status": "COMPUTED" if terminal == planned else "PENDING_TERMINAL_EVIDENCE",
            }
        )
    return pd.DataFrame(rows)


def build_resume_validation(
    request_universe: pd.DataFrame,
    g0_reuse: pd.DataFrame,
    initial_request_records: Sequence[Mapping[str, Any]],
    initial_response_records: Sequence[Mapping[str, Any]],
    request_records: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
    sent_now: int,
    *,
    resume: bool,
    invocation_start: str,
    invocation_end: str,
) -> pd.DataFrame:
    planned = len(request_universe)
    reused = int(g0_reuse["reused_in_g1"].eq("true").sum()) if not g0_reuse.empty else 0
    current_req_counts = count_by_id(request_records)
    current_resp_counts = count_by_id(response_records)
    duplicate_requests = sum(max(0, count - 1) for count in current_req_counts.values())
    duplicate_responses = sum(max(0, count - 1) for count in current_resp_counts.values())
    completed_new = sum(1 for value in current_resp_counts.values() if value == 1)
    terminal = reused + completed_new
    raw_hash_before = response_hash_manifest(initial_response_records)
    raw_hash_after = response_hash_manifest(response_records)
    status = "PENDING_FULL_COMPLETION"
    reason = "G1 is not complete; explicit end-of-run resume validation must be repeated after all canonical requests are terminal."
    if resume:
        if terminal == planned and sent_now == 0 and raw_hash_before == raw_hash_after and duplicate_requests == 0 and duplicate_responses == 0:
            status = "PASS"
            reason = "Resume mode skipped all completed canonical G1 requests and left raw hashes unchanged."
        else:
            status = "FAIL"
            reason = "Resume mode did not satisfy the zero-new-call idempotence gate."
    return pd.DataFrame(
        [
            {
                "resume_mode": bool_text(resume),
                "invocation_started_at_utc": invocation_start,
                "invocation_completed_at_utc": invocation_end,
                "planned_requests": planned,
                "g0_reused_requests": reused,
                "completed_new_g1_requests": completed_new,
                "completed_canonical_requests": terminal,
                "first_run_llm_calls": len(response_records) if not resume else len(initial_response_records),
                "resume_run_llm_calls": sent_now if resume else "",
                "target_resume_run_calls": 0,
                "raw_response_hashes_unchanged": bool_text(raw_hash_before == raw_hash_after),
                "duplicate_request_rows": duplicate_requests,
                "duplicate_response_rows": duplicate_responses,
                "no_new_llm_calls_for_completed_requests": bool_text(resume and sent_now == 0),
                "status": status,
                "reason": reason,
            }
        ]
    )


def response_hash_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(record.get("canonical_request_id") or record.get("request_id")): str(record.get("raw_response_hash", ""))
        for record in records
    }


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    inputs: G1Inputs,
    request_universe: pd.DataFrame,
    physical_execution_order: pd.DataFrame,
    g0_reuse: pd.DataFrame,
    execution_state: pd.DataFrame,
    raw_inventory: pd.DataFrame,
    strict_parsing: pd.DataFrame,
    utility: pd.DataFrame,
    beyond: pd.DataFrame,
    foodcom: pd.DataFrame,
    runtime: pd.DataFrame,
    completeness: pd.DataFrame,
    resume_validation: pd.DataFrame,
    model_preflight: pd.DataFrame,
    protocol_defects: Sequence[str],
    initial_git_status: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any, terminal_required: bool = True) -> None:
        rows.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "terminal_required": bool_text(terminal_required),
                "evidence": stable_json(evidence) if isinstance(evidence, (dict, list)) else str(evidence),
            }
        )

    budget = {str(row["budget_item"]): int(row["request_count"]) for row in inputs.f0_budget.to_dict("records")}
    dataset_users = request_universe.drop_duplicates(["dataset", "user_id"]).groupby("dataset").size().to_dict()
    cells = request_universe.groupby(["dataset", "model", "prompt_strategy"], sort=False).ngroups
    terminal = int(execution_state["terminal"].eq("true").sum())
    new_terminal = int(execution_state.loc[execution_state["evidence_origin"].eq("G1_NEW"), "terminal"].eq("true").sum())
    reused = int(g0_reuse["reused_in_g1"].eq("true").sum()) if not g0_reuse.empty else 0
    strict_terminal = strict_parsing.loc[strict_parsing["status"].eq("COMPUTED")]
    forbidden_enabled = {
        "RERANKING": bool(config["execution"].get("allow_reranking_execution")),
        "HISTORY": bool(config["execution"].get("allow_history_grid_execution")),
        "CANDDIF": bool(config["execution"].get("allow_canddif_execution")),
        "PROFILE": bool(config["execution"].get("allow_profile_generation")),
        "POOL_SIZE_SENSITIVITY": bool(config["execution"].get("allow_pool_size_sensitivity")),
    }

    current_head = git_output(root, ["rev-parse", "HEAD"])
    base_checkpoint = str(config["phase"].get("expected_base_checkpoint", ""))
    g1_paths = [CONFIG_PATH, Path("src/extension/g1_primary_ranking_execution.py"), Path("tests/test_extension_g1_primary_ranking_execution.py")]
    check("scientific_base_checkpoint_configured", base_checkpoint == SCIENTIFIC_BASE_CHECKPOINT, base_checkpoint)
    check("scientific_base_is_ancestor_of_execution_head", git_is_ancestor(root, base_checkpoint, current_head), {"base": base_checkpoint, "head": current_head})
    check("execution_head_recorded", len(current_head) == 40, current_head)
    check("precondition_working_tree_clean_at_start", initial_git_status.strip() == "", initial_git_status or "clean")
    check("g1_implementation_paths_tracked", all(git_tracked(root, path) for path in g1_paths), [path.as_posix() for path in g1_paths])
    check("f0_manifest_validated", inputs.f0_manifest.get("decision_token") == "FINAL_EXTENSION_FREEZE_VALIDATED", inputs.f0_manifest.get("decision_token"))
    check("f0r_manifest_validated", inputs.f0r_manifest.get("decision_token") == "F0_FINAL_SANITY_VALIDATED", inputs.f0r_manifest.get("decision_token"))
    check("g0_manifest_validated", inputs.g0_execution_manifest.get("decision_token") == "G0_CANONICAL_LLM_CANARY_VALIDATED", inputs.g0_execution_manifest.get("decision_token"))
    check("f1a_manifest_validated", inputs.f1a_manifest.get("decision_token") == "F1A_OFFLINE_METRIC_CLOSURE_VALIDATED", inputs.f1a_manifest.get("decision_token"))
    check("f1arr_manifest_validated", inputs.f1arr_manifest.get("decision_token") == "F1ARR_POPREO_DPD_RECOVERED", inputs.f1arr_manifest.get("decision_token"))
    check("f1bzq_manifest_validated", inputs.f1bzq_manifest.get("decision_token") == "F1BZQ_AUTHORITATIVE_RERANKING_POOLS_VALIDATED", inputs.f1bzq_manifest.get("decision_token"))
    check("active_datasets_exact", tuple(request_universe["dataset"].drop_duplicates()) == DATASET_ORDER, request_universe["dataset"].drop_duplicates().tolist())
    check("video_games_absent", "video_games" not in set(request_universe["dataset"].astype(str)) and not request_universe.astype(str).apply(lambda col: col.str.contains("video_games|video games|videogames", case=False, regex=True)).any().any(), "absent")
    check("primary_universe_29292", len(request_universe) == 29292 and request_universe["canonical_request_id"].nunique() == 29292, {"rows": len(request_universe), "unique": request_universe["canonical_request_id"].nunique()})
    check("dataset_user_counts", {str(key): int(value) for key, value in dataset_users.items()} == EXPECTED_DATASET_USERS, {str(key): int(value) for key, value in dataset_users.items()})
    check("thirty_cells", cells == 30 and len(completeness) == 30, {"cells": cells, "completeness_rows": len(completeness)})
    check("logical_request_order_dataset_user_prompt_model", logical_order_is_dataset_user_prompt_model(request_universe), request_universe[["request_order", "dataset", "user_id", "prompt_strategy", "model"]].head(12).to_dict("records"))
    check("physical_execution_order_model_grouped", physical_order_is_model_grouped(physical_execution_order), {"rows": len(physical_execution_order), "model_switches": physical_model_switch_count(physical_execution_order)})
    check("physical_order_covers_logical_universe_once", set(physical_execution_order["canonical_request_id"].astype(str)) == set(request_universe["canonical_request_id"].astype(str)) and physical_execution_order["canonical_request_id"].nunique() == len(request_universe), {"physical_rows": len(physical_execution_order), "universe_rows": len(request_universe)})
    check("models_prompts_exact", set(request_universe["model"]) == set(MODEL_ORDER) and set(request_universe["prompt_strategy"]) == set(PROMPT_ORDER), request_universe[["model", "prompt_strategy"]].nunique().to_dict())
    check("ranking_only", request_universe["task"].eq(TASK).all(), request_universe["task"].value_counts().to_dict())
    check("ranking_contract_frozen", request_universe["candidate_count"].astype(int).eq(20).all() and request_universe["requested_output_length"].astype(int).eq(10).all() and request_universe["parser_expected_length"].astype(int).eq(10).all() and request_universe["metric_k"].astype(int).eq(5).all(), "20/10/10/5")
    check("candidate_hashes_frozen_present", request_universe["candidate_hash"].astype(str).str.len().eq(64).all(), request_universe["candidate_hash"].head().tolist())
    check("prompt_hashes_frozen_present", request_universe["prompt_hash"].astype(str).str.len().eq(64).all(), request_universe["prompt_hash"].head().tolist())
    check("model_identities_frozen", model_identity_matches_g0(inputs, model_preflight), model_preflight[["model", "available", "digest"]].to_dict("records") if not model_preflight.empty else [])
    check("g0_ranking_reuse_36", len(g0_reuse) == 36 and reused == 36, {"rows": len(g0_reuse), "reused": reused})
    check("expected_new_calls_29256", len(request_universe) - reused == 29256, len(request_universe) - reused)
    check("no_g0_reranking_reuse", set(g0_reuse["task"]) == {TASK}, g0_reuse["task"].value_counts().to_dict())
    check("g0_identity_match_exact", g0_reuse["all_identity_fields_match"].eq("true").all(), g0_reuse.loc[g0_reuse["all_identity_fields_match"].ne("true")].head().to_dict("records"))
    check("no_duplicate_canonical_execution", execution_state["canonical_request_id"].is_unique and execution_state["request_state"].ne("invalid_duplicate_evidence").all(), execution_state["request_state"].value_counts().to_dict())
    check("no_interrupted_raw_state", execution_state["request_state"].ne("ambiguous_interrupted").all(), execution_state["request_state"].value_counts().to_dict())
    check("terminal_requests_29292", terminal == 29292, terminal)
    check("new_g1_terminal_requests_29256", new_terminal == 29256, new_terminal)
    check("strict_parser_accounted_terminal", len(strict_terminal) == terminal, {"strict_rows": len(strict_terminal), "terminal": terminal})
    check("invalid_outputs_not_retried", strict_parsing["invalid_output_retried"].eq("false").all(), strict_parsing["invalid_output_retried"].value_counts().to_dict())
    check("raw_inventory_matches_terminal", len(raw_inventory) == terminal, {"raw_inventory": len(raw_inventory), "terminal": terminal})
    check("utility_30_cells_computed", len(utility) == 90 and utility["status"].eq("COMPUTED").all(), utility["status"].value_counts().to_dict())
    check("beyond_utility_30_cells_accounted", len(beyond) == 30 * len(BEYOND_METRICS), {"rows": len(beyond), "metrics": len(BEYOND_METRICS)})
    check("foodcom_warm_cold_scopes", set(foodcom["scope"]) == {"ALL_USERS", "WARM_USERS", "TRUE_ZERO_HISTORY_USERS"} if not foodcom.empty else False, sorted(foodcom["scope"].unique()) if not foodcom.empty else [])
    check("runtime_recorded", not runtime.empty and terminal == 29292, {"runtime_rows": len(runtime), "terminal": terminal})
    check("completeness_30_computed", len(completeness) == 30 and completeness["cell_status"].eq("COMPUTED").all(), completeness["cell_status"].value_counts().to_dict())
    check("resume_idempotent_after_full_run", resume_validation["status"].iloc[0] == "PASS", resume_validation.to_dict("records"))
    check("budget_invariance", all(budget.get(key) == value for key, value in REQUIRED_BUDGET.items()), budget)
    check("forbidden_execution_families_zero", not any(forbidden_enabled.values()), forbidden_enabled)
    check("zero_external_api_download_training", not bool(config["execution"]["allow_external_api_calls"]) and not bool(config["execution"]["allow_downloads"]) and not bool(config["execution"]["allow_network_downloads"]) and not bool(config["execution"]["allow_training"]) and not bool(config["execution"]["allow_tuning"]), config["execution"])
    check("protocol_defects_absent", not protocol_defects, list(protocol_defects))
    return pd.DataFrame(rows)


def model_identity_matches_g0(inputs: G1Inputs, model_preflight: pd.DataFrame) -> bool:
    if model_preflight.empty or "digest" not in model_preflight:
        return False
    expected = {
        str(row["model"]): str(row["digest"])
        for row in inputs.g0_execution_manifest.get("model_preflight", [])
        if str(row.get("model")) in MODEL_ORDER
    }
    observed = model_preflight.set_index("model")["digest"].astype(str).to_dict()
    statuses = model_preflight.set_index("model")["current_model_identity_status"].astype(str).to_dict() if "current_model_identity_status" in model_preflight else {}
    return bool(expected) and all(
        observed.get(model) == expected.get(model) and statuses.get(model) == MODEL_AVAILABLE_MATCHED
        for model in MODEL_ORDER
    )


def build_manifests(
    root: Path,
    config: Mapping[str, Any],
    inputs: G1Inputs,
    decision: str,
    request_universe: pd.DataFrame,
    physical_execution_order: pd.DataFrame,
    g0_reuse: pd.DataFrame,
    execution_state: pd.DataFrame,
    raw_inventory: pd.DataFrame,
    strict_parsing: pd.DataFrame,
    utility: pd.DataFrame,
    beyond: pd.DataFrame,
    foodcom: pd.DataFrame,
    runtime: pd.DataFrame,
    completeness: pd.DataFrame,
    resume_validation: pd.DataFrame,
    validation: pd.DataFrame,
    model_preflight: pd.DataFrame,
    sent_now: int,
) -> dict[str, dict[str, Any]]:
    budget = {str(row["budget_item"]): int(row["request_count"]) for row in inputs.f0_budget.to_dict("records")}
    reused = int(g0_reuse["reused_in_g1"].eq("true").sum()) if not g0_reuse.empty else 0
    terminal = int(execution_state["terminal"].eq("true").sum())
    new_terminal = int(execution_state.loc[execution_state["evidence_origin"].eq("G1_NEW"), "terminal"].eq("true").sum())
    strict_valid = int(strict_parsing["strict_parse_valid"].eq("true").sum())
    structurally_invalid = int((strict_parsing["strict_parse_valid"].ne("true") & strict_parsing["terminal_structural_status"].ne("BACKEND_FAILURE") & strict_parsing["terminal_structural_status"].ne("NOT_TERMINAL")).sum())
    current_head = git_output(root, ["rev-parse", "HEAD"])
    git_status = git_output(root, ["status", "--short", "--untracked-files=all"])
    source_paths = {
        "config": CONFIG_PATH,
        "runner": Path("src/extension/g1_primary_ranking_execution.py"),
        "tests": Path("tests/test_extension_g1_primary_ranking_execution.py"),
    }
    source_hashes = {
        key: sha256_file(root / path)
        for key, path in source_paths.items()
        if (root / path).exists()
    }
    base = {
        "phase_id": PHASE_ID,
        "generated_at": utc_now(),
        "decision_token": decision,
        "scientific_base_checkpoint": SCIENTIFIC_BASE_CHECKPOINT,
        "execution_code_commit": current_head,
        "head": current_head,
        "base_checkpoint_is_ancestor": git_is_ancestor(root, SCIENTIFIC_BASE_CHECKPOINT, current_head),
        "working_tree_clean": git_status.strip() == "",
        "working_tree_status": git_status,
        "config_hash": source_hashes.get("config", ""),
        "runner_source_hash": source_hashes.get("runner", ""),
        "test_source_hash": source_hashes.get("tests", ""),
        "source_hashes": source_hashes,
        "active_datasets": list(DATASET_ORDER),
        "models": list(MODEL_ORDER),
        "prompt_strategies": list(PROMPT_ORDER),
        "tasks": [TASK],
        "planned_requests": int(len(request_universe)),
        "canonical_universe": int(len(request_universe)),
        "g0_reused_ranking_requests": reused,
        "new_g1_ollama_calls_expected": int(len(request_universe) - reused),
        "new_g1_terminal_requests": new_terminal,
        "terminal_requests": terminal,
        "backend_successful_requests": int(execution_state["backend_success"].eq("true").sum()),
        "strict_valid_requests": strict_valid,
        "structurally_invalid_requests": structurally_invalid,
        "llm_inference_calls_executed_in_this_invocation": int(sent_now),
        "incremental_experiment_identities_created": 0,
        "training_runs_executed": 0,
        "tuning_runs_executed": 0,
        "reranking_requests_executed": 0,
        "history_requests_executed": 0,
        "canddif_requests_executed": 0,
        "profile_requests_executed": 0,
        "pool_size_sensitivity_requests_executed": 0,
        "downloads_executed": 0,
        "external_api_calls_executed": 0,
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
        "tables": {key: path.as_posix() for key, path in OUTPUT_TABLES.items()},
        "raw_paths": {
            "raw_dir": RAW_DIR.as_posix(),
            "request_jsonl": REQUEST_JSONL.as_posix(),
            "response_jsonl": RESPONSE_JSONL.as_posix(),
            "event_jsonl": EVENT_JSONL.as_posix(),
            "response_dir": RESPONSE_DIR.as_posix(),
        },
        "frame_hashes": {
            "request_universe": frame_hash(request_universe),
            "physical_execution_order": frame_hash(physical_execution_order),
            "g0_reuse_inventory": frame_hash(g0_reuse),
            "execution_state": frame_hash(execution_state),
            "raw_response_inventory": frame_hash(raw_inventory),
            "strict_parsing": frame_hash(strict_parsing),
            "utility": frame_hash(utility),
            "beyond_utility": frame_hash(beyond),
            "foodcom_warm_cold": frame_hash(foodcom),
            "runtime": frame_hash(runtime),
            "completeness": frame_hash(completeness),
            "resume_validation": frame_hash(resume_validation),
            "validation": frame_hash(validation),
        },
    }
    return {
        "request_manifest": {
            **base,
            "canonical_request_ids": request_universe["canonical_request_id"].tolist(),
            "request_order_hash": sha256_json(request_universe["canonical_request_id"].tolist()),
            "physical_execution_order_hash": sha256_json(physical_execution_order.sort_values("physical_execution_order", kind="mergesort")["canonical_request_id"].tolist()),
        },
        "execution_manifest": {
            **base,
            "execution_accounting": execution_accounting_summary(execution_state, g0_reuse),
            "parse_status_counts": strict_parsing["parse_status"].value_counts().to_dict(),
        },
        "model_runtime_provenance": {
            **base,
            "model_preflight": model_preflight.to_dict("records"),
            "runtime_summary": runtime.to_dict("records"),
        },
        "g0_reuse": {
            **base,
            "g0_reuse_rows": g0_reuse.to_dict("records"),
        },
        "resume_idempotence": {
            **base,
            "resume_validation": resume_validation.to_dict("records"),
        },
        "budget_invariance": {
            **base,
            "f0_budget": budget,
            "required_budget": REQUIRED_BUDGET,
            "budget_invariant": all(budget.get(key) == value for key, value in REQUIRED_BUDGET.items()),
        },
    }


def execution_accounting_summary(execution_state: pd.DataFrame, g0_reuse: pd.DataFrame) -> dict[str, Any]:
    reused = int(g0_reuse["reused_in_g1"].eq("true").sum()) if not g0_reuse.empty else 0
    return {
        "planned": int(len(execution_state)),
        "reused_from_G0": reused,
        "new_backend_calls": int(execution_state.loc[execution_state["evidence_origin"].eq("G1_NEW"), "terminal"].eq("true").sum()),
        "terminal_requests": int(execution_state["terminal"].eq("true").sum()),
        "backend_successes": int(execution_state["backend_success"].eq("true").sum()),
        "backend_failures": int(execution_state["backend_status"].eq("BACKEND_FAILURE").sum()),
        "retries": int(pd.to_numeric(execution_state["retry_count"], errors="coerce").fillna(0).sum()),
        "remaining": int(execution_state["terminal"].ne("true").sum()),
    }


def render_report(
    decision: str,
    model_preflight: pd.DataFrame,
    request_universe: pd.DataFrame,
    g0_reuse: pd.DataFrame,
    execution_state: pd.DataFrame,
    strict_parsing: pd.DataFrame,
    utility: pd.DataFrame,
    beyond: pd.DataFrame,
    foodcom: pd.DataFrame,
    runtime: pd.DataFrame,
    reliability: pd.DataFrame,
    resume_validation: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    reused = int(g0_reuse["reused_in_g1"].eq("true").sum()) if not g0_reuse.empty else 0
    terminal = int(execution_state["terminal"].eq("true").sum()) if not execution_state.empty else 0
    strict_valid = int(strict_parsing["strict_parse_valid"].eq("true").sum()) if not strict_parsing.empty else 0
    invalid = int((strict_parsing["status"].eq("COMPUTED") & strict_parsing["strict_parse_valid"].ne("true")).sum()) if not strict_parsing.empty else 0
    failures = validation.loc[validation["status"].ne("PASS"), "check"].tolist() if not validation.empty else []
    dataset_counts = request_universe.groupby("dataset").size().to_dict()
    model_counts = execution_state.groupby("model")["terminal"].apply(lambda values: int(values.eq("true").sum())).to_dict() if not execution_state.empty else {}
    prompt_counts = execution_state.groupby("prompt_strategy")["terminal"].apply(lambda values: int(values.eq("true").sum())).to_dict() if not execution_state.empty else {}
    current_head = git_output(find_project_root(), ["rev-parse", "HEAD"])
    base_is_ancestor = git_is_ancestor(find_project_root(), SCIENTIFIC_BASE_CHECKPOINT, current_head)
    lines = [
        "# G1 Primary Ranking Execution",
        "",
        "## PRECONDITION",
        f"- Scientific base checkpoint: `{SCIENTIFIC_BASE_CHECKPOINT}`.",
        f"- Execution code commit: `{current_head}`.",
        f"- Base checkpoint ancestor of execution commit: `{bool_text(base_is_ancestor)}`.",
        f"- Decision: `{decision}`.",
        f"- Validation failures: `{len(failures)}`.",
        "",
        "## MODEL PREFLIGHT",
        model_preflight.to_csv(index=False, lineterminator="\n").strip() if not model_preflight.empty else "No model preflight rows.",
        "",
        "## REQUEST UNIVERSE",
        f"- Canonical primary-ranking requests: `{len(request_universe)}`.",
        f"- Dataset request counts: `{stable_json({str(k): int(v) for k, v in dataset_counts.items()})}`.",
        f"- Cells: `{request_universe.groupby(['dataset', 'model', 'prompt_strategy']).ngroups if not request_universe.empty else 0}`.",
        "- Canonical logical order: `dataset_user_prompt_model`.",
        "- Physical execution order: `model_dataset_prompt_user`.",
        "",
        "## G0 REUSE",
        f"- Reused ranking requests: `{reused}`.",
        f"- Identity mismatches: `{int(g0_reuse['all_identity_fields_match'].ne('true').sum()) if not g0_reuse.empty else 0}`.",
        "",
        "## EXECUTION ACCOUNTING",
        f"- Terminal requests: `{terminal}` / `{len(request_universe)}`.",
        f"- New G1 terminal requests: `{int(execution_state.loc[execution_state['evidence_origin'].eq('G1_NEW'), 'terminal'].eq('true').sum()) if not execution_state.empty else 0}` / `29256`.",
        f"- Remaining: `{int(execution_state['terminal'].ne('true').sum()) if not execution_state.empty else len(request_universe)}`.",
        "",
        "## BEAUTY",
        dataset_report_line(execution_state, "beauty"),
        "## SPORTS",
        dataset_report_line(execution_state, "sports"),
        "## ML1M",
        dataset_report_line(execution_state, "ml1m"),
        "## LASTFM",
        dataset_report_line(execution_state, "lastfm"),
        "## FOOD.COM",
        dataset_report_line(execution_state, "foodcom"),
        "## QWEN",
        f"- Terminal requests: `{model_counts.get('qwen2:7b-instruct', 0)}`.",
        "## MISTRAL",
        f"- Terminal requests: `{model_counts.get('mistral:7b-instruct', 0)}`.",
        "## GEMMA",
        f"- Terminal requests: `{model_counts.get('gemma2:9b', 0)}`.",
        "## BASE",
        f"- Terminal requests: `{prompt_counts.get('BASE', 0)}`.",
        "## LLMRANK",
        f"- Terminal requests: `{prompt_counts.get('LLMRANK', 0)}`.",
        "## STRICT VALIDITY",
        f"- Strict valid: `{strict_valid}`.",
        f"- Structurally invalid or nonterminal: `{invalid}`.",
        f"- Parse status counts: `{stable_json({str(k): int(v) for k, v in strict_parsing['parse_status'].value_counts().sort_index().items()}) if not strict_parsing.empty else '{}'} `.",
        "## UTILITY",
        utility_summary(utility),
        "## BEYOND-UTILITY",
        f"- Beyond-utility rows: `{len(beyond)}`.",
        f"- Computed rows: `{int(beyond['status'].eq('COMPUTED').sum()) if not beyond.empty else 0}`.",
        "## FOOD.COM WARM/COLD",
        foodcom_summary(foodcom),
        "## RUNTIME",
        runtime_summary(runtime),
        "## FAILURE / RETRY AUDIT",
        retry_summary(execution_state, reliability),
        "## RESUME",
        resume_validation.to_csv(index=False, lineterminator="\n").strip() if not resume_validation.empty else "No resume validation row.",
        "## BUDGET INVARIANCE",
        "- F0 conceptual budget is checked in `g1_primary_ranking_budget_invariance_manifest.json`.",
        "## TESTS",
        "- Focused G1 tests are in `tests/test_extension_g1_primary_ranking_execution.py`.",
        "## FILES",
        f"- Raw dir: `{RAW_DIR.as_posix()}`.",
        f"- Request universe: `{OUTPUT_TABLES['request_universe'].as_posix()}`.",
        f"- Report: `{REPORT_PATH.as_posix()}`.",
        "## GIT STATUS",
        f"- Status at report render: `{git_output(find_project_root(), ['status', '--short']) or 'clean'}`.",
        "## NEXT STAGE",
        "- Do not start G2 history, G3 CandDif, or G4 reranking from this phase.",
        "",
        decision,
    ]
    return "\n".join(lines) + "\n"


def dataset_report_line(execution_state: pd.DataFrame, dataset: str) -> str:
    if execution_state.empty:
        return "- Terminal requests: `0`."
    group = execution_state.loc[execution_state["dataset"].eq(dataset)]
    return f"- Terminal requests: `{int(group['terminal'].eq('true').sum())}` / `{len(group)}`."


def utility_summary(utility: pd.DataFrame) -> str:
    if utility.empty:
        return "- Utility rows: `0`."
    computed = int(utility["status"].eq("COMPUTED").sum())
    return f"- Utility rows: `{len(utility)}`; computed: `{computed}`."


def foodcom_summary(foodcom: pd.DataFrame) -> str:
    if foodcom.empty:
        return "- Food.com diagnostic rows: `0`."
    scopes = {str(key): int(value) for key, value in foodcom.groupby("scope").size().items()}
    return f"- Diagnostic rows by scope: `{stable_json(scopes)}`."


def runtime_summary(runtime: pd.DataFrame) -> str:
    if runtime.empty:
        return "- Runtime rows: `0`."
    overall = runtime.loc[runtime["runtime_scope"].eq("overall")]
    if overall.empty:
        return f"- Runtime rows: `{len(runtime)}`."
    row = overall.iloc[0]
    return f"- Requests timed: `{row['request_count']}`; total seconds: `{row['total_wall_clock_inference_seconds']}`; throughput/hour: `{row['throughput_per_hour']}`."


def retry_summary(execution_state: pd.DataFrame, reliability: pd.DataFrame) -> str:
    if execution_state.empty:
        return "- Retries: `0`; backend failures: `0`."
    retries = int(pd.to_numeric(execution_state["retry_count"], errors="coerce").fillna(0).sum())
    failures = int(execution_state["backend_status"].eq("BACKEND_FAILURE").sum())
    structural = int(reliability["structurally_invalid_N"].astype(int).sum()) if not reliability.empty else 0
    return f"- Retries: `{retries}`; backend failures: `{failures}`; structurally invalid responses: `{structural}`."


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    ensure_output_dirs(root)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")
    artifact_paths = [*OUTPUT_TABLES.values(), REPORT_PATH, REQUEST_JSONL, RESPONSE_JSONL, EVENT_JSONL]
    for key, rel_path in OUTPUT_MANIFESTS.items():
        payload = dict(outputs["manifests"][key])
        payload["artifact_hashes"] = {
            path.as_posix(): sha256_file(root / path) for path in artifact_paths if (root / path).exists()
        }
        (root / rel_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_phase_notes(root: Path, outputs: Mapping[str, Any], *, validate_only: bool) -> None:
    manifest = outputs["manifests"]["execution_manifest"]
    lines = [
        "",
        f"## {datetime.now(timezone.utc).isoformat(timespec='seconds')} - G1 primary ranking execution",
        "",
        f"- Decision: `{manifest['decision_token']}`.",
        f"- Invocation mode: `{'validate-only' if validate_only else 'execute-or-resume'}`.",
        f"- Planned canonical ranking requests: `{manifest['planned_requests']}`.",
        f"- G0 reused ranking requests: `{manifest['g0_reused_ranking_requests']}`.",
        f"- New G1 terminal requests: `{manifest['new_g1_terminal_requests']}` / `{manifest['new_g1_ollama_calls_expected']}`.",
        f"- Ollama calls in this invocation: `{manifest['llm_inference_calls_executed_in_this_invocation']}`.",
        "- No training, tuning, reranking, history-grid, CandDif, profile, pool-size sensitivity, external API, download, network download, commit, reset, clean, or checkout was requested by this phase.",
    ]
    mode = "a" if (root / NOTES_PATH).exists() else "w"
    with (root / NOTES_PATH).open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def ensure_output_dirs(root: Path) -> None:
    for rel_path in (TABLE_DIR, MANIFEST_DIR, REPORT_PATH.parent, RAW_DIR, RESPONSE_DIR):
        (root / rel_path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

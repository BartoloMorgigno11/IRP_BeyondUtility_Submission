"""G4 F1BZQ reranking protocol freeze and zero-call dry-run.

G4 is an adapted extension reranking experiment. This module materializes the
request-level universe over the already-frozen F1BZQ authoritative candidate
pools, freezes the metric/statistical contracts, and provides a raw-first
resume runner for a later explicitly approved execution.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

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
from src.extension.f1a_offline_metric_closure import format_float
from src.extension.f1bx_execution_stage1 import BEYOND_METRICS, dataset_specs, frame_hash, read_csv, read_json, sort_key
from src.extension.f1bz_final_traditional_execution import foodcom_final_training_events, original_four_training_events
from src.extension.f1bzp_reranking_protocol_resolution import ADAPTED_ALGORITHM, ADAPTED_POLICY
from src.extension.f1bzq_authoritative_reranking_pool_materialization import POOL_STATUS_COMPUTED, SOURCE_POLICY, load_title_map
from src.extension.g0_canonical_llm_canary import (
    G0_RAW_PROTOCOL,
    MODEL_FAMILY,
    RequestResult,
    call_ollama_generate,
    canonical_request_id,
    candidate_items_from_frame,
    classify_state,
    count_by_id,
    single_record_by_id,
    title_for_item,
    write_raw_response_file,
)
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync, positive_label_exposed, read_jsonl
from src.utils.config import find_project_root


PHASE_ID = "g4_f1bzq_reranking_protocol_freeze"
DECISION_VALIDATED = "G4_F1BZQ_RERANKING_PROTOCOL_FROZEN"
DECISION_BLOCKED = "G4_F1BZQ_RERANKING_PROTOCOL_BLOCKED"
SCIENTIFIC_BASE_CHECKPOINT = "0465d7aa5f9fe9231be94dbe8d45649864bb4878"
TASK = "reranking"
SCIENTIFIC_STATUS = "ADAPTED_EXTENSION_NOT_PAPER_EXACT"
HISTORY_CONTRACT = "visible_last_10_or_available"
CANDIDATE_SOURCE_PROTOCOL = "F1BZQ_AUTHORITATIVE_BALANCED_RERANKING_POOL"
NO_REUSE_POLICY = "G4_RESPONSE_REUSE_EXPLICITLY_ZERO"

CONFIG_PATH = Path("configs/g4_f1bzq_reranking_execution.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_PATH = Path("report/g4_f1bzq_reranking_protocol_freeze.md")
NOTES_PATH = Path("report/notes.md")
RAW_DIR = Path("results/llm/extension_canonical/g4_f1bzq_reranking")
REQUEST_JSONL = RAW_DIR / "requests.jsonl"
RESPONSE_JSONL = RAW_DIR / "responses.jsonl"
EVENT_JSONL = RAW_DIR / "events.jsonl"
RESPONSE_DIR = RAW_DIR / "responses"

DATASET_ORDER = tuple(ACTIVE_EXTENSION_DATASETS)
MODEL_ORDER = tuple(FUTURE_MODELS)
PROMPT_ORDER = tuple(PROMPT_STRATEGIES)
EXPECTED_USERS = {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}
EXPECTED_LOGICAL_REQUESTS = 29_292
EXPECTED_CELLS = 30

PRIMARY_UTILITY_METRICS = ("hr_at_5", "ndcg_at_5", "mrr")
RERANKING_DIAGNOSTICS = ("candidate_recall_at_20", "conditional_hr_at_5", "conditional_ndcg_at_5")
EXCLUDED_METRICS = {
    "canddif_hr_at_5": "excluded_from_g4_by_protocol_freeze",
    "canddif_ndcg_at_5": "excluded_from_g4_by_protocol_freeze",
    "popreo_at_5": "blocked_global_protocol",
    "dpd_at_5": "blocked_global_protocol",
    "monetary_marginal_cost": "not_applicable_local_ollama",
}
RELIABILITY_METRICS = (
    "strict_parse_validity",
    "structural_invalidity_rate",
    "invalid_reason_distribution",
    "duplicate_rate",
    "hallucination",
    "backend_failure_rate",
    "latency",
    "mean_latency",
    "median_latency",
    "p90_latency",
    "p95_latency",
    "p99_latency",
    "throughput",
    "token_counts",
    "hardware_runtime_provenance",
)

OUTPUT_TABLES = {
    "request_universe": TABLE_DIR / "g4_f1bzq_reranking_request_universe.csv",
    "execution_order": TABLE_DIR / "g4_f1bzq_reranking_execution_order.csv",
    "candidate_hashes": TABLE_DIR / "g4_f1bzq_candidate_hashes.csv",
    "model_digests": TABLE_DIR / "g4_f1bzq_model_digest_freeze.csv",
    "metric_contract": TABLE_DIR / "g4_f1bzq_metric_denominator_contract.csv",
    "candidate_recall": TABLE_DIR / "g4_f1bzq_candidate_recall_at_20.csv",
    "statistical_comparisons": TABLE_DIR / "g4_f1bzq_statistical_comparisons.csv",
    "response_reuse_policy": TABLE_DIR / "g4_f1bzq_response_reuse_policy.csv",
    "dry_run_accounting": TABLE_DIR / "g4_f1bzq_zero_call_dry_run_accounting.csv",
    "retention_plan": TABLE_DIR / "g4_f1bzq_retention_plan.csv",
    "validation": TABLE_DIR / "g4_f1bzq_validation.csv",
}

OUTPUT_MANIFESTS = {
    "protocol": MANIFEST_DIR / "g4_f1bzq_reranking_protocol_freeze.json",
    "request_universe": MANIFEST_DIR / "g4_f1bzq_request_universe_manifest.json",
    "zero_call_dry_run": MANIFEST_DIR / "g4_f1bzq_zero_call_dry_run_manifest.json",
}

SUPPORT_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/g4_f1bzq_reranking_execution.py"),
    Path("tests/test_extension_g4_f1bzq_reranking_execution.py"),
    REPORT_PATH,
)


@dataclass(frozen=True)
class G4Inputs:
    config: dict[str, Any]
    specs: dict[str, Any]
    pools: dict[str, pd.DataFrame]
    title_maps: dict[str, dict[str, str]]
    histories: dict[str, dict[str, list[str]]]
    f0_inference_contract: dict[str, Any]
    f0_metric_matrix: pd.DataFrame
    f0_request_budget: pd.DataFrame
    g0_execution_manifest: dict[str, Any]
    f1bzq_pool_manifest: dict[str, Any]
    f1bzq_source_manifest: dict[str, Any]
    f1bzq_readiness: pd.DataFrame
    f1bzq_candidate_manifest: pd.DataFrame
    f1bzq_candidate_recall: pd.DataFrame
    f1bzq_foodcom_warm_cold: pd.DataFrame
    f1bzq_pool_contributions: pd.DataFrame
    f1bzq_supersession: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or execute G4 F1BZQ reranking protocol.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--dry-run", action="store_true", help="Build the zero-call protocol artifacts. This is the default mode.")
    parser.add_argument("--execute", action="store_true", help="Execute pending G4 Ollama calls after the frozen dry-run gates.")
    parser.add_argument("--resume", action="store_true", help="Skip completed canonical G4 requests by exact raw evidence identity.")
    parser.add_argument("--allow-api-calls", action="store_true", help="Required before any local Ollama generation call.")
    parser.add_argument("--max-new-calls", type=int, default=None, help="Optional cap for a later approved execution batch.")
    parser.add_argument("--no-write", action="store_true", help="Build in memory without writing artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    if args.execute and not args.allow_api_calls:
        print("error=--allow-api-calls is required before any G4 local Ollama inference path is enabled")
        return 1
    outputs = run_g4(
        root,
        Path(args.config),
        execute=bool(args.execute),
        resume=bool(args.resume),
        max_new_calls=args.max_new_calls,
    )
    if not args.no_write:
        write_outputs(root, outputs)
        append_phase_notes(root, outputs, execute=bool(args.execute))
    manifest = outputs["manifests"]["protocol"]
    print(f"decision={manifest['decision_token']}")
    print(f"logical_requests={manifest['logical_requests']}")
    print(f"response_reuse={manifest['response_reuse']}")
    print(f"future_ollama_calls={manifest['future_ollama_calls']}")
    print(f"cells={manifest['cells']}")
    print(f"existing_terminal_g4_evidence={manifest['existing_terminal_g4_evidence']}")
    print(f"llm_inference_calls_executed_in_this_invocation={manifest['llm_inference_calls_executed_in_this_invocation']}")
    print(f"validation_passed={str(manifest['validation_passed']).lower()}")
    return 0 if manifest["decision_token"] == DECISION_VALIDATED else 1


def run_g4(
    root: Path,
    config_path: Path,
    *,
    execute: bool,
    resume: bool,
    max_new_calls: int | None,
) -> dict[str, Any]:
    ensure_output_dirs(root, include_raw=execute)
    config = load_config(root, config_path)
    validate_execution_config(config)
    inputs = load_inputs(root, config)
    model_digests = build_model_digest_freeze(inputs)
    request_universe, contexts = build_request_universe(root, inputs, model_digests)
    execution_order = build_execution_order(request_universe)
    candidate_hashes = build_candidate_hashes(request_universe)
    metric_contract = build_metric_contract()
    candidate_recall = build_candidate_recall_contract(inputs)
    statistical_comparisons = build_statistical_comparisons()
    response_reuse_policy = build_response_reuse_policy(root, request_universe)
    retention_plan = build_retention_plan()
    request_records_before = read_jsonl(root / REQUEST_JSONL)
    response_records_before = read_jsonl(root / RESPONSE_JSONL)
    sent_now = 0
    if execute:
        protocol_defects = protocol_defects_for_execution(root, inputs, request_universe, response_reuse_policy)
        if protocol_defects:
            append_jsonl_fsync(
                root / EVENT_JSONL,
                {"event": "pre_execution_protocol_block", "phase_id": PHASE_ID, "timestamp_utc": utc_now(), "defects": protocol_defects},
            )
        else:
            sent_now = execute_pending_requests(root, config, request_universe, execution_order, contexts, max_new_calls=max_new_calls)
    request_records_after = read_jsonl(root / REQUEST_JSONL)
    response_records_after = read_jsonl(root / RESPONSE_JSONL)
    dry_run_accounting = build_dry_run_accounting(
        request_universe,
        response_reuse_policy,
        request_records_before,
        response_records_before,
        request_records_after,
        response_records_after,
        sent_now,
        execute=execute,
    )
    validation = build_validation(
        root,
        config,
        inputs,
        request_universe,
        execution_order,
        candidate_hashes,
        model_digests,
        metric_contract,
        candidate_recall,
        statistical_comparisons,
        response_reuse_policy,
        dry_run_accounting,
        retention_plan,
        require_zero_call_freeze=not execute,
    )
    decision = DECISION_VALIDATED if bool(validation["status"].eq("PASS").all()) else DECISION_BLOCKED
    manifests = build_manifests(
        root,
        config,
        inputs,
        decision,
        request_universe,
        execution_order,
        candidate_hashes,
        model_digests,
        metric_contract,
        candidate_recall,
        statistical_comparisons,
        response_reuse_policy,
        dry_run_accounting,
        retention_plan,
        validation,
        sent_now,
    )
    report = render_report(decision, request_universe, candidate_hashes, candidate_recall, metric_contract, statistical_comparisons, dry_run_accounting, validation)
    return {
        "request_universe": request_universe,
        "execution_order": execution_order,
        "candidate_hashes": candidate_hashes,
        "model_digests": model_digests,
        "metric_contract": metric_contract,
        "candidate_recall": candidate_recall,
        "statistical_comparisons": statistical_comparisons,
        "response_reuse_policy": response_reuse_policy,
        "dry_run_accounting": dry_run_accounting,
        "retention_plan": retention_plan,
        "validation": validation,
        "manifests": manifests,
        "report": report,
    }


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def validate_execution_config(config: Mapping[str, Any]) -> None:
    execution = config["execution"]
    if bool(execution.get("allow_external_api_calls", False)):
        raise ValueError("G4 forbids external API calls")
    if bool(execution.get("allow_downloads", False)) or bool(execution.get("allow_network_downloads", False)):
        raise ValueError("G4 forbids downloads")
    if bool(execution.get("allow_training", False)) or bool(execution.get("allow_tuning", False)):
        raise ValueError("G4 forbids training and tuning")
    if bool(execution.get("allow_f1br_f1bzc_pool_use", False)):
        raise ValueError("G4 forbids superseded F1BR/F1BZC pool use")
    if int(execution.get("response_reuse", -1)) != 0:
        raise ValueError("G4 response reuse must remain zero")
    validate_loopback_url(str(execution["ollama_base_url"]))


def validate_loopback_url(url: str) -> None:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("G4 Ollama endpoint must be an http(s) loopback URL")
    if parsed.hostname.lower().strip("[]") not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("G4 Ollama endpoint must be loopback")


def load_inputs(root: Path, config: Mapping[str, Any]) -> G4Inputs:
    f1bx_config = load_config(root, Path(str(config["inputs"]["f1bx_config"])))
    specs = {spec.name: spec for spec in dataset_specs(f1bx_config)}
    candidate_manifest = read_csv(root / str(config["inputs"]["f1bzq_reranking_candidate_manifest"]))
    pools = {str(row["dataset"]): read_csv(root / str(row["candidate_artifact_path"])) for row in candidate_manifest.to_dict("records")}
    title_maps: dict[str, dict[str, str]] = {}
    for dataset in DATASET_ORDER:
        title_maps[dataset], _source = load_title_map(root, specs[dataset])
    histories = load_histories(root, specs, pools)
    return G4Inputs(
        config=dict(config),
        specs=specs,
        pools=pools,
        title_maps=title_maps,
        histories=histories,
        f0_inference_contract=read_json(root / str(config["inputs"]["f0_inference_contract"])),
        f0_metric_matrix=read_csv(root / str(config["inputs"]["f0_metric_matrix"])),
        f0_request_budget=read_csv(root / str(config["inputs"]["f0_request_budget"])),
        g0_execution_manifest=read_json(root / str(config["inputs"]["g0_execution_manifest"])),
        f1bzq_pool_manifest=read_json(root / str(config["inputs"]["f1bzq_authoritative_pool_manifest"])),
        f1bzq_source_manifest=read_json(root / str(config["inputs"]["f1bzq_source_manifest"])),
        f1bzq_readiness=read_csv(root / str(config["inputs"]["f1bzq_reranking_readiness"])),
        f1bzq_candidate_manifest=candidate_manifest,
        f1bzq_candidate_recall=read_csv(root / str(config["inputs"]["f1bzq_candidate_recall"])),
        f1bzq_foodcom_warm_cold=read_csv(root / str(config["inputs"]["f1bzq_foodcom_warm_cold_recall"])),
        f1bzq_pool_contributions=read_csv(root / str(config["inputs"]["f1bzq_pool_source_contributions"])),
        f1bzq_supersession=read_csv(root / str(config["inputs"]["f1bzq_supersession_validation"])),
    )


def load_histories(root: Path, specs: Mapping[str, Any], pools: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, list[str]]]:
    histories: dict[str, dict[str, list[str]]] = {}
    for dataset in DATASET_ORDER:
        spec = specs[dataset]
        if dataset == "foodcom":
            users = sorted(pools[dataset]["user_id"].astype(str).unique(), key=sort_key)
            positive_by_user = pools[dataset].drop_duplicates("user_id", keep="first").set_index("user_id")["positive_item"].astype(str).to_dict()
            events, _definition = foodcom_final_training_events(root, spec, users, positive_by_user)
        else:
            events, _definition = original_four_training_events(root, spec)
        histories[dataset] = {
            str(user_id): group.sort_values("prefix_position", kind="mergesort")["item_id"].astype(str).tolist()
            for user_id, group in events.groupby("user_id", sort=False)
        }
    return histories


def build_model_digest_freeze(inputs: G4Inputs) -> pd.DataFrame:
    rows = []
    by_model = {
        str(row.get("model")): row
        for row in inputs.g0_execution_manifest.get("model_preflight", [])
        if str(row.get("model")) in MODEL_ORDER
    }
    for order, model in enumerate(MODEL_ORDER, start=1):
        source = dict(by_model.get(model, {}))
        rows.append(
            {
                "model_order": order,
                "model": model,
                "model_family": MODEL_FAMILY[model],
                "frozen_digest": str(source.get("digest", "")),
                "size_bytes": str(source.get("size_bytes", "")),
                "modified_at": str(source.get("modified_at", "")),
                "source_manifest": "results/manifests/g0_canary_execution_manifest.json",
                "metadata_call_during_g4_freeze": "false",
                "status": "FROZEN_FROM_G0_PROVENANCE",
            }
        )
    return pd.DataFrame(rows)


def build_request_universe(
    root: Path,
    inputs: G4Inputs,
    model_digests: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    digests = model_digests.set_index("model")["frozen_digest"].to_dict()
    candidate_artifacts = inputs.f1bzq_candidate_manifest.set_index("dataset").to_dict("index")
    readiness_counts = {
        (str(row["dataset"]), str(row["model_name"]), str(row["prompt"])): int(row["future_requests_unlocked"])
        for row in inputs.f1bzq_readiness.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    order = 1
    for dataset in DATASET_ORDER:
        users = sorted(inputs.pools[dataset]["user_id"].astype(str).unique(), key=sort_key)
        dataset_label = str(inputs.specs[dataset].display_name)
        for user_id in users:
            base_context = build_reranking_user_context(inputs, dataset, user_id)
            candidate_items = list(base_context["candidate_items"])
            candidate_hash = str(base_context["candidate_hash"])
            candidate_artifact = candidate_artifacts[dataset]
            for prompt_strategy in PROMPT_ORDER:
                bundle = render_extension_prompt(
                    prompt_strategy=prompt_strategy,
                    dataset_label=dataset_label,
                    candidate_items=candidate_items,
                    history_items=base_context["history_items"],
                    history_length=int(base_context["history_length"]),
                )
                payload_template_hash = payload_hash_for_request(inputs.config, "", bundle.system_message, bundle.user_message)
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
                    payload_hash = payload_hash_for_request(inputs.config, model, bundle.system_message, bundle.user_message)
                    row = {
                        "canonical_request_id": request_id,
                        "request_id": request_id,
                        "request_order": order,
                        "dataset": dataset,
                        "dataset_label": dataset_label,
                        "task": TASK,
                        "scientific_status": SCIENTIFIC_STATUS,
                        "model": model,
                        "model_family": MODEL_FAMILY[model],
                        "model_digest": str(digests.get(model, "")),
                        "prompt_strategy": prompt_strategy,
                        "prompt_contract": bundle.prompt_contract,
                        "parser_contract": PARSER_CONTRACT_ID,
                        "protocol_hash": protocol_hash_for_row(inputs),
                        "generation_config_id": str(inputs.f0_inference_contract.get("generation_config_id", "")),
                        "generation_config_hash": str(inputs.f0_inference_contract.get("generation_config_hash", "")),
                        "temperature": str(inputs.config["execution"]["temperature"]),
                        "max_output_tokens": int(inputs.config["execution"]["max_output_tokens"][TASK]),
                        "timeout_seconds": int(inputs.config["execution"]["timeout_seconds"]),
                        "backend_retry_count": int(inputs.config["execution"]["backend_retry_count"]),
                        "scientific_retry_count": int(inputs.config["execution"]["scientific_retry_count"]),
                        "user_id": user_id,
                        "candidate_set_id": base_context["candidate_set_id"],
                        "pool_id": base_context["pool_id"],
                        "pool_hash": base_context["pool_hash"],
                        "candidate_source": str(candidate_artifact["candidate_artifact_path"]),
                        "candidate_source_hash": str(candidate_artifact["candidate_artifact_hash"]),
                        "candidate_source_protocol": CANDIDATE_SOURCE_PROTOCOL,
                        "candidate_pool_policy": ADAPTED_POLICY,
                        "candidate_pool_algorithm": ADAPTED_ALGORITHM,
                        "first_stage_recommender": SOURCE_POLICY,
                        "source_recommenders": stable_json(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]),
                        "source_depth": 20,
                        "candidate_hash": candidate_hash,
                        "candidate_count": CANDIDATE_COUNT,
                        "candidate_items": stable_json([item["item_id"] for item in candidate_items]),
                        "candidate_titles_hash": sha256_json([item["title"] for item in candidate_items]),
                        "candidate_ordering": "F1BZQ_BALANCED_RANK_ROUND_ROBIN_ORDER",
                        "candidate_order_immutable_for_llm": "true",
                        "force_positive": "false",
                        "target_identity_used_for_ranking": "false",
                        "positive_item_id": str(base_context["positive_item_id"]),
                        "positive_present": base_context["positive_present"],
                        "positive_position": base_context["positive_position"],
                        "history_contract": HISTORY_CONTRACT,
                        "history_length": int(base_context["history_length"]),
                        "full_history_length": int(base_context["full_history_length"]),
                        "history_hash": sha256_json([item["item_id"] for item in base_context["history_items"]]),
                        "foodcom_history_class": base_context["foodcom_history_class"],
                        "prompt_hash": bundle.prompt_hash,
                        "payload_template_hash": payload_template_hash,
                        "payload_hash": payload_hash,
                        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                        "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
                        "metric_k": METRIC_K,
                        "canonical_cell_request_count": readiness_counts[(dataset, model, prompt_strategy)],
                        "raw_response_protocol": G0_RAW_PROTOCOL,
                        "raw_response_path": (RESPONSE_DIR / f"{request_id}.json").as_posix(),
                        "response_reuse_allowed": "false",
                        "planned_new_ollama_call": "true",
                        "manifest_status": "PLANNED_G4_F1BZQ_RERANKING_REQUEST",
                    }
                    row["request_identity_hash"] = semantic_hash_for_row(
                        row,
                        exclude=("request_order", "manifest_status", "raw_response_path", "planned_new_ollama_call"),
                    )
                    rows.append(row)
                    contexts[request_id] = {
                        **base_context,
                        "system_message": bundle.system_message,
                        "user_message": bundle.user_message,
                        "prompt_bundle": bundle,
                        "candidate_items_for_task": candidate_items,
                    }
                    order += 1
    universe = pd.DataFrame(rows)
    if len(universe) != EXPECTED_LOGICAL_REQUESTS:
        raise ValueError(f"G4 request universe expected {EXPECTED_LOGICAL_REQUESTS}, observed {len(universe)}")
    if universe["canonical_request_id"].nunique() != EXPECTED_LOGICAL_REQUESTS:
        raise ValueError("G4 canonical request ids are not unique")
    return universe, contexts


def build_reranking_user_context(inputs: G4Inputs, dataset: str, user_id: str) -> dict[str, Any]:
    pool_frame = inputs.pools[dataset].loc[inputs.pools[dataset]["user_id"].astype(str).eq(str(user_id))].copy()
    candidate_items = candidate_items_from_frame(pool_frame, inputs.title_maps[dataset], title_column="candidate_title")
    candidate_ids = [item["item_id"] for item in candidate_items]
    positives = pool_frame["positive_item"].astype(str).drop_duplicates().tolist()
    pool_ids = pool_frame["pool_id"].astype(str).drop_duplicates().tolist()
    pool_hashes = pool_frame["pool_hash"].astype(str).drop_duplicates().tolist()
    if len(candidate_items) != CANDIDATE_COUNT or len(set(candidate_ids)) != CANDIDATE_COUNT:
        raise ValueError(f"G4 candidate contract broken for {dataset}/{user_id}")
    if len(positives) != 1 or len(pool_ids) != 1 or len(pool_hashes) != 1:
        raise ValueError(f"G4 pool identity contract broken for {dataset}/{user_id}")
    if not all(item["title"] for item in candidate_items):
        raise ValueError(f"G4 candidate title missing for {dataset}/{user_id}")
    history_ids = inputs.histories[dataset].get(str(user_id), [])
    visible_history_ids = list(history_ids)[-10:]
    history_items = [{"item_id": item_id, "title": title_for_item(inputs.title_maps[dataset], item_id)} for item_id in visible_history_ids]
    if not all(item["title"] for item in history_items):
        raise ValueError(f"G4 history title missing for {dataset}/{user_id}")
    positive_item = str(positives[0])
    positive_present = bool_text(positive_item in candidate_ids)
    positive_position = str(candidate_ids.index(positive_item) + 1) if positive_item in candidate_ids else ""
    observed_positive_present = str(pool_frame["positive_present"].iloc[0]).lower()
    if positive_present != observed_positive_present:
        raise ValueError(f"G4 positive-present mismatch for {dataset}/{user_id}")
    if positive_position != str(pool_frame["positive_position"].iloc[0]):
        raise ValueError(f"G4 positive-position mismatch for {dataset}/{user_id}")
    foodcom_class = ""
    if dataset == "foodcom":
        foodcom_class = "TRUE_ZERO_HISTORY_USERS" if not history_ids else "WARM_USERS"
    return {
        "dataset": dataset,
        "user_id": str(user_id),
        "candidate_items": candidate_items,
        "candidate_hash": item_list_hash(candidate_ids),
        "candidate_set_id": pool_ids[0],
        "pool_id": pool_ids[0],
        "pool_hash": pool_hashes[0],
        "positive_item_id": positive_item,
        "positive_present": positive_present,
        "positive_position": positive_position,
        "history_items": history_items,
        "history_length": len(history_items),
        "full_history_length": len(history_ids),
        "foodcom_history_class": foodcom_class,
    }


def payload_hash_for_request(config: Mapping[str, Any], model: str, system_message: str, user_message: str) -> str:
    payload = {
        "model": model,
        "system": system_message,
        "prompt": user_message,
        "stream": bool(config["execution"].get("stream", False)),
        "format": str(config["execution"].get("response_format", "json")),
        "options": {
            "temperature": float(config["execution"]["temperature"]),
            "num_predict": int(config["execution"]["max_output_tokens"][TASK]),
        },
    }
    return sha256_json(payload)


def protocol_hash_for_row(inputs: G4Inputs) -> str:
    return sha256_json(
        {
            "phase_id": PHASE_ID,
            "scientific_status": SCIENTIFIC_STATUS,
            "task": TASK,
            "history_contract": HISTORY_CONTRACT,
            "candidate_source_protocol": CANDIDATE_SOURCE_PROTOCOL,
            "candidate_pool_policy": ADAPTED_POLICY,
            "candidate_pool_algorithm": ADAPTED_ALGORITHM,
            "candidate_count": CANDIDATE_COUNT,
            "requested_output_length": REQUESTED_OUTPUT_LENGTH,
            "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
            "metric_k": METRIC_K,
            "parser_contract": PARSER_CONTRACT_ID,
            "generation_config_hash": str(inputs.f0_inference_contract.get("generation_config_hash", "")),
            "response_reuse": 0,
        }
    )


def build_execution_order(request_universe: pd.DataFrame) -> pd.DataFrame:
    ordered = request_universe.copy()
    ordered["_model_order"] = ordered["model"].map({model: index for index, model in enumerate(MODEL_ORDER)})
    ordered["_dataset_order"] = ordered["dataset"].map({dataset: index for index, dataset in enumerate(DATASET_ORDER)})
    ordered["_prompt_order"] = ordered["prompt_strategy"].map({prompt: index for index, prompt in enumerate(PROMPT_ORDER)})
    ordered["_user_sort"] = ordered["user_id"].map(sort_key)
    ordered = ordered.sort_values(["_model_order", "_dataset_order", "_prompt_order", "_user_sort"], kind="mergesort").reset_index(drop=True)
    rows = []
    for physical_order, row in enumerate(ordered.to_dict("records"), start=1):
        rows.append(
            {
                "physical_execution_order": physical_order,
                "canonical_request_id": row["canonical_request_id"],
                "request_order": int(row["request_order"]),
                "dataset": row["dataset"],
                "user_id": row["user_id"],
                "task": row["task"],
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "pool_id": row["pool_id"],
                "pool_hash": row["pool_hash"],
                "candidate_hash": row["candidate_hash"],
                "prompt_hash": row["prompt_hash"],
                "execution_action": "G4_LOCAL_OLLAMA_CALL_WHEN_EXPLICITLY_ALLOWED",
                "response_reuse_allowed": "false",
                "status": "PLANNED_NOT_EXECUTED",
            }
        )
    return pd.DataFrame(rows)


def build_candidate_hashes(request_universe: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "user_id",
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
    ]
    return request_universe.drop_duplicates(["dataset", "user_id"], keep="first")[cols].reset_index(drop=True)


def build_metric_contract() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_UTILITY_METRICS:
        rows.append(metric_row(metric, "primary_unconditional_utility", "planned_applicable_users_invalid_terminal_zero", "INCLUDED_TO_COMPUTE_AFTER_RAW"))
    for metric in RERANKING_DIAGNOSTICS:
        denominator = "all_users" if metric == "candidate_recall_at_20" else "positive_present_users_only"
        status = "COMPUTED_FROM_F1BZQ_POOL_ARTIFACT" if metric == "candidate_recall_at_20" else "INCLUDED_TO_COMPUTE_AFTER_RAW"
        rows.append(metric_row(metric, "reranking_diagnostic", denominator, status))
    included_beyond = [metric for metric in BEYOND_METRICS if metric not in {"popreo_at_5", "dpd_at_5"}]
    for metric in included_beyond:
        rows.append(metric_row(metric, "beyond_utility_g1_g2_compatible", "strict_valid_users_metric_specific_defined_subset", "INCLUDED_TO_COMPUTE_AFTER_RAW"))
    for metric in RELIABILITY_METRICS:
        rows.append(metric_row(metric, "reliability_efficiency", "all_planned_requests", "INCLUDED_TO_COMPUTE_AFTER_RAW"))
    for metric, reason in EXCLUDED_METRICS.items():
        rows.append(metric_row(metric, "excluded", "not_applicable_to_g4", "EXCLUDED", reason=reason))
    return pd.DataFrame(rows)


def metric_row(metric: str, family: str, denominator: str, status: str, *, reason: str = "") -> dict[str, Any]:
    return {
        "metric_id": metric,
        "task": TASK,
        "metric_family": family,
        "included_in_g4": bool_text(status != "EXCLUDED"),
        "denominator": denominator,
        "invalid_terminal_policy": "zero_for_utility_only;excluded_from_strict_valid_beyond_utility;counted_in_reliability",
        "status": status,
        "reason": reason or "Frozen by G4 F1BZQ protocol.",
    }


def build_candidate_recall_contract(inputs: G4Inputs) -> pd.DataFrame:
    rows = []
    for row in inputs.f1bzq_candidate_recall.to_dict("records"):
        rows.append(
            {
                "metric_id": "candidate_recall_at_20",
                "dataset": row["dataset"],
                "pool_policy": row["pool_policy"],
                "denominator": "all_users",
                "evaluation_N": int(row["evaluation_N"]),
                "positive_present_N": int(row["positive_present_N"]),
                "positive_absent_N": int(row["positive_absent_N"]),
                "candidate_recall_at_20": row["candidate_recall_at_20"],
                "force_positive": row["force_positive"],
                "target_identity_used_for_ranking": row["target_identity_used_for_ranking"],
                "candidate_recall_tautological": row["candidate_recall_tautological"],
                "source": "results/tables/f1bzq_candidate_recall.csv",
                "status": "FROZEN_FROM_F1BZQ",
            }
        )
    return pd.DataFrame(rows)


def build_statistical_comparisons() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            rows.append(stat_row("g4_prompt_strategy", dataset, model, "ALL", "BASE_vs_LLMRANK", inferential=True))
        for prompt in PROMPT_ORDER:
            for left, right in model_pairs():
                rows.append(stat_row("g4_model_pair", dataset, f"{left}_vs_{right}", prompt, f"{left}_vs_{right}", inferential=True))
            for model in MODEL_ORDER:
                rows.append(stat_row("g1_vs_g4_pipeline_descriptive", dataset, model, prompt, "G1_ranking_vs_G4_reranking", inferential=False))
    return pd.DataFrame(rows)


def model_pairs() -> list[tuple[str, str]]:
    return [
        ("qwen2:7b-instruct", "mistral:7b-instruct"),
        ("qwen2:7b-instruct", "gemma2:9b"),
        ("mistral:7b-instruct", "gemma2:9b"),
    ]


def stat_row(family: str, dataset: str, model_scope: str, prompt_scope: str, comparison: str, *, inferential: bool) -> dict[str, Any]:
    return {
        "comparison_family": family,
        "dataset": dataset,
        "model_scope": model_scope,
        "prompt_scope": prompt_scope,
        "comparison": comparison,
        "paired_unit": "dataset/user/candidate_pool_identity" if inferential else "pipeline_level_descriptive_only",
        "method": "paired_user_bootstrap_mean_delta" if inferential else "descriptive_summary_no_inference",
        "bootstrap_resamples": 10_000 if inferential else 0,
        "seed": 20260901 if inferential else "",
        "ci": "two_sided_95_percentile" if inferential else "",
        "alpha": "0.05" if inferential else "",
        "p_value_method": "NOT_FROZEN_DO_NOT_COMPUTE",
        "holm": "false",
        "inferential_test_allowed": bool_text(inferential),
        "ranking_vs_reranking_policy": "descriptive_pipeline_level_only" if family == "g1_vs_g4_pipeline_descriptive" else "",
        "status": "FROZEN",
    }


def build_response_reuse_policy(root: Path, request_universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sources = [
        ("g0_canary", "results/llm/extension_canonical/g0_canary/responses.jsonl", "excluded_by_g4_response_reuse_zero"),
        ("g1_primary_ranking", "results/llm/extension_canonical/g1_primary_ranking/responses.jsonl", "different_task_candidate_source"),
        ("f6m_lastfm_reranking", "results/llm/lastfm_reranking_v4_1/responses.jsonl", "explicitly_excluded_not_f1bzq_g4"),
    ]
    g4_ids = set(request_universe["canonical_request_id"].astype(str))
    for source_id, rel_path, reason in sources:
        path = root / rel_path
        records = read_jsonl(path)
        ids = {str(record.get("canonical_request_id") or record.get("request_id", "")) for record in records}
        rows.append(
            {
                "source_id": source_id,
                "source_path": rel_path,
                "source_exists": bool_text(path.exists()),
                "terminal_rows_observed": len(records),
                "canonical_id_overlap_with_g4": len(ids & g4_ids),
                "reuse_allowed": "false",
                "exact_reused_rows": 0,
                "reason": reason,
                "status": NO_REUSE_POLICY,
            }
        )
    return pd.DataFrame(rows)


def build_retention_plan() -> pd.DataFrame:
    rows = [
        ("raw_requests", REQUEST_JSONL, "append_only_request_records_before_send"),
        ("raw_responses", RESPONSE_JSONL, "append_only_response_records_before_parse"),
        ("raw_response_files", RESPONSE_DIR, "one_json_file_per_canonical_request_no_overwrite"),
        ("runner_events", EVENT_JSONL, "append_only_pre_send_backend_attempt_response_events"),
        ("request_universe", OUTPUT_TABLES["request_universe"], "canonical_csv_freeze_artifact"),
        ("metrics_future", Path("results/tables/g4_f1bzq_reranking_metrics.csv"), "future_computed_from_raw_responses_no_manual_values"),
    ]
    return pd.DataFrame(
        [
            {
                "artifact_class": artifact_class,
                "path": path.as_posix(),
                "retention_policy": policy,
                "overwrite_policy": "no_overwrite_for_raw;freeze_artifacts_replaceable_only_before_commit",
                "status": "FROZEN",
            }
            for artifact_class, path, policy in rows
        ]
    )


def build_dry_run_accounting(
    request_universe: pd.DataFrame,
    response_reuse_policy: pd.DataFrame,
    request_records_before: Sequence[Mapping[str, Any]],
    response_records_before: Sequence[Mapping[str, Any]],
    request_records_after: Sequence[Mapping[str, Any]],
    response_records_after: Sequence[Mapping[str, Any]],
    sent_now: int,
    *,
    execute: bool,
) -> pd.DataFrame:
    existing_terminal = len(response_records_before)
    current_terminal = len(response_records_after)
    return pd.DataFrame(
        [
            {
                "design": "G4_F1BZQ_ADAPTED_RERANKING",
                "logical_requests": len(request_universe),
                "exact_response_reuse": int(response_reuse_policy["exact_reused_rows"].astype(int).sum()),
                "future_ollama_calls": len(request_universe),
                "cells": request_universe.groupby(["dataset", "model", "prompt_strategy"]).ngroups,
                "existing_terminal_g4_evidence_before": existing_terminal,
                "existing_terminal_g4_evidence_after": current_terminal,
                "request_jsonl_rows_before": len(request_records_before),
                "request_jsonl_rows_after": len(request_records_after),
                "llm_inference_calls_executed_in_this_invocation": int(sent_now),
                "dry_run": bool_text(not execute),
                "status": "ZERO_CALL_DRY_RUN_FROZEN" if not execute else "EXECUTION_ATTEMPT_RECORDED",
            }
        ]
    )


def protocol_defects_for_execution(
    root: Path,
    inputs: G4Inputs,
    request_universe: pd.DataFrame,
    response_reuse_policy: pd.DataFrame,
) -> list[str]:
    validation = build_validation(
        root,
        inputs.config,
        inputs,
        request_universe,
        build_execution_order(request_universe),
        build_candidate_hashes(request_universe),
        build_model_digest_freeze(inputs),
        build_metric_contract(),
        build_candidate_recall_contract(inputs),
        build_statistical_comparisons(),
        response_reuse_policy,
        build_dry_run_accounting(request_universe, response_reuse_policy, [], [], [], [], 0, execute=False),
        build_retention_plan(),
        require_zero_call_freeze=True,
    )
    return validation.loc[validation["status"].ne("PASS"), "check"].astype(str).tolist()


def execute_pending_requests(
    root: Path,
    config: Mapping[str, Any],
    request_universe: pd.DataFrame,
    execution_order: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    *,
    max_new_calls: int | None,
) -> int:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    verify_existing_g4_raw_state(request_universe, request_records, response_records)
    req_counts = count_by_id(request_records)
    resp_counts = count_by_id(response_records)
    order = execution_order.set_index("canonical_request_id")["physical_execution_order"].to_dict()
    pending = []
    for row in request_universe.to_dict("records"):
        request_id = str(row["canonical_request_id"])
        if classify_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0)) == "not_started":
            pending.append({**row, "physical_execution_order": int(order[request_id])})
    pending = sorted(pending, key=lambda row: int(row["physical_execution_order"]))
    if max_new_calls is not None:
        if max_new_calls < 0:
            raise ValueError("--max-new-calls must be non-negative")
        pending = pending[:max_new_calls]
    sent = 0
    for row in pending:
        context = contexts[row["canonical_request_id"]]
        payload = build_payload(config, row, context)
        request_record = build_request_record(row, payload)
        append_jsonl_fsync(root / REQUEST_JSONL, request_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "request_recorded_before_send", "phase_id": PHASE_ID, "canonical_request_id": row["canonical_request_id"], "timestamp_utc": utc_now()})
        result, attempts = call_with_backend_retries(root, config, row, payload)
        response_record = build_response_record(row, result, attempts)
        write_raw_response_file(root / row["raw_response_path"], response_record)
        append_jsonl_fsync(root / RESPONSE_JSONL, response_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "response_recorded_before_parse", "phase_id": PHASE_ID, "canonical_request_id": row["canonical_request_id"], "backend_success": result.backend_success, "timestamp_utc": utc_now()})
        sent += 1
        print(f"g4_request physical_order={row['physical_execution_order']} dataset={row['dataset']} prompt={row['prompt_strategy']} model={row['model']} backend={bool_text(result.backend_success)}", flush=True)
    return sent


def verify_existing_g4_raw_state(
    request_universe: pd.DataFrame,
    request_records: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
) -> None:
    expected = set(request_universe["canonical_request_id"].astype(str))
    req_counts = count_by_id(request_records)
    resp_counts = count_by_id(response_records)
    extra = sorted((set(req_counts) | set(resp_counts)) - expected)
    if extra:
        raise RuntimeError(f"raw evidence contains request outside G4 universe: {extra[0]}")
    request_by_id = request_universe.set_index("canonical_request_id").to_dict("index")
    for request_id in sorted(expected):
        state = classify_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0))
        if state in {"ambiguous_interrupted", "invalid_duplicate_evidence"}:
            raise RuntimeError(f"{state}: {request_id}")
    for record in response_records:
        request_id = str(record.get("canonical_request_id") or record.get("request_id") or "")
        row = request_by_id[request_id]
        for key in ("dataset", "task", "model", "prompt_strategy", "user_id", "candidate_hash", "prompt_hash", "pool_hash"):
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
            "num_predict": int(config["execution"]["max_output_tokens"][TASK]),
        },
    }
    visible_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if positive_label_exposed(visible_payload, str(context["positive_item_id"])):
        raise ValueError(f"positive label exposed in G4 payload: {row['canonical_request_id']}")
    return payload


def build_request_record(row: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "canonical_request_id": row["canonical_request_id"],
        "request_id": row["canonical_request_id"],
        "request_order": int(row["request_order"]),
        "physical_execution_order": int(row["physical_execution_order"]),
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
        "pool_id": row["pool_id"],
        "pool_hash": row["pool_hash"],
        "candidate_hash": row["candidate_hash"],
        "prompt_hash": row["prompt_hash"],
        "history_contract": row["history_contract"],
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
        result = call_ollama_generate(str(config["execution"]["ollama_base_url"]), payload, int(config["execution"]["timeout_seconds"]))
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
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "backend_attempt_completed", "phase_id": PHASE_ID, "canonical_request_id": row["canonical_request_id"], **attempts[-1], "timestamp_utc": utc_now()})
        if result.backend_success:
            break
        if attempt <= retry_count:
            time.sleep(backoffs[attempt - 1])
    if result is None:
        raise RuntimeError("backend retry loop did not execute")
    return result, attempts


def build_response_record(row: Mapping[str, Any], result: RequestResult, attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parse_status = ""
    strict_valid = ""
    ranked_indices: Sequence[int] = []
    ranked_items: Sequence[str] = []
    if result.backend_success:
        parsed = parse_extension_top10_response(str(result.response_text), candidate_items=json.loads(str(row["candidate_items"])))
        parse_status = parsed.status
        strict_valid = bool_text(parsed.parse_valid)
        ranked_indices = parsed.output_indices
        ranked_items = parsed.output_items
    return {
        "phase_id": PHASE_ID,
        "canonical_request_id": row["canonical_request_id"],
        "request_id": row["canonical_request_id"],
        "request_order": int(row["request_order"]),
        "physical_execution_order": int(row["physical_execution_order"]),
        "dataset": row["dataset"],
        "task": row["task"],
        "model": row["model"],
        "model_digest": row["model_digest"],
        "prompt_strategy": row["prompt_strategy"],
        "user_id": row["user_id"],
        "candidate_set_id": row["candidate_set_id"],
        "pool_id": row["pool_id"],
        "pool_hash": row["pool_hash"],
        "candidate_hash": row["candidate_hash"],
        "prompt_hash": row["prompt_hash"],
        "parser_contract": row["parser_contract"],
        "protocol_hash": row["protocol_hash"],
        "generation_config_hash": row["generation_config_hash"],
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
        "strict_parse_valid": strict_valid,
        "parse_status": parse_status,
        "ranked_candidate_indices": stable_json(list(ranked_indices)),
        "ranked_item_ids": stable_json(list(ranked_items)),
        "raw_backend_response": result.raw_backend_response,
        "raw_response_recorded_before_parse": True,
    }


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    inputs: G4Inputs,
    request_universe: pd.DataFrame,
    execution_order: pd.DataFrame,
    candidate_hashes: pd.DataFrame,
    model_digests: pd.DataFrame,
    metric_contract: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    statistical_comparisons: pd.DataFrame,
    response_reuse_policy: pd.DataFrame,
    dry_run_accounting: pd.DataFrame,
    retention_plan: pd.DataFrame,
    *,
    require_zero_call_freeze: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": stable_json(evidence) if isinstance(evidence, (dict, list)) else str(evidence)})

    dataset_users = request_universe.drop_duplicates(["dataset", "user_id"]).groupby("dataset").size().to_dict()
    cell_counts = request_universe.groupby(["dataset", "model", "prompt_strategy"]).size().to_dict()
    readiness_counts = {
        (str(row["dataset"]), str(row["model_name"]), str(row["prompt"])): int(row["future_requests_unlocked"])
        for row in inputs.f1bzq_readiness.to_dict("records")
    }
    artifact_hashes_match = {
        str(row["candidate_artifact_path"]): sha256_file(root / str(row["candidate_artifact_path"])) == str(row["candidate_artifact_hash"])
        for row in inputs.f1bzq_candidate_manifest.to_dict("records")
    }
    dry = dry_run_accounting.iloc[0].to_dict()

    check("scientific_base_checkpoint_is_ancestor", git_is_ancestor(root, SCIENTIFIC_BASE_CHECKPOINT, git_output(root, ["rev-parse", "HEAD"])), git_output(root, ["rev-parse", "HEAD"]))
    check("phase_start_clean_verified", bool(config["phase"].get("phase_start_clean_verified", False)), config["phase"].get("phase_start_clean_evidence", ""))
    check("config_phase_identity", config["phase"]["id"] == PHASE_ID and config["phase"]["scientific_status"] == SCIENTIFIC_STATUS, config["phase"])
    check("execution_zero_reuse_no_external", not bool(config["execution"]["allow_external_api_calls"]) and int(config["execution"]["response_reuse"]) == 0, config["execution"])
    check("no_training_tuning_downloads_pool_regeneration", not any(bool(config["execution"].get(key, False)) for key in ["allow_downloads", "allow_network_downloads", "allow_training", "allow_tuning", "allow_traditional_ranking_regeneration", "allow_f1bzq_pool_regeneration", "allow_f1br_f1bzc_pool_use"]), config["execution"])
    check("f1bzq_pool_manifest_validated", inputs.f1bzq_pool_manifest.get("decision_token") == "F1BZQ_AUTHORITATIVE_RERANKING_POOLS_VALIDATED", inputs.f1bzq_pool_manifest.get("decision_token"))
    check("f1bzq_only_candidate_artifacts", all("f1bzq_reranking_pools" in str(path) and "f1bzc" not in str(path) for path in inputs.f1bzq_candidate_manifest["candidate_artifact_path"]), inputs.f1bzq_candidate_manifest["candidate_artifact_path"].tolist())
    check("f1bzq_candidate_artifact_hashes_match", all(artifact_hashes_match.values()), artifact_hashes_match)
    check("old_pools_superseded_not_authoritative", inputs.f1bzq_supersession["authoritative_for_downstream"].eq("false").all(), inputs.f1bzq_supersession[["artifact_id", "authoritative_for_downstream"]].to_dict("records"))
    check("datasets_users_exact", {str(k): int(v) for k, v in dataset_users.items()} == EXPECTED_USERS, {str(k): int(v) for k, v in dataset_users.items()})
    check("models_prompts_exact", set(request_universe["model"]) == set(MODEL_ORDER) and set(request_universe["prompt_strategy"]) == set(PROMPT_ORDER), request_universe[["model", "prompt_strategy"]].nunique().to_dict())
    check("task_reranking_only", request_universe["task"].eq(TASK).all(), request_universe["task"].value_counts().to_dict())
    check("logical_requests_29292_unique", len(request_universe) == EXPECTED_LOGICAL_REQUESTS and request_universe["canonical_request_id"].nunique() == EXPECTED_LOGICAL_REQUESTS, {"rows": len(request_universe), "unique": request_universe["canonical_request_id"].nunique()})
    check("cells_30_match_f1bzq", len(cell_counts) == EXPECTED_CELLS and all(int(count) == readiness_counts[key] for key, count in cell_counts.items()), {str(key): int(value) for key, value in cell_counts.items()})
    check("candidate_contract_20_unique", request_universe["candidate_count"].astype(int).eq(20).all() and candidate_hashes["candidate_hash"].astype(str).str.len().eq(64).all(), {"candidate_hash_rows": len(candidate_hashes)})
    check("pool_contract_f1bzq_balanced", request_universe["candidate_pool_policy"].eq(ADAPTED_POLICY).all() and request_universe["candidate_pool_algorithm"].eq(ADAPTED_ALGORITHM).all(), {"policy": ADAPTED_POLICY, "algorithm": ADAPTED_ALGORITHM})
    check("no_positive_forcing", request_universe["force_positive"].eq("false").all() and request_universe["target_identity_used_for_ranking"].eq("false").all(), "false")
    check("history_contract_visible_last_10", request_universe["history_contract"].eq(HISTORY_CONTRACT).all() and request_universe["history_length"].astype(int).le(10).all(), request_universe["history_length"].astype(int).describe().to_dict())
    check("prompt_parser_generation_hashes_present", request_universe["prompt_hash"].astype(str).str.len().eq(64).all() and request_universe["payload_hash"].astype(str).str.len().eq(64).all() and request_universe["generation_config_hash"].astype(str).str.len().eq(64).all(), request_universe[["prompt_hash", "payload_hash", "generation_config_hash"]].head().to_dict("records"))
    check("model_digests_frozen", len(model_digests) == 3 and model_digests["frozen_digest"].astype(str).str.len().eq(64).all(), model_digests[["model", "frozen_digest"]].to_dict("records"))
    check("execution_order_covers_universe_once", len(execution_order) == len(request_universe) and execution_order["canonical_request_id"].nunique() == len(request_universe), {"execution_order_rows": len(execution_order)})
    check("metric_contract_excludes_canddif_popreo_dpd", metric_contract.loc[metric_contract["metric_id"].isin(["canddif_hr_at_5", "canddif_ndcg_at_5", "popreo_at_5", "dpd_at_5"]), "included_in_g4"].eq("false").all(), metric_contract.loc[metric_contract["included_in_g4"].eq("false"), "metric_id"].tolist())
    check("candidate_recall_frozen_all_users", len(candidate_recall) == 5 and candidate_recall["denominator"].eq("all_users").all() and candidate_recall["force_positive"].eq("false").all(), candidate_recall[["dataset", "evaluation_N", "positive_present_N"]].to_dict("records"))
    check("statistical_contract_no_pvalues_no_holm", statistical_comparisons["p_value_method"].eq("NOT_FROZEN_DO_NOT_COMPUTE").all() and statistical_comparisons["holm"].eq("false").all(), statistical_comparisons[["comparison_family", "p_value_method", "holm"]].drop_duplicates().to_dict("records"))
    check("ranking_vs_reranking_descriptive_only", statistical_comparisons.loc[statistical_comparisons["comparison_family"].eq("g1_vs_g4_pipeline_descriptive"), "inferential_test_allowed"].eq("false").all(), statistical_comparisons["comparison_family"].value_counts().to_dict())
    check("response_reuse_zero", int(response_reuse_policy["exact_reused_rows"].astype(int).sum()) == 0 and response_reuse_policy["reuse_allowed"].eq("false").all(), response_reuse_policy.to_dict("records"))
    check("zero_call_dry_run_accounting", int(dry["logical_requests"]) == EXPECTED_LOGICAL_REQUESTS and int(dry["exact_response_reuse"]) == 0 and int(dry["future_ollama_calls"]) == EXPECTED_LOGICAL_REQUESTS and int(dry["cells"]) == EXPECTED_CELLS, dry)
    check("existing_terminal_g4_evidence_zero", int(dry["existing_terminal_g4_evidence_before"]) == 0 if require_zero_call_freeze else True, dry)
    check("llm_calls_in_this_invocation_zero", int(dry["llm_inference_calls_executed_in_this_invocation"]) == 0 if require_zero_call_freeze else True, dry)
    check("retention_plan_raw_first", set(retention_plan["artifact_class"]) >= {"raw_requests", "raw_responses", "raw_response_files", "runner_events"}, retention_plan.to_dict("records"))
    check("original_repo_paper_data_raw_clean", git_output(root, ["status", "--short", "--", "original_repo", "paper", "data/raw"]) == "", "clean")
    return pd.DataFrame(rows)


def build_manifests(
    root: Path,
    config: Mapping[str, Any],
    inputs: G4Inputs,
    decision: str,
    request_universe: pd.DataFrame,
    execution_order: pd.DataFrame,
    candidate_hashes: pd.DataFrame,
    model_digests: pd.DataFrame,
    metric_contract: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    statistical_comparisons: pd.DataFrame,
    response_reuse_policy: pd.DataFrame,
    dry_run_accounting: pd.DataFrame,
    retention_plan: pd.DataFrame,
    validation: pd.DataFrame,
    sent_now: int,
) -> dict[str, dict[str, Any]]:
    dry = dry_run_accounting.iloc[0]
    current_head = git_output(root, ["rev-parse", "HEAD"])
    git_status = git_output(root, ["status", "--short", "--untracked-files=all"])
    base = {
        "phase_id": PHASE_ID,
        "generated_at": utc_now(),
        "decision_token": decision,
        "scientific_status": SCIENTIFIC_STATUS,
        "scientific_objective": "LLM reranking over frozen F1BZQ authoritative balanced candidate pools.",
        "execution_code_commit": current_head,
        "head": current_head,
        "scientific_base_checkpoint": SCIENTIFIC_BASE_CHECKPOINT,
        "base_checkpoint_is_ancestor": git_is_ancestor(root, SCIENTIFIC_BASE_CHECKPOINT, current_head),
        "working_tree_status_at_manifest_build": git_status,
        "f1bzq_pool_manifest_decision": inputs.f1bzq_pool_manifest.get("decision_token", ""),
        "candidate_source": "F1BZQ_ONLY",
        "candidate_pool_policy": ADAPTED_POLICY,
        "candidate_pool_algorithm": ADAPTED_ALGORITHM,
        "source_policy": SOURCE_POLICY,
        "history_contract": HISTORY_CONTRACT,
        "task": TASK,
        "active_datasets": list(DATASET_ORDER),
        "models": list(MODEL_ORDER),
        "prompt_strategies": list(PROMPT_ORDER),
        "logical_requests": int(dry["logical_requests"]),
        "response_reuse": int(dry["exact_response_reuse"]),
        "future_ollama_calls": int(dry["future_ollama_calls"]),
        "cells": int(dry["cells"]),
        "existing_terminal_g4_evidence": int(dry["existing_terminal_g4_evidence_before"]),
        "llm_inference_calls_executed_in_this_invocation": int(sent_now),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
        "tables": {key: path.as_posix() for key, path in OUTPUT_TABLES.items()},
        "raw_paths": {
            "raw_dir": RAW_DIR.as_posix(),
            "requests_jsonl": REQUEST_JSONL.as_posix(),
            "responses_jsonl": RESPONSE_JSONL.as_posix(),
            "events_jsonl": EVENT_JSONL.as_posix(),
            "response_dir": RESPONSE_DIR.as_posix(),
        },
        "frame_hashes": {
            "request_universe": frame_hash(request_universe),
            "execution_order": frame_hash(execution_order),
            "candidate_hashes": frame_hash(candidate_hashes),
            "model_digests": frame_hash(model_digests),
            "metric_contract": frame_hash(metric_contract),
            "candidate_recall": frame_hash(candidate_recall),
            "statistical_comparisons": frame_hash(statistical_comparisons),
            "response_reuse_policy": frame_hash(response_reuse_policy),
            "dry_run_accounting": frame_hash(dry_run_accounting),
            "retention_plan": frame_hash(retention_plan),
            "validation": frame_hash(validation),
        },
    }
    return {
        "protocol": {
            **base,
            "model_digests": model_digests.to_dict("records"),
            "candidate_artifacts": inputs.f1bzq_candidate_manifest.to_dict("records"),
            "metric_contract_summary": {
                f"{str(family)}|included={str(included)}": int(count)
                for (family, included), count in metric_contract.groupby(["metric_family", "included_in_g4"]).size().items()
            },
        },
        "request_universe": {
            **base,
            "canonical_request_ids": request_universe["canonical_request_id"].tolist(),
            "request_order_hash": sha256_json(request_universe["canonical_request_id"].tolist()),
            "physical_execution_order_hash": sha256_json(execution_order.sort_values("physical_execution_order", kind="mergesort")["canonical_request_id"].tolist()),
        },
        "zero_call_dry_run": {
            **base,
            "dry_run_accounting": dry_run_accounting.to_dict("records"),
            "response_reuse_policy": response_reuse_policy.to_dict("records"),
        },
    }


def render_report(
    decision: str,
    request_universe: pd.DataFrame,
    candidate_hashes: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    metric_contract: pd.DataFrame,
    statistical_comparisons: pd.DataFrame,
    dry_run_accounting: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    dry = dry_run_accounting.iloc[0]
    included_metrics = int(metric_contract["included_in_g4"].eq("true").sum())
    excluded_metrics = int(metric_contract["included_in_g4"].eq("false").sum())
    lines = [
        "# G4 F1BZQ Reranking Protocol Freeze",
        "",
        f"Decision token: `{decision}`",
        "",
        "## Scientific Status",
        "",
        "- G4 is an adapted extension reranking experiment, not paper-exact.",
        "- F1BZQ is the sole authoritative candidate-pool source.",
        "- Superseded F1BR/F1BZC pools and old F6M reranking responses are excluded.",
        "",
        "## Universe",
        "",
        f"- Logical requests: `{int(dry['logical_requests'])}`.",
        f"- Cells: `{int(dry['cells'])}`.",
        f"- Response reuse: `{int(dry['exact_response_reuse'])}`.",
        f"- Future Ollama calls: `{int(dry['future_ollama_calls'])}`.",
        f"- Candidate hash rows: `{len(candidate_hashes)}`.",
        f"- Dataset user counts: `{stable_json({str(k): int(v) for k, v in candidate_hashes.groupby('dataset').size().to_dict().items()})}`.",
        "",
        "## Contract",
        "",
        f"- Task: `{TASK}`.",
        f"- History: `{HISTORY_CONTRACT}`.",
        f"- Candidate pool: `{ADAPTED_ALGORITHM}`, 20 items, no positive forcing.",
        f"- Output: exactly `{REQUESTED_OUTPUT_LENGTH}` unique 1-based candidate indices.",
        f"- Parser: `{PARSER_CONTRACT_ID}`.",
        "",
        "## Candidate Recall",
        "",
        candidate_recall.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Metrics",
        "",
        f"- Included metric rows: `{included_metrics}`.",
        f"- Excluded metric rows: `{excluded_metrics}`.",
        "- CandDif, PopREO, DPD, monetary marginal cost, and unimplemented metrics are excluded.",
        "",
        "## Statistics",
        "",
        f"- Statistical comparison rows: `{len(statistical_comparisons)}`.",
        "- Internal comparisons use paired-user bootstrap with 10000 resamples, seed 20260901, and two-sided 95% percentile CI.",
        "- P-values are not frozen and Holm correction is not applied.",
        "- G1 vs G4 is descriptive pipeline-level comparison only.",
        "",
        "## Zero-Call Dry Run",
        "",
        dry_run_accounting.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Validation",
        "",
        f"- Validation checks: `{len(validation)}`.",
        f"- Failures: `{int(validation['status'].ne('PASS').sum())}`.",
        "",
        decision,
    ]
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    ensure_output_dirs(root, include_raw=False)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")
    artifact_paths = [*OUTPUT_TABLES.values(), *SUPPORT_ARTIFACTS]
    for key, rel_path in OUTPUT_MANIFESTS.items():
        payload = dict(outputs["manifests"][key])
        payload["artifact_hashes"] = {
            path.as_posix(): sha256_file(root / path) for path in artifact_paths if (root / path).exists()
        }
        (root / rel_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_phase_notes(root: Path, outputs: Mapping[str, Any], *, execute: bool) -> None:
    manifest = outputs["manifests"]["protocol"]
    lines = [
        "",
        f"## {datetime.now(timezone.utc).isoformat(timespec='seconds')} - G4 F1BZQ reranking protocol freeze",
        "",
        f"- Decision: `{manifest['decision_token']}`.",
        f"- Invocation mode: `{'execute' if execute else 'zero-call dry-run freeze'}`.",
        f"- Logical requests: `{manifest['logical_requests']}`.",
        f"- Response reuse: `{manifest['response_reuse']}`.",
        f"- Future Ollama calls: `{manifest['future_ollama_calls']}`.",
        f"- Existing terminal G4 evidence: `{manifest['existing_terminal_g4_evidence']}`.",
        f"- Ollama/API calls in this invocation: `{manifest['llm_inference_calls_executed_in_this_invocation']}`.",
        "- No G1/G2/G3 artifacts, F1BZQ pool artifacts, original_repo files, paper files, raw data, training outputs, or superseded reranking pools were modified.",
    ]
    mode = "a" if (root / NOTES_PATH).exists() else "w"
    with (root / NOTES_PATH).open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def ensure_output_dirs(root: Path, *, include_raw: bool) -> None:
    for rel_path in (TABLE_DIR, MANIFEST_DIR, REPORT_PATH.parent):
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    if include_raw:
        for rel_path in (RAW_DIR, RESPONSE_DIR):
            (root / rel_path).mkdir(parents=True, exist_ok=True)


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    try:
        completed = subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.returncode == 0
    except OSError:
        return False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

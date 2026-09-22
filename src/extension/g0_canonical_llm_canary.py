"""G0 canonical local LLM canary over frozen extension requests.

The phase executes 72 real canonical requests drawn from the frozen F0/F1BZQ
universe. It uses local Ollama only, writes raw evidence before parsing, and
supports resume/idempotence through canonical request ids.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.extension.e0_parser import parse_extension_top10_response
from src.extension.e0_prompts import PromptBundle, render_extension_prompt
from src.extension.e0_protocol import (
    ACTIVE_EXTENSION_DATASETS,
    CANDIDATE_COUNT,
    FUTURE_MODELS,
    METRIC_K,
    PARSER_CONTRACT_ID,
    PROMPT_BASE,
    PROMPT_LLMRANK,
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
from src.extension.f1bx_execution_stage1 import dataset_specs, load_candidates, read_csv, read_json, sort_key
from src.extension.f1bz_final_traditional_execution import (
    foodcom_final_training_events,
    load_config as load_yaml_config,
    original_four_training_events,
)
from src.extension.f1bzq_authoritative_reranking_pool_materialization import load_title_map
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync, positive_label_exposed, read_jsonl
from src.utils.config import find_project_root


PHASE_ID = "g0_canonical_llm_canary"
DECISION_VALIDATED = "G0_CANONICAL_LLM_CANARY_VALIDATED"
DECISION_BLOCKED = "G0_CANONICAL_LLM_CANARY_BLOCKED"
CONFIG_PATH = Path("configs/g0_canonical_llm_canary.yaml")

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_PATH = Path("report/g0_canonical_llm_canary.md")
RAW_DIR = Path("results/llm/extension_canonical/g0_canary")
REQUEST_JSONL = RAW_DIR / "requests.jsonl"
RESPONSE_JSONL = RAW_DIR / "responses.jsonl"
EVENT_JSONL = RAW_DIR / "events.jsonl"
RESPONSE_DIR = RAW_DIR / "responses"

OUTPUT_TABLES = {
    "selection": TABLE_DIR / "g0_canary_selection.csv",
    "request_matrix": TABLE_DIR / "g0_canary_request_matrix.csv",
    "execution": TABLE_DIR / "g0_canary_execution.csv",
    "parse_validation": TABLE_DIR / "g0_canary_parse_validation.csv",
    "runtime": TABLE_DIR / "g0_canary_runtime.csv",
    "resume_validation": TABLE_DIR / "g0_canary_resume_validation.csv",
    "validation": TABLE_DIR / "g0_canary_validation.csv",
}

OUTPUT_MANIFESTS = {
    "request_manifest": MANIFEST_DIR / "g0_canary_request_manifest.json",
    "execution_manifest": MANIFEST_DIR / "g0_canary_execution_manifest.json",
}

TASKS = ("ranking", "reranking")
DATASET_ORDER = tuple(ACTIVE_EXTENSION_DATASETS)
MODEL_ORDER = tuple(FUTURE_MODELS)
PROMPT_ORDER = tuple(PROMPT_STRATEGIES)
EXPECTED_HEAD = "727340acdcfe29bf03a75ee309cdc6b848f967c4"
G0_RAW_PROTOCOL = "extension_canonical_raw_response_jsonl_v1"
MODEL_FAMILY = {
    "qwen2:7b-instruct": "QWEN",
    "mistral:7b-instruct": "MISTRAL",
    "gemma2:9b": "GEMMA",
}


@dataclass(frozen=True)
class RequestResult:
    backend_success: bool
    response_text: str
    raw_backend_response: dict[str, Any]
    backend_error_type: str
    backend_error_message: str
    started_at: str
    completed_at: str
    duration_seconds: float


@dataclass(frozen=True)
class G0Inputs:
    config: dict[str, Any]
    specs: dict[str, Any]
    ranking_candidates: dict[str, pd.DataFrame]
    reranking_pools: dict[str, pd.DataFrame]
    title_maps: dict[str, dict[str, str]]
    history_by_dataset: dict[str, dict[str, list[str]]]
    f0_primary: pd.DataFrame
    f1bzq_manifest: dict[str, Any]
    f1bzq_source_manifest: dict[str, Any]
    f1bzq_readiness: pd.DataFrame
    f1bzq_candidate_manifest: pd.DataFrame
    f1bzq_supersession: pd.DataFrame
    f0_budget: pd.DataFrame
    f1bzq_pool_hashes_before: dict[str, str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate the G0 canonical local LLM canary.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--resume", action="store_true", help="Resume existing canonical requests and skip completed ones.")
    parser.add_argument("--validate-only", action="store_true", help="Rebuild status from existing G0 artifacts without inference.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    if args.validate_only:
        outputs = load_existing_outputs(root)
    else:
        outputs = run_g0(root, Path(args.config), resume=args.resume)
        write_outputs(root, outputs)
    manifest = outputs["manifests"]["execution_manifest"]
    validation = outputs["validation"]
    print(f"decision={manifest['decision_token']}")
    print(f"planned_requests={manifest['planned_requests']}")
    print(f"terminal_requests={manifest['terminal_requests']}")
    print(f"backend_successful={manifest['backend_successful_requests']}")
    print(f"strict_valid={manifest['strict_valid_requests']}")
    print(f"structurally_invalid={manifest['structurally_invalid_requests']}")
    print(f"first_run_llm_calls={manifest['first_run_llm_calls']}")
    print(f"resume_run_llm_calls={manifest['resume_run_llm_calls']}")
    print(f"validation_passed={str(validation['status'].eq('PASS').all()).lower()}")
    return 0 if manifest["decision_token"] == DECISION_VALIDATED else 1


def run_g0(root: Path, config_path: Path, *, resume: bool) -> dict[str, Any]:
    ensure_output_dirs(root)
    config = load_config(root, config_path)
    inputs = load_inputs(root, config)
    model_preflight = ollama_model_preflight(config)
    selection, contexts_by_user = select_canary_users(root, inputs)
    request_matrix, contexts = build_request_matrix(root, inputs, selection, contexts_by_user, model_preflight)
    protocol_defects = pre_execution_protocol_defects(root, inputs, request_matrix, contexts, model_preflight)
    invocation_start = utc_now()
    initial_request_records = read_jsonl(root / REQUEST_JSONL)
    initial_response_records = read_jsonl(root / RESPONSE_JSONL)
    sent_now = 0
    if not protocol_defects:
        sent_now = execute_pending_requests(root, config, request_matrix, contexts, model_preflight)
    else:
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "pre_execution_protocol_block",
                "phase_id": PHASE_ID,
                "timestamp_utc": utc_now(),
                "defects": protocol_defects,
            },
        )
    invocation_end = utc_now()
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    execution = build_execution_table(request_matrix, request_records, response_records)
    parse_validation = build_parse_validation(request_matrix, contexts, response_records)
    runtime = build_runtime_table(request_matrix, execution)
    resume_validation = build_resume_validation(
        root,
        request_matrix,
        initial_request_records,
        initial_response_records,
        request_records,
        response_records,
        sent_now,
        resume=resume,
        invocation_start=invocation_start,
        invocation_end=invocation_end,
    )
    f1bzq_pool_hashes_after = current_f1bzq_pool_hashes(root, inputs)
    validation = build_validation(
        root,
        config,
        inputs,
        model_preflight,
        selection,
        request_matrix,
        execution,
        parse_validation,
        runtime,
        resume_validation,
        protocol_defects,
        f1bzq_pool_hashes_after,
    )
    decision = DECISION_VALIDATED if bool(validation["status"].eq("PASS").all()) else DECISION_BLOCKED
    manifests = build_manifests(
        root,
        config,
        inputs,
        decision,
        model_preflight,
        selection,
        request_matrix,
        execution,
        parse_validation,
        runtime,
        resume_validation,
        validation,
        request_records,
        response_records,
        sent_now,
    )
    report = render_report(
        decision,
        model_preflight,
        selection,
        request_matrix,
        execution,
        parse_validation,
        runtime,
        resume_validation,
        validation,
    )
    return {
        "selection": selection,
        "request_matrix": request_matrix,
        "execution": execution,
        "parse_validation": parse_validation,
        "runtime": runtime,
        "resume_validation": resume_validation,
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


def load_inputs(root: Path, config: Mapping[str, Any]) -> G0Inputs:
    f1bx_config = load_config(root, Path(str(config["inputs"]["f1bx_config"])))
    specs = {spec.name: spec for spec in dataset_specs(f1bx_config)}
    ranking_candidates = {dataset: load_candidates(root, specs[dataset]) for dataset in DATASET_ORDER}
    title_maps: dict[str, dict[str, str]] = {}
    for dataset in DATASET_ORDER:
        title_maps[dataset], _title_source = load_title_map(root, specs[dataset])
    history_by_dataset = load_histories(root, specs, ranking_candidates)
    f1bzq_manifest = read_json(root / str(config["inputs"]["f1bzq_authoritative_pool_manifest"]))
    f1bzq_source_manifest = read_json(root / str(config["inputs"]["f1bzq_source_manifest"]))
    f1bzq_candidate_manifest = read_csv(root / str(config["inputs"]["f1bzq_reranking_candidate_manifest"]))
    reranking_pools = {
        str(row["dataset"]): read_csv(root / str(row["candidate_artifact_path"]))
        for row in f1bzq_candidate_manifest.to_dict("records")
    }
    return G0Inputs(
        config=dict(config),
        specs=specs,
        ranking_candidates=ranking_candidates,
        reranking_pools=reranking_pools,
        title_maps=title_maps,
        history_by_dataset=history_by_dataset,
        f0_primary=read_csv(root / str(config["inputs"]["f0_primary_ranking_matrix"])),
        f1bzq_manifest=f1bzq_manifest,
        f1bzq_source_manifest=f1bzq_source_manifest,
        f1bzq_readiness=read_csv(root / str(config["inputs"]["f1bzq_reranking_readiness"])),
        f1bzq_candidate_manifest=f1bzq_candidate_manifest,
        f1bzq_supersession=read_csv(root / str(config["inputs"]["f1bzq_supersession_validation"])),
        f0_budget=read_csv(root / str(config["inputs"]["f0_request_budget"])),
        f1bzq_pool_hashes_before=current_f1bzq_pool_hashes_from_manifest(root, f1bzq_candidate_manifest),
    )


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


def select_canary_users(root: Path, inputs: G0Inputs) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    order = 1
    for dataset in DATASET_ORDER:
        if dataset == "foodcom":
            strata = [("FOODCOM_WARM", "WARM_USERS"), ("FOODCOM_TRUE_ZERO_HISTORY", "TRUE_ZERO_HISTORY_USERS")]
        else:
            strata = [("PRIMARY", "WARM_USERS")]
        for logical_group, pool_group in strata:
            selected_user, context = select_one_user_for_stratum(root, inputs, dataset, logical_group, pool_group)
            history_length = int(context["visible_history_length"])
            selection_hash = sha256_json(
                {
                    "salt": inputs.config["selection"]["deterministic_salt"],
                    "dataset": dataset,
                    "logical_group": logical_group,
                    "user_id": selected_user,
                }
            )
            rows.append(
                {
                    "logical_user_order": order,
                    "dataset": dataset,
                    "logical_user_group": logical_group,
                    "pool_group_filter": pool_group,
                    "user_id": selected_user,
                    "selection_rule": "min_sha256_over_protocol_renderable_final_users",
                    "selection_hash": selection_hash,
                    "selection_used_recommendation_success": "false",
                    "selection_used_candidate_recall": "false",
                    "ranking_candidate_hash": context["ranking_candidate_hash"],
                    "reranking_candidate_hash": context["reranking_candidate_hash"],
                    "ranking_candidate_source": context["ranking_candidate_source"],
                    "reranking_candidate_source": context["reranking_candidate_source"],
                    "full_history_length": int(context["full_history_length"]),
                    "visible_history_length": history_length,
                    "foodcom_history_class": context["foodcom_history_class"],
                    "history_representation": "(no visible user history)" if history_length == 0 else "numbered_title_list",
                    "status": "SELECTED",
                    "reason": "Deterministic final-user selection from renderable frozen inputs.",
                }
            )
            contexts[(dataset, selected_user)] = context
            order += 1
    selection = pd.DataFrame(rows)
    if len(selection) != 6 or selection["user_id"].nunique() != 6:
        raise ValueError("G0 selection must contain exactly 6 distinct logical users")
    return selection, contexts


def select_one_user_for_stratum(
    root: Path,
    inputs: G0Inputs,
    dataset: str,
    logical_group: str,
    pool_group: str,
) -> tuple[str, dict[str, Any]]:
    ranking_users = set(inputs.ranking_candidates[dataset]["user_id"].astype(str))
    pool = inputs.reranking_pools[dataset]
    group_pool = pool.loc[pool["warm_cold_group"].eq(pool_group)].copy() if dataset == "foodcom" else pool.copy()
    pool_users = set(group_pool["user_id"].astype(str))
    users = sorted(ranking_users & pool_users, key=sort_key)
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for user in users:
        context = build_user_context(root, inputs, dataset, user)
        if logical_group == "FOODCOM_WARM" and int(context["visible_history_length"]) == 0:
            continue
        if logical_group == "FOODCOM_TRUE_ZERO_HISTORY" and int(context["visible_history_length"]) != 0:
            continue
        if not context["prompt_renderable"]:
            continue
        key = sha256_json(
            {
                "salt": inputs.config["selection"]["deterministic_salt"],
                "dataset": dataset,
                "logical_group": logical_group,
                "user_id": user,
            }
        )
        candidates.append((key, user, context))
    if not candidates:
        raise ValueError(f"no renderable G0 user for {dataset} {logical_group}")
    _key, selected_user, context = sorted(candidates, key=lambda item: (item[0], sort_key(item[1])))[0]
    return selected_user, context


def build_user_context(root: Path, inputs: G0Inputs, dataset: str, user_id: str) -> dict[str, Any]:
    ranking_frame = inputs.ranking_candidates[dataset].loc[inputs.ranking_candidates[dataset]["user_id"].eq(user_id)].copy()
    ranking_items = candidate_items_from_frame(ranking_frame, inputs.title_maps[dataset], title_column="")
    ranking_candidate_ids = [item["item_id"] for item in ranking_items]
    positive_items = ranking_frame.loc[ranking_frame["is_ground_truth"].astype(bool), "positive_item_id"].astype(str).unique().tolist()
    if len(positive_items) != 1:
        raise ValueError(f"ranking candidate positive contract broken for {dataset}/{user_id}")
    positive_item = positive_items[0]
    pool_frame = inputs.reranking_pools[dataset].loc[inputs.reranking_pools[dataset]["user_id"].eq(user_id)].copy()
    reranking_items = candidate_items_from_frame(pool_frame, inputs.title_maps[dataset], title_column="candidate_title")
    reranking_candidate_ids = [item["item_id"] for item in reranking_items]
    history_ids = inputs.history_by_dataset[dataset].get(str(user_id), [])
    visible_history_ids = list(history_ids)[-10:]
    history_items = [{"item_id": item_id, "title": title_for_item(inputs.title_maps[dataset], item_id)} for item_id in visible_history_ids]
    prompt_renderable = (
        len(ranking_items) == CANDIDATE_COUNT
        and len(reranking_items) == CANDIDATE_COUNT
        and len(set(ranking_candidate_ids)) == CANDIDATE_COUNT
        and len(set(reranking_candidate_ids)) == CANDIDATE_COUNT
        and all(item["title"] for item in ranking_items)
        and all(item["title"] for item in reranking_items)
        and all(item["title"] for item in history_items)
    )
    foodcom_class = ""
    if dataset == "foodcom":
        foodcom_class = "TRUE_ZERO_HISTORY_USERS" if not history_ids else "WARM_USERS"
    return {
        "dataset": dataset,
        "user_id": str(user_id),
        "positive_item_id": str(positive_item),
        "ranking_candidates": ranking_items,
        "reranking_candidates": reranking_items,
        "history_items": history_items,
        "full_history_length": len(history_ids),
        "visible_history_length": len(history_items),
        "ranking_candidate_hash": item_list_hash(ranking_candidate_ids),
        "reranking_candidate_hash": item_list_hash(reranking_candidate_ids),
        "ranking_candidate_source": inputs.specs[dataset].candidate_source.as_posix(),
        "reranking_candidate_source": str(pool_frame["pool_id"].iloc[0]) if not pool_frame.empty else "",
        "reranking_pool_hash": str(pool_frame["pool_hash"].iloc[0]) if not pool_frame.empty else "",
        "reranking_candidate_artifact": pool_artifact_path_for_dataset(inputs, dataset),
        "prompt_renderable": prompt_renderable,
        "foodcom_history_class": foodcom_class,
    }


def candidate_items_from_frame(frame: pd.DataFrame, title_map: Mapping[str, str], *, title_column: str) -> list[dict[str, Any]]:
    rows = []
    ordered = frame.assign(_candidate_position_int=pd.to_numeric(frame["candidate_position"], errors="raise")).sort_values(
        "_candidate_position_int",
        kind="mergesort",
    )
    for expected, row in enumerate(ordered.to_dict("records"), start=1):
        position = int(row["_candidate_position_int"])
        item_id = str(row.get("candidate_item_id") or row.get("item_id") or "")
        if position != expected:
            raise ValueError(f"candidate positions are not 1..20 for user {row.get('user_id', '')}")
        title = str(row.get(title_column, "")).strip() if title_column else ""
        if not title:
            title = title_for_item(title_map, item_id)
        rows.append({"candidate_position": position, "item_id": item_id, "title": title})
    return rows


def title_for_item(title_map: Mapping[str, str], item_id: str) -> str:
    return str(title_map.get(str(item_id), str(item_id))).strip()


def pool_artifact_path_for_dataset(inputs: G0Inputs, dataset: str) -> str:
    match = inputs.f1bzq_candidate_manifest.loc[inputs.f1bzq_candidate_manifest["dataset"].eq(dataset)]
    return str(match.iloc[0]["candidate_artifact_path"]) if not match.empty else ""


def build_request_matrix(
    root: Path,
    inputs: G0Inputs,
    selection: pd.DataFrame,
    contexts_by_user: Mapping[tuple[str, str], Mapping[str, Any]],
    model_preflight: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    digests = model_preflight.set_index("model")["digest"].to_dict() if not model_preflight.empty else {}
    order = 1
    for selected in selection.sort_values("logical_user_order").to_dict("records"):
        dataset = str(selected["dataset"])
        user_id = str(selected["user_id"])
        base_context = contexts_by_user[(dataset, user_id)]
        dataset_label = str(inputs.specs[dataset].display_name)
        for task in TASKS:
            candidate_key = "ranking_candidates" if task == "ranking" else "reranking_candidates"
            candidate_items = list(base_context[candidate_key])
            candidate_hash = item_list_hash(item["item_id"] for item in candidate_items)
            canonical_cell_counts = canonical_cell_request_count(inputs, dataset, task)
            for prompt_strategy in PROMPT_ORDER:
                bundle = render_extension_prompt(
                    prompt_strategy=prompt_strategy,
                    dataset_label=dataset_label,
                    candidate_items=candidate_items,
                    history_items=base_context["history_items"],
                    history_length=int(base_context["visible_history_length"]),
                )
                for model in MODEL_ORDER:
                    canonical_id = canonical_request_id(
                        dataset=dataset,
                        task=task,
                        model=model,
                        prompt_strategy=prompt_strategy,
                        user_id=user_id,
                        candidate_hash=candidate_hash,
                        prompt_hash=bundle.prompt_hash,
                    )
                    canary_id = f"G0-CANARY-{order:04d}-{canonical_id[-10:]}"
                    row = {
                        "canary_request_id": canary_id,
                        "canonical_request_id": canonical_id,
                        "request_id": canonical_id,
                        "request_order": order,
                        "dataset": dataset,
                        "dataset_label": dataset_label,
                        "task": task,
                        "model": model,
                        "model_family": MODEL_FAMILY[model],
                        "model_digest": str(digests.get(model, "")),
                        "prompt_strategy": prompt_strategy,
                        "prompt_contract": bundle.prompt_contract,
                        "parser_contract": PARSER_CONTRACT_ID,
                        "user_id": user_id,
                        "logical_user_group": selected["logical_user_group"],
                        "candidate_source": base_context["ranking_candidate_source"] if task == "ranking" else base_context["reranking_candidate_artifact"],
                        "candidate_source_protocol": "F0_FROZEN_RANKING_CANDIDATES" if task == "ranking" else "F1BZQ_AUTHORITATIVE_RERANKING_POOL",
                        "candidate_hash": candidate_hash,
                        "candidate_count": CANDIDATE_COUNT,
                        "candidate_items": stable_json([item["item_id"] for item in candidate_items]),
                        "candidate_titles_hash": sha256_json([item["title"] for item in candidate_items]),
                        "reranking_pool_hash": base_context["reranking_pool_hash"] if task == "reranking" else "",
                        "prompt_hash": bundle.prompt_hash,
                        "history_length": int(base_context["visible_history_length"]),
                        "full_history_length": int(base_context["full_history_length"]),
                        "history_hash": sha256_json([item["item_id"] for item in base_context["history_items"]]),
                        "positive_item_id": base_context["positive_item_id"],
                        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                        "metric_k": METRIC_K,
                        "canonical_cell_request_count": canonical_cell_counts[(task, model, prompt_strategy)],
                        "raw_response_protocol": G0_RAW_PROTOCOL,
                        "raw_response_path": (RESPONSE_DIR / f"{canonical_id}.json").as_posix(),
                        "manifest_status": "PLANNED_CANONICAL_REQUEST",
                    }
                    row["request_identity_hash"] = semantic_hash_for_row(
                        row,
                        exclude=("canary_request_id", "request_order", "manifest_status"),
                    )
                    rows.append(row)
                    contexts[canonical_id] = {
                        **base_context,
                        "system_message": bundle.system_message,
                        "user_message": bundle.user_message,
                        "prompt_bundle": bundle,
                        "candidate_items_for_task": candidate_items,
                        "candidate_hash": candidate_hash,
                    }
                    order += 1
    matrix = pd.DataFrame(rows)
    if len(matrix) != 72:
        raise ValueError(f"G0 request matrix expected 72 rows, observed {len(matrix)}")
    if matrix["canonical_request_id"].nunique() != 72:
        raise ValueError("G0 canonical request ids are not unique")
    return matrix, contexts


def canonical_cell_request_count(inputs: G0Inputs, dataset: str, task: str) -> dict[tuple[str, str, str], int]:
    if task == "ranking":
        source = inputs.f0_primary.loc[inputs.f0_primary["dataset"].eq(dataset)]
    else:
        source = inputs.f1bzq_readiness.loc[inputs.f1bzq_readiness["dataset"].eq(dataset)]
    rows: dict[tuple[str, str, str], int] = {}
    for row in source.to_dict("records"):
        rows[(task, str(row["model_name"]), str(row["prompt"]))] = int(row.get("request_count") or row.get("future_requests_unlocked"))
    return rows


def canonical_request_id(
    *,
    dataset: str,
    task: str,
    model: str,
    prompt_strategy: str,
    user_id: str,
    candidate_hash: str,
    prompt_hash: str,
) -> str:
    digest = sha256_json(
        {
            "phase": "F0_F1BZQ_FINAL_EXTENSION_UNIVERSE",
            "dataset": dataset,
            "task": task,
            "model": model,
            "prompt_strategy": prompt_strategy,
            "user_id": str(user_id),
            "candidate_hash": candidate_hash,
            "prompt_hash": prompt_hash,
            "output_length": REQUESTED_OUTPUT_LENGTH,
            "parser": PARSER_CONTRACT_ID,
        }
    )
    return "EXTREQ-" + digest.upper()[:32]


def execute_pending_requests(
    root: Path,
    config: Mapping[str, Any],
    request_matrix: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    model_preflight: pd.DataFrame,
) -> int:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    req_counts = count_by_id(request_records)
    resp_counts = count_by_id(response_records)
    verify_existing_raw_state(request_matrix, req_counts, resp_counts)
    pending = []
    for row in request_matrix.sort_values("request_order", kind="mergesort").to_dict("records"):
        request_id = row["canonical_request_id"]
        state = classify_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0))
        if state == "not_started":
            pending.append(row)
    model_available = model_preflight.set_index("model")["available"].to_dict()
    sent = 0
    for row in pending:
        if str(model_available.get(row["model"], "false")) != "true":
            raise RuntimeError(f"model not locally available: {row['model']}")
        context = contexts[row["canonical_request_id"]]
        payload = build_payload(config, row, context)
        request_record = build_request_record(row, context, payload)
        append_jsonl_fsync(root / REQUEST_JSONL, request_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "request_recorded_before_send", "phase_id": PHASE_ID, "canonical_request_id": row["canonical_request_id"], "timestamp_utc": utc_now()})
        result = call_ollama_generate(
            str(config["execution"]["ollama_base_url"]),
            payload,
            int(config["execution"]["timeout_seconds"]),
        )
        response_record = build_response_record(row, result)
        write_raw_response_file(root / row["raw_response_path"], response_record)
        append_jsonl_fsync(root / RESPONSE_JSONL, response_record)
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "response_recorded_before_parse",
                "phase_id": PHASE_ID,
                "canonical_request_id": row["canonical_request_id"],
                "backend_success": result.backend_success,
                "timestamp_utc": utc_now(),
            },
        )
        sent += 1
        print(
            "g0_request "
            f"order={row['request_order']} dataset={row['dataset']} task={row['task']} "
            f"prompt={row['prompt_strategy']} model={row['model']} "
            f"backend={bool_text(result.backend_success)} duration={format_float(result.duration_seconds)}",
            flush=True,
        )
    return sent


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
        "canary_request_id": row["canary_request_id"],
        "canonical_request_id": row["canonical_request_id"],
        "request_id": row["canonical_request_id"],
        "request_order": int(row["request_order"]),
        "dataset": row["dataset"],
        "task": row["task"],
        "model": row["model"],
        "model_digest": row["model_digest"],
        "prompt_strategy": row["prompt_strategy"],
        "prompt_contract": row["prompt_contract"],
        "parser_contract": row["parser_contract"],
        "user_id": row["user_id"],
        "candidate_hash": row["candidate_hash"],
        "prompt_hash": row["prompt_hash"],
        "history_length": int(row["history_length"]),
        "raw_response_path": row["raw_response_path"],
        "payload_hash": sha256_json(payload),
        "request_payload": payload,
        "timestamp_before_send_utc": utc_now(),
    }


def build_response_record(row: Mapping[str, Any], result: RequestResult) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "canary_request_id": row["canary_request_id"],
        "canonical_request_id": row["canonical_request_id"],
        "request_id": row["canonical_request_id"],
        "request_order": int(row["request_order"]),
        "dataset": row["dataset"],
        "task": row["task"],
        "model": row["model"],
        "model_digest": row["model_digest"],
        "prompt_strategy": row["prompt_strategy"],
        "user_id": row["user_id"],
        "candidate_hash": row["candidate_hash"],
        "prompt_hash": row["prompt_hash"],
        "backend_success": result.backend_success,
        "backend_status": "BACKEND_SUCCESS" if result.backend_success else "BACKEND_FAILURE",
        "backend_error_type": result.backend_error_type,
        "backend_error_message": result.backend_error_message,
        "request_started_at_utc": result.started_at,
        "request_completed_at_utc": result.completed_at,
        "runtime_seconds": result.duration_seconds,
        "raw_response": result.response_text,
        "raw_response_hash": sha256_text(result.response_text),
        "raw_backend_response": result.raw_backend_response,
        "raw_response_recorded_before_parse": True,
    }


def call_ollama_generate(base_url: str, payload: Mapping[str, Any], timeout_seconds: int) -> RequestResult:
    started = utc_now()
    start_clock = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
        raw = json.loads(raw_body)
        response_text = str(raw.get("response", ""))
        success = bool(response_text.strip()) and bool(raw.get("done", True))
        return RequestResult(success, response_text, raw, "", "", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.HTTPError as exc:
        return RequestResult(False, "", {}, "HTTPError", f"{exc.code}: {exc.reason}", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.URLError as exc:
        return RequestResult(False, "", {}, "URLError", str(exc.reason), started, utc_now(), time.perf_counter() - start_clock)
    except TimeoutError as exc:
        return RequestResult(False, "", {}, "TimeoutError", str(exc), started, utc_now(), time.perf_counter() - start_clock)
    except Exception as exc:  # pragma: no cover - depends on local backend state.
        return RequestResult(False, "", {}, type(exc).__name__, str(exc), started, utc_now(), time.perf_counter() - start_clock)


def write_raw_response_file(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if sha256_text(existing) != sha256_text(payload):
            raise RuntimeError(f"conflicting raw response file exists: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def build_execution_table(
    request_matrix: pd.DataFrame,
    request_records: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    req_counts = count_by_id(request_records)
    response_by_id = single_record_by_id(response_records)
    rows = []
    for row in request_matrix.sort_values("request_order", kind="mergesort").to_dict("records"):
        request_id = row["canonical_request_id"]
        response = response_by_id.get(request_id, {})
        state = classify_state(req_counts.get(request_id, 0), 1 if response else 0)
        backend_success = bool(response.get("backend_success", False))
        parse = {}
        terminal = "false"
        terminal_state = state
        if response:
            terminal = "true"
            terminal_state = "BACKEND_SUCCESS_UNPARSED" if backend_success else "BACKEND_FAILURE"
        rows.append(
            {
                "canary_request_id": row["canary_request_id"],
                "canonical_request_id": request_id,
                "request_order": int(row["request_order"]),
                "dataset": row["dataset"],
                "task": row["task"],
                "model": row["model"],
                "model_family": row["model_family"],
                "prompt_strategy": row["prompt_strategy"],
                "user_id": row["user_id"],
                "candidate_hash": row["candidate_hash"],
                "prompt_hash": row["prompt_hash"],
                "history_length": int(row["history_length"]),
                "request_record_count": req_counts.get(request_id, 0),
                "response_record_count": 1 if response else 0,
                "request_state": state,
                "terminal": terminal,
                "terminal_state": terminal_state,
                "backend_success": bool_text(backend_success),
                "backend_status": str(response.get("backend_status", "")),
                "request_started_at_utc": str(response.get("request_started_at_utc", "")),
                "request_completed_at_utc": str(response.get("request_completed_at_utc", "")),
                "runtime_seconds": format_float(float(response.get("runtime_seconds", math.nan))) if response else "",
                "raw_response": str(response.get("raw_response", "")),
                "raw_response_hash": str(response.get("raw_response_hash", "")),
                "raw_response_path": row["raw_response_path"],
                "raw_response_recorded_before_parse": bool_text(bool(response.get("raw_response_recorded_before_parse", False))),
                "backend_error_type": str(response.get("backend_error_type", "")),
                "backend_error_message": str(response.get("backend_error_message", "")),
                "strict_parse_valid": str(parse.get("strict_parse_valid", "")),
            }
        )
    return pd.DataFrame(rows)


def build_parse_validation(
    request_matrix: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    response_by_id = single_record_by_id(response_records)
    request_rows: list[dict[str, Any]] = []
    for row in request_matrix.sort_values("request_order", kind="mergesort").to_dict("records"):
        request_id = row["canonical_request_id"]
        response = response_by_id.get(request_id, {})
        candidate_items = [item["item_id"] for item in contexts[request_id]["candidate_items_for_task"]]
        if not response:
            parse_status = "not_executed"
            parse_valid = False
            output_indices: Sequence[int] = []
            output_items: Sequence[str] = []
            duplicate_count = 0
            unknown_count = 0
            output_length = 0
        elif not bool(response.get("backend_success", False)):
            parse_status = "backend_failure_not_parsed"
            parse_valid = False
            output_indices = []
            output_items = []
            duplicate_count = 0
            unknown_count = 0
            output_length = 0
        else:
            parsed = parse_extension_top10_response(str(response.get("raw_response", "")), candidate_items=candidate_items)
            parse_status = parsed.status
            parse_valid = parsed.parse_valid
            output_indices = parsed.output_indices
            output_items = parsed.output_items
            duplicate_count = parsed.duplicate_count
            unknown_count = parsed.unknown_item_count
            output_length = parsed.output_length
        length_violation = output_length != REQUESTED_OUTPUT_LENGTH
        request_rows.append(
            {
                "canary_request_id": row["canary_request_id"],
                "canonical_request_id": request_id,
                "request_order": int(row["request_order"]),
                "dataset": row["dataset"],
                "task": row["task"],
                "model": row["model"],
                "model_family": row["model_family"],
                "prompt_strategy": row["prompt_strategy"],
                "user_id": row["user_id"],
                "candidate_hash": row["candidate_hash"],
                "prompt_hash": row["prompt_hash"],
                "raw_response_hash": str(response.get("raw_response_hash", "")),
                "parser_contract": PARSER_CONTRACT_ID,
                "expected_output_length": REQUESTED_OUTPUT_LENGTH,
                "output_length": output_length,
                "strict_parse_valid": bool_text(parse_valid),
                "parse_status": parse_status,
                "ranked_candidate_indices": stable_json(list(output_indices)),
                "ranked_item_ids": stable_json(list(output_items)),
                "duplicate_count": int(duplicate_count),
                "invalid_candidate_reference_count": int(unknown_count),
                "hallucination_count": int(unknown_count),
                "length_violation": bool_text(length_violation),
                "malformed_numbering": bool_text(parse_status in {"parse_failed", "invalid_schema", "invalid_length"}),
                "structurally_invalid_preserved": bool_text(not parse_valid and bool(response.get("backend_success", False))),
                "invalid_output_retried": "false",
                "status": "COMPUTED",
                "reason": "Strict parser result preserved without repair or scientific retry.",
            }
        )
    request_level = pd.DataFrame(request_rows)
    aggregate_rows = []
    group_cols = ["model", "task", "prompt_strategy", "dataset"]
    for keys, group in request_level.groupby(group_cols, sort=False):
        model, task, prompt, dataset = keys
        planned = len(group)
        valid = int(group["strict_parse_valid"].eq("true").sum())
        backend_successful = int((~group["parse_status"].isin({"not_executed", "backend_failure_not_parsed"})).sum())
        aggregate_rows.append(
            {
                "canary_request_id": "AGGREGATE",
                "canonical_request_id": "",
                "request_order": "",
                "dataset": dataset,
                "task": task,
                "model": model,
                "model_family": MODEL_FAMILY[str(model)],
                "prompt_strategy": prompt,
                "user_id": "",
                "candidate_hash": "",
                "prompt_hash": "",
                "raw_response_hash": "",
                "parser_contract": PARSER_CONTRACT_ID,
                "expected_output_length": REQUESTED_OUTPUT_LENGTH,
                "output_length": "",
                "strict_parse_valid": "",
                "parse_status": "AGGREGATE",
                "ranked_candidate_indices": "",
                "ranked_item_ids": "",
                "planned_requests": planned,
                "backend_successful": backend_successful,
                "structurally_valid": valid,
                "structurally_invalid": planned - valid,
                "strict_valid_rate": format_float(valid / planned if planned else math.nan),
                "duplicate_count": int(group["duplicate_count"].astype(int).sum()),
                "invalid_candidate_reference_count": int(group["invalid_candidate_reference_count"].astype(int).sum()),
                "hallucination_count": int(group["hallucination_count"].astype(int).sum()),
                "length_violations": int(group["length_violation"].eq("true").sum()),
                "malformed_numbering": int(group["malformed_numbering"].eq("true").sum()),
                "structurally_invalid_preserved": bool_text(group["structurally_invalid_preserved"].eq("true").any()),
                "invalid_output_retried": "false",
                "status": "COMPUTED",
                "reason": "Aggregate canary parser diagnostics; not a scientific estimate.",
            }
        )
    return pd.concat([request_level, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False).fillna("")


def build_runtime_table(request_matrix: pd.DataFrame, execution: pd.DataFrame) -> pd.DataFrame:
    merged = execution.merge(
        request_matrix[["canonical_request_id", "canonical_cell_request_count"]],
        on="canonical_request_id",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for group_cols in (["model", "task"], ["model", "task", "prompt_strategy", "dataset"]):
        for keys, group in merged.groupby(group_cols, sort=False):
            values = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(group_cols, keys))
            full_count = int(pd.to_numeric(group["canonical_cell_request_count"], errors="coerce").fillna(0).sum())
            mean_seconds = float(values.mean())
            rows.append(
                {
                    "runtime_scope": "model_task" if group_cols == ["model", "task"] else "model_task_prompt_dataset",
                    "model": key_map.get("model", "ALL"),
                    "task": key_map.get("task", "ALL"),
                    "prompt_strategy": key_map.get("prompt_strategy", "ALL"),
                    "dataset": key_map.get("dataset", "ALL"),
                    "observed_requests": int(len(values)),
                    "mean_seconds": format_float(mean_seconds),
                    "median_seconds": format_float(float(np.median(values))),
                    "min_seconds": format_float(float(values.min())),
                    "max_seconds": format_float(float(values.max())),
                    "p90_seconds": format_float(float(np.percentile(values, 90))),
                    "p95_seconds": format_float(float(np.percentile(values, 95))),
                    "requests_per_hour": format_float(3600.0 / mean_seconds if mean_seconds > 0 else math.nan),
                    "full_cell_request_count": full_count,
                    "approx_full_runtime_hours": format_float((mean_seconds * full_count) / 3600.0 if full_count else math.nan),
                    "status": "DIAGNOSTIC_ONLY",
                    "reason": "Runtime estimate is diagnostic and does not alter the experiment universe.",
                }
            )
    return pd.DataFrame(rows)


def build_resume_validation(
    root: Path,
    request_matrix: pd.DataFrame,
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
    planned = len(request_matrix)
    current_req_counts = count_by_id(request_records)
    current_resp_counts = count_by_id(response_records)
    duplicate_requests = sum(max(0, count - 1) for count in current_req_counts.values())
    duplicate_responses = sum(max(0, count - 1) for count in current_resp_counts.values())
    completed = sum(
        1
        for request_id in request_matrix["canonical_request_id"].astype(str)
        if classify_state(current_req_counts.get(request_id, 0), current_resp_counts.get(request_id, 0)) == "completed"
    )
    raw_hash_before = response_hash_manifest(initial_response_records)
    raw_hash_after = response_hash_manifest(response_records)
    if resume:
        first_run_calls = len(initial_response_records)
        resume_run_calls = sent_now
        status = "PASS" if first_run_calls == planned and resume_run_calls == 0 and raw_hash_before == raw_hash_after and completed == planned and duplicate_requests == 0 and duplicate_responses == 0 else "FAIL"
    else:
        first_run_calls = sent_now
        resume_run_calls = ""
        status = "PENDING_RESUME_CHECK"
    return pd.DataFrame(
        [
            {
                "resume_mode": bool_text(resume),
                "invocation_started_at_utc": invocation_start,
                "invocation_completed_at_utc": invocation_end,
                "planned_requests": planned,
                "completed_canonical_requests": completed,
                "first_run_llm_calls": first_run_calls,
                "resume_run_llm_calls": resume_run_calls,
                "target_resume_run_calls": 0,
                "raw_response_hashes_unchanged": bool_text(raw_hash_before == raw_hash_after if resume else True),
                "duplicate_request_rows": duplicate_requests,
                "duplicate_response_rows": duplicate_responses,
                "no_new_llm_calls_for_completed_requests": bool_text(resume and sent_now == 0),
                "status": status,
                "reason": "Resume mode skipped already completed canonical requests." if resume else "First execution complete; run --resume to perform the explicit idempotence checkpoint.",
            }
        ]
    )


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    inputs: G0Inputs,
    model_preflight: pd.DataFrame,
    selection: pd.DataFrame,
    request_matrix: pd.DataFrame,
    execution: pd.DataFrame,
    parse_validation: pd.DataFrame,
    runtime: pd.DataFrame,
    resume_validation: pd.DataFrame,
    protocol_defects: Sequence[str],
    f1bzq_pool_hashes_after: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any, severity: str = "ERROR") -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "severity": severity, "evidence": str(evidence)})

    budget = {str(row["budget_item"]): int(row["request_count"]) for row in inputs.f0_budget.to_dict("records")}
    request_level_parse = parse_validation.loc[parse_validation["canary_request_id"].ne("AGGREGATE")]
    readiness = inputs.f1bzq_readiness
    source_execution = read_csv(root / "results/tables/f1bzq_source_execution.csv")
    source_rows = int(source_execution["source_artifact_rows"].astype(int).sum())
    pool_contributions = read_csv(root / "results/tables/f1bzq_pool_source_contributions.csv")
    warm_cold = read_csv(root / "results/tables/f1bzq_foodcom_warm_cold_recall.csv").set_index("scope")
    terminal = execution["terminal"].eq("true").all() and len(execution) == 72
    backend_successful = execution["backend_success"].eq("true").all() and len(execution) == 72
    aggregate_parse = parse_validation.loc[parse_validation["canary_request_id"].eq("AGGREGATE")]
    reranking = request_matrix.loc[request_matrix["task"].eq("reranking")]
    rerank_hash_groups = reranking.groupby(["dataset", "user_id"], sort=False)["candidate_hash"].nunique().to_dict()

    check("precondition_head", git_output(root, ["rev-parse", "HEAD"]) == str(config["phase"]["expected_head"]) == EXPECTED_HEAD, git_output(root, ["rev-parse", "HEAD"]))
    check("precondition_phase_start_clean", bool(config["phase"].get("phase_start_clean_verified", False)), config["phase"].get("phase_start_clean_evidence", ""))
    check("f1bzq_state_validated", inputs.f1bzq_manifest.get("decision_token") == "F1BZQ_AUTHORITATIVE_RERANKING_POOLS_VALIDATED", inputs.f1bzq_manifest.get("decision_token"))
    check("active_datasets_exact", inputs.f1bzq_manifest.get("active_datasets") == list(DATASET_ORDER), inputs.f1bzq_manifest.get("active_datasets"))
    check("video_games_absent", "video_games" not in set(request_matrix["dataset"].astype(str)), sorted(request_matrix["dataset"].unique()))
    check("source_cells_20_computed", len(source_execution) == 20 and source_execution["status"].eq("COMPUTED").all(), source_execution["status"].value_counts().to_dict())
    check("source_user_model_cases_19528", int(inputs.f1bzq_manifest.get("user_model_source_cases", -1)) == 19528, inputs.f1bzq_manifest.get("user_model_source_cases"))
    check("source_ranking_rows_390560", source_rows == 390560, source_rows)
    check("authoritative_pool_users_4882", int(inputs.f1bzq_manifest.get("authoritative_pool_users", -1)) == 4882 and len(pool_contributions) == 4882, {"manifest": inputs.f1bzq_manifest.get("authoritative_pool_users"), "rows": len(pool_contributions)})
    check("pool_candidate_rows_97640", int(inputs.f1bzq_manifest.get("pool_candidate_rows", -1)) == 97640, inputs.f1bzq_manifest.get("pool_candidate_rows"))
    check("reranking_readiness_30_ready", len(readiness) == 30 and readiness["readiness_status"].eq("READY_TO_INFER").all() and readiness["future_requests_unlocked"].astype(int).sum() == 29292, {"rows": len(readiness), "requests": int(readiness["future_requests_unlocked"].astype(int).sum())})
    check("checkpoint_hashes_unchanged", int(inputs.f1bzq_manifest.get("checkpoint_hashes_changed", -1)) == 0, inputs.f1bzq_manifest.get("checkpoint_hashes_changed"))
    check("foodcom_true_zero_727", int(warm_cold.loc["TRUE_ZERO_HISTORY_USERS", "evaluation_N"]) == 727, warm_cold.loc["TRUE_ZERO_HISTORY_USERS"].to_dict())
    check("foodcom_warm_273", int(warm_cold.loc["WARM_USERS", "evaluation_N"]) == 273, warm_cold.loc["WARM_USERS"].to_dict())
    check("old_pools_superseded", inputs.f1bzq_supersession["superseded_status"].eq("SUPERSEDED_RERANKING_SOURCE_DEFECT").all() and inputs.f1bzq_supersession["authoritative_for_downstream"].eq("false").all(), inputs.f1bzq_supersession[["artifact_id", "superseded_status"]].to_dict("records"))
    check("g0_no_superseded_pool_consumers", no_superseded_pool_consumers(request_matrix), request_matrix[["task", "candidate_source"]].drop_duplicates().to_dict("records"))
    check("protected_paths_unchanged", git_output(root, ["status", "--short", "--untracked-files=all", "--", "original_repo", "paper", "data/raw", "results/checkpoints", "results/models"]) == "", "clean")
    check("models_available", len(model_preflight) == 3 and model_preflight["available"].eq("true").all(), model_preflight[["model", "available", "digest"]].to_dict("records"))
    check("request_matrix_72", len(request_matrix) == 72 and request_matrix["canonical_request_id"].nunique() == 72, {"rows": len(request_matrix), "ids": request_matrix["canonical_request_id"].nunique()})
    check("six_deterministic_users", len(selection) == 6 and selection["user_id"].nunique() == 6, selection[["dataset", "logical_user_group", "user_id"]].to_dict("records"))
    check("foodcom_warm_and_zero_selected", set(selection.loc[selection["dataset"].eq("foodcom"), "logical_user_group"]) == {"FOODCOM_WARM", "FOODCOM_TRUE_ZERO_HISTORY"} and int(selection.loc[selection["logical_user_group"].eq("FOODCOM_WARM"), "visible_history_length"].iloc[0]) > 0 and int(selection.loc[selection["logical_user_group"].eq("FOODCOM_TRUE_ZERO_HISTORY"), "visible_history_length"].iloc[0]) == 0, selection.loc[selection["dataset"].eq("foodcom")].to_dict("records"))
    check("models_prompts_tasks_crossed", set(request_matrix["model"]) == set(MODEL_ORDER) and set(request_matrix["prompt_strategy"]) == set(PROMPT_ORDER) and set(request_matrix["task"]) == set(TASKS), request_matrix[["model", "prompt_strategy", "task"]].nunique().to_dict())
    check("ranking_contract", request_matrix.loc[request_matrix["task"].eq("ranking"), "candidate_count"].astype(int).eq(20).all() and request_matrix.loc[request_matrix["task"].eq("ranking"), "requested_output_length"].astype(int).eq(10).all(), "20 candidates and top10 output")
    check("reranking_contract_f1bzq_only", reranking["candidate_source_protocol"].eq("F1BZQ_AUTHORITATIVE_RERANKING_POOL").all() and reranking["reranking_pool_hash"].astype(str).str.len().eq(64).all(), reranking[["dataset", "user_id", "candidate_source"]].drop_duplicates().to_dict("records"))
    check("identical_reranking_pool_across_llms", all(value == 1 for value in rerank_hash_groups.values()), rerank_hash_groups)
    check("candidate_hashes_present", request_matrix["candidate_hash"].astype(str).str.len().eq(64).all(), request_matrix["candidate_hash"].head().tolist())
    check("prompt_hashes_present", request_matrix["prompt_hash"].astype(str).str.len().eq(64).all(), request_matrix["prompt_hash"].head().tolist())
    check("protocol_defects_absent", not protocol_defects, protocol_defects)
    check("terminal_requests_72", terminal, execution["terminal"].value_counts().to_dict())
    check("backend_success_72", backend_successful, execution["backend_success"].value_counts().to_dict())
    check("raw_preservation_complete", execution["raw_response_recorded_before_parse"].eq("true").all() and execution["raw_response_hash"].astype(str).str.len().eq(64).all(), execution[["canonical_request_id", "raw_response_hash"]].head().to_dict("records"))
    check("strict_parser_executed", len(request_level_parse) == 72 and request_level_parse["status"].eq("COMPUTED").all(), request_level_parse["parse_status"].value_counts().to_dict())
    check("invalid_outputs_not_retried", request_level_parse["invalid_output_retried"].eq("false").all(), request_level_parse["invalid_output_retried"].value_counts().to_dict())
    expected_aggregate_rows = request_matrix.groupby(["model", "task", "prompt_strategy", "dataset"], sort=False).ngroups
    check("parse_aggregates_present", len(aggregate_parse) == expected_aggregate_rows, {"observed": len(aggregate_parse), "expected": expected_aggregate_rows})
    check("runtime_recorded", not runtime.empty and pd.to_numeric(execution["runtime_seconds"], errors="coerce").notna().sum() == 72, len(runtime))
    check("resume_idempotent", resume_validation["status"].iloc[0] == "PASS" and str(resume_validation["resume_run_llm_calls"].iloc[0]) == "0", resume_validation.to_dict("records"))
    check("f1bzq_pool_hashes_unchanged", dict(inputs.f1bzq_pool_hashes_before) == dict(f1bzq_pool_hashes_after), len(inputs.f1bzq_pool_hashes_before))
    check("budget_unchanged", budget.get("CORE_MANDATORY_NOW_REQUESTS") == 29292 and budget.get("TRAINING_DEPENDENT_REQUESTS") == 29292 and budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292, budget)
    check("request_accounting_subset", int(request_matrix["canonical_cell_request_count"].astype(int).sum()) > 72 and len(request_matrix) == 72, {"executed_canonical_requests": len(request_matrix), "incremental_experiment_requests": 0})
    check("zero_external_network_download_training", not bool(config["execution"]["allow_external_api_calls"]) and not bool(config["execution"]["allow_downloads"]) and not bool(config["execution"]["allow_network_downloads"]) and not bool(config["execution"]["allow_training"]) and not bool(config["execution"]["allow_tuning"]), config["execution"])
    return pd.DataFrame(rows)


def pre_execution_protocol_defects(
    root: Path,
    inputs: G0Inputs,
    request_matrix: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    model_preflight: pd.DataFrame,
) -> list[str]:
    defects: list[str] = []
    if git_output(root, ["rev-parse", "HEAD"]) != str(inputs.config["phase"]["expected_head"]):
        defects.append("precondition_head_mismatch")
    if not (len(model_preflight) == 3 and model_preflight["available"].eq("true").all()):
        defects.append("missing_required_local_model")
    if len(request_matrix) != 72:
        defects.append("request_matrix_not_72")
    if not no_superseded_pool_consumers(request_matrix):
        defects.append("superseded_pool_consumer_detected")
    for row in request_matrix.to_dict("records"):
        candidate_items = contexts[row["canonical_request_id"]]["candidate_items_for_task"]
        if len(candidate_items) != 20 or len({item["item_id"] for item in candidate_items}) != 20:
            defects.append(f"candidate_contract_broken:{row['canonical_request_id']}")
        if not all(item["title"] for item in candidate_items):
            defects.append(f"missing_candidate_title:{row['canonical_request_id']}")
    return defects


def no_superseded_pool_consumers(request_matrix: pd.DataFrame) -> bool:
    superseded_exact = {
        "results/tables/f1bz_reranking_pools.csv",
        "results/tables/f1bz_candidate_recall.csv",
        "results/raw_outputs/f1bzc_reranking_candidates/f1bzc_lastfm_attempted_adapted_pools.csv",
    }
    reranking_sources = set(request_matrix.loc[request_matrix["task"].eq("reranking"), "candidate_source"].astype(str))
    if reranking_sources & superseded_exact:
        return False
    return all("f1bzq" in source and "f1bzc" not in source for source in reranking_sources)


def ollama_model_preflight(config: Mapping[str, Any]) -> pd.DataFrame:
    base_url = str(config["execution"]["ollama_base_url"]).rstrip("/")
    version_payload = http_json("GET", base_url + "/api/version", None, timeout=5)
    tags_payload = http_json("GET", base_url + "/api/tags", None, timeout=10)
    ps_payload = http_json("GET", base_url + "/api/ps", None, timeout=5)
    tags = {str(model.get("name") or model.get("model")): model for model in tags_payload.get("models", [])}
    gpu_info = gpu_summary()
    ram_gb = physical_ram_gb()
    rows = []
    for model in config["models"]:
        model = str(model)
        tag = tags.get(model, {})
        rows.append(
            {
                "model": model,
                "model_family": MODEL_FAMILY.get(model, model),
                "available": bool_text(bool(tag)),
                "digest": str(tag.get("digest") or tag.get("id") or ""),
                "size_bytes": str(tag.get("size", "")),
                "modified_at": str(tag.get("modified_at", "")),
                "ollama_version": str(version_payload.get("version", "")),
                "ollama_models_loaded": stable_json([str(row.get("name") or row.get("model")) for row in ps_payload.get("models", [])]),
                "gpu": gpu_info,
                "ram_gb": format_float(ram_gb),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "status": "AVAILABLE_LOCAL" if tag else "BLOCKED_MISSING_LOCAL_MODEL",
            }
        )
    return pd.DataFrame(rows)


def http_json(method: str, url: str, payload: Mapping[str, Any] | None, *, timeout: int) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}", "models": []}


def gpu_summary() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def physical_ram_gb() -> float:
    if platform.system().lower() == "windows":
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
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullTotalPhys) / (1024 ** 3)
    return math.nan


def verify_existing_raw_state(request_matrix: pd.DataFrame, req_counts: Mapping[str, int], resp_counts: Mapping[str, int]) -> None:
    expected = set(request_matrix["canonical_request_id"].astype(str))
    extra = sorted((set(req_counts) | set(resp_counts)) - expected)
    if extra:
        raise RuntimeError(f"raw evidence contains request outside G0 matrix: {extra[0]}")
    for request_id in sorted(expected):
        state = classify_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0))
        if state in {"ambiguous_interrupted", "invalid_duplicate_evidence"}:
            raise RuntimeError(f"{state}: {request_id}")


def count_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        request_id = str(record.get("canonical_request_id") or record.get("request_id") or "")
        counts[request_id] = counts.get(request_id, 0) + 1
    return counts


def single_record_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    counts = count_by_id(records)
    for record in records:
        request_id = str(record.get("canonical_request_id") or record.get("request_id") or "")
        if counts.get(request_id, 0) == 1:
            by_id[request_id] = record
    return by_id


def classify_state(request_count: int, response_count: int) -> str:
    if request_count == 0 and response_count == 0:
        return "not_started"
    if request_count == 1 and response_count == 1:
        return "completed"
    if request_count == 1 and response_count == 0:
        return "ambiguous_interrupted"
    return "invalid_duplicate_evidence"


def response_hash_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(record.get("canonical_request_id") or record.get("request_id")): str(record.get("raw_response_hash", ""))
        for record in records
    }


def current_f1bzq_pool_hashes_from_manifest(root: Path, manifest: pd.DataFrame) -> dict[str, str]:
    return {
        str(row["candidate_artifact_path"]): sha256_file(root / str(row["candidate_artifact_path"]))
        for row in manifest.to_dict("records")
    }


def current_f1bzq_pool_hashes(root: Path, inputs: G0Inputs) -> dict[str, str]:
    return current_f1bzq_pool_hashes_from_manifest(root, inputs.f1bzq_candidate_manifest)


def build_manifests(
    root: Path,
    config: Mapping[str, Any],
    inputs: G0Inputs,
    decision: str,
    model_preflight: pd.DataFrame,
    selection: pd.DataFrame,
    request_matrix: pd.DataFrame,
    execution: pd.DataFrame,
    parse_validation: pd.DataFrame,
    runtime: pd.DataFrame,
    resume_validation: pd.DataFrame,
    validation: pd.DataFrame,
    request_records: Sequence[Mapping[str, Any]],
    response_records: Sequence[Mapping[str, Any]],
    sent_now: int,
) -> dict[str, dict[str, Any]]:
    request_level_parse = parse_validation.loc[parse_validation["canary_request_id"].ne("AGGREGATE")]
    backend_success = int(execution["backend_success"].eq("true").sum())
    strict_valid = int(request_level_parse["strict_parse_valid"].eq("true").sum())
    structurally_invalid = int((request_level_parse["strict_parse_valid"].ne("true") & execution["backend_success"].eq("true")).sum())
    resume_row = resume_validation.iloc[0].to_dict()
    base = {
        "phase_id": PHASE_ID,
        "generated_at": utc_now(),
        "decision_token": decision,
        "expected_head": config["phase"]["expected_head"],
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "active_datasets": list(DATASET_ORDER),
        "models": list(MODEL_ORDER),
        "prompt_strategies": list(PROMPT_ORDER),
        "tasks": list(TASKS),
        "planned_requests": int(len(request_matrix)),
        "terminal_requests": int(execution["terminal"].eq("true").sum()),
        "backend_successful_requests": backend_success,
        "strict_valid_requests": strict_valid,
        "structurally_invalid_requests": structurally_invalid,
        "executed_canonical_requests": int(len(request_matrix)),
        "incremental_experiment_requests": 0,
        "first_run_llm_calls": resume_row.get("first_run_llm_calls", ""),
        "resume_run_llm_calls": resume_row.get("resume_run_llm_calls", ""),
        "llm_inference_calls_executed_in_this_invocation": sent_now,
        "training_runs_executed": 0,
        "tuning_runs_executed": 0,
        "downloads_executed": 0,
        "external_api_calls_executed": 0,
        "f1bzq_pool_hashes_unchanged": dict(inputs.f1bzq_pool_hashes_before) == current_f1bzq_pool_hashes(root, inputs),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
        "model_preflight": model_preflight.to_dict("records"),
        "selected_users": selection.to_dict("records"),
        "tables": {key: path.as_posix() for key, path in OUTPUT_TABLES.items()},
        "raw_paths": {
            "raw_dir": RAW_DIR.as_posix(),
            "request_jsonl": REQUEST_JSONL.as_posix(),
            "response_jsonl": RESPONSE_JSONL.as_posix(),
            "event_jsonl": EVENT_JSONL.as_posix(),
            "response_dir": RESPONSE_DIR.as_posix(),
        },
        "frame_hashes": {
            "selection": frame_hash(selection),
            "request_matrix": frame_hash(request_matrix),
            "execution": frame_hash(execution),
            "parse_validation": frame_hash(parse_validation),
            "runtime": frame_hash(runtime),
            "resume_validation": frame_hash(resume_validation),
            "validation": frame_hash(validation),
        },
    }
    request_manifest = {
        **base,
        "canonical_request_ids": request_matrix["canonical_request_id"].tolist(),
        "request_jsonl_records": len(request_records),
        "response_jsonl_records": len(response_records),
    }
    execution_manifest = {
        **base,
        "runtime_summary": runtime.to_dict("records"),
        "parse_status_counts": request_level_parse["parse_status"].value_counts().to_dict(),
    }
    return {"request_manifest": request_manifest, "execution_manifest": execution_manifest}


def render_report(
    decision: str,
    model_preflight: pd.DataFrame,
    selection: pd.DataFrame,
    request_matrix: pd.DataFrame,
    execution: pd.DataFrame,
    parse_validation: pd.DataFrame,
    runtime: pd.DataFrame,
    resume_validation: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    request_level_parse = parse_validation.loc[parse_validation["canary_request_id"].ne("AGGREGATE")]
    model_counts = execution.groupby("model")["backend_success"].apply(lambda s: int(s.eq("true").sum())).to_dict()
    prompt_counts = request_matrix["prompt_strategy"].value_counts().to_dict()
    task_counts = request_matrix["task"].value_counts().to_dict()
    invalid_count = int(request_level_parse["strict_parse_valid"].ne("true").sum())
    lines = [
        "# G0 Canonical Local LLM Canary",
        "",
        f"Decision: `{decision}`",
        "",
        "## Model Preflight",
        "",
        model_preflight.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Canary Users",
        "",
        selection.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Request Matrix",
        "",
        f"- Planned canonical requests: `{len(request_matrix)}`.",
        f"- Tasks: `{stable_json(task_counts)}`.",
        f"- Prompts: `{stable_json(prompt_counts)}`.",
        "- Request ids are canonical extension request ids, not canary-only ids.",
        "",
        "## Execution",
        "",
        f"- Backend successes by model: `{stable_json(model_counts)}`.",
        f"- Terminal requests: `{int(execution['terminal'].eq('true').sum())}` / `{len(execution)}`.",
        "",
        "## Strict Parser",
        "",
        f"- Strict valid requests: `{int(request_level_parse['strict_parse_valid'].eq('true').sum())}`.",
        f"- Structurally invalid preserved requests: `{invalid_count}`.",
        "- Invalid LLM content is not retried or repaired.",
        "",
        "## Runtime",
        "",
        runtime.head(12).to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Resume",
        "",
        resume_validation.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Validation",
        "",
        f"- Checks: `{len(validation)}`.",
        f"- Failures: `{int(validation['status'].ne('PASS').sum())}`.",
        "",
        "No comparative scientific conclusion is drawn from this six-user canary.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    ensure_output_dirs(root)
    for key, rel_path in OUTPUT_TABLES.items():
        outputs[key].to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")
    response_files = [
        path.relative_to(root)
        for path in sorted((root / RESPONSE_DIR).glob("*.json"))
    ]
    artifact_paths = [
        *OUTPUT_TABLES.values(),
        REPORT_PATH,
        REQUEST_JSONL,
        RESPONSE_JSONL,
        EVENT_JSONL,
        *response_files,
    ]
    for key, rel_path in OUTPUT_MANIFESTS.items():
        payload = dict(outputs["manifests"][key])
        payload["artifact_hashes"] = {
            path.as_posix(): sha256_file(root / path) for path in artifact_paths if (root / path).exists()
        }
        (root / rel_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_existing_outputs(root: Path) -> dict[str, Any]:
    return {
        **{key: read_csv(root / path) for key, path in OUTPUT_TABLES.items()},
        "manifests": {key: read_json(root / path) for key, path in OUTPUT_MANIFESTS.items()},
        "report": (root / REPORT_PATH).read_text(encoding="utf-8"),
    }


def ensure_output_dirs(root: Path) -> None:
    for rel_path in (TABLE_DIR, MANIFEST_DIR, REPORT_PATH.parent, RAW_DIR, RESPONSE_DIR):
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return sha256_json([])
    records = frame.fillna("").astype(str).to_dict("records")
    return sha256_json(records)


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

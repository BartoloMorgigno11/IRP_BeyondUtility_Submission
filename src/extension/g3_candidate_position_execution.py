"""G3 candidate-position FIRST/RANDOM protocol freeze and execution runner.

The default path is offline: it materializes the G3 protocol, request universe,
execution order, accounting, statistical plan, and retention plan without
calling Ollama. Future live execution remains behind --allow-api-calls and uses
raw-first persistence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
    sha256_json,
    sha256_text,
    stable_json,
    unordered_item_set_hash,
)
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1bx_execution_stage1 import frame_hash, sort_key
from src.extension.g0_canonical_llm_canary import MODEL_FAMILY, RequestResult, call_ollama_generate, write_raw_response_file
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync, positive_label_exposed, read_jsonl
from src.utils.config import find_project_root


PHASE_ID = "g3_candidate_position_first_random_protocol_freeze"
EXECUTION_PHASE_ID = "g3_candidate_position_first_random_execution_freeze"
DECISION_FROZEN = "G3_BASE_LLMRANK_FIRST_RANDOM_PROTOCOL_FROZEN"
DECISION_BLOCKED = "G3_BASE_LLMRANK_FIRST_RANDOM_PROTOCOL_BLOCKED"
EXECUTION_DECISION_FROZEN = "G3_EXECUTION_RUNNER_FROZEN"
EXECUTION_DECISION_BLOCKED = "G3_EXECUTION_RUNNER_BLOCKED"

CONFIG_PATH = Path("configs/g3_candidate_position_execution.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
NOTES_PATH = Path("report/notes.md")
PROTOCOL_REPORT_PATH = Path("report/g3_candidate_position_protocol.md")
EXECUTION_REPORT_PATH = Path("report/g3_candidate_position_execution_runner.md")
PROTOCOL_MANIFEST_PATH = MANIFEST_DIR / "g3_candidate_position_protocol_freeze.json"
EXECUTION_MANIFEST_PATH = MANIFEST_DIR / "g3_candidate_position_execution_freeze.json"

OUTPUT_TABLES = {
    "cohort": TABLE_DIR / "g3_candidate_position_cohort.csv",
    "request_universe": TABLE_DIR / "g3_candidate_position_request_universe.csv",
    "request_accounting": TABLE_DIR / "g3_candidate_position_request_accounting.csv",
    "pair_integrity": TABLE_DIR / "g3_candidate_position_pair_integrity.csv",
    "metric_contract": TABLE_DIR / "g3_candidate_position_metric_contract.csv",
    "statistical_plan": TABLE_DIR / "g3_candidate_position_statistical_plan.csv",
    "retention_plan": TABLE_DIR / "g3_candidate_position_retention_plan.csv",
    "validation": TABLE_DIR / "g3_candidate_position_validation.csv",
    "execution_order": TABLE_DIR / "g3_candidate_position_execution_order.csv",
    "execution_accounting": TABLE_DIR / "g3_candidate_position_execution_accounting.csv",
}

FREEZE_SOURCE_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/g3_candidate_position_execution.py"),
    Path("tests/test_extension_g3_candidate_position_execution.py"),
    NOTES_PATH,
)

DATASET_ORDER = tuple(ACTIVE_EXTENSION_DATASETS)
MODEL_ORDER = tuple(FUTURE_MODELS)
PROMPT_ORDER = tuple(PROMPT_STRATEGIES)
ORDER_FIRST = "FIRST"
ORDER_RANDOM = "RANDOM"
ORDER_CONDITIONS = (ORDER_FIRST, ORDER_RANDOM)
TASK = "ranking"
COHORT_SIZE = 200
RANDOM_SEED = "R5A_CANDDIF_RANDOM_POSITION_V1"
RANDOM_ALGORITHM = "sha256_seeded_sort_rotate_if_positive_first"
FIRST_ALGORITHM = "positive_at_position_1_then_remaining_negatives_in_frozen_primary_candidate_order"
HISTORY_POLICY = "visible_last_10_or_available"
NO_REUSE_POLICY = "NO_G1_G2_RESPONSE_REUSE_PROTOCOL_DIFFERENT_CANDIDATE_ORDER_IDENTITY"
PENDING_STATUS = "PENDING_NEW_G3_OLLAMA_CALL"
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
    validation_failures: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or execute the G3 FIRST/RANDOM candidate-position protocol.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--dry-run", action="store_true", help="Validate the frozen runner without inference.")
    parser.add_argument("--resume", action="store_true", help="Skip already-terminal G3 evidence during future execution.")
    parser.add_argument("--max-new-calls", type=int, default=None, help="Maximum new Ollama calls for a future live invocation.")
    parser.add_argument("--allow-api-calls", action="store_true", help="Required before local Ollama metadata or generation calls.")
    parser.add_argument("--no-write", action="store_true", help="Build in memory without writing freeze artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    config = load_config(root, Path(args.config))
    validate_config(config)

    if args.dry_run:
        outputs = build_all(root, config)
        if not args.no_write:
            write_outputs(root, outputs)
            append_phase_notes(root, outputs)
        print_summary(outputs, new_calls_this_invocation=0)
        return 0 if outputs["protocol_manifest"]["decision_token"] == DECISION_FROZEN and outputs["execution_manifest"]["decision_token"] == EXECUTION_DECISION_FROZEN else 1

    if not args.allow_api_calls:
        print("error=--allow-api-calls is required before any local Ollama execution path is enabled")
        return 1

    sent = execute_pending_requests(
        root,
        config,
        max_new_calls=args.max_new_calls,
        resume=bool(args.resume),
        call_fn=call_ollama_generate,
        sleep_fn=time.sleep,
    )
    print(f"new_ollama_calls_in_this_invocation={sent}")
    return 0


def print_summary(outputs: Mapping[str, Any], *, new_calls_this_invocation: int) -> None:
    protocol = outputs["protocol_manifest"]
    execution = outputs["execution_manifest"]
    print(f"decision={protocol['decision_token']}")
    print(f"logical_requests={protocol['logical_requests']}")
    print(f"g1_g2_reuse_requests={protocol['g1_g2_reuse_requests']}")
    print(f"new_ollama_calls_required={protocol['new_ollama_calls_required']}")
    print(f"execution_decision={execution['decision_token']}")
    print(f"pending_new_requests={execution['pending_new_requests']}")
    print(f"new_ollama_calls_in_this_invocation={new_calls_this_invocation}")
    print("llm_inference_calls_executed=0")


def build_all(root: Path | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    project_root = root or find_project_root()
    cfg = dict(config or load_config(project_root, CONFIG_PATH))
    validate_config(cfg)
    protocol_outputs = build_protocol_freeze(project_root, cfg)
    execution_outputs = build_execution_freeze(project_root, cfg, protocol_outputs["tables"]["request_universe"])
    tables = {**protocol_outputs["tables"], **execution_outputs["tables"]}
    return {
        "config": cfg,
        "tables": tables,
        "protocol_manifest": protocol_outputs["manifest"],
        "execution_manifest": execution_outputs["manifest"],
        "protocol_report": protocol_outputs["report"],
        "execution_report": execution_outputs["report"],
    }


def build_protocol_freeze(root: Path | None = None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    project_root = root or find_project_root()
    cfg = dict(config or load_config(project_root, CONFIG_PATH))
    validate_config(cfg)
    phase_start_head = git_output(project_root, ["rev-parse", "HEAD"])
    phase_start_status = git_output(project_root, ["status", "--short", "--untracked-files=all"])

    g1_config = g1.load_config(project_root, Path(str(cfg["inputs"]["g1_config"])))
    g1_inputs = g1.load_inputs(project_root, g1_config)
    g1_request_universe = read_csv(project_root / str(cfg["inputs"]["g1_request_universe"]))
    g1_execution_manifest = read_json(project_root / str(cfg["inputs"]["g1_execution_manifest"]))
    g1_retention_manifest = read_json(project_root / str(cfg["inputs"]["g1_retention_manifest"]))
    g2_retention_manifest = read_json(project_root / str(cfg["inputs"]["g2_execution_retention_manifest"]))
    g2_synthesis_manifest = read_json(project_root / str(cfg["inputs"]["g2_scientific_synthesis_manifest"]))
    f0_inference_contract = read_json(project_root / str(cfg["inputs"]["f0_inference_contract"]))

    model_digests = model_digests_from_g1(g1_request_universe)
    generation_contract = build_generation_contract(cfg, f0_inference_contract)
    parser_hash = parser_contract_hash()
    user_contexts = build_user_contexts(g1_inputs)
    cohort = build_cohort(g1_inputs, user_contexts)
    request_universe = build_request_universe(g1_inputs, user_contexts, cohort, model_digests, generation_contract, parser_hash, cfg)
    pair_integrity = build_pair_integrity(request_universe)
    request_accounting = build_request_accounting(request_universe)
    metric_contract = build_metric_contract(cfg)
    statistical_plan = build_statistical_plan(cfg)
    retention_plan = build_retention_plan(cfg)

    dependencies = {
        "g1_execution_manifest": g1_execution_manifest,
        "g1_retention_manifest": g1_retention_manifest,
        "g2_retention_manifest": g2_retention_manifest,
        "g2_synthesis_manifest": g2_synthesis_manifest,
    }
    validation = build_validation(
        cfg,
        phase_start_head=phase_start_head,
        phase_start_status=phase_start_status,
        cohort=cohort,
        request_universe=request_universe,
        pair_integrity=pair_integrity,
        request_accounting=request_accounting,
        statistical_plan=statistical_plan,
        dependencies=dependencies,
    )
    tables = {
        "cohort": cohort,
        "request_universe": request_universe,
        "request_accounting": request_accounting,
        "pair_integrity": pair_integrity,
        "metric_contract": metric_contract,
        "statistical_plan": statistical_plan,
        "retention_plan": retention_plan,
        "validation": validation,
    }
    manifest = build_protocol_manifest(
        project_root,
        cfg,
        tables,
        generation_contract,
        parser_hash,
        model_digests,
        phase_start_head,
        phase_start_status,
    )
    report = render_protocol_report(tables, manifest)
    return {"tables": tables, "manifest": manifest, "report": report}


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def validate_config(config: Mapping[str, Any]) -> None:
    if str(config["phase"]["id"]) != PHASE_ID:
        raise ValueError("unexpected G3 phase id")
    if tuple(config["datasets"]) != DATASET_ORDER:
        raise ValueError("G3 datasets must match the active extension datasets")
    if tuple(config["models"]) != MODEL_ORDER:
        raise ValueError("G3 models must match the frozen extension model set")
    if tuple(config["prompt_strategies"]) != PROMPT_ORDER:
        raise ValueError("G3 prompts must be BASE and LLMRANK")
    if tuple(config["candidate_orders"]["conditions"]) != ORDER_CONDITIONS:
        raise ValueError("G3 candidate orders must be FIRST then RANDOM")
    forbidden = [
        "allow_local_ollama_inference",
        "allow_local_ollama_metadata",
        "allow_external_api_calls",
        "allow_downloads",
        "allow_network_downloads",
        "allow_training",
        "allow_tuning",
        "allow_g4",
        "allow_g5",
    ]
    if any(bool(config["execution"].get(key, False)) for key in forbidden):
        raise ValueError("default G3 config must be no-inference/no-network")
    if int(config["execution"]["scientific_retry_count"]) != 0:
        raise ValueError("scientific retries must be zero")
    if int(config["execution"]["structural_invalidity_retry_count"]) != 0:
        raise ValueError("structural invalidity retries must be zero")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def allowed_freeze_artifact_paths() -> set[str]:
    paths = set(FREEZE_SOURCE_ARTIFACTS)
    paths.update(OUTPUT_TABLES.values())
    paths.update({PROTOCOL_MANIFEST_PATH, EXECUTION_MANIFEST_PATH, PROTOCOL_REPORT_PATH, EXECUTION_REPORT_PATH})
    return {path.as_posix() for path in paths}


def unexpected_git_status_lines(status: str) -> list[str]:
    allowed_paths = allowed_freeze_artifact_paths()
    unexpected: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path_text = line[3:].strip().replace("\\", "/")
        if " -> " in path_text or path_text not in allowed_paths:
            unexpected.append(line)
    return unexpected


def model_digests_from_g1(g1_request_universe: pd.DataFrame) -> dict[str, str]:
    digests: dict[str, str] = {}
    for model in MODEL_ORDER:
        rows = g1_request_universe.loc[g1_request_universe["model"].eq(model), "model_digest"].astype(str)
        unique = sorted(set(value for value in rows if value))
        if len(unique) != 1:
            raise ValueError(f"G1 model digest is not uniquely frozen for {model}")
        digests[model] = unique[0]
    return digests


def build_generation_contract(config: Mapping[str, Any], f0_inference_contract: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(f0_inference_contract.get("generation_parameters", {}))
    return {
        "generation_config_id": str(f0_inference_contract.get("generation_config_id", "")),
        "generation_config_hash": str(f0_inference_contract["generation_config_hash"]),
        "temperature": parameters.get("temperature", config["execution"]["temperature"]),
        "num_predict": parameters.get("num_predict", config["execution"]["max_output_tokens"]["ranking"]),
        "timeout_seconds": parameters.get("timeout_seconds", config["execution"]["timeout_seconds"]),
        "stream": parameters.get("stream", config["execution"]["stream"]),
        "response_format": config["execution"]["response_format"],
        "backend_retry_count": int(config["execution"]["backend_retry_count"]),
        "backend_retry_backoff_seconds": list(config["execution"]["backend_retry_backoff_seconds"]),
        "scientific_retry_count": int(config["execution"]["scientific_retry_count"]),
        "structural_invalidity_retry_count": int(config["execution"]["structural_invalidity_retry_count"]),
        "seed": parameters.get("seed", ""),
        "top_p": parameters.get("top_p", ""),
        "top_k": parameters.get("top_k", ""),
        "num_ctx": parameters.get("context_window", ""),
        "stop": parameters.get("stop_sequences", ""),
        "repeat_penalty": parameters.get("repeat_penalty", ""),
        "keep_alive": parameters.get("keep_alive", ""),
    }


def parser_contract_hash() -> str:
    return sha256_json(
        {
            "parser_contract": PARSER_CONTRACT_ID,
            "candidate_count": CANDIDATE_COUNT,
            "requested_output_length": REQUESTED_OUTPUT_LENGTH,
            "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
            "metric_k": METRIC_K,
            "position_base": 1,
            "position_domain": list(range(1, CANDIDATE_COUNT + 1)),
            "malformed_response_policy": "fail_strict_invalid_scores_zero",
        }
    )


def build_user_contexts(inputs: g1.G1Inputs) -> dict[tuple[str, str], dict[str, Any]]:
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        candidate_frame = inputs.ranking_candidates[dataset]
        users = sorted(candidate_frame["user_id"].astype(str).unique(), key=sort_key)
        for user_id in users:
            contexts[(dataset, str(user_id))] = g1.build_ranking_user_context(inputs, dataset, str(user_id))
    return contexts


def build_cohort(inputs: g1.G1Inputs, user_contexts: Mapping[tuple[str, str], Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_order = 1
    for dataset in DATASET_ORDER:
        users = sorted([user_id for (context_dataset, user_id) in user_contexts if context_dataset == dataset], key=sort_key)
        selected = users[:COHORT_SIZE]
        if len(selected) != COHORT_SIZE:
            raise ValueError(f"G3 cohort needs {COHORT_SIZE} users for {dataset}, observed {len(selected)}")
        for dataset_order, user_id in enumerate(selected, start=1):
            context = user_contexts[(dataset, user_id)]
            candidate_ids = [str(item["item_id"]) for item in context["ranking_candidates"]]
            history_ids = [str(item["item_id"]) for item in context["history_items"]]
            positive = str(context["positive_item_id"])
            rows.append(
                {
                    "cohort_id": "G3CP-COHORT-" + sha256_json({"dataset": dataset, "user_id": user_id}).upper()[:20],
                    "cohort_order": global_order,
                    "dataset_order": dataset_order,
                    "dataset": dataset,
                    "dataset_label": str(inputs.specs[dataset].display_name),
                    "user_id": user_id,
                    "selection_rule": "first_200_users_by_sort_key_from_frozen_G1_primary_candidate_users_after_candidate_contract_validation",
                    "same_users_across_models_prompts_orders": "true",
                    "outcome_dependent_filtering": "false",
                    "candidate_source": str(context["candidate_source"]),
                    "candidate_source_protocol": "F0_FROZEN_RANKING_CANDIDATES_G1_LINEAGE",
                    "candidate_count": CANDIDATE_COUNT,
                    "candidate_membership_hash": unordered_item_set_hash(candidate_ids),
                    "g1_canonical_candidate_order_hash": item_list_hash(candidate_ids),
                    "candidate_set_id": g1.candidate_set_id_for(dataset, user_id, str(context["candidate_hash"])),
                    "positive_item_id": positive,
                    "g1_positive_position": positive_position(candidate_ids, positive),
                    "history_policy": HISTORY_POLICY,
                    "visible_history_length": int(context["visible_history_length"]),
                    "full_history_length": int(context["full_history_length"]),
                    "history_hash": item_list_hash(history_ids),
                    "history_items": stable_json(history_ids),
                    "foodcom_history_class": str(context["foodcom_history_class"]),
                    "target_excluded_from_history": bool_text(positive not in set(history_ids)),
                    "cohort_status": "G3_SELECTED_SHARED_200_USER_COHORT",
                }
            )
            global_order += 1
    return pd.DataFrame(rows)


def positive_position(item_ids: Sequence[str], positive_item_id: str) -> int:
    for index, item_id in enumerate(item_ids, start=1):
        if str(item_id) == str(positive_item_id):
            return index
    raise ValueError("positive item is not present in candidate list")


def order_candidates(
    candidate_items: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    dataset: str,
    user_id: str,
    positive_item_id: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for original_position, item in enumerate(candidate_items, start=1):
        item_id = str(item["item_id"])
        title = str(item["title"])
        normalized.append(
            {
                "item_id": item_id,
                "title": title,
                "is_positive": item_id == str(positive_item_id),
                "original_candidate_position": int(item.get("candidate_position", original_position)),
            }
        )
    positives = [item for item in normalized if item["is_positive"]]
    if len(positives) != 1 or len(normalized) != CANDIDATE_COUNT or len({item["item_id"] for item in normalized}) != CANDIDATE_COUNT:
        raise ValueError("G3 candidate membership contract requires 20 unique items and one positive")
    if condition == ORDER_FIRST:
        ordered = positives + [item for item in normalized if not item["is_positive"]]
    elif condition == ORDER_RANDOM:
        ordered = sorted(
            normalized,
            key=lambda item: sha256_text(f"{RANDOM_SEED}|{dataset}|{user_id}|{item['item_id']}|{item['title']}"),
        )
        if ordered[0]["is_positive"]:
            ordered = ordered[1:] + ordered[:1]
    else:
        raise ValueError(f"unknown G3 order condition: {condition}")
    result: list[dict[str, Any]] = []
    for position, item in enumerate(ordered, start=1):
        copied = dict(item)
        copied["candidate_position"] = position
        result.append(copied)
    return result


def build_request_universe(
    inputs: g1.G1Inputs,
    user_contexts: Mapping[tuple[str, str], Mapping[str, Any]],
    cohort: pd.DataFrame,
    model_digests: Mapping[str, str],
    generation_contract: Mapping[str, Any],
    parser_hash: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    order = 1
    for cohort_row in cohort.sort_values("cohort_order", kind="mergesort").to_dict("records"):
        dataset = str(cohort_row["dataset"])
        user_id = str(cohort_row["user_id"])
        context = user_contexts[(dataset, user_id)]
        history_items = list(context["history_items"])
        history_ids = [str(item["item_id"]) for item in history_items]
        positive = str(context["positive_item_id"])
        for condition in ORDER_CONDITIONS:
            ordered = order_candidates(
                context["ranking_candidates"],
                condition=condition,
                dataset=dataset,
                user_id=user_id,
                positive_item_id=positive,
            )
            ordered_ids = [str(item["item_id"]) for item in ordered]
            ordered_titles = [str(item["title"]) for item in ordered]
            candidate_order_hash = item_list_hash(ordered_ids)
            membership_hash = unordered_item_set_hash(ordered_ids)
            candidate_order_algorithm = FIRST_ALGORITHM if condition == ORDER_FIRST else RANDOM_ALGORITHM
            for prompt_strategy in PROMPT_ORDER:
                bundle = render_extension_prompt(
                    prompt_strategy=prompt_strategy,
                    dataset_label=str(inputs.specs[dataset].display_name),
                    candidate_items=ordered,
                    history_items=history_items,
                    history_length=int(context["visible_history_length"]),
                )
                for model in MODEL_ORDER:
                    model_digest = str(model_digests[model])
                    identity_fields = {
                        "dataset": dataset,
                        "user_id": user_id,
                        "model": model,
                        "prompt_strategy": prompt_strategy,
                        "candidate_order_condition": condition,
                        "candidate_order_algorithm": candidate_order_algorithm,
                        "candidate_order_seed": RANDOM_SEED if condition == ORDER_RANDOM else "",
                        "history_policy": HISTORY_POLICY,
                        "history_length": int(context["visible_history_length"]),
                        "history_hash": item_list_hash(history_ids),
                        "candidate_membership_hash": membership_hash,
                        "candidate_order_hash": candidate_order_hash,
                        "candidate_items": stable_json(ordered_ids),
                        "positive_item_id": positive,
                        "prompt_hash": bundle.prompt_hash,
                        "model_digest": model_digest,
                        "generation_config_hash": str(generation_contract["generation_config_hash"]),
                        "parser_contract_hash": parser_hash,
                    }
                    request_identity_hash = sha256_json(identity_fields)
                    request_id = "G3CP-" + request_identity_hash.upper()[:32]
                    row = {
                        "canonical_request_id": request_id,
                        "request_id": request_id,
                        "request_order": order,
                        "dataset": dataset,
                        "dataset_label": str(inputs.specs[dataset].display_name),
                        "task": TASK,
                        "model": model,
                        "model_family": MODEL_FAMILY[model],
                        "model_digest": model_digest,
                        "prompt_strategy": prompt_strategy,
                        "prompt_contract": bundle.prompt_contract,
                        "prompt_hash": bundle.prompt_hash,
                        "parser_contract": PARSER_CONTRACT_ID,
                        "parser_contract_hash": parser_hash,
                        "generation_config_id": str(generation_contract["generation_config_id"]),
                        "generation_config_hash": str(generation_contract["generation_config_hash"]),
                        "generation_contract_hash": str(generation_contract["generation_config_hash"]),
                        "user_id": user_id,
                        "cohort_id": str(cohort_row["cohort_id"]),
                        "cohort_order": int(cohort_row["cohort_order"]),
                        "dataset_user_order": int(cohort_row["dataset_order"]),
                        "cohort_selection_rule": str(cohort_row["selection_rule"]),
                        "history_policy": HISTORY_POLICY,
                        "history_length": int(context["visible_history_length"]),
                        "full_history_length": int(context["full_history_length"]),
                        "history_hash": item_list_hash(history_ids),
                        "history_items": stable_json(history_ids),
                        "candidate_order_condition": condition,
                        "candidate_order_algorithm": candidate_order_algorithm,
                        "candidate_order_seed": RANDOM_SEED if condition == ORDER_RANDOM else "",
                        "candidate_order_key_derivation": "seed|dataset|user_id|candidate_item_id|candidate_title" if condition == ORDER_RANDOM else "not_applicable",
                        "candidate_set_id": str(cohort_row["candidate_set_id"]),
                        "candidate_source": str(cohort_row["candidate_source"]),
                        "candidate_source_protocol": "F0_FROZEN_RANKING_CANDIDATES_G1_LINEAGE",
                        "candidate_membership_hash": membership_hash,
                        "candidate_order_hash": candidate_order_hash,
                        "g1_canonical_candidate_order_hash": str(cohort_row["g1_canonical_candidate_order_hash"]),
                        "candidate_count": CANDIDATE_COUNT,
                        "candidate_items": stable_json(ordered_ids),
                        "candidate_titles_hash": sha256_json(ordered_titles),
                        "positive_item_id": positive,
                        "positive_candidate_position": positive_position(ordered_ids, positive),
                        "target_excluded_from_history": bool_text(positive not in set(history_ids)),
                        "only_candidate_order_may_differ_in_pair": "true",
                        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                        "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
                        "metric_k": METRIC_K,
                        "utility_denominator_policy": str(config["metrics"]["denominator_policy"]["utility"]),
                        "beyond_utility_denominator_policy": str(config["metrics"]["denominator_policy"]["beyond_utility"]),
                        "exact_g1_g2_reuse": "false",
                        "planned_action": "NEW_G3_OLLAMA_CALL_REQUIRED",
                        "reuse_policy": NO_REUSE_POLICY,
                        "raw_response_protocol": str(config["raw_evidence"]["raw_response_protocol"]),
                        "g3_future_raw_response_path": (Path(str(config["raw_evidence"]["response_dir"])) / f"{request_id}.json").as_posix(),
                        "request_identity_hash": request_identity_hash,
                        "request_identity_fields": stable_json(identity_fields),
                        "retention_policy": "TRACKED_FOREGROUND_UNIVERSE_EXTERNALIZED_RAW_EVIDENCE_AFTER_EXECUTION",
                    }
                    rows.append(row)
                    order += 1
    return pd.DataFrame(rows)


def build_pair_integrity(request_universe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pair_columns = ["dataset", "user_id", "model", "prompt_strategy"]
    for key, group in request_universe.groupby(pair_columns, sort=False):
        dataset, user_id, model, prompt = [str(value) for value in key]
        by_condition = {str(row["candidate_order_condition"]): row for row in group.to_dict("records")}
        first = by_condition.get(ORDER_FIRST)
        random = by_condition.get(ORDER_RANDOM)
        has_pair = first is not None and random is not None and len(group) == 2
        same_membership = has_pair and str(first["candidate_membership_hash"]) == str(random["candidate_membership_hash"])
        same_history = has_pair and str(first["history_hash"]) == str(random["history_hash"])
        same_target = has_pair and str(first["positive_item_id"]) == str(random["positive_item_id"])
        same_contract = has_pair and str(first["prompt_contract"]) == str(random["prompt_contract"]) and str(first["parser_contract_hash"]) == str(random["parser_contract_hash"])
        different_order = has_pair and str(first["candidate_order_hash"]) != str(random["candidate_order_hash"])
        first_position_ok = has_pair and int(first["positive_candidate_position"]) == 1
        random_position_ok = has_pair and 2 <= int(random["positive_candidate_position"]) <= CANDIDATE_COUNT
        rows.append(
            {
                "pair_id": "G3CP-PAIR-" + sha256_json({"dataset": dataset, "user_id": user_id, "model": model, "prompt": prompt}).upper()[:24],
                "dataset": dataset,
                "user_id": user_id,
                "model": model,
                "prompt_strategy": prompt,
                "first_request_id": "" if first is None else str(first["canonical_request_id"]),
                "random_request_id": "" if random is None else str(random["canonical_request_id"]),
                "pair_row_count": len(group),
                "has_exactly_first_and_random": bool_text(has_pair),
                "same_candidate_membership": bool_text(same_membership),
                "same_history": bool_text(same_history),
                "same_target": bool_text(same_target),
                "same_prompt_parser_generation_contract": bool_text(same_contract),
                "candidate_order_differs": bool_text(different_order),
                "first_positive_position": "" if first is None else int(first["positive_candidate_position"]),
                "random_positive_position": "" if random is None else int(random["positive_candidate_position"]),
                "first_positive_position_ok": bool_text(first_position_ok),
                "random_positive_position_ok": bool_text(random_position_ok),
                "only_candidate_order_differs": bool_text(
                    has_pair
                    and same_membership
                    and same_history
                    and same_target
                    and same_contract
                    and different_order
                    and first_position_ok
                    and random_position_ok
                ),
            }
        )
    return pd.DataFrame(rows)


def build_request_accounting(request_universe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(scope: str, value: str, frame: pd.DataFrame) -> None:
        rows.append(
            {
                "accounting_scope": scope,
                "scope_value": value,
                "logical_requests": len(frame),
                "new_ollama_calls_required": int(frame["planned_action"].eq("NEW_G3_OLLAMA_CALL_REQUIRED").sum()),
                "exact_g1_g2_reuse": int(frame["exact_g1_g2_reuse"].astype(str).str.lower().eq("true").sum()),
                "users": int(frame["user_id"].nunique()) if value != "ALL" else "",
                "datasets": int(frame["dataset"].nunique()),
                "models": int(frame["model"].nunique()),
                "prompts": int(frame["prompt_strategy"].nunique()),
                "candidate_orders": int(frame["candidate_order_condition"].nunique()),
            }
        )

    add("GRAND_TOTAL", "ALL", request_universe)
    for dataset, group in request_universe.groupby("dataset", sort=False):
        add("BY_DATASET", str(dataset), group)
    for model, group in request_universe.groupby("model", sort=False):
        add("BY_MODEL", str(model), group)
    for prompt, group in request_universe.groupby("prompt_strategy", sort=False):
        add("BY_PROMPT", str(prompt), group)
    for condition, group in request_universe.groupby("candidate_order_condition", sort=False):
        add("BY_ORDER", str(condition), group)
    return pd.DataFrame(rows)


def build_metric_contract(config: Mapping[str, Any]) -> pd.DataFrame:
    utility_denominator = str(config["metrics"]["denominator_policy"]["utility"])
    strict_denominator = str(config["metrics"]["denominator_policy"]["beyond_utility"])
    rows = [
        {
            "metric": "hr_at_5",
            "metric_family": "utility",
            "directionality": "higher_is_better",
            "formula": "1 if positive item appears in ranks 1..5 else 0",
            "denominator_policy": utility_denominator,
            "undefined_policy": "invalid_terminal_response_counts_zero",
        },
        {
            "metric": "ndcg_at_5",
            "metric_family": "utility",
            "directionality": "higher_is_better",
            "formula": "1/log2(rank+1) if positive item appears in ranks 1..5 else 0",
            "denominator_policy": utility_denominator,
            "undefined_policy": "invalid_terminal_response_counts_zero",
        },
        {
            "metric": "mrr",
            "metric_family": "utility",
            "directionality": "higher_is_better",
            "formula": "1/rank if positive item appears in returned top-10 else 0",
            "denominator_policy": utility_denominator,
            "undefined_policy": "invalid_terminal_response_counts_zero",
        },
        {
            "metric": "canddif_hr_at_5",
            "metric_family": "candidate_position_sensitivity",
            "directionality": "positive_means_first_better_than_random",
            "formula": "log((1-HR_random)/(1-HR_first)) with epsilon 1e-12 clamp when an error rate is zero",
            "denominator_policy": utility_denominator,
            "undefined_policy": "same_numerical_safety_policy_as_R5B_canddif_value",
        },
        {
            "metric": "canddif_ndcg_at_5",
            "metric_family": "candidate_position_sensitivity",
            "directionality": "positive_means_first_better_than_random",
            "formula": "log((1-NDCG_random)/(1-NDCG_first)) with epsilon 1e-12 clamp when an error rate is zero",
            "denominator_policy": utility_denominator,
            "undefined_policy": "same_numerical_safety_policy_as_R5B_canddif_value",
        },
        {
            "metric": "beyond_utility_metrics",
            "metric_family": "beyond_utility",
            "directionality": "reuse_frozen_metric_specific_contract",
            "formula": "reuse frozen G1 definitions where applicable after strict-valid parsing",
            "denominator_policy": strict_denominator,
            "undefined_policy": "metric_specific_defined_user_subset",
        },
    ]
    for row in rows:
        row["metric_contract_hash"] = sha256_json(row)
    return pd.DataFrame(rows)


def build_statistical_plan(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = list(config["metrics"]["utility"])
    order = 1
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            for prompt in PROMPT_ORDER:
                for metric in metrics:
                    canddif_metric = f"canddif_{metric}" if metric in {"hr_at_5", "ndcg_at_5"} else "not_applicable"
                    row = {
                        "plan_order": order,
                        "comparison_id": f"G3_FIRST_VS_RANDOM_{dataset.upper()}_{MODEL_FAMILY[model]}_{prompt}_{metric.upper()}",
                        "dataset": dataset,
                        "model": model,
                        "prompt_strategy": prompt,
                        "comparison_group": "g3_candidate_position_first_random",
                        "metric": metric,
                        "condition_a": ORDER_RANDOM,
                        "condition_b": ORDER_FIRST,
                        "contrast": "FIRST_minus_RANDOM",
                        "canddif_metric": canddif_metric,
                        "canddif_formula": "log((1-Acc_random)/(1-Acc_first))" if canddif_metric != "not_applicable" else "not_applicable",
                        "canddif_epsilon_policy": "clamp_zero_error_rate_to_1e-12" if canddif_metric != "not_applicable" else "not_applicable",
                        "paired_unit": "user_id",
                        "planned_paired_users": COHORT_SIZE,
                        "denominator_policy": str(config["metrics"]["denominator_policy"]["utility"]),
                        "primary_test": str(config["statistics"]["primary_test"]),
                        "bootstrap_resamples": int(config["statistics"]["bootstrap_resamples"]),
                        "bootstrap_seed": int(config["statistics"]["bootstrap_seed"]),
                        "confidence_interval": str(config["statistics"]["confidence_interval"]),
                        "effect_size": str(config["statistics"]["effect_size"]),
                        "secondary_test": str(config["statistics"]["secondary_test"]),
                        "multiple_comparison_correction": str(config["statistics"]["multiple_comparison_correction"]),
                        "alpha": float(config["statistics"]["alpha"]),
                        "planned_status": "FROZEN_PRE_INFERENCE",
                    }
                    row["plan_hash"] = sha256_json(row)
                    rows.append(row)
                    order += 1
    return pd.DataFrame(rows)


def build_retention_plan(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    tracked = list(config["retention"]["tracked_scientific_foreground"])
    for artifact in tracked:
        rows.append(
            {
                "path": artifact,
                "retention_class": "TRACKED_SCIENTIFIC_FOREGROUND",
                "required_to_interpret_or_reproduce_g3": "true",
                "retention_policy": "commit_with_g3_protocol_freeze",
                "stable_hash_location": "results/manifests/g3_candidate_position_protocol_freeze.json;results/manifests/g3_candidate_position_execution_freeze.json",
            }
        )
    rows.append(
        {
            "path": str(config["retention"]["externalized_raw_evidence_future_root"]),
            "retention_class": "EXTERNALIZED_RAW_EVIDENCE_AFTER_FUTURE_EXECUTION",
            "required_to_interpret_or_reproduce_g3": "true_after_execution",
            "retention_policy": "ignored_runtime_evidence_referenced_by_future_retention_manifest",
            "stable_hash_location": "future_G3_retention_manifest",
        }
    )
    return pd.DataFrame(rows)


def build_validation(
    config: Mapping[str, Any],
    *,
    phase_start_head: str,
    phase_start_status: str,
    cohort: pd.DataFrame,
    request_universe: pd.DataFrame,
    pair_integrity: pd.DataFrame,
    request_accounting: pd.DataFrame,
    statistical_plan: pd.DataFrame,
    dependencies: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

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

    expected_total = int(config["expected_accounting"]["logical_requests"])
    by_dataset_counts = request_universe.groupby("dataset", sort=False).size().astype(int).to_dict()
    by_model_counts = request_universe.groupby("model", sort=False).size().astype(int).to_dict()
    by_prompt_counts = request_universe.groupby("prompt_strategy", sort=False).size().astype(int).to_dict()
    by_order_counts = request_universe.groupby("candidate_order_condition", sort=False).size().astype(int).to_dict()
    cohort_counts = cohort.groupby("dataset", sort=False)["user_id"].nunique().astype(int).to_dict()
    positive_by_order = request_universe.groupby("candidate_order_condition", sort=False)["positive_candidate_position"].apply(lambda s: sorted(set(int(v) for v in s))).to_dict()

    unexpected_status = unexpected_git_status_lines(phase_start_status)
    add("G3-VAL-001", "precondition HEAD", phase_start_head == str(config["phase"]["required_head"]), phase_start_head, config["phase"]["required_head"])
    add(
        "G3-VAL-002",
        "git status limited to current G3 freeze artifacts",
        not unexpected_status,
        {"unexpected": unexpected_status, "full_status": phase_start_status or "clean"},
        "no non-G3 freeze artifacts",
    )
    add("G3-VAL-003", "G1 execution frozen", dependencies["g1_execution_manifest"].get("decision_token") == "G1_PRIMARY_RANKING_VALIDATED", dependencies["g1_execution_manifest"].get("decision_token"), "G1_PRIMARY_RANKING_VALIDATED")
    add("G3-VAL-004", "G1 retention frozen", dependencies["g1_retention_manifest"].get("decision_source") == "G1_PRIMARY_RANKING_VALIDATED", dependencies["g1_retention_manifest"].get("decision_source"), "G1_PRIMARY_RANKING_VALIDATED")
    add("G3-VAL-005", "G2 execution retention frozen", dependencies["g2_retention_manifest"].get("decision_token") == "G2_EXECUTION_RETENTION_FROZEN", dependencies["g2_retention_manifest"].get("decision_token"), "G2_EXECUTION_RETENTION_FROZEN")
    add("G3-VAL-006", "G2 scientific synthesis frozen", dependencies["g2_synthesis_manifest"].get("decision_token") == "G2_SCIENTIFIC_SYNTHESIS_FROZEN", dependencies["g2_synthesis_manifest"].get("decision_token"), "G2_SCIENTIFIC_SYNTHESIS_FROZEN")
    add("G3-VAL-007", "cohort 200 users per dataset", cohort_counts == {dataset: COHORT_SIZE for dataset in DATASET_ORDER}, cohort_counts, {dataset: COHORT_SIZE for dataset in DATASET_ORDER})
    add("G3-VAL-008", "request universe row count", len(request_universe) == expected_total, len(request_universe), expected_total)
    add("G3-VAL-009", "request IDs unique", request_universe["canonical_request_id"].nunique() == len(request_universe), request_universe["canonical_request_id"].nunique(), len(request_universe))
    add("G3-VAL-010", "no G1/G2 response reuse", int(request_universe["exact_g1_g2_reuse"].astype(str).str.lower().eq("true").sum()) == 0, int(request_universe["exact_g1_g2_reuse"].astype(str).str.lower().eq("true").sum()), 0)
    add("G3-VAL-011", "new Ollama call count", int(request_universe["planned_action"].eq("NEW_G3_OLLAMA_CALL_REQUIRED").sum()) == expected_total, int(request_universe["planned_action"].eq("NEW_G3_OLLAMA_CALL_REQUIRED").sum()), expected_total)
    add("G3-VAL-012", "dataset accounting", by_dataset_counts == {dataset: int(config["expected_accounting"]["by_dataset"]) for dataset in DATASET_ORDER}, by_dataset_counts, {dataset: int(config["expected_accounting"]["by_dataset"]) for dataset in DATASET_ORDER})
    add("G3-VAL-013", "model accounting", by_model_counts == {model: int(config["expected_accounting"]["by_model"]) for model in MODEL_ORDER}, by_model_counts, {model: int(config["expected_accounting"]["by_model"]) for model in MODEL_ORDER})
    add("G3-VAL-014", "prompt accounting", by_prompt_counts == {prompt: int(config["expected_accounting"]["by_prompt"]) for prompt in PROMPT_ORDER}, by_prompt_counts, {prompt: int(config["expected_accounting"]["by_prompt"]) for prompt in PROMPT_ORDER})
    add("G3-VAL-015", "order accounting", by_order_counts == {condition: int(config["expected_accounting"]["by_order"]) for condition in ORDER_CONDITIONS}, by_order_counts, {condition: int(config["expected_accounting"]["by_order"]) for condition in ORDER_CONDITIONS})
    add("G3-VAL-016", "FIRST positive position fixed", positive_by_order.get(ORDER_FIRST) == [1], positive_by_order.get(ORDER_FIRST), [1])
    random_positions = positive_by_order.get(ORDER_RANDOM, [])
    add("G3-VAL-017", "RANDOM positive never first", bool(random_positions) and min(random_positions) >= 2 and max(random_positions) <= CANDIDATE_COUNT, random_positions, "positions 2..20")
    add("G3-VAL-018", "all FIRST/RANDOM pairs valid", pair_integrity["only_candidate_order_differs"].eq("true").all(), int(pair_integrity["only_candidate_order_differs"].eq("true").sum()), len(pair_integrity))
    add("G3-VAL-019", "pair count", len(pair_integrity) == 5 * 3 * 2 * COHORT_SIZE, len(pair_integrity), 5 * 3 * 2 * COHORT_SIZE)
    add("G3-VAL-020", "target excluded from visible history", request_universe["target_excluded_from_history"].eq("true").all(), int(request_universe["target_excluded_from_history"].eq("true").sum()), len(request_universe))
    add("G3-VAL-021", "strict G1 prompt/parser contract reused", set(request_universe["parser_contract"]) == {PARSER_CONTRACT_ID} and set(request_universe["requested_output_length"].astype(int)) == {REQUESTED_OUTPUT_LENGTH}, sorted(set(request_universe["parser_contract"])), PARSER_CONTRACT_ID)
    add("G3-VAL-022", "statistical plan rows", len(statistical_plan) == 5 * 3 * 2 * 3, len(statistical_plan), 5 * 3 * 2 * 3)
    add("G3-VAL-023", "bootstrap settings reused", set(statistical_plan["bootstrap_seed"].astype(int)) == {20260901} and set(statistical_plan["bootstrap_resamples"].astype(int)) == {10000}, {"seed": sorted(set(statistical_plan["bootstrap_seed"].astype(int))), "resamples": sorted(set(statistical_plan["bootstrap_resamples"].astype(int)))}, {"seed": 20260901, "resamples": 10000})
    add("G3-VAL-024", "LLMSRec excluded", not request_universe.astype(str).apply(lambda col: col.str.contains("LLMSRec|llmsrec", regex=True).any()).any(), "absent", "absent")
    add("G3-VAL-025", "zero inference during freeze", True, 0, 0)
    return pd.DataFrame(rows)


def build_protocol_manifest(
    root: Path,
    config: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    generation_contract: Mapping[str, Any],
    parser_hash: str,
    model_digests: Mapping[str, str],
    phase_start_head: str,
    phase_start_status: str,
) -> dict[str, Any]:
    validation = tables["validation"]
    request_universe = tables["request_universe"]
    request_accounting = tables["request_accounting"]
    grand = request_accounting.loc[request_accounting["accounting_scope"].eq("GRAND_TOTAL")].iloc[0].to_dict()
    validation_passed = bool(validation["status"].eq("PASS").all())
    return {
        "phase_id": PHASE_ID,
        "created_at_utc": utc_now(),
        "decision_token": DECISION_FROZEN if validation_passed else DECISION_BLOCKED,
        "validation_passed": validation_passed,
        "validation_pass": int(validation["status"].eq("PASS").sum()),
        "validation_fail": int(validation["status"].eq("FAIL").sum()),
        "precondition_head": str(config["phase"]["required_head"]),
        "head_before_generation": phase_start_head,
        "phase_start_git_status": phase_start_status,
        "datasets": list(DATASET_ORDER),
        "models": list(MODEL_ORDER),
        "prompt_strategies": list(PROMPT_ORDER),
        "candidate_order_conditions": list(ORDER_CONDITIONS),
        "llmsrec_excluded": True,
        "cohort_rule": str(config["cohort"]["selection_rule"]),
        "users_per_dataset": COHORT_SIZE,
        "history_policy": HISTORY_POLICY,
        "first_algorithm": FIRST_ALGORITHM,
        "random_algorithm": RANDOM_ALGORITHM,
        "random_seed": RANDOM_SEED,
        "random_key_derivation": str(config["candidate_orders"]["random"]["key_derivation"]),
        "prompt_contracts": {
            "parser_contract": PARSER_CONTRACT_ID,
            "requested_output_length": REQUESTED_OUTPUT_LENGTH,
            "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
            "metric_k": METRIC_K,
        },
        "generation_contract": dict(generation_contract),
        "parser_contract_hash": parser_hash,
        "model_digests": dict(model_digests),
        "logical_requests": int(grand["logical_requests"]),
        "g1_g2_reuse_requests": int(grand["exact_g1_g2_reuse"]),
        "new_ollama_calls_required": int(grand["new_ollama_calls_required"]),
        "llm_inference_calls_executed": 0,
        "g3_execution_started": False,
        "table_hashes": {OUTPUT_TABLES[name].as_posix(): frame_hash(frame) for name, frame in tables.items() if name in OUTPUT_TABLES},
        "source_artifact_hashes": {
            str(path): sha256_file(root / str(path))
            for path in [
                config["inputs"]["g1_request_universe"],
                config["inputs"]["g1_execution_manifest"],
                config["inputs"]["g1_retention_manifest"],
                config["inputs"]["g2_execution_retention_manifest"],
                config["inputs"]["g2_scientific_synthesis_manifest"],
                config["inputs"]["f0_inference_contract"],
            ]
            if (root / str(path)).exists()
        },
    }


def render_protocol_report(tables: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]) -> str:
    accounting = tables["request_accounting"]
    validation = tables["validation"]
    by_dataset = accounting.loc[accounting["accounting_scope"].eq("BY_DATASET")]
    by_model = accounting.loc[accounting["accounting_scope"].eq("BY_MODEL")]
    by_prompt = accounting.loc[accounting["accounting_scope"].eq("BY_PROMPT")]
    by_order = accounting.loc[accounting["accounting_scope"].eq("BY_ORDER")]
    return "\n".join(
        [
            "# G3 Candidate-Position FIRST/RANDOM Protocol Freeze",
            "",
            f"Decision token: `{manifest['decision_token']}`.",
            "",
            "## Scope",
            "",
            "G3 freezes BASE and LLMRANK candidate-position/order sensitivity over FIRST versus RANDOM. LLMSRec is excluded because repository evidence classifies it as reconstructed and not a faithful frozen executable strategy.",
            "",
            "## Cohort",
            "",
            f"- Rule: `{manifest['cohort_rule']}`.",
            f"- Users: `{manifest['users_per_dataset']}` per dataset, shared across all models, prompts, and orders.",
            "- Outcome-dependent filtering: `false`.",
            "",
            "## Candidate Orders",
            "",
            f"- FIRST: `{manifest['first_algorithm']}`.",
            f"- RANDOM: `{manifest['random_algorithm']}`.",
            f"- RANDOM seed: `{manifest['random_seed']}`.",
            f"- RANDOM key: `{manifest['random_key_derivation']}`.",
            "",
            "## Prompt And Parser",
            "",
            f"- Prompts: `{';'.join(PROMPT_ORDER)}`.",
            f"- Parser: `{PARSER_CONTRACT_ID}`.",
            f"- Output: strict JSON top-10 one-based candidate indices.",
            f"- History: `{HISTORY_POLICY}`.",
            "",
            "## Accounting",
            "",
            markdown_table(accounting.loc[accounting["accounting_scope"].eq("GRAND_TOTAL")], ["accounting_scope", "scope_value", "logical_requests", "new_ollama_calls_required", "exact_g1_g2_reuse"]),
            "",
            "## By Dataset",
            "",
            markdown_table(by_dataset, ["scope_value", "logical_requests", "new_ollama_calls_required", "users"]),
            "",
            "## By Model",
            "",
            markdown_table(by_model, ["scope_value", "logical_requests", "new_ollama_calls_required"]),
            "",
            "## By Prompt",
            "",
            markdown_table(by_prompt, ["scope_value", "logical_requests", "new_ollama_calls_required"]),
            "",
            "## By Order",
            "",
            markdown_table(by_order, ["scope_value", "logical_requests", "new_ollama_calls_required"]),
            "",
            "## Statistics",
            "",
            "- Primary test: paired-user bootstrap mean delta, FIRST minus RANDOM.",
            "- Bootstrap: 10000 resamples, seed 20260901, two-sided 95 percent percentile CI.",
            "- Multiplicity: Holm-Bonferroni within metric/dataset/model/prompt family, reused from the frozen G2 convention.",
            "- CandDif HR/NDCG: log((1-Acc_random)/(1-Acc_first)) with the R5B epsilon safety policy.",
            "",
            "## Validation",
            "",
            f"- PASS: `{int(validation['status'].eq('PASS').sum())}`.",
            f"- FAIL: `{int(validation['status'].eq('FAIL').sum())}`.",
        ]
    ) + "\n"


def build_execution_freeze(root: Path, config: Mapping[str, Any], request_universe: pd.DataFrame | None = None) -> dict[str, Any]:
    universe = request_universe if request_universe is not None else read_csv(root / str(config["outputs"]["request_universe"]))
    existing = inspect_existing_evidence(root, config, universe)
    execution_order = build_execution_order(universe, existing)
    execution_accounting = build_execution_accounting(universe, execution_order, existing)
    validation_failures = validate_execution_freeze(config, universe, execution_order, execution_accounting, existing)
    tables = {"execution_order": execution_order, "execution_accounting": execution_accounting}
    manifest = build_execution_manifest(root, config, tables, existing, validation_failures)
    report = render_execution_report(manifest, execution_accounting)
    return {"tables": tables, "manifest": manifest, "report": report}


def inspect_existing_evidence(root: Path, config: Mapping[str, Any], request_universe: pd.DataFrame) -> ExistingEvidenceState:
    request_path = root / str(config["raw_evidence"]["request_jsonl"])
    response_path = root / str(config["raw_evidence"]["response_jsonl"])
    response_dir = root / str(config["raw_evidence"]["response_dir"])
    request_records = [dict(row) for row in read_jsonl(request_path)] if request_path.exists() else []
    response_records = [dict(row) for row in read_jsonl(response_path)] if response_path.exists() else []
    request_ids = set(request_universe["canonical_request_id"].astype(str))
    failures: list[str] = []
    terminal_by_id: dict[str, dict[str, Any]] = {}

    raw_ids = [str(row.get("canonical_request_id", row.get("request_id", ""))) for row in response_records]
    outside = sorted(set(raw_ids) - request_ids)
    if outside:
        failures.append(f"G3_RAW_EVIDENCE_OUTSIDE_UNIVERSE:{outside[0]}")
    duplicates = sorted(request_id for request_id in set(raw_ids) if raw_ids.count(request_id) > 1)
    if duplicates:
        failures.append(f"G3_DUPLICATE_TERMINAL_LEDGER_ROWS:{duplicates[0]}")
    for record in response_records:
        request_id = str(record.get("canonical_request_id", record.get("request_id", "")))
        if request_id:
            terminal_by_id[request_id] = dict(record)
    if response_dir.exists():
        raw_file_ids = {path.stem for path in response_dir.glob("*.json")}
        outside_files = sorted(raw_file_ids - request_ids)
        if outside_files:
            failures.append(f"G3_RAW_FILE_OUTSIDE_UNIVERSE:{outside_files[0]}")
    return ExistingEvidenceState(request_records, response_records, terminal_by_id, failures)


def build_execution_order(request_universe: pd.DataFrame, existing: ExistingEvidenceState) -> pd.DataFrame:
    frame = request_universe.copy()
    model_order = {model: index for index, model in enumerate(MODEL_ORDER)}
    dataset_order = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    prompt_order = {prompt: index for index, prompt in enumerate(PROMPT_ORDER)}
    condition_order = {condition: index for index, condition in enumerate(ORDER_CONDITIONS)}
    frame["_model_order"] = frame["model"].map(model_order).astype(int)
    frame["_dataset_order"] = frame["dataset"].map(dataset_order).astype(int)
    frame["_prompt_order"] = frame["prompt_strategy"].map(prompt_order).astype(int)
    frame["_condition_order"] = frame["candidate_order_condition"].map(condition_order).astype(int)
    frame["_user_sort"] = frame["user_id"].map(sort_key)
    frame = frame.sort_values(["_model_order", "_dataset_order", "_prompt_order", "_condition_order", "_user_sort"], kind="mergesort").copy()
    frame["physical_execution_order"] = range(1, len(frame) + 1)
    frame["execution_status"] = frame["canonical_request_id"].map(lambda request_id: "G3_TERMINAL_SKIP_ZERO_CALL" if str(request_id) in existing.terminal_by_id else PENDING_STATUS)
    frame["existing_terminal_evidence"] = frame["canonical_request_id"].map(lambda request_id: bool_text(str(request_id) in existing.terminal_by_id))
    keep_columns = [
        "physical_execution_order",
        "canonical_request_id",
        "dataset",
        "user_id",
        "model",
        "prompt_strategy",
        "candidate_order_condition",
        "history_length",
        "candidate_order_hash",
        "candidate_membership_hash",
        "positive_candidate_position",
        "prompt_hash",
        "model_digest",
        "generation_config_hash",
        "parser_contract_hash",
        "g3_future_raw_response_path",
        "execution_status",
        "existing_terminal_evidence",
    ]
    return frame[keep_columns].reset_index(drop=True)


def build_execution_accounting(request_universe: pd.DataFrame, execution_order: pd.DataFrame, existing: ExistingEvidenceState) -> pd.DataFrame:
    terminal = int(execution_order["existing_terminal_evidence"].eq("true").sum())
    pending = int(execution_order["execution_status"].eq(PENDING_STATUS).sum())
    return pd.DataFrame(
        [
            {
                "accounting_scope": "GRAND_TOTAL",
                "logical_requests": len(request_universe),
                "new_ollama_calls_required": len(request_universe),
                "existing_g3_terminal_requests": terminal,
                "pending_new_requests": pending,
                "g1_g2_reuse_requests": 0,
                "request_ledger_rows": len(existing.request_records),
                "response_ledger_rows": len(existing.response_records),
                "new_ollama_calls_in_dry_run": 0,
            }
        ]
    )


def validate_execution_freeze(
    config: Mapping[str, Any],
    request_universe: pd.DataFrame,
    execution_order: pd.DataFrame,
    execution_accounting: pd.DataFrame,
    existing: ExistingEvidenceState,
) -> list[str]:
    failures = list(existing.validation_failures)
    expected = int(config["expected_accounting"]["logical_requests"])
    if len(request_universe) != expected:
        failures.append(f"G3_EXECUTION_UNIVERSE_COUNT_MISMATCH:{len(request_universe)}")
    if len(execution_order) != expected:
        failures.append(f"G3_EXECUTION_ORDER_COUNT_MISMATCH:{len(execution_order)}")
    if execution_order["canonical_request_id"].nunique() != expected:
        failures.append("G3_EXECUTION_ORDER_IDS_NOT_UNIQUE")
    grand = execution_accounting.iloc[0]
    if int(grand["existing_g3_terminal_requests"]) != 0:
        failures.append(f"G3_TERMINAL_EVIDENCE_ALREADY_EXISTS:{grand['existing_g3_terminal_requests']}")
    if int(grand["pending_new_requests"]) != expected:
        failures.append(f"G3_PENDING_COUNT_MISMATCH:{grand['pending_new_requests']}")
    return failures


def build_execution_manifest(
    root: Path,
    config: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    existing: ExistingEvidenceState,
    validation_failures: Sequence[str],
) -> dict[str, Any]:
    accounting = tables["execution_accounting"].iloc[0].to_dict()
    return {
        "phase_id": EXECUTION_PHASE_ID,
        "created_at_utc": utc_now(),
        "decision_token": EXECUTION_DECISION_FROZEN if not validation_failures else EXECUTION_DECISION_BLOCKED,
        "validation_passed": not bool(validation_failures),
        "validation_failures": list(validation_failures),
        "protocol_decision_required": DECISION_FROZEN,
        "logical_requests": int(accounting["logical_requests"]),
        "new_ollama_calls_required": int(accounting["new_ollama_calls_required"]),
        "existing_g3_terminal_requests": int(accounting["existing_g3_terminal_requests"]),
        "pending_new_requests": int(accounting["pending_new_requests"]),
        "g1_g2_reuse_requests": 0,
        "llm_inference_calls_executed": 0,
        "raw_first_persistence": True,
        "resume_policy": "skip_only_matching_terminal_g3_raw_evidence_by_canonical_request_id",
        "structural_invalidity_retry_count": 0,
        "backend_retry_count": int(config["execution"]["backend_retry_count"]),
        "table_hashes": {OUTPUT_TABLES[name].as_posix(): frame_hash(frame) for name, frame in tables.items()},
        "raw_evidence": dict(config["raw_evidence"]),
        "request_ledger_rows": len(existing.request_records),
        "response_ledger_rows": len(existing.response_records),
        "g3_execution_started": False,
        "zero_inference": True,
        "head_before_generation": git_output(root, ["rev-parse", "HEAD"]),
    }


def render_execution_report(manifest: Mapping[str, Any], accounting: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# G3 Candidate-Position Execution Runner Freeze",
            "",
            f"Decision token: `{manifest['decision_token']}`.",
            "",
            "## Accounting",
            "",
            markdown_table(accounting, ["accounting_scope", "logical_requests", "new_ollama_calls_required", "existing_g3_terminal_requests", "pending_new_requests", "g1_g2_reuse_requests"]),
            "",
            "## Runner Contract",
            "",
            "- Live execution requires `--allow-api-calls`.",
            "- Dry-run performs zero Ollama calls.",
            "- Raw response file is written before terminal parsing ledger append.",
            "- Structural invalid responses are terminal scientific evidence and are not retried.",
            "- Resume skips only matching terminal G3 evidence by canonical request ID.",
        ]
    ) + "\n"


def select_pending_execution_rows(execution_order: pd.DataFrame, *, max_new_calls: int | None, resume: bool) -> pd.DataFrame:
    frame = execution_order
    if resume:
        frame = frame.loc[frame["execution_status"].eq(PENDING_STATUS)].copy()
    if max_new_calls is not None:
        frame = frame.head(max_new_calls).copy()
    return frame


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
    protocol_manifest = read_json(root / str(config["outputs"]["protocol_manifest"]))
    if str(protocol_manifest.get("decision_token", "")) != DECISION_FROZEN:
        raise RuntimeError("G3 protocol is not frozen")
    request_universe = read_csv(root / str(config["outputs"]["request_universe"]))
    existing = inspect_existing_evidence(root, config, request_universe)
    if existing.validation_failures:
        raise RuntimeError(";".join(existing.validation_failures))
    order = build_execution_order(request_universe, existing)
    pending = select_pending_execution_rows(order, max_new_calls=max_new_calls, resume=resume)
    contexts = build_prompt_context_cache(root, config)
    sent = 0
    for order_row in pending.to_dict("records"):
        row = request_universe.loc[request_universe["canonical_request_id"].eq(str(order_row["canonical_request_id"]))].iloc[0].to_dict()
        context = build_context_for_request(row, contexts)
        payload = build_payload(config, row, context)
        request_record = build_request_record(order_row, row, payload)
        append_jsonl_fsync(root / str(config["raw_evidence"]["request_jsonl"]), request_record)
        append_jsonl_fsync(root / str(config["raw_evidence"]["event_jsonl"]), {"event": "request_recorded_before_send", "phase_id": EXECUTION_PHASE_ID, "canonical_request_id": row["canonical_request_id"], "timestamp_utc": utc_now()})
        result, attempts = call_with_backend_retries(root, config, row, payload, call_fn=call_fn, sleep_fn=sleep_fn)
        raw_record = build_unparsed_raw_record(row, result, attempts)
        write_raw_response_file(root / str(row["g3_future_raw_response_path"]), raw_record)
        terminal_record = build_terminal_record(row, raw_record)
        append_jsonl_fsync(root / str(config["raw_evidence"]["response_jsonl"]), terminal_record)
        append_jsonl_fsync(root / str(config["raw_evidence"]["event_jsonl"]), {"event": "terminal_response_recorded_after_raw_persistence", "phase_id": EXECUTION_PHASE_ID, "canonical_request_id": row["canonical_request_id"], "terminal_state": terminal_record["terminal_state"], "timestamp_utc": utc_now()})
        sent += 1
    return sent


def build_prompt_context_cache(root: Path, config: Mapping[str, Any]) -> dict[Any, Any]:
    g1_config = g1.load_config(root, Path(str(config["inputs"]["g1_config"])))
    g1_inputs = g1.load_inputs(root, g1_config)
    return {"_g1_inputs": {"value": g1_inputs}, **build_user_contexts(g1_inputs)}


def build_context_for_request(row: Mapping[str, Any], contexts: Mapping[Any, Any]) -> dict[str, Any]:
    g1_inputs = contexts["_g1_inputs"]["value"]
    dataset = str(row["dataset"])
    user_id = str(row["user_id"])
    base = contexts[(dataset, user_id)]
    ordered = order_candidates(
        base["ranking_candidates"],
        condition=str(row["candidate_order_condition"]),
        dataset=dataset,
        user_id=user_id,
        positive_item_id=str(base["positive_item_id"]),
    )
    bundle = render_extension_prompt(
        prompt_strategy=str(row["prompt_strategy"]),
        dataset_label=str(g1_inputs.specs[dataset].display_name),
        candidate_items=ordered,
        history_items=list(base["history_items"]),
        history_length=int(base["visible_history_length"]),
    )
    if bundle.prompt_hash != str(row["prompt_hash"]):
        raise RuntimeError(f"G3 prompt hash mismatch for {row['canonical_request_id']}")
    return {"system_message": bundle.system_message, "user_message": bundle.user_message, "positive_item_id": str(base["positive_item_id"])}


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
        raise ValueError(f"positive item id exposed in prompt payload: {row['canonical_request_id']}")
    return payload


def build_request_record(order_row: Mapping[str, Any], row: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": EXECUTION_PHASE_ID,
        "canonical_request_id": str(row["canonical_request_id"]),
        "request_id": str(row["canonical_request_id"]),
        "physical_execution_order": int(order_row["physical_execution_order"]),
        "dataset": str(row["dataset"]),
        "task": str(row["task"]),
        "user_id": str(row["user_id"]),
        "model": str(row["model"]),
        "prompt_strategy": str(row["prompt_strategy"]),
        "candidate_order_condition": str(row["candidate_order_condition"]),
        "history_length": int(row["history_length"]),
        "history_hash": str(row["history_hash"]),
        "candidate_order_hash": str(row["candidate_order_hash"]),
        "candidate_membership_hash": str(row["candidate_membership_hash"]),
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
        append_jsonl_fsync(root / str(config["raw_evidence"]["event_jsonl"]), {"event": "backend_attempt_completed", "phase_id": EXECUTION_PHASE_ID, "canonical_request_id": row["canonical_request_id"], **attempts[-1], "timestamp_utc": utc_now()})
        if result.backend_success:
            break
        if attempt <= retry_count:
            sleep_fn(backoffs[attempt - 1])
    if result is None:
        raise RuntimeError("backend retry loop did not execute")
    return result, attempts


def build_unparsed_raw_record(row: Mapping[str, Any], result: RequestResult, attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    record = {
        "phase_id": EXECUTION_PHASE_ID,
        "canonical_request_id": str(row["canonical_request_id"]),
        "request_id": str(row["canonical_request_id"]),
        "dataset": str(row["dataset"]),
        "task": str(row["task"]),
        "user_id": str(row["user_id"]),
        "model": str(row["model"]),
        "model_digest": str(row["model_digest"]),
        "prompt_strategy": str(row["prompt_strategy"]),
        "candidate_order_condition": str(row["candidate_order_condition"]),
        "history_length": int(row["history_length"]),
        "history_hash": str(row["history_hash"]),
        "candidate_order_hash": str(row["candidate_order_hash"]),
        "candidate_membership_hash": str(row["candidate_membership_hash"]),
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
    ranked_candidate_indices: list[int] = []
    ranked_item_ids: list[str] = []
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


def normalize_terminal_state(parse_status: str, backend_success: bool) -> str:
    if not backend_success:
        return "BACKEND_FAILURE"
    return STRICT_STATUS_TO_TERMINAL.get(str(parse_status), "INVALID_FORMAT")


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for path in [TABLE_DIR, MANIFEST_DIR, PROTOCOL_REPORT_PATH.parent]:
        (root / path).mkdir(parents=True, exist_ok=True)
    for name, frame in outputs["tables"].items():
        if name in OUTPUT_TABLES:
            frame.to_csv(root / OUTPUT_TABLES[name], index=False, lineterminator="\n")
    (root / PROTOCOL_REPORT_PATH).write_text(str(outputs["protocol_report"]), encoding="utf-8", newline="\n")
    (root / EXECUTION_REPORT_PATH).write_text(str(outputs["execution_report"]), encoding="utf-8", newline="\n")

    protocol_manifest = dict(outputs["protocol_manifest"])
    execution_manifest = dict(outputs["execution_manifest"])
    artifact_paths = [CONFIG_PATH, Path("src/extension/g3_candidate_position_execution.py"), Path("tests/test_extension_g3_candidate_position_execution.py")]
    artifact_paths.extend(OUTPUT_TABLES.values())
    artifact_paths.extend([PROTOCOL_REPORT_PATH, EXECUTION_REPORT_PATH])
    artifact_hashes = {
        path.as_posix(): sha256_file(root / path)
        for path in artifact_paths
        if (root / path).exists()
    }
    protocol_manifest["artifact_hashes"] = artifact_hashes
    execution_manifest["artifact_hashes"] = artifact_hashes
    write_json(root / PROTOCOL_MANIFEST_PATH, protocol_manifest)
    write_json(root / EXECUTION_MANIFEST_PATH, execution_manifest)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def append_phase_notes(root: Path, outputs: Mapping[str, Any]) -> None:
    manifest = outputs["protocol_manifest"]
    execution = outputs["execution_manifest"]
    notes = (
        f"\n## {utc_now()} - G3 candidate-position FIRST/RANDOM protocol freeze\n\n"
        f"- Decision: `{manifest['decision_token']}`; runner decision: `{execution['decision_token']}`.\n"
        f"- Scope: BASE + LLMRANK, FIRST vs RANDOM, LLMSRec excluded, `5 x 3 x 2 x 200 x 2 = {manifest['logical_requests']}` logical requests.\n"
        f"- Cohort rule: `{manifest['cohort_rule']}`.\n"
        f"- RANDOM algorithm/seed: `{manifest['random_algorithm']}` / `{manifest['random_seed']}`.\n"
        f"- Reuse: `{manifest['g1_g2_reuse_requests']}`; future new Ollama calls: `{manifest['new_ollama_calls_required']}`.\n"
        "- No Ollama/API calls, G3 execution, G1/G2 evidence edits, downloads, training, tuning, or metric interpretation were run.\n"
    )
    with (root / NOTES_PATH).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(notes)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    values = frame.loc[:, list(columns)].astype(str)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in values.to_dict("records"):
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

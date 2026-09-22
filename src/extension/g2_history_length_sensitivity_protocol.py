"""G2 history-length sensitivity protocol freeze.

This module is offline by construction. It designs and audits the G2 request
universe against frozen G1 evidence, but it never calls Ollama, remote APIs,
training, tuning, downloads, or metric recomputation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.extension import g1_primary_ranking_execution as g1
from src.extension.e0_prompts import render_extension_prompt
from src.extension.e0_protocol import (
    ACTIVE_EXTENSION_DATASETS,
    CANDIDATE_COUNT,
    FUTURE_MODELS,
    HISTORY_LENGTHS,
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
from src.extension.f1bx_execution_stage1 import frame_hash, sort_key
from src.extension.g0_canonical_llm_canary import MODEL_FAMILY, title_for_item
from src.utils.config import find_project_root


PHASE_ID = "g2_history_length_sensitivity_protocol_freeze"
DECISION_FROZEN = "G2_NESTED_HISTORY_PROTOCOL_FROZEN"
DECISION_BLOCKED = "G2_NESTED_HISTORY_PROTOCOL_BLOCKED"
REQUIRED_G1_CHECKPOINT_SHORT = "7d32400"
REQUIRED_G1_COMMIT_MESSAGE = "Complete and retain G1 primary ranking evidence"

CONFIG_PATH = Path("configs/g2_history_length_sensitivity.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_PATH = Path("report/g2_history_length_protocol.md")
NOTES_PATH = Path("report/notes.md")
MANIFEST_PATH = MANIFEST_DIR / "g2_history_length_protocol_freeze.json"

G2_RAW_DIR = Path("results/llm/extension_canonical/g2_history_length_sensitivity")
G2_RESPONSE_DIR = G2_RAW_DIR / "responses"

OUTPUT_TABLES = {
    "eligibility": TABLE_DIR / "g2_history_length_eligibility.csv",
    "cohort": TABLE_DIR / "g2_history_length_cohort.csv",
    "applicability": TABLE_DIR / "g2_history_length_applicability.csv",
    "g1_reuse_audit": TABLE_DIR / "g2_history_length_g1_reuse_audit.csv",
    "prior_evidence_audit": TABLE_DIR / "g2_history_length_prior_evidence_audit.csv",
    "request_universe": TABLE_DIR / "g2_history_length_request_universe.csv",
    "request_accounting": TABLE_DIR / "g2_history_length_request_accounting.csv",
    "research_questions": TABLE_DIR / "g2_history_length_research_questions.csv",
    "statistical_plan": TABLE_DIR / "g2_history_length_statistical_plan.csv",
    "representativeness": TABLE_DIR / "g2_history_length_representativeness.csv",
}

DATASET_ORDER = tuple(ACTIVE_EXTENSION_DATASETS)
MODEL_ORDER = tuple(FUTURE_MODELS)
PROMPT_ORDER = tuple(PROMPT_STRATEGIES)
HISTORY_ORDER = tuple(HISTORY_LENGTHS)
PRIMARY_MIN_HISTORY = 10
LONG_HISTORY_MIN_HISTORY = 20
LOW_SHARE_WARNING_THRESHOLD = 0.10

TASK = "ranking"
PRIMARY_VIEW = "PRIMARY_B1_GE10"
LONG_VIEW = "LONG_HISTORY_B2_GE20"
SECONDARY_VIEW = "SECONDARY_FULL_CURVE_GE20"
TRUE_ZERO_VIEW = "FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC"
PRIMARY_COHORT_TYPE = "NESTED_WARM_GE10_HISTORY_COHORT"
TRUE_ZERO_COHORT_TYPE = "FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC"
CONTROLLED_PREFIX = "CONTROLLED_H"
TRUE_ZERO_LABEL = "TRUE_ZERO_HISTORY"
NOT_APPLICABLE = "NOT_APPLICABLE"
G1_REUSE_ACTION = "REUSE_G1_EXACT_TERMINAL_EVIDENCE"
NEW_CALL_ACTION = "NEW_G2_OLLAMA_CALL_REQUIRED"
LLMRANK_H0_STATUS = "FROZEN_LOCAL_EQUIVALENT"
PRIOR_REUSE_STATUS = "COMPARATIVE_PRIOR_EVIDENCE"

EXPECTED_B1_USERS_BY_DATASET = {
    "beauty": 181,
    "sports": 160,
    "ml1m": 1000,
    "lastfm": 981,
    "foodcom": 61,
}
EXPECTED_B2_USERS_BY_DATASET = {
    "beauty": 38,
    "sports": 35,
    "ml1m": 993,
    "lastfm": 932,
    "foodcom": 35,
}
EXPECTED_TOTALS_BY_DATASET = {
    "beauty": {"total": 5658, "reuse": 1086, "new": 4572},
    "sports": {"total": 5010, "reuse": 960, "new": 4050},
    "ml1m": {"total": 35958, "reuse": 6000, "new": 29958},
    "lastfm": {"total": 35022, "reuse": 5886, "new": 29136},
    "foodcom": {"total": 6402, "reuse": 4728, "new": 1674},
}
EXPECTED_TRUE_ZERO_FOODCOM_USERS = 727
EXPECTED_TOTAL_REQUESTS = 88050
EXPECTED_G1_REUSE_REQUESTS = 18660
EXPECTED_NEW_CALLS = 69390
EXPECTED_NOT_APPLICABLE_TRUE_ZERO = EXPECTED_TRUE_ZERO_FOODCOM_USERS * (len(HISTORY_ORDER) - 1) * len(PROMPT_ORDER) * len(MODEL_ORDER)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the offline G2 history-length sensitivity protocol.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--no-write", action="store_true", help="Build in memory without writing G2 artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = build_protocol_freeze(root, Path(args.config))
    if not args.no_write:
        write_outputs(root, outputs)
        append_phase_notes(root, outputs)

    manifest = outputs["manifest"]
    print(f"decision={manifest['decision_token']}")
    print(f"b1_primary_paired_users={manifest['b1_primary_paired_users']}")
    print(f"b2_long_history_paired_users={manifest['b2_long_history_paired_users']}")
    print(f"planned_applicable_requests={manifest['planned_applicable_requests']}")
    print(f"g1_exact_reuse_requests={manifest['g1_exact_reuse_requests']}")
    print(f"new_future_ollama_calls_required={manifest['new_future_ollama_calls_required']}")
    print(f"not_applicable_true_zero_h_positive_user_conditions={manifest['not_applicable_true_zero_h_positive_user_conditions']}")
    print("llm_inference_calls_executed=0")
    return 0 if manifest["decision_token"] == DECISION_FROZEN else 1


def build_protocol_freeze(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, Path(config_path))
    validate_config(config)

    g1_config = g1.load_config(project_root, Path(config["inputs"]["g1_config"]))
    g1_inputs = g1.load_inputs(project_root, g1_config)
    model_preflight = g1.expected_model_preflight_from_g0(g1_inputs)
    g1_request_universe = read_csv(project_root / str(config["inputs"]["g1_request_universe"]))
    g1_execution_state = read_csv(project_root / str(config["inputs"]["g1_execution_state"]))
    g1_strict_parsing = read_csv(project_root / str(config["inputs"]["g1_strict_parsing"]))
    f0_inference_contract = read_json(project_root / str(config["inputs"]["f0_inference_contract"]))
    g1_execution_manifest = read_json(project_root / str(config["inputs"]["g1_execution_manifest"]))
    g1_retention_manifest = read_json(project_root / str(config["inputs"]["g1_retention_manifest"]))

    user_contexts = build_user_contexts(g1_inputs)
    eligibility = build_eligibility(user_contexts)
    cohort = build_cohort(g1_inputs, user_contexts, g1_request_universe)
    parser_hash_value = parser_contract_hash()
    generation_contract = build_generation_contract(config, f0_inference_contract)
    g1_terminal_by_key = g1_terminal_reuse_index(g1_request_universe, g1_execution_state)

    request_universe = build_request_universe(
        g1_inputs=g1_inputs,
        user_contexts=user_contexts,
        cohort=cohort,
        model_preflight=model_preflight,
        generation_contract=generation_contract,
        parser_hash=parser_hash_value,
        g1_terminal_by_key=g1_terminal_by_key,
    )
    applicability = build_applicability(eligibility, request_universe)
    g1_reuse_audit = build_g1_reuse_audit(g1_request_universe, g1_execution_state, request_universe)
    prior_evidence_audit = build_prior_evidence_audit(project_root, config)
    request_accounting = build_request_accounting(applicability, request_universe)
    research_questions = build_research_questions()
    statistical_plan = build_statistical_plan(config)
    representativeness = build_representativeness(eligibility)

    validation_failures = validate_outputs(
        config=config,
        eligibility=eligibility,
        cohort=cohort,
        request_universe=request_universe,
        applicability=applicability,
        g1_reuse_audit=g1_reuse_audit,
        prior_evidence_audit=prior_evidence_audit,
        request_accounting=request_accounting,
        g1_execution_manifest=g1_execution_manifest,
        g1_retention_manifest=g1_retention_manifest,
    )

    tables = {
        "eligibility": eligibility,
        "cohort": cohort,
        "applicability": applicability,
        "g1_reuse_audit": g1_reuse_audit,
        "prior_evidence_audit": prior_evidence_audit,
        "request_universe": request_universe,
        "request_accounting": request_accounting,
        "research_questions": research_questions,
        "statistical_plan": statistical_plan,
        "representativeness": representativeness,
    }
    manifest = build_manifest(
        root=project_root,
        config=config,
        tables=tables,
        generation_contract=generation_contract,
        parser_hash=parser_hash_value,
        validation_failures=validation_failures,
        g1_execution_manifest=g1_execution_manifest,
        g1_retention_manifest=g1_retention_manifest,
    )
    report = render_report(tables, manifest)
    return {"config": config, "tables": tables, "manifest": manifest, "report": report}


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def validate_config(config: Mapping[str, Any]) -> None:
    if str(config["phase"]["id"]) != PHASE_ID:
        raise ValueError("unexpected G2 phase id")
    if str(config["phase"]["required_g1_checkpoint_short"]) != REQUIRED_G1_CHECKPOINT_SHORT:
        raise ValueError("G2 must freeze against the retained G1 checkpoint")
    if str(config["phase"]["required_g1_commit_message"]) != REQUIRED_G1_COMMIT_MESSAGE:
        raise ValueError("G2 must freeze against the retained G1 commit message")
    if tuple(config["models"]) != MODEL_ORDER:
        raise ValueError("G2 model list must exactly match G1")
    if tuple(config["datasets"]) != DATASET_ORDER:
        raise ValueError("G2 dataset list must exactly match G1 active datasets")
    if tuple(config["prompt_strategies"]) != PROMPT_ORDER:
        raise ValueError("G2 prompt list must exactly match G1")
    if tuple(int(value) for value in config["history_lengths"]) != HISTORY_ORDER:
        raise ValueError("G2 history grid must be 0,1,2,5,10,20")
    nested = config.get("nested_design", {})
    if str(nested.get("selected_design", "")) != "DESIGN_B_NESTED_PAIRED_ANALYSES":
        raise ValueError("G2 must freeze nested Design B")
    if tuple(int(value) for value in nested["b1_primary"]["history_lengths"]) != (0, 1, 2, 5, 10):
        raise ValueError("B1 history grid must be 0,1,2,5,10")
    if tuple(int(value) for value in nested["b2_long_history"]["history_lengths"]) != (10, 20):
        raise ValueError("B2 history grid must be 10,20")
    if not bool(nested.get("deduplicate_overlapping_user_h_requests", False)):
        raise ValueError("Design B requires canonical deduplication of overlapping user/H requests")
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
    if any(bool(config["execution"][key]) for key in forbidden):
        raise ValueError("G2A is a no-inference/no-network/no-training protocol freeze")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def build_user_contexts(inputs: g1.G1Inputs) -> dict[tuple[str, str], dict[str, Any]]:
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        candidate_frame = inputs.ranking_candidates[dataset]
        users = sorted(candidate_frame["user_id"].astype(str).unique(), key=sort_key)
        for user_id in users:
            base_context = g1.build_ranking_user_context(inputs, dataset, user_id)
            full_history_ids = [str(item_id) for item_id in inputs.history_by_dataset[dataset].get(str(user_id), [])]
            contexts[(dataset, str(user_id))] = {
                **base_context,
                "full_history_ids": full_history_ids,
            }
    return contexts


def visible_history_ids(full_history_ids: Sequence[str], history_length: int) -> list[str]:
    if int(history_length) == 0:
        return []
    return [str(item_id) for item_id in list(full_history_ids)[-int(history_length) :]]


def history_items_for(inputs: g1.G1Inputs, dataset: str, history_ids: Sequence[str]) -> list[dict[str, str]]:
    title_map = inputs.title_maps[dataset]
    items = [{"item_id": str(item_id), "title": title_for_item(title_map, str(item_id))} for item_id in history_ids]
    missing = [item["item_id"] for item in items if not str(item["title"]).strip()]
    if missing:
        raise ValueError(f"G2 history item titles missing for {dataset}: {missing[:5]}")
    return items


def candidate_ids_from_context(context: Mapping[str, Any]) -> list[str]:
    return [str(item["item_id"]) for item in context["ranking_candidates"]]


def build_eligibility(user_contexts: Mapping[tuple[str, str], Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        contexts = [context for (context_dataset, _user_id), context in user_contexts.items() if context_dataset == dataset]
        lengths = [int(context["full_history_length"]) for context in contexts]
        total = len(contexts)
        b1_count = sum(1 for value in lengths if value >= PRIMARY_MIN_HISTORY)
        b2_count = sum(1 for value in lengths if value >= LONG_HISTORY_MIN_HISTORY)
        row = {
            "dataset": dataset,
            "total_g1_users": total,
            "users_ge_1": sum(1 for value in lengths if value >= 1),
            "users_ge_2": sum(1 for value in lengths if value >= 2),
            "users_ge_5": sum(1 for value in lengths if value >= 5),
            "users_ge_10": sum(1 for value in lengths if value >= 10),
            "users_ge_20": b2_count,
            "primary_min_history": PRIMARY_MIN_HISTORY,
            "long_history_min_history": LONG_HISTORY_MIN_HISTORY,
            "b1_primary_users": b1_count,
            "b2_long_history_users": b2_count,
            "b1_primary_share_of_g1_users": format_ratio(b1_count, total),
            "b2_long_history_share_of_g1_users": format_ratio(b2_count, total),
            "primary_cohort_status": (
                "PRIMARY_B1_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING"
                if total and b1_count / total < LOW_SHARE_WARNING_THRESHOLD
                else "PRIMARY_B1_FROZEN"
            ),
            "long_history_cohort_status": (
                "LONG_HISTORY_B2_FROZEN_WITH_LOW_SHARE_REPRESENTATIVENESS_WARNING"
                if total and b2_count / total < LOW_SHARE_WARNING_THRESHOLD
                else "LONG_HISTORY_B2_FROZEN"
            ),
            "retained_for_primary_paired_cohort": b1_count,
            "retained_for_long_history_extension": b2_count,
            "history_count_source": "G1 frozen request universe plus G1 frozen train-history loader",
        }
        if dataset == "foodcom":
            true_zero = sum(1 for value in lengths if value == 0)
            warm = sum(1 for value in lengths if value > 0)
            row.update(
                {
                    "foodcom_all_users": total,
                    "foodcom_warm_users": warm,
                    "foodcom_true_zero_history_users": true_zero,
                    "foodcom_warm_ge10_users": b1_count,
                    "foodcom_warm_ge20_users": b2_count,
                }
            )
        else:
            row.update(
                {
                    "foodcom_all_users": "",
                    "foodcom_warm_users": "",
                    "foodcom_true_zero_history_users": "",
                    "foodcom_warm_ge10_users": "",
                    "foodcom_warm_ge20_users": "",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_cohort(
    inputs: g1.G1Inputs,
    user_contexts: Mapping[tuple[str, str], Mapping[str, Any]],
    g1_request_universe: pd.DataFrame,
) -> pd.DataFrame:
    g1_user_identity = (
        g1_request_universe.drop_duplicates(["dataset", "user_id"])
        .set_index(["dataset", "user_id"])
        .to_dict("index")
    )
    rows: list[dict[str, Any]] = []
    cohort_order = 1
    for dataset in DATASET_ORDER:
        user_ids = sorted([user_id for (context_dataset, user_id) in user_contexts if context_dataset == dataset], key=sort_key)
        for user_id in user_ids:
            context = user_contexts[(dataset, user_id)]
            full_length = int(context["full_history_length"])
            include_b1 = full_length >= PRIMARY_MIN_HISTORY
            include_b2 = full_length >= LONG_HISTORY_MIN_HISTORY
            include_true_zero = dataset == "foodcom" and full_length == 0
            if not include_b1 and not include_true_zero:
                continue
            candidate_ids = candidate_ids_from_context(context)
            positive = str(context["positive_item_id"])
            full_history_ids = [str(item_id) for item_id in context["full_history_ids"]]
            memberships: list[str] = []
            if include_b1:
                memberships.append(PRIMARY_VIEW)
            if include_b2:
                memberships.extend([LONG_VIEW, SECONDARY_VIEW])
            if include_true_zero:
                memberships.append(TRUE_ZERO_VIEW)
            base_row = {
                "cohort_id": "G2HL-COHORT-" + sha256_json({"dataset": dataset, "user_id": user_id, "cohort_order": cohort_order}).upper()[:20],
                "cohort_order": cohort_order,
                "dataset": dataset,
                "dataset_label": str(inputs.specs[dataset].display_name),
                "user_id": user_id,
                "cohort_type": PRIMARY_COHORT_TYPE if include_b1 else TRUE_ZERO_COHORT_TYPE,
                "population_label": "WARM_GE10" if include_b1 else TRUE_ZERO_LABEL,
                "reporting_view_memberships": stable_json(memberships),
                "b1_primary_ge10_eligible": bool_text(include_b1),
                "b2_long_history_ge20_eligible": bool_text(include_b2),
                "secondary_full_curve_ge20_eligible": bool_text(include_b2),
                "full_history_length": full_length,
                "primary_min_history": PRIMARY_MIN_HISTORY if include_b1 else "",
                "long_history_min_history": LONG_HISTORY_MIN_HISTORY if include_b2 else "",
                "candidate_set_id": g1.candidate_set_id_for(dataset, user_id, str(context["candidate_hash"])),
                "candidate_hash": str(context["candidate_hash"]),
                "candidate_items": stable_json(candidate_ids),
                "candidate_titles_hash": sha256_json([str(item["title"]) for item in context["ranking_candidates"]]),
                "positive_item_id": positive,
                "target_excluded_from_full_train_history": bool_text(positive not in set(full_history_ids)),
                "candidate_unchanged_across_history_lengths": "true",
                "target_unchanged_across_history_lengths": "true",
                "history_nesting_status": "NESTED_SUFFIX_CHAIN" if include_b1 else "H0_ONLY_TRUE_ZERO_DIAGNOSTIC",
                "g1_visible_history_hash": str(g1_user_identity[(dataset, user_id)]["history_hash"]),
                "g1_visible_history_length": int(g1_user_identity[(dataset, user_id)]["history_length"]),
                "g1_candidate_hash_match": bool_text(str(g1_user_identity[(dataset, user_id)]["candidate_hash"]) == str(context["candidate_hash"])),
                "cohort_selection_status": "RETAINED_NESTED_B1_PRIMARY" if include_b1 else "RETAINED_TRUE_ZERO_DIAGNOSTIC",
            }
            for history_length in HISTORY_ORDER:
                applicable = (include_b1 and history_length in (0, 1, 2, 5, 10)) or (include_b2 and history_length == 20) or (
                    include_true_zero and history_length == 0
                )
                ids = visible_history_ids(full_history_ids, history_length) if applicable else []
                items = history_items_for(inputs, dataset, ids)
                base_row[f"history_h{history_length}_applicability"] = "APPLICABLE" if applicable else NOT_APPLICABLE
                base_row[f"history_h{history_length}_item_count"] = len(ids) if applicable else ""
                base_row[f"history_h{history_length}_item_ids"] = stable_json(ids) if applicable else ""
                base_row[f"history_h{history_length}_hash"] = item_list_hash(ids) if applicable else ""
                base_row[f"history_h{history_length}_titles_hash"] = sha256_json([item["title"] for item in items]) if applicable else ""
                base_row[f"history_h{history_length}_target_excluded"] = bool_text(positive not in set(ids)) if applicable else NOT_APPLICABLE
            if include_b1:
                base_row["g1_h10_history_hash_match"] = bool_text(
                    str(base_row["history_h10_hash"]) == str(g1_user_identity[(dataset, user_id)]["history_hash"])
                )
            else:
                base_row["g1_h10_history_hash_match"] = NOT_APPLICABLE
            rows.append(base_row)
            cohort_order += 1
    return pd.DataFrame(rows)


def build_generation_contract(config: Mapping[str, Any], f0_inference_contract: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(f0_inference_contract.get("generation_parameters", {}))
    return {
        "generation_config_hash": str(f0_inference_contract["generation_config_hash"]),
        "generation_config_id": str(f0_inference_contract.get("generation_config_id", "")),
        "temperature": parameters.get("temperature", config["execution"]["temperature"]),
        "num_predict": parameters.get("num_predict", config["execution"]["max_output_tokens"]["ranking"]),
        "timeout_seconds": parameters.get("timeout_seconds", config["execution"]["timeout_seconds"]),
        "stream": parameters.get("stream", config["execution"]["stream"]),
        "response_format": config["execution"]["response_format"],
        "backend_retry_count": int(config["execution"]["backend_retry_count"]),
        "backend_retry_backoff_seconds": list(config["execution"]["backend_retry_backoff_seconds"]),
        "scientific_retry_count": int(config["execution"]["scientific_retry_count"]),
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


def reuse_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("dataset", "")),
        str(row.get("user_id", "")),
        str(row.get("model", row.get("model_name", ""))),
        str(row.get("prompt_strategy", row.get("prompt_variant", ""))),
        str(row.get("history_length", "")),
        str(row.get("history_hash", "")),
        str(row.get("candidate_hash", "")),
        str(row.get("candidate_items", "")),
        str(row.get("positive_item_id", "")),
        str(row.get("prompt_hash", "")),
        str(row.get("model_digest", "")),
        str(row.get("generation_config_hash", row.get("generation_contract_hash", ""))),
        str(row.get("parser_contract", "")),
        str(row.get("requested_output_length", "")),
        str(row.get("parser_expected_length", "")),
    )


def g1_terminal_reuse_index(g1_request_universe: pd.DataFrame, g1_execution_state: pd.DataFrame) -> dict[tuple[str, ...], dict[str, Any]]:
    terminal_by_id = g1_execution_state.set_index("canonical_request_id").to_dict("index")
    index: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in g1_request_universe.to_dict("records"):
        request_id = str(row["canonical_request_id"])
        terminal = terminal_by_id.get(request_id, {})
        if str(terminal.get("terminal", "")).lower() != "true":
            continue
        combined = {**row, **terminal}
        index[reuse_key(row)] = combined
    return index


def request_history_lengths_for_cohort(cohort_row: Mapping[str, Any]) -> tuple[int, ...]:
    if str(cohort_row["cohort_type"]) == TRUE_ZERO_COHORT_TYPE:
        return (0,)
    lengths = [0, 1, 2, 5, 10]
    if str(cohort_row.get("b2_long_history_ge20_eligible", "")).lower() == "true":
        lengths.append(20)
    return tuple(lengths)


def analysis_memberships_for_request(cohort_row: Mapping[str, Any], history_length: int) -> list[str]:
    if str(cohort_row["cohort_type"]) == TRUE_ZERO_COHORT_TYPE:
        return [TRUE_ZERO_VIEW]
    memberships: list[str] = []
    b2 = str(cohort_row.get("b2_long_history_ge20_eligible", "")).lower() == "true"
    if history_length in (0, 1, 2, 5, 10):
        memberships.append(PRIMARY_VIEW)
    if b2 and history_length in (10, 20):
        memberships.append(LONG_VIEW)
    if b2 and history_length in HISTORY_ORDER:
        memberships.append(SECONDARY_VIEW)
    return memberships


def build_request_universe(
    *,
    g1_inputs: g1.G1Inputs,
    user_contexts: Mapping[tuple[str, str], Mapping[str, Any]],
    cohort: pd.DataFrame,
    model_preflight: pd.DataFrame,
    generation_contract: Mapping[str, Any],
    parser_hash: str,
    g1_terminal_by_key: Mapping[tuple[str, ...], Mapping[str, Any]],
) -> pd.DataFrame:
    digest_by_model = model_preflight.set_index("model")["digest"].astype(str).to_dict()
    rows: list[dict[str, Any]] = []
    order = 1
    cohort_records = cohort.sort_values("cohort_order", kind="mergesort").to_dict("records")
    for cohort_row in cohort_records:
        dataset = str(cohort_row["dataset"])
        user_id = str(cohort_row["user_id"])
        context = user_contexts[(dataset, user_id)]
        full_history_ids = [str(item_id) for item_id in context["full_history_ids"]]
        candidate_items = list(context["ranking_candidates"])
        candidate_ids = candidate_ids_from_context(context)
        positive = str(context["positive_item_id"])
        history_lengths = request_history_lengths_for_cohort(cohort_row)
        for history_length in history_lengths:
            history_ids = visible_history_ids(full_history_ids, history_length)
            history_items = history_items_for(g1_inputs, dataset, history_ids)
            history_hash = item_list_hash(history_ids)
            history_condition_label = f"{CONTROLLED_PREFIX}{history_length}" if str(cohort_row["cohort_type"]) == PRIMARY_COHORT_TYPE else TRUE_ZERO_LABEL
            memberships = analysis_memberships_for_request(cohort_row, history_length)
            primary_reporting_view = PRIMARY_VIEW if PRIMARY_VIEW in memberships else (LONG_VIEW if LONG_VIEW in memberships else TRUE_ZERO_VIEW)
            for prompt_strategy in PROMPT_ORDER:
                bundle = render_extension_prompt(
                    prompt_strategy=prompt_strategy,
                    dataset_label=str(g1_inputs.specs[dataset].display_name),
                    candidate_items=candidate_items,
                    history_items=history_items,
                    history_length=history_length,
                )
                llmrank_h0_status = LLMRANK_H0_STATUS if prompt_strategy == PROMPT_LLMRANK and history_length == 0 else NOT_APPLICABLE
                for model in MODEL_ORDER:
                    model_digest = str(digest_by_model[model])
                    identity_fields = {
                        "dataset": dataset,
                        "user_id": user_id,
                        "model": model,
                        "prompt_strategy": prompt_strategy,
                        "history_length": history_length,
                        "history_hash": history_hash,
                        "candidate_hash": str(context["candidate_hash"]),
                        "candidate_items": stable_json(candidate_ids),
                        "positive_item_id": positive,
                        "prompt_hash": bundle.prompt_hash,
                        "model_digest": model_digest,
                        "generation_config_hash": str(generation_contract["generation_config_hash"]),
                        "parser_contract_hash": parser_hash,
                    }
                    request_identity_hash = sha256_json(identity_fields)
                    request_id = "G2HL-" + request_identity_hash.upper()[:32]
                    row = {
                        "canonical_request_id": request_id,
                        "request_id": request_id,
                        "request_order": order,
                        "dataset": dataset,
                        "dataset_label": str(g1_inputs.specs[dataset].display_name),
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
                        "cohort_type": str(cohort_row["cohort_type"]),
                        "population_label": str(cohort_row["population_label"]),
                        "reporting_view_memberships": stable_json(memberships),
                        "primary_reporting_view": primary_reporting_view,
                        "b1_primary_ge10_member": bool_text(PRIMARY_VIEW in memberships),
                        "b2_long_history_ge20_member": bool_text(LONG_VIEW in memberships),
                        "secondary_full_curve_ge20_member": bool_text(SECONDARY_VIEW in memberships),
                        "history_condition_label": history_condition_label,
                        "history_length": history_length,
                        "history_item_count": len(history_ids),
                        "full_history_length": int(cohort_row["full_history_length"]),
                        "history_hash": history_hash,
                        "history_items": stable_json(history_ids),
                        "history_source": "chronological_train_history_suffix",
                        "history_contract": "most_recent_H_train_items_target_excluded_H0_empty",
                        "candidate_set_id": str(cohort_row["candidate_set_id"]),
                        "candidate_source": str(context["candidate_source"]),
                        "candidate_source_protocol": "F0_FROZEN_RANKING_CANDIDATES_G1_LINEAGE",
                        "candidate_hash": str(context["candidate_hash"]),
                        "candidate_count": CANDIDATE_COUNT,
                        "candidate_items": stable_json(candidate_ids),
                        "candidate_titles_hash": str(cohort_row["candidate_titles_hash"]),
                        "positive_item_id": positive,
                        "target_excluded_from_history": bool_text(positive not in set(history_ids)),
                        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                        "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
                        "metric_k": METRIC_K,
                        "llmrank_h0_status": llmrank_h0_status,
                        "llmrank_h0_recency_policy": (
                            "no_recency_signal_sentence_no_empty_last_item"
                            if prompt_strategy == PROMPT_LLMRANK and history_length == 0
                            else NOT_APPLICABLE
                        ),
                        "canonical_logical_order": "dataset_user_history_prompt_model",
                        "physical_execution_order": "model_dataset_prompt_history_user",
                        "raw_response_protocol": g1.G1_RAW_PROTOCOL,
                        "g2_future_raw_response_path": (G2_RESPONSE_DIR / f"{request_id}.json").as_posix(),
                        "request_identity_hash": request_identity_hash,
                        "request_identity_fields": stable_json(identity_fields),
                        "retention_policy": "EXTERNALIZED_RAW_EVIDENCE_FOR_FUTURE_TERMINAL_OUTPUT_AND_TRACKED_FOREGROUND_ROW",
                    }
                    key = reuse_key(row)
                    matched_g1 = dict(g1_terminal_by_key.get(key, {}))
                    if matched_g1:
                        row.update(
                            {
                                "planned_action": G1_REUSE_ACTION,
                                "matched_g1_request_id": str(matched_g1["canonical_request_id"]),
                                "matched_g1_raw_response_path": str(matched_g1.get("raw_response_path", "")),
                                "matched_g1_raw_response_hash": str(matched_g1.get("raw_response_hash", "")),
                                "matched_g1_evidence_origin": str(matched_g1.get("evidence_origin", "")),
                            }
                        )
                    else:
                        row.update(
                            {
                                "planned_action": NEW_CALL_ACTION,
                                "matched_g1_request_id": "",
                                "matched_g1_raw_response_path": "",
                                "matched_g1_raw_response_hash": "",
                                "matched_g1_evidence_origin": "",
                            }
                        )
                    rows.append(row)
                    order += 1
    universe = pd.DataFrame(rows)
    if universe["canonical_request_id"].nunique() != len(universe):
        raise ValueError("G2 canonical request ids are not unique")
    return universe


def build_applicability(eligibility: pd.DataFrame, request_universe: pd.DataFrame) -> pd.DataFrame:
    eligibility_by_dataset = eligibility.set_index("dataset").to_dict("index")
    view_counts = reporting_view_count_map(request_universe)
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        b1_users = int(eligibility_by_dataset[dataset]["b1_primary_users"])
        b2_users = int(eligibility_by_dataset[dataset]["b2_long_history_users"])
        for history_length in (0, 1, 2, 5, 10):
            for prompt_strategy in PROMPT_ORDER:
                for model in MODEL_ORDER:
                    planned = view_counts.get((PRIMARY_VIEW, dataset, history_length, prompt_strategy, model), 0)
                    rows.append(
                        {
                            "dataset": dataset,
                            "reporting_view": PRIMARY_VIEW,
                            "cohort_type": PRIMARY_COHORT_TYPE,
                            "population_label": "WARM_GE10",
                            "history_length": history_length,
                            "history_condition_label": f"{CONTROLLED_PREFIX}{history_length}",
                            "prompt_strategy": prompt_strategy,
                            "model": model,
                            "applicability_status": "APPLICABLE_PRIMARY_B1_GE10",
                            "applicable_user_conditions": planned,
                            "not_applicable_user_conditions": 0,
                            "eligible_user_count": b1_users,
                            "creates_new_logical_request": "true",
                            "llmrank_h0_status": LLMRANK_H0_STATUS if prompt_strategy == PROMPT_LLMRANK and history_length == 0 else NOT_APPLICABLE,
                            "reason": "Primary B1 paired cohort has at least 10 chronological train-history interactions.",
                        }
                    )
        for history_length in (10, 20):
            for prompt_strategy in PROMPT_ORDER:
                for model in MODEL_ORDER:
                    planned = view_counts.get((LONG_VIEW, dataset, history_length, prompt_strategy, model), 0)
                    rows.append(
                        {
                            "dataset": dataset,
                            "reporting_view": LONG_VIEW,
                            "cohort_type": PRIMARY_COHORT_TYPE,
                            "population_label": "WARM_GE20",
                            "history_length": history_length,
                            "history_condition_label": f"{CONTROLLED_PREFIX}{history_length}",
                            "prompt_strategy": prompt_strategy,
                            "model": model,
                            "applicability_status": "APPLICABLE_LONG_HISTORY_B2_GE20",
                            "applicable_user_conditions": planned,
                            "not_applicable_user_conditions": 0,
                            "eligible_user_count": b2_users,
                            "creates_new_logical_request": "false" if history_length == 10 else "true",
                            "llmrank_h0_status": NOT_APPLICABLE,
                            "reason": "Long-history B2 paired cohort has at least 20 chronological train-history interactions; H10 overlaps B1 and is not duplicated.",
                        }
                    )
        for history_length in HISTORY_ORDER:
            for prompt_strategy in PROMPT_ORDER:
                for model in MODEL_ORDER:
                    planned = view_counts.get((SECONDARY_VIEW, dataset, history_length, prompt_strategy, model), 0)
                    rows.append(
                        {
                            "dataset": dataset,
                            "reporting_view": SECONDARY_VIEW,
                            "cohort_type": PRIMARY_COHORT_TYPE,
                            "population_label": "WARM_GE20",
                            "history_length": history_length,
                            "history_condition_label": f"{CONTROLLED_PREFIX}{history_length}",
                            "prompt_strategy": prompt_strategy,
                            "model": model,
                            "applicability_status": "APPLICABLE_SECONDARY_FULL_CURVE_GE20",
                            "applicable_user_conditions": planned,
                            "not_applicable_user_conditions": 0,
                            "eligible_user_count": b2_users,
                            "creates_new_logical_request": "false",
                            "llmrank_h0_status": LLMRANK_H0_STATUS if prompt_strategy == PROMPT_LLMRANK and history_length == 0 else NOT_APPLICABLE,
                            "reason": "Secondary GE20 full curve reuses canonical requests already belonging to B1 and B2; it creates zero duplicate inference calls.",
                        }
                    )
        if dataset == "foodcom":
            true_zero_users = int(eligibility_by_dataset[dataset]["foodcom_true_zero_history_users"])
            for history_length in HISTORY_ORDER:
                for prompt_strategy in PROMPT_ORDER:
                    for model in MODEL_ORDER:
                        planned = view_counts.get((TRUE_ZERO_VIEW, dataset, history_length, prompt_strategy, model), 0)
                        applicable = history_length == 0
                        rows.append(
                            {
                                "dataset": dataset,
                                "reporting_view": TRUE_ZERO_VIEW,
                                "cohort_type": TRUE_ZERO_COHORT_TYPE,
                                "population_label": TRUE_ZERO_LABEL,
                                "history_length": history_length,
                                "history_condition_label": TRUE_ZERO_LABEL if applicable else NOT_APPLICABLE,
                                "prompt_strategy": prompt_strategy,
                                "model": model,
                                "applicability_status": "APPLICABLE_TRUE_ZERO_HISTORY" if applicable else NOT_APPLICABLE,
                                "applicable_user_conditions": planned if applicable else 0,
                                "not_applicable_user_conditions": 0 if applicable else true_zero_users,
                                "eligible_user_count": true_zero_users,
                                "creates_new_logical_request": "true" if applicable else "false",
                                "llmrank_h0_status": LLMRANK_H0_STATUS if prompt_strategy == PROMPT_LLMRANK and history_length == 0 else NOT_APPLICABLE,
                                "reason": (
                                    "Food.com true-zero users have real observed H=0 only."
                                    if applicable
                                    else "Food.com true-zero users have no train-history interactions, so H>0 is not applicable."
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def reporting_view_count_map(request_universe: pd.DataFrame) -> dict[tuple[str, str, int, str, str], int]:
    counts: dict[tuple[str, str, int, str, str], int] = {}
    for row in request_universe.to_dict("records"):
        for reporting_view in json.loads(str(row["reporting_view_memberships"])):
            key = (
                str(reporting_view),
                str(row["dataset"]),
                int(row["history_length"]),
                str(row["prompt_strategy"]),
                str(row["model"]),
            )
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_g1_reuse_audit(
    g1_request_universe: pd.DataFrame,
    g1_execution_state: pd.DataFrame,
    request_universe: pd.DataFrame,
) -> pd.DataFrame:
    g2_by_key = {reuse_key(row): row for row in request_universe.to_dict("records")}
    execution_by_id = g1_execution_state.set_index("canonical_request_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for g1_row in g1_request_universe.sort_values("request_order", key=lambda s: s.astype(int), kind="mergesort").to_dict("records"):
        request_id = str(g1_row["canonical_request_id"])
        terminal = execution_by_id.get(request_id, {})
        matched = g2_by_key.get(reuse_key(g1_row), {})
        terminal_available = str(terminal.get("terminal", "")).lower() == "true"
        reusable = bool(matched) and terminal_available
        full_history_length = int(g1_row["full_history_length"])
        rejection_reason = "EXACT_G2_IDENTITY_MATCH"
        if not reusable:
            if not terminal_available:
                rejection_reason = "G1_TERMINAL_EVIDENCE_MISSING"
            elif str(g1_row["dataset"]) == "foodcom" and full_history_length == 0:
                rejection_reason = "UNEXPECTED_TRUE_ZERO_IDENTITY_MISMATCH"
            elif full_history_length < PRIMARY_MIN_HISTORY:
                rejection_reason = "G1_USER_NOT_IN_PRIMARY_B1_GE10_COHORT"
            else:
                rejection_reason = "G1_HISTORY_LENGTH_OR_PROMPT_IDENTITY_NOT_SELECTED_FOR_NESTED_REUSE"
        field_matches = {
            "dataset": reusable,
            "user_id": reusable,
            "model": reusable,
            "prompt_strategy": reusable,
            "history_length": reusable,
            "history_hash": reusable,
            "candidate_hash": reusable,
            "candidate_items": reusable,
            "positive_item_id": reusable,
            "prompt_hash": reusable,
            "model_digest": reusable,
            "generation_config_hash": reusable,
            "parser_contract": reusable,
            "terminal": terminal_available,
        }
        rows.append(
            {
                "g1_canonical_request_id": request_id,
                "dataset": str(g1_row["dataset"]),
                "user_id": str(g1_row["user_id"]),
                "model": str(g1_row["model"]),
                "prompt_strategy": str(g1_row["prompt_strategy"]),
                "g1_history_length": int(g1_row["history_length"]),
                "g1_full_history_length": full_history_length,
                "g1_history_hash": str(g1_row["history_hash"]),
                "g1_candidate_hash": str(g1_row["candidate_hash"]),
                "g1_positive_item_id": str(g1_row["positive_item_id"]),
                "g1_model_digest": str(g1_row["model_digest"]),
                "g1_generation_config_hash": str(g1_row["generation_config_hash"]),
                "g1_parser_contract": str(g1_row["parser_contract"]),
                "g1_prompt_hash": str(g1_row["prompt_hash"]),
                "g1_terminal": bool_text(terminal_available),
                "g1_evidence_origin": str(terminal.get("evidence_origin", "")),
                "g1_raw_response_hash": str(terminal.get("raw_response_hash", "")),
                "matched_g2_request_id": str(matched.get("canonical_request_id", "")),
                "matched_g2_history_length": str(matched.get("history_length", "")),
                "matched_g2_population_label": str(matched.get("population_label", "")),
                "reuse_status": "REUSABLE_G2_TERMINAL_EVIDENCE" if reusable else "NOT_REUSABLE_FOR_G2",
                "reusable_g2_terminal_evidence": bool_text(reusable),
                "field_match_vector": stable_json(field_matches),
                "reuse_rejection_reason": rejection_reason,
            }
        )
    return pd.DataFrame(rows)


def build_prior_evidence_audit(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in config.get("prior_evidence", []):
        source_id = str(source["id"])
        rel_path = Path(str(source["path"]))
        abs_path = root / rel_path
        exists = abs_path.exists()
        source_hash = sha256_file(abs_path) if exists else ""
        frame = read_csv(abs_path) if exists else pd.DataFrame()
        rows.append(
            {
                "source_id": source_id,
                "path": rel_path.as_posix(),
                "exists": bool_text(exists),
                "size_bytes": abs_path.stat().st_size if exists else "",
                "sha256": source_hash,
                "source_rows": len(frame) if exists else 0,
                "datasets_observed": joined_unique(frame, ["dataset"]) if exists else "",
                "models_observed": joined_unique(frame, ["model", "model_name"]) if exists else "",
                "prompts_observed": joined_unique(frame, ["prompt_strategy", "prompt_template_id"]) if exists else "",
                "history_lengths_observed": joined_unique(frame, ["history_length", "history_length_used"]) if exists else "",
                "parser_contracts_observed": joined_unique(frame, ["parser_contract"]) if exists else "",
                "expected_output_schemas_observed": joined_unique(frame, ["expected_output_schema"]) if exists else "",
                "max_new_tokens_observed": joined_unique(frame, ["max_new_tokens"]) if exists else "",
                "seed_observed": joined_unique(frame, ["seed"]) if exists else "",
                "candidate_counts_observed": joined_unique(frame, ["candidate_count", "candidate_pool_size"]) if exists else "",
                "reuse_status": PRIOR_REUSE_STATUS if exists else "MISSING_PRIOR_EVIDENCE_FILE",
                "reusable_g2_terminal_evidence": "false",
                "retention_policy": "TRACKED_COMPARATIVE_PRIOR_AUDIT_ONLY",
                "rejection_reasons": stable_json(prior_rejection_reasons(source_id, frame if exists else None)),
            }
        )
    return pd.DataFrame(rows)


def joined_unique(frame: pd.DataFrame, candidate_columns: Sequence[str]) -> str:
    values: list[str] = []
    for column in candidate_columns:
        if column in frame.columns:
            values.extend(str(value) for value in frame[column].dropna().astype(str).unique() if str(value).strip())
    return ";".join(sorted(set(values), key=sort_key))


def prior_rejection_reasons(source_id: str, frame: pd.DataFrame | None) -> list[str]:
    if frame is None:
        return ["source_file_missing"]
    reasons = ["not_frozen_g2_request_identity"]
    if "parser_contract" in frame.columns and not set(frame["parser_contract"].astype(str)).issubset({PARSER_CONTRACT_ID}):
        reasons.append("parser_contract_differs_from_g2_strict_top10")
    if "expected_output_schema" in frame.columns:
        schemas = set(frame["expected_output_schema"].astype(str))
        if any("[1,2,3,4,5]" in schema for schema in schemas):
            reasons.append("requested_output_schema_top5_not_top10")
    if "max_new_tokens" in frame.columns and any(str(value) != "256" for value in frame["max_new_tokens"].astype(str).unique()):
        reasons.append("generation_num_predict_differs_from_g1_g2")
    if "model_name" in frame.columns:
        observed = set(frame["model_name"].astype(str).unique())
        if not set(MODEL_ORDER).issubset(observed):
            reasons.append("model_coverage_not_exact_g1_three_model_set")
    if "dataset" in frame.columns:
        observed_datasets = {normalize_dataset_label(value) for value in frame["dataset"].astype(str).unique()}
        if not set(DATASET_ORDER).issubset(observed_datasets):
            reasons.append("dataset_coverage_not_exact_g1_five_dataset_set")
    if source_id == "R1_OFFICIAL_HISTORY_LENGTH":
        reasons.append("paper_official_summary_not_local_terminal_request_evidence")
    if source_id == "PLAN_D_ML1M_HISTORY_LENGTH":
        reasons.append("ml1m_only_non_g1_prompt_family")
    return sorted(set(reasons))


def normalize_dataset_label(value: str) -> str:
    text = str(value).strip().lower()
    return {
        "beauty": "beauty",
        "amazon beauty": "beauty",
        "sports": "sports",
        "amazon sports": "sports",
        "ml1m": "ml1m",
        "movielens": "ml1m",
        "lastfm": "lastfm",
        "last.fm": "lastfm",
        "foodcom": "foodcom",
        "food.com": "foodcom",
    }.get(text, text)


def build_request_accounting(applicability: pd.DataFrame, request_universe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "accounting_scope",
        "reporting_view",
        "dataset",
        "cohort_type",
        "population_label",
        "history_length",
        "history_condition_label",
        "prompt_strategy",
        "model",
        "applicability_status",
        "eligible_user_count",
        "applicable_user_conditions",
        "not_applicable_user_conditions",
        "planned_applicable_requests",
        "g1_exact_reuse_requests",
        "new_future_ollama_calls_required",
        "llmrank_h0_status",
        "future_raw_evidence_retention_policy",
        "reason",
    ]
    rows: list[dict[str, Any]] = []

    def add_account_row(
        *,
        scope: str,
        reporting_view: str,
        dataset: str,
        cohort_type: str = "ALL",
        population_label: str = "ALL",
        history_length: str | int = "ALL",
        history_condition_label: str = "ALL",
        prompt_strategy: str = "ALL",
        model: str = "ALL",
        planned: int,
        reuse: int,
        new: int,
        not_applicable: int = 0,
        eligible_user_count: str | int = "",
        applicable_user_conditions: str | int = "",
        llmrank_h0_status: str = NOT_APPLICABLE,
        reason: str = "",
    ) -> None:
        rows.append(
            {
                "accounting_scope": scope,
                "reporting_view": reporting_view,
                "dataset": dataset,
                "cohort_type": cohort_type,
                "population_label": population_label,
                "history_length": history_length,
                "history_condition_label": history_condition_label,
                "prompt_strategy": prompt_strategy,
                "model": model,
                "applicability_status": "SUMMARY" if scope != "DEDUP_EXECUTION_CELL" else "APPLICABLE_DEDUPLICATED_CANONICAL_REQUEST",
                "eligible_user_count": eligible_user_count,
                "applicable_user_conditions": planned if applicable_user_conditions == "" else applicable_user_conditions,
                "not_applicable_user_conditions": not_applicable,
                "planned_applicable_requests": planned,
                "g1_exact_reuse_requests": reuse,
                "new_future_ollama_calls_required": new,
                "llmrank_h0_status": llmrank_h0_status,
                "future_raw_evidence_retention_policy": "EXTERNALIZED_RAW_EVIDENCE_REQUIRED" if new else "NO_NEW_RAW_EVIDENCE_REQUIRED",
                "reason": reason,
            }
        )

    for (dataset, history_length, prompt_strategy, model), group in request_universe.groupby(
        ["dataset", "history_length", "prompt_strategy", "model"], sort=False
    ):
        planned = len(group)
        reuse = int(group["planned_action"].eq(G1_REUSE_ACTION).sum())
        new = int(group["planned_action"].eq(NEW_CALL_ACTION).sum())
        add_account_row(
            scope="DEDUP_EXECUTION_CELL",
            reporting_view="DEDUPLICATED_EXECUTION",
            dataset=str(dataset),
            history_length=int(history_length),
            history_condition_label=f"H{history_length}",
            prompt_strategy=str(prompt_strategy),
            model=str(model),
            planned=planned,
            reuse=reuse,
            new=new,
            llmrank_h0_status=LLMRANK_H0_STATUS if str(prompt_strategy) == PROMPT_LLMRANK and int(history_length) == 0 else NOT_APPLICABLE,
            reason="Deduplicated canonical user/H/model/prompt execution cell; overlapping H10 analysis membership does not duplicate execution.",
        )

    for dataset, group in request_universe.groupby("dataset", sort=False):
        add_account_row(
            scope="DEDUP_BY_DATASET",
            reporting_view="DEDUPLICATED_EXECUTION",
            dataset=str(dataset),
            planned=len(group),
            reuse=int(group["planned_action"].eq(G1_REUSE_ACTION).sum()),
            new=int(group["planned_action"].eq(NEW_CALL_ACTION).sum()),
            reason="Deduplicated execution requests by dataset.",
        )

    for model, group in request_universe.groupby("model", sort=False):
        add_account_row(
            scope="DEDUP_BY_MODEL",
            reporting_view="DEDUPLICATED_EXECUTION",
            dataset="ALL",
            model=str(model),
            planned=len(group),
            reuse=int(group["planned_action"].eq(G1_REUSE_ACTION).sum()),
            new=int(group["planned_action"].eq(NEW_CALL_ACTION).sum()),
            reason="Deduplicated execution requests by model.",
        )

    for prompt_strategy, group in request_universe.groupby("prompt_strategy", sort=False):
        add_account_row(
            scope="DEDUP_BY_PROMPT",
            reporting_view="DEDUPLICATED_EXECUTION",
            dataset="ALL",
            prompt_strategy=str(prompt_strategy),
            planned=len(group),
            reuse=int(group["planned_action"].eq(G1_REUSE_ACTION).sum()),
            new=int(group["planned_action"].eq(NEW_CALL_ACTION).sum()),
            reason="Deduplicated execution requests by prompt.",
        )

    for (reporting_view, dataset, prompt_strategy, model), group in applicability.loc[
        applicability["applicability_status"].ne(NOT_APPLICABLE)
    ].groupby(["reporting_view", "dataset", "prompt_strategy", "model"], sort=False):
        planned = int(pd.to_numeric(group["applicable_user_conditions"]).sum())
        add_account_row(
            scope="REPORTING_VIEW_DENOMINATOR",
            reporting_view=str(reporting_view),
            dataset=str(dataset),
            prompt_strategy=str(prompt_strategy),
            model=str(model),
            planned=planned,
            reuse=0,
            new=0,
            eligible_user_count=";".join(sorted(set(group["eligible_user_count"].astype(str)), key=sort_key)),
            applicable_user_conditions=planned,
            reason="Reporting-view denominator only; these rows are not summed into deduplicated execution totals.",
        )

    not_applicable_total = int(pd.to_numeric(applicability["not_applicable_user_conditions"]).sum())
    total_row = {
        "accounting_scope": "GRAND_TOTAL",
        "reporting_view": "DEDUPLICATED_EXECUTION",
        "dataset": "ALL",
        "cohort_type": "ALL",
        "population_label": "ALL",
        "history_length": "ALL",
        "history_condition_label": "ALL",
        "prompt_strategy": "ALL",
        "model": "ALL",
        "applicability_status": "SUMMARY",
        "eligible_user_count": "",
        "applicable_user_conditions": len(request_universe),
        "not_applicable_user_conditions": not_applicable_total,
        "planned_applicable_requests": len(request_universe),
        "g1_exact_reuse_requests": int(request_universe["planned_action"].eq(G1_REUSE_ACTION).sum()),
        "new_future_ollama_calls_required": int(request_universe["planned_action"].eq(NEW_CALL_ACTION).sum()),
        "llmrank_h0_status": LLMRANK_H0_STATUS,
        "future_raw_evidence_retention_policy": "EXTERNALIZED_RAW_EVIDENCE_REQUIRED_FOR_NEW_CALLS",
        "reason": "Grand total across deduplicated Design B canonical requests and Food.com true-zero H0 diagnostic.",
    }
    rows.append(total_row)
    return pd.DataFrame(rows, columns=columns)


def build_research_questions() -> pd.DataFrame:
    rows = [
        {
            "rq_id": "RQ-G2.1",
            "scope": PRIMARY_VIEW,
            "question": "How does utility change as visible chronological train history increases from H=0 to H=10 among GE10 users?",
            "primary_metrics": "hr_at_5;ndcg_at_5;mrr",
            "planned_status": "FROZEN_PRE_INFERENCE",
        },
        {
            "rq_id": "RQ-G2.2",
            "scope": PRIMARY_VIEW,
            "question": "What is the controlled value of any visible history relative to no visible history?",
            "primary_metrics": "hr_at_5;ndcg_at_5;mrr",
            "planned_status": "FROZEN_PRE_INFERENCE",
        },
        {
            "rq_id": "RQ-G2.3",
            "scope": PRIMARY_VIEW,
            "question": "Do marginal gains saturate between H=5 and H=10 in the primary B1 short/medium design?",
            "primary_metrics": "hr_at_5;ndcg_at_5;mrr",
            "planned_status": "FROZEN_PRE_INFERENCE",
        },
        {
            "rq_id": "RQ-G2.4",
            "scope": f"{PRIMARY_VIEW}_prompt_interaction",
            "question": "Does BASE versus LLMRANK interact with short/medium history length under the frozen prompt contracts?",
            "primary_metrics": "hr_at_5;ndcg_at_5;mrr;strict_parse_valid_rate",
            "planned_status": "FROZEN_PRE_INFERENCE",
        },
        {
            "rq_id": "RQ-G2.5",
            "scope": f"{PRIMARY_VIEW}_model_interaction",
            "question": "Are B1 short/medium history-length effects consistent across the three frozen G1 local models?",
            "primary_metrics": "hr_at_5;ndcg_at_5;mrr;strict_parse_valid_rate",
            "planned_status": "FROZEN_PRE_INFERENCE",
        },
        {
            "rq_id": "RQ-G2.6",
            "scope": f"{PRIMARY_VIEW}_dataset_interaction",
            "question": "Are B1 short/medium history-length effects consistent across Beauty, Sports, ML1M, LastFM, and Food.com?",
            "primary_metrics": "hr_at_5;ndcg_at_5;mrr",
            "planned_status": "FROZEN_PRE_INFERENCE",
        },
        {
            "rq_id": "RQ-G2.LONG",
            "scope": LONG_VIEW,
            "question": "Among users with at least 20 historical interactions, does increasing context from H=10 to H=20 produce additional utility/reliability benefit?",
            "primary_metrics": "hr_at_5;ndcg_at_5;mrr;strict_parse_valid_rate",
            "planned_status": "FROZEN_PRE_INFERENCE_LONG_HISTORY_EXTENSION",
        },
        {
            "rq_id": "RQ-G2.7",
            "scope": TRUE_ZERO_VIEW,
            "question": "How does Food.com real true-zero H=0 behavior compare descriptively with Food.com controlled H=0 warm users?",
            "primary_metrics": "hr_at_5;ndcg_at_5;mrr;strict_parse_valid_rate",
            "planned_status": "DESCRIPTIVE_ONLY_NOT_PRIMARY_PAIRED",
        },
    ]
    return pd.DataFrame(rows)


def build_statistical_plan(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparison_groups = [
        (PRIMARY_VIEW, "primary_b1_ge10", [(int(pair[0]), int(pair[1])) for pair in config["statistics"]["planned_history_comparisons"]["primary_b1_ge10"]]),
        (LONG_VIEW, "long_history_b2_ge20", [(int(pair[0]), int(pair[1])) for pair in config["statistics"]["planned_history_comparisons"]["long_history_b2_ge20"]]),
        (
            SECONDARY_VIEW,
            "secondary_full_curve_ge20",
            [(int(pair[0]), int(pair[1])) for pair in config["statistics"]["planned_history_comparisons"]["secondary_full_curve_ge20"]],
        ),
    ]
    metrics = list(config["metrics"]["utility"])
    order = 1
    for reporting_view, comparison_group, comparisons in comparison_groups:
        for metric in metrics:
            for h_a, h_b in comparisons:
                rows.append(
                    {
                        "plan_order": order,
                        "comparison_id": f"{reporting_view}_H{h_a}_VS_H{h_b}_{metric.upper()}",
                        "reporting_view": reporting_view,
                        "comparison_group": comparison_group,
                        "scope": f"within_dataset_model_prompt_{reporting_view.lower()}",
                        "metric": metric,
                        "history_length_a": h_a,
                        "history_length_b": h_b,
                        "contrast": f"H{h_b}_minus_H{h_a}",
                        "paired_unit": "user_id",
                        "denominator_policy": "fail_strict_all_planned_users_never_collapse_b1_b2_denominators",
                        "primary_test": "paired_user_bootstrap_mean_delta",
                        "bootstrap_resamples": int(config["statistics"]["bootstrap_resamples"]),
                        "bootstrap_seed": int(config["statistics"]["bootstrap_seed"]),
                        "confidence_interval": "two_sided_95_percent_percentile_ci",
                        "effect_size": "paired_mean_delta",
                        "secondary_test": "mcnemar_for_hr_at_5_only_else_wilcoxon_signed_rank_sensitivity",
                        "multiple_comparison_correction": str(config["statistics"]["multiple_comparison_correction"]),
                        "alpha": float(config["statistics"]["alpha"]),
                        "planned_status": "FROZEN_PRE_INFERENCE_SECONDARY" if reporting_view == SECONDARY_VIEW else "FROZEN_PRE_INFERENCE",
                    }
                )
                order += 1
    return pd.DataFrame(rows)


def build_representativeness(eligibility: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in eligibility.to_dict("records"):
        dataset = str(row["dataset"])
        total = int(row["total_g1_users"])
        ge10 = int(row["users_ge_10"])
        ge20 = int(row["users_ge_20"])
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "design": "DESIGN_A_RETIRED_SINGLE_GE20_FULL_CURVE",
                    "reporting_view": SECONDARY_VIEW,
                    "users": ge20,
                    "share_of_g1_users": format_ratio(ge20, total),
                    "status": "RETIRED_AS_PRIMARY_PRE_INFERENCE",
                    "interpretation": "Clean full H0..H20 curve, but Beauty/Sports/Food.com GE20 cohorts are small.",
                },
                {
                    "dataset": dataset,
                    "design": "DESIGN_B_SELECTED_PRIMARY_B1_GE10",
                    "reporting_view": PRIMARY_VIEW,
                    "users": ge10,
                    "share_of_g1_users": format_ratio(ge10, total),
                    "status": str(row["primary_cohort_status"]),
                    "interpretation": "Selected before G2 inference because it improves Beauty/Sports/Food.com representativeness while preserving paired H0..H10 inference.",
                },
                {
                    "dataset": dataset,
                    "design": "DESIGN_B_SELECTED_LONG_HISTORY_B2_GE20",
                    "reporting_view": LONG_VIEW,
                    "users": ge20,
                    "share_of_g1_users": format_ratio(ge20, total),
                    "status": str(row["long_history_cohort_status"]),
                    "interpretation": "Selected before G2 inference as a separate H10/H20 long-history extension with its own denominator.",
                },
                {
                    "dataset": dataset,
                    "design": "DESIGN_B_SECONDARY_FULL_CURVE_GE20",
                    "reporting_view": SECONDARY_VIEW,
                    "users": ge20,
                    "share_of_g1_users": format_ratio(ge20, total),
                    "status": "SECONDARY_PRE_INFERENCE",
                    "interpretation": "Complete H0..H20 curve remains available as a secondary GE20 analysis using only existing B1/B2 request identities.",
                },
            ]
        )
    return pd.DataFrame(rows)


def validate_outputs(
    *,
    config: Mapping[str, Any],
    eligibility: pd.DataFrame,
    cohort: pd.DataFrame,
    request_universe: pd.DataFrame,
    applicability: pd.DataFrame,
    g1_reuse_audit: pd.DataFrame,
    prior_evidence_audit: pd.DataFrame,
    request_accounting: pd.DataFrame,
    g1_execution_manifest: Mapping[str, Any],
    g1_retention_manifest: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    head_short = git_output(find_project_root(), ["rev-parse", "--short", "HEAD"])
    if head_short != str(config["phase"]["required_g1_checkpoint_short"]):
        failures.append(f"HEAD_SHORT_MISMATCH:{head_short}")
    if str(g1_execution_manifest.get("decision_token", "")) != "G1_PRIMARY_RANKING_VALIDATED":
        failures.append("G1_PRIMARY_RANKING_NOT_VALIDATED")
    retention_valid = (
        str(g1_retention_manifest.get("decision_token", "")) == "G1_RETENTION_VALIDATED"
        or (
            str(g1_retention_manifest.get("phase_id", "")) == "g1_primary_ranking_retention_freeze"
            and str(g1_retention_manifest.get("decision_source", "")) == "G1_PRIMARY_RANKING_VALIDATED"
            and not bool(g1_retention_manifest.get("retention_gap_detected", False))
        )
    )
    if not retention_valid:
        failures.append("G1_RETENTION_NOT_VALIDATED")
    b1_counts = eligibility.set_index("dataset")["users_ge_10"].astype(int).to_dict()
    if b1_counts != EXPECTED_B1_USERS_BY_DATASET:
        failures.append(f"PRIMARY_B1_GE10_COUNTS_CHANGED:{b1_counts}")
    b2_counts = eligibility.set_index("dataset")["users_ge_20"].astype(int).to_dict()
    if b2_counts != EXPECTED_B2_USERS_BY_DATASET:
        failures.append(f"LONG_HISTORY_B2_GE20_COUNTS_CHANGED:{b2_counts}")
    food_true_zero = int(eligibility.set_index("dataset").loc["foodcom", "foodcom_true_zero_history_users"])
    if food_true_zero != EXPECTED_TRUE_ZERO_FOODCOM_USERS:
        failures.append(f"FOODCOM_TRUE_ZERO_COUNT_CHANGED:{food_true_zero}")
    primary = cohort.loc[cohort["cohort_type"].eq(PRIMARY_COHORT_TYPE)]
    if len(primary) != sum(EXPECTED_B1_USERS_BY_DATASET.values()):
        failures.append("PRIMARY_B1_COHORT_ROW_COUNT_MISMATCH")
    b2 = primary.loc[primary["b2_long_history_ge20_eligible"].eq("true")]
    if len(b2) != sum(EXPECTED_B2_USERS_BY_DATASET.values()):
        failures.append("LONG_HISTORY_B2_COHORT_ROW_COUNT_MISMATCH")
    if not primary["target_excluded_from_full_train_history"].eq("true").all():
        failures.append("TARGET_PRESENT_IN_FULL_TRAIN_HISTORY")
    for history_length in HISTORY_ORDER:
        column = f"history_h{history_length}_target_excluded"
        applicability_column = f"history_h{history_length}_applicability"
        applicable_targets = primary.loc[primary[applicability_column].eq("APPLICABLE"), column]
        if not applicable_targets.eq("true").all():
            failures.append(f"TARGET_PRESENT_IN_HISTORY_H{history_length}")
    if not primary["g1_h10_history_hash_match"].eq("true").all():
        failures.append("G1_H10_HISTORY_HASH_NOT_MATCHED_FOR_PRIMARY")
    if request_universe["canonical_request_id"].nunique() != len(request_universe):
        failures.append("G2_REQUEST_IDS_NOT_UNIQUE")
    identity_columns = ["dataset", "user_id", "history_length", "prompt_strategy", "model", "population_label"]
    if request_universe.duplicated(identity_columns).any():
        failures.append("DUPLICATE_USER_H_PROMPT_MODEL_IDENTITY_PRESENT")
    if len(request_universe) != EXPECTED_TOTAL_REQUESTS:
        failures.append("G2_REQUEST_UNIVERSE_ROW_COUNT_MISMATCH")
    exact_reuse = int(request_universe["planned_action"].eq(G1_REUSE_ACTION).sum())
    if exact_reuse != EXPECTED_G1_REUSE_REQUESTS:
        failures.append(f"G1_REUSE_COUNT_MISMATCH:{exact_reuse}")
    new_calls = int(request_universe["planned_action"].eq(NEW_CALL_ACTION).sum())
    if new_calls != EXPECTED_NEW_CALLS:
        failures.append(f"NEW_CALL_COUNT_MISMATCH:{new_calls}")
    grand = request_accounting.loc[request_accounting["accounting_scope"].eq("GRAND_TOTAL")].iloc[0]
    if int(grand["planned_applicable_requests"]) != EXPECTED_TOTAL_REQUESTS:
        failures.append("GRAND_TOTAL_REQUEST_COUNT_MISMATCH")
    if int(grand["not_applicable_user_conditions"]) != EXPECTED_NOT_APPLICABLE_TRUE_ZERO:
        failures.append("TRUE_ZERO_NOT_APPLICABLE_COUNT_MISMATCH")
    dataset_totals = {
        str(row["dataset"]): {
            "total": int(row["planned_applicable_requests"]),
            "reuse": int(row["g1_exact_reuse_requests"]),
            "new": int(row["new_future_ollama_calls_required"]),
        }
        for row in request_accounting.loc[request_accounting["accounting_scope"].eq("DEDUP_BY_DATASET")].to_dict("records")
    }
    if dataset_totals != EXPECTED_TOTALS_BY_DATASET:
        failures.append(f"DATASET_ACCOUNTING_MISMATCH:{dataset_totals}")
    model_totals = {
        str(row["model"]): (
            int(row["planned_applicable_requests"]),
            int(row["g1_exact_reuse_requests"]),
            int(row["new_future_ollama_calls_required"]),
        )
        for row in request_accounting.loc[request_accounting["accounting_scope"].eq("DEDUP_BY_MODEL")].to_dict("records")
    }
    if set(model_totals.values()) != {(29350, 6220, 23130)} or set(model_totals) != set(MODEL_ORDER):
        failures.append(f"MODEL_ACCOUNTING_MISMATCH:{model_totals}")
    prompt_totals = {
        str(row["prompt_strategy"]): (
            int(row["planned_applicable_requests"]),
            int(row["g1_exact_reuse_requests"]),
            int(row["new_future_ollama_calls_required"]),
        )
        for row in request_accounting.loc[request_accounting["accounting_scope"].eq("DEDUP_BY_PROMPT")].to_dict("records")
    }
    if set(prompt_totals.values()) != {(44025, 9330, 34695)} or set(prompt_totals) != set(PROMPT_ORDER):
        failures.append(f"PROMPT_ACCOUNTING_MISMATCH:{prompt_totals}")
    secondary_rows = request_universe.loc[request_universe["secondary_full_curve_ge20_member"].eq("true")]
    if not secondary_rows["reporting_view_memberships"].str.contains(PRIMARY_VIEW, regex=False).any():
        failures.append("SECONDARY_FULL_CURVE_NOT_REUSING_B1_REQUESTS")
    if not secondary_rows["reporting_view_memberships"].str.contains(LONG_VIEW, regex=False).any():
        failures.append("SECONDARY_FULL_CURVE_NOT_REUSING_B2_REQUESTS")
    if not applicability.loc[
        (applicability["prompt_strategy"].eq(PROMPT_LLMRANK)) & (applicability["history_length"].astype(str).eq("0")),
        "llmrank_h0_status",
    ].eq(LLMRANK_H0_STATUS).all():
        failures.append("LLMRANK_H0_STATUS_NOT_FROZEN_LOCAL_EQUIVALENT")
    if not prior_evidence_audit["reusable_g2_terminal_evidence"].eq("false").all():
        failures.append("PRIOR_EVIDENCE_MARKED_AS_REUSABLE")
    if int(g1_reuse_audit["g1_terminal"].eq("true").sum()) != len(g1_reuse_audit):
        failures.append("G1_NONTERMINAL_REQUEST_PRESENT")
    return failures


def build_manifest(
    *,
    root: Path,
    config: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    generation_contract: Mapping[str, Any],
    parser_hash: str,
    validation_failures: Sequence[str],
    g1_execution_manifest: Mapping[str, Any],
    g1_retention_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    request_universe = tables["request_universe"]
    request_accounting = tables["request_accounting"]
    grand = request_accounting.loc[request_accounting["accounting_scope"].eq("GRAND_TOTAL")].iloc[0].to_dict()
    table_hashes = {OUTPUT_TABLES[name].as_posix(): frame_hash(frame) for name, frame in tables.items()}
    manifest = {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_FROZEN if not validation_failures else DECISION_BLOCKED,
        "generated_at": utc_now(),
        "required_g1_checkpoint_short": str(config["phase"]["required_g1_checkpoint_short"]),
        "required_g1_commit_message": str(config["phase"]["required_g1_commit_message"]),
        "head_short": git_output(root, ["rev-parse", "--short", "HEAD"]),
        "head_oneline": git_output(root, ["log", "-1", "--oneline"]),
        "phase_start_clean_required": bool(config["phase"]["phase_start_clean_required"]),
        "phase_start_clean_verified_before_g2_edits": True,
        "models": list(MODEL_ORDER),
        "datasets": list(DATASET_ORDER),
        "prompt_strategies": list(PROMPT_ORDER),
        "history_lengths": list(HISTORY_ORDER),
        "selected_design": "DESIGN_B_NESTED_PAIRED_ANALYSES",
        "reporting_views": [PRIMARY_VIEW, LONG_VIEW, SECONDARY_VIEW, TRUE_ZERO_VIEW],
        "primary_min_history": PRIMARY_MIN_HISTORY,
        "long_history_min_history": LONG_HISTORY_MIN_HISTORY,
        "b1_primary_paired_users": int(sum(EXPECTED_B1_USERS_BY_DATASET.values())),
        "b1_primary_paired_users_by_dataset": EXPECTED_B1_USERS_BY_DATASET,
        "b2_long_history_paired_users": int(sum(EXPECTED_B2_USERS_BY_DATASET.values())),
        "b2_long_history_paired_users_by_dataset": EXPECTED_B2_USERS_BY_DATASET,
        "secondary_full_curve_paired_users": int(sum(EXPECTED_B2_USERS_BY_DATASET.values())),
        "primary_paired_users": int(sum(EXPECTED_B1_USERS_BY_DATASET.values())),
        "primary_paired_users_by_dataset": EXPECTED_B1_USERS_BY_DATASET,
        "foodcom_true_zero_history_users": EXPECTED_TRUE_ZERO_FOODCOM_USERS,
        "planned_applicable_requests": int(grand["planned_applicable_requests"]),
        "g1_exact_reuse_requests": int(grand["g1_exact_reuse_requests"]),
        "new_future_ollama_calls_required": int(grand["new_future_ollama_calls_required"]),
        "not_applicable_true_zero_h_positive_user_conditions": int(grand["not_applicable_user_conditions"]),
        "expected_totals_by_dataset": EXPECTED_TOTALS_BY_DATASET,
        "expected_total_requests": EXPECTED_TOTAL_REQUESTS,
        "expected_g1_reuse_requests": EXPECTED_G1_REUSE_REQUESTS,
        "expected_new_calls": EXPECTED_NEW_CALLS,
        "llmrank_h0_status": LLMRANK_H0_STATUS,
        "generation_contract": dict(generation_contract),
        "parser_contract": PARSER_CONTRACT_ID,
        "parser_contract_hash": parser_hash,
        "request_universe_hash": frame_hash(request_universe),
        "table_hashes": table_hashes,
        "tables": {name: path.as_posix() for name, path in OUTPUT_TABLES.items()},
        "report": REPORT_PATH.as_posix(),
        "config_path": CONFIG_PATH.as_posix(),
        "source_paths": {
            "runner": Path("src/extension/g2_history_length_sensitivity_protocol.py").as_posix(),
            "tests": Path("tests/test_extension_g2_history_length_sensitivity_protocol.py").as_posix(),
        },
        "g1_validation": {
            "primary_ranking_decision": str(g1_execution_manifest.get("decision_token", "")),
            "retention_decision": str(g1_retention_manifest.get("decision_token", "G1_RETENTION_VALIDATED")),
            "retention_phase_id": str(g1_retention_manifest.get("phase_id", "")),
            "g1_planned_requests": int(g1_execution_manifest.get("planned_requests", 0)),
            "g1_terminal_requests": int(g1_execution_manifest.get("terminal_requests", 0)),
        },
        "retention": {
            "tracked_scientific_foreground": list(config["retention"]["tracked_scientific_foreground"]),
            "tracked_provenance_manifest": MANIFEST_PATH.as_posix(),
            "future_raw_evidence_class": "EXTERNALIZED_RAW_EVIDENCE",
            "future_per_request_audit_class": "EXTERNALIZED_PER_REQUEST_AUDIT",
            "all_future_externalized_evidence_must_be_referenced_by_tracked_manifest_with_stable_hashes": True,
        },
        "no_inference_guards": {
            "llm_inference_calls_executed": 0,
            "local_ollama_metadata_calls_executed": 0,
            "external_api_calls_executed": 0,
            "downloads_executed": 0,
            "network_downloads_executed": 0,
            "training_runs_executed": 0,
            "tuning_runs_executed": 0,
            "g3_requests_executed": 0,
            "g4_requests_executed": 0,
            "g5_requests_executed": 0,
            "metrics_recomputed": False,
            "g1_protocol_modified": False,
            "g1_scientific_results_modified": False,
        },
        "validation_failures": list(validation_failures),
        "validation_passed": not validation_failures,
    }
    return manifest


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for path in [TABLE_DIR, MANIFEST_DIR, REPORT_PATH.parent]:
        (root / path).mkdir(parents=True, exist_ok=True)
    for name, frame in outputs["tables"].items():
        frame.to_csv(root / OUTPUT_TABLES[name], index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8", newline="\n")

    manifest = dict(outputs["manifest"])
    artifact_paths = [CONFIG_PATH, Path("src/extension/g2_history_length_sensitivity_protocol.py"), Path("tests/test_extension_g2_history_length_sensitivity_protocol.py")]
    artifact_paths.extend(OUTPUT_TABLES.values())
    artifact_paths.append(REPORT_PATH)
    artifact_hashes = {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in artifact_paths
        if (root / rel_path).exists()
    }
    manifest["artifact_hashes"] = artifact_hashes
    manifest["source_hashes"] = {
        "config": artifact_hashes.get(CONFIG_PATH.as_posix(), ""),
        "runner": artifact_hashes.get("src/extension/g2_history_length_sensitivity_protocol.py", ""),
        "tests": artifact_hashes.get("tests/test_extension_g2_history_length_sensitivity_protocol.py", ""),
        "report": artifact_hashes.get(REPORT_PATH.as_posix(), ""),
    }
    (root / MANIFEST_PATH).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    outputs["manifest"].clear()
    outputs["manifest"].update(manifest)


def append_phase_notes(root: Path, outputs: Mapping[str, Any]) -> None:
    manifest = outputs["manifest"]
    notes = (
        f"\n## {utc_now()} - G2 nested history-length sensitivity protocol refreeze\n\n"
        f"- Decision: `{manifest['decision_token']}`.\n"
        "- Scope: nested Design B protocol update, G1 reuse audit, prior-evidence rejection audit, request accounting, and retention planning only.\n"
        f"- B1 primary paired cohort: `{manifest['b1_primary_paired_users']}` users with at least `{PRIMARY_MIN_HISTORY}` train-history interactions.\n"
        f"- B2 long-history paired cohort: `{manifest['b2_long_history_paired_users']}` users with at least `{LONG_HISTORY_MIN_HISTORY}` train-history interactions.\n"
        f"- Planned applicable G2 requests: `{manifest['planned_applicable_requests']}`; exact G1 terminal reuse: `{manifest['g1_exact_reuse_requests']}`; future new Ollama calls required: `{manifest['new_future_ollama_calls_required']}`.\n"
        f"- Food.com true-zero H>0 not-applicable user conditions: `{manifest['not_applicable_true_zero_h_positive_user_conditions']}`.\n"
        "- No inference, Ollama metadata, remote API calls, downloads, training, tuning, metric recomputation, G1 protocol edits, commit, reset, clean, checkout, G3, G4, or G5 actions were run.\n"
    )
    with (root / NOTES_PATH).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(notes)


def render_report(tables: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]) -> str:
    eligibility = tables["eligibility"]
    request_accounting = tables["request_accounting"]
    representativeness = tables["representativeness"]
    prior = tables["prior_evidence_audit"]
    grand = request_accounting.loc[request_accounting["accounting_scope"].eq("GRAND_TOTAL")].iloc[0]
    eligibility_summary = eligibility[
        [
            "dataset",
            "users_ge_1",
            "users_ge_2",
            "users_ge_5",
            "users_ge_10",
            "users_ge_20",
            "primary_cohort_status",
            "long_history_cohort_status",
        ]
    ]
    dataset_totals = request_accounting.loc[
        request_accounting["accounting_scope"].eq("DEDUP_BY_DATASET"),
        ["dataset", "planned_applicable_requests", "g1_exact_reuse_requests", "new_future_ollama_calls_required"],
    ]
    model_totals = request_accounting.loc[
        request_accounting["accounting_scope"].eq("DEDUP_BY_MODEL"),
        ["model", "planned_applicable_requests", "g1_exact_reuse_requests", "new_future_ollama_calls_required"],
    ]
    prompt_totals = request_accounting.loc[
        request_accounting["accounting_scope"].eq("DEDUP_BY_PROMPT"),
        ["prompt_strategy", "planned_applicable_requests", "g1_exact_reuse_requests", "new_future_ollama_calls_required"],
    ]
    b1_representativeness = representativeness.loc[
        representativeness["design"].eq("DESIGN_B_SELECTED_PRIMARY_B1_GE10"),
        ["dataset", "users", "share_of_g1_users", "status"],
    ]
    b2_representativeness = representativeness.loc[
        representativeness["design"].eq("DESIGN_B_SELECTED_LONG_HISTORY_B2_GE20"),
        ["dataset", "users", "share_of_g1_users", "status"],
    ]
    prior_summary = prior[["source_id", "exists", "source_rows", "reuse_status", "reusable_g2_terminal_evidence"]]

    return "\n".join(
        [
            "# G2 Nested History-Length Sensitivity Protocol",
            "",
            "## PRECONDITION",
            f"- Required checkpoint: `{manifest['required_g1_checkpoint_short']}` / `{manifest['required_g1_commit_message']}`.",
            f"- Observed HEAD: `{manifest['head_oneline']}`.",
            "- G1 primary ranking and G1 retention manifests are validated. No G0/G1 artifact is modified by this refreeze.",
            "",
            "## DESIGN DECISION",
            "- `ADOPT_DESIGN_B` from `G2_COHORT_DESIGN_REVIEW_COMPLETE` replaces the prior single GE20 primary design before any G2 inference.",
            "- B1 is primary short/medium history sensitivity; B2 is a separate long-history extension; the GE20 full curve is secondary.",
            "",
            "## B1 PRIMARY COHORT",
            f"- Reporting view `{PRIMARY_VIEW}`; eligibility train history >= `{PRIMARY_MIN_HISTORY}`; H=0,1,2,5,10.",
            f"- Paired users by dataset: `{stable_json(manifest['b1_primary_paired_users_by_dataset'])}`.",
            "",
            "## B2 LONG-HISTORY COHORT",
            f"- Reporting view `{LONG_VIEW}`; eligibility train history >= `{LONG_HISTORY_MIN_HISTORY}`; H=10,20.",
            f"- Paired users by dataset: `{stable_json(manifest['b2_long_history_paired_users_by_dataset'])}`.",
            "",
            "## SECONDARY FULL CURVE",
            f"- Reporting view `{SECONDARY_VIEW}`; GE20 users; H=0,1,2,5,10,20.",
            "- It reuses B1/B2 canonical request identities and creates zero duplicate inference calls.",
            "",
            "## FOOD.COM",
            f"- Food.com true-zero users: `{manifest['foodcom_true_zero_history_users']}`.",
            "- Warm GE10 users may enter B1; warm GE20 users may enter B2/secondary. True-zero users remain outside B1/B2 paired warm-user estimands.",
            "- Controlled warm H0 rows use `CONTROLLED_H0`; true-zero diagnostic rows use `TRUE_ZERO_HISTORY`; true-zero H>0 is `NOT_APPLICABLE`.",
            "",
            "## LLMRANK H0",
            f"- LLMRANK H=0 status: `{manifest['llmrank_h0_status']}`.",
            "- The frozen prompt renderer emits the no-recency sentence and does not fabricate a last item.",
            "",
            "## REQUEST UNIVERSE",
            f"- Deduplicated applicable request rows: `{manifest['planned_applicable_requests']}`.",
            "- A GE20/H10 request can belong to B1, B2, and secondary views, but keeps one canonical identity.",
            "Dataset totals:",
            dataset_totals.to_csv(index=False, lineterminator="\n").strip(),
            "",
            "## G1 REUSE",
            f"- Exact G1 terminal evidence reuse: `{manifest['g1_exact_reuse_requests']}` requests.",
            "- Reuse requires exact dataset, user, model, prompt, history content/order/hash, candidate content/order/hash, target, generation contract, parser contract, prompt hash, and model digest.",
            "- B1 H10 rows reuse G1 for GE10 users; Food.com true-zero H0 rows reuse G1.",
            "",
            "## NEW CALL ACCOUNTING",
            f"- Future new Ollama calls required: `{manifest['new_future_ollama_calls_required']}`.",
            f"- Not-applicable true-zero H>0 user conditions: `{int(grand['not_applicable_user_conditions'])}`.",
            "Model totals:",
            model_totals.to_csv(index=False, lineterminator="\n").strip(),
            "Prompt totals:",
            prompt_totals.to_csv(index=False, lineterminator="\n").strip(),
            "",
            "## STATISTICAL PLAN",
            "- B1 primary paired comparisons: H0-H1, H0-H5, H0-H10, H1-H2, H2-H5, H5-H10.",
            "- B2 long-history comparison: H10-H20.",
            "- Secondary GE20 full-curve comparisons: H0-H20 and H5-H20.",
            "- Multiplicity, CI, effect-size, alpha, and fail-strict denominator policies remain pre-outcome frozen.",
            "",
            "## REPRESENTATIVENESS",
            "Eligibility counts:",
            eligibility_summary.to_csv(index=False, lineterminator="\n").strip(),
            "B1 selected view:",
            b1_representativeness.to_csv(index=False, lineterminator="\n").strip(),
            "B2 long-history view:",
            b2_representativeness.to_csv(index=False, lineterminator="\n").strip(),
            "",
            "## RESEARCH QUESTIONS",
            "- RQ-G2.1 through RQ-G2.6 now refer primarily to B1 short/medium sensitivity.",
            "- `RQ-G2.LONG` covers H10 versus H20 among GE20 users.",
            "- The Food.com controlled-zero versus true-zero diagnostic RQ is preserved.",
            "",
            "## RETENTION",
            "- Retention classes are preserved: TRACKED_SCIENTIFIC_FOREGROUND, TRACKED_PROVENANCE_MANIFEST, EXTERNALIZED_RAW_EVIDENCE, EXTERNALIZED_PER_REQUEST_AUDIT.",
            "- Future G2 raw responses and per-request execution audit remain externalized and must be referenced by tracked manifests with stable SHA256 hashes before any commit.",
            "",
            "## TESTS",
            "- Focused tests are in `tests/test_extension_g2_history_length_sensitivity_protocol.py`.",
            "",
            "## FILES",
            "\n".join(f"- `{path.as_posix()}`" for path in [CONFIG_PATH, Path("src/extension/g2_history_length_sensitivity_protocol.py"), *OUTPUT_TABLES.values(), MANIFEST_PATH, REPORT_PATH]),
            "",
            "## PRIOR HISTORY EVIDENCE",
            prior_summary.to_csv(index=False, lineterminator="\n").strip(),
            "- Prior history-length evidence remains comparative only, not G2 reusable terminal evidence.",
            "",
            "## GIT STATUS",
            "- This refreeze does not commit automatically. Use `git status --short --untracked-files=all` for the current tree.",
            "",
            "## NEXT STAGE",
            "- G2B may execute only after this nested protocol is reviewed. No G2B/G3/G4/G5 execution is started here.",
            "",
            str(manifest["decision_token"]),
            "",
        ]
    )


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{numerator / denominator:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())

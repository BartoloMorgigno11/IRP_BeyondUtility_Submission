"""R5D generation-involved gated recovery.

R5D executes the frozen R5A GENERATION_INVOLVED_RECOVERY family in a separate
namespace. The family has two stages: profile generation for ML-1M/LastFM, then
profile-plus-history ranking. Beauty/Sports reuse existing F6L profile context.
No selective retry policy is introduced; max_retries remains zero.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import reproduction_r5c_history_length_recovery as r5c
from src.analysis.reproduction_r5b_canddif_recovery import (
    EXPECTED_R5A_MANIFEST_SHA256,
    FAIL_STRICT_DENOMINATOR_POLICY,
    candidate_item_ids,
    canonical_json,
    git_stdout,
    hr_at_5,
    load_json,
    ndcg_at_5,
    positive_rank,
    protected_hash_check,
    read_csv,
    rel_path,
    sha256_file,
    sha256_text,
    write_csv,
    write_json_atomic,
)
from src.llm.canonical_llm_contract import parse_strict_index_response


PHASE = "R5D_GENERATION_INVOLVED_RECOVERY_V2"
DEFAULT_CONFIG_PATH = Path("configs") / "reproduction_r5d_generation_involved_recovery.yaml"
NOTES_PATH = Path("report") / "notes.md"

PROFILE_PROMPT_TEMPLATE_ID = "r5d_profile_generation_json_profile_v2"
RANKING_PROMPT_TEMPLATE_ID = "r5d_profile_plus_history_json_index_exact10_v1"
PROFILE_PARSER_CONTRACT = "r5d_profile_json_profile_parser_v1"
RANKING_PARSER_CONTRACT = "r5d_ranked_index_json_top10_v1"
EVIDENCE_CLASS = "PAPER_ALIGNED_ADAPTED_GENERATION_INVOLVED_RECOVERY"
GLOBAL_STATUS = "CRITICAL_EXPERIMENTAL_PATH_R5D_PENDING_FREEZE"

DECISION_COMPLETE = "R5D_GENERATION_INVOLVED_RECOVERY_V2_COMPLETE"
DECISION_PREFLIGHT_READY = "R5D_PREFLIGHT_READY"
DECISION_PREFLIGHT_FAILED = "R5D_STATIC_PREFLIGHT_FAILED"
DECISION_CONTEXT_FAILED = "R5D_CONTEXT_LENGTH_CHECK_FAILED"
DECISION_CANARY_PASS = "R5D_CANARY_PASS"
DECISION_CANARY_FAILED = "R5D_CANARY_FAILED"
DECISION_EARLY_PASS = "R5D_EARLY_HEALTH_PASS"
DECISION_EARLY_FAILED = "R5D_EARLY_RUN_FAILED"
DECISION_EXECUTION_FAILED = "R5D_EXECUTION_FAILED"
DECISION_SCIENTIFIC_FAILED = "R5D_SCIENTIFIC_VALIDITY_FAILED"

TERMINAL_STATUSES = {
    "SUCCEEDED",
    "BACKEND_FAILED",
    "BACKEND_INCOMPLETE",
}

OUTPUT_KEYS = (
    "execution_manifest",
    "request_execution_status",
    "dependency_validation",
    "output_validity",
    "metrics_by_condition",
    "generation_results",
    "conditional_valid_metrics",
    "dataset_model_summary",
    "gap_resolution",
    "runtime_provenance",
    "validation",
    "scope_summary",
)

ORIGINAL_R5_NAMESPACE = "results/recovery/r5/"
R5B_CORRECTED_NAMESPACE = "results/recovery/r5b_corrected/"
R5C_ORIGINAL_NAMESPACE = "results/recovery/r5c_history_length/"
R5C_CORRECTED_NAMESPACE = "results/recovery/r5c_history_length_corrected/"
R5D_FAILED_V1_NAMESPACE = "results/recovery/r5d_generation_involved/"


class R5DError(RuntimeError):
    """Raised when the R5D protocol, dependency, or gate contract fails."""


@dataclass(frozen=True)
class R5DBuildResult:
    decision: str
    frames: dict[str, pd.DataFrame]
    report: str
    manifest_sha256: str
    protocol_hash: str
    executed_this_run: int
    full_run_authorized: bool
    blocked_reason: str


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def normalize_rel(path: str | Path) -> str:
    return Path(path).as_posix().replace("\\", "/")


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def output_path(config: Mapping[str, Any], key: str) -> Path:
    return Path((config.get("outputs") or {})[key])


def configured_outputs(config: Mapping[str, Any]) -> dict[str, Path]:
    return {key: output_path(config, key) for key in OUTPUT_KEYS}


def diagnostics_dir(project_root: Path, config: Mapping[str, Any]) -> Path:
    return project_path(project_root, (config.get("outputs") or {})["diagnostics_dir"])


def result_namespace(project_root: Path, config: Mapping[str, Any]) -> Path:
    return project_path(project_root, (config.get("outputs") or {})["result_namespace"])


def expected_k(config: Mapping[str, Any]) -> int:
    return int((config.get("scope") or {}).get("expected_k", 10))


def metric_k(config: Mapping[str, Any]) -> int:
    return int((config.get("scope") or {}).get("metric_k", 5))


def row_expected_count(row: Mapping[str, Any]) -> int:
    if str(row.get("task", "")) == "profile_generation":
        return 1
    schema = str(row.get("expected_output_schema", "")).strip()
    try:
        parsed = json.loads(schema)
    except json.JSONDecodeError:
        return 10
    values = parsed.get("ranked_candidate_indices") if isinstance(parsed, dict) else None
    return len(values) if isinstance(values, list) and values else 10


def corrected_profile_instruction() -> str:
    return (
        "Generate a concise user preference profile from the train-only history. "
        "The profile must be one short natural-language paragraph of at most 80 words. "
        "Return only valid JSON with exactly one key user_profile whose value is a non-empty plain text string. "
        "The string must not contain JSON, arrays, objects, numbered lists, or bullet lists. "
        "Do not include rationale, Markdown, prose outside JSON, or extra keys."
    )


def corrected_ranking_instruction(expected_count: int = 10, candidate_count: int = 20) -> str:
    return (
        "Use only the train-history, profile, and candidate context below. "
        f"Return exactly {expected_count} 1-based candidate indices. "
        f"Valid outputs are candidate POSITIONS 1..{candidate_count}, not dataset item IDs. "
        "Use only integer indices corresponding to positions in the candidate list supplied in this prompt. "
        "Do not return item IDs. "
        f"Return only valid JSON with key ranked_candidate_indices containing EXACTLY {expected_count} "
        f"unique integers ordered from rank 1 to rank {expected_count}: "
        '{"ranked_candidate_indices":[2,5,1,7,4,9,3,8,6,10]}. '
        f"Do not return more than {expected_count} indices. "
        f"Do not return fewer than {expected_count} indices. "
        "Do not include rationale, explanation, prose, Markdown, or extra keys."
    )


def render_profile_prompt(old_prompt: str) -> str:
    old = "Generate a concise user preference profile from the train-only history. Return JSON with key user_profile."
    instruction = corrected_profile_instruction()
    if old in old_prompt:
        return old_prompt.replace(old, instruction, 1)
    marker = "\n\nUser history:\n"
    if marker not in old_prompt:
        raise R5DError("cannot locate profile-generation history block")
    head, tail = old_prompt.split(marker, 1)
    head_lines = head.rstrip().splitlines()
    if len(head_lines) < 2:
        raise R5DError("unexpected profile-generation prompt header")
    return "\n".join(head_lines[:2] + [instruction]) + marker + tail


def render_ranking_template(old_prompt: str, expected_count: int = 10, candidate_count: int = 20) -> str:
    old = (
        "Use only the train-history and candidate context below. "
        "Return JSON with key ranked_candidate_indices containing unique 1-based candidate indices."
    )
    instruction = corrected_ranking_instruction(expected_count, candidate_count)
    if old in old_prompt:
        return old_prompt.replace(old, instruction, 1)
    marker = "\nProfile context:\n"
    if marker not in old_prompt:
        marker = "\nUser history:\n"
    if marker not in old_prompt:
        raise R5DError("cannot locate R5D ranking prompt context block")
    head, tail = old_prompt.split(marker, 1)
    head_lines = head.rstrip().splitlines()
    if len(head_lines) < 3:
        raise R5DError("unexpected R5D ranking prompt header")
    return "\n".join(head_lines[:3] + [instruction]) + marker + tail


def r5d_request_id(row: Mapping[str, Any]) -> str:
    task = str(row.get("task", ""))
    prefix = "R5D-PROF" if task == "profile_generation" else "R5D-RANK"
    suffix = sha256_text(f"{PHASE}|{row.get('request_id', '')}")[:16]
    return f"{prefix}-{suffix}"


def r5d_request_key(row: Mapping[str, Any]) -> str:
    fields = [
        "phase",
        "r5a_request_id",
        "recovery_family",
        "dataset",
        "model_name",
        "task",
        "experimental_condition",
        "history_length",
        "candidate_list_hash",
        "history_hash",
        "profile_hash",
        "r5a_profile_dependency_request_id",
        "r5d_profile_dependency_request_id",
        "prompt_template_id",
        "parser_contract",
        "expected_output_schema",
        "prompt_hash",
    ]
    return canonical_json({field: row.get(field, "") for field in fields})


def r5d_material_context_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "phase": PHASE,
                "r5a_request_id": row.get("r5a_request_id", ""),
                "dataset": row.get("dataset", ""),
                "user_id": row.get("user_id", ""),
                "model_name": row.get("model_name", ""),
                "task": row.get("task", ""),
                "experimental_condition": row.get("experimental_condition", ""),
                "history_hash": row.get("history_hash", ""),
                "candidate_list_hash": row.get("candidate_list_hash", ""),
                "unordered_candidate_item_set_hash": row.get("unordered_candidate_item_set_hash", ""),
                "positive_item_id": row.get("positive_item_id", ""),
                "profile_hash": row.get("profile_hash", ""),
                "r5a_profile_dependency_request_id": row.get("r5a_profile_dependency_request_id", ""),
                "r5d_profile_dependency_request_id": row.get("r5d_profile_dependency_request_id", ""),
                "prompt_hash": row.get("prompt_hash", ""),
                "parser_contract": row.get("parser_contract", ""),
                "expected_output_schema": row.get("expected_output_schema", ""),
            }
        )
    )


def candidate_count_int(row: Mapping[str, Any]) -> int:
    value = str(row.get("candidate_count", "0")).strip()
    return int(value) if value.isdigit() else 0


def load_r5d_manifest(
    project_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    frozen = config.get("frozen_r5a") or {}
    path = project_path(project_root, frozen["manifest"])
    if not path.exists():
        raise FileNotFoundError(path)
    manifest_sha = sha256_file(path)
    expected_sha = str(frozen.get("manifest_sha256", EXPECTED_R5A_MANIFEST_SHA256)).upper()
    if manifest_sha != expected_sha:
        raise R5DError(f"R5A manifest SHA mismatch observed={manifest_sha} expected={expected_sha}")
    manifest = read_csv(path)
    expected_rows = int(frozen.get("manifest_rows", 4800))
    if len(manifest) != expected_rows:
        raise R5DError(f"R5A manifest row count changed: {len(manifest)} != {expected_rows}")

    scope = config.get("scope") or {}
    family = str(scope.get("recovery_family", "GENERATION_INVOLVED_RECOVERY"))
    generation = manifest.loc[manifest["recovery_family"].eq(family)].copy()
    selected_rows = int(scope.get("selected_rows", 1200))
    if len(generation) != selected_rows:
        raise R5DError(f"R5D subset row count changed: {len(generation)} != {selected_rows}")
    if not generation["request_id"].is_unique:
        raise R5DError("R5D source request IDs are not unique")
    if generation["recovery_family"].isin(set(scope.get("forbidden_recovery_families") or [])).any():
        raise R5DError("R5D subset contains forbidden recovery families")
    if set(generation["dataset"]) != set(scope.get("datasets") or []):
        raise R5DError(f"R5D datasets changed: {sorted(set(generation['dataset']))}")
    if set(generation["model_name"]) != set(scope.get("models") or []):
        raise R5DError(f"R5D models changed: {sorted(set(generation['model_name']))}")

    profile_rows = generation.loc[generation["task"].eq("profile_generation")].copy()
    ranking_rows = generation.loc[generation["task"].eq("ranking")].copy()
    if len(profile_rows) != int(scope.get("profile_generation_rows", 400)):
        raise R5DError("R5D profile-generation row count mismatch")
    if len(ranking_rows) != int(scope.get("downstream_recommendation_rows", 800)):
        raise R5DError("R5D downstream ranking row count mismatch")

    old_to_new_profile = {
        str(row["request_id"]): r5d_request_id(row)
        for row in profile_rows.to_dict("records")
    }
    old_to_new_profile_path = {
        old_id: normalize_rel(Path((config.get("outputs") or {})["result_namespace"]) / f"{new_id}.json")
        for old_id, new_id in old_to_new_profile.items()
    }

    rows: list[dict[str, Any]] = []
    for original in generation.to_dict("records"):
        row = dict(original)
        original_request_id = str(row["request_id"])
        row["phase"] = PHASE
        row["r5a_request_id"] = original_request_id
        row["original_output_destination"] = str(row["output_destination"])
        row["r5a_profile_dependency_request_id"] = str(row.get("profile_dependency_request_id", ""))
        row["r5a_profile_artifact_path"] = str(row.get("profile_artifact_path", ""))
        row["r5d_profile_dependency_request_id"] = ""
        row["r5d_profile_artifact_path"] = ""
        row["request_id"] = r5d_request_id(original)
        if str(row["task"]) == "profile_generation":
            row["prompt_template_id"] = PROFILE_PROMPT_TEMPLATE_ID
            row["parser_contract"] = PROFILE_PARSER_CONTRACT
            row["expected_output_schema"] = canonical_json({"user_profile": "string"})
            row["prompt_render"] = render_profile_prompt(str(row["prompt_render"]))
        elif str(row["task"]) == "ranking":
            old_dep = str(row.get("r5a_profile_dependency_request_id", ""))
            if old_dep in old_to_new_profile:
                row["r5d_profile_dependency_request_id"] = old_to_new_profile[old_dep]
                row["r5d_profile_artifact_path"] = old_to_new_profile_path[old_dep]
                row["profile_artifact_path"] = row["r5d_profile_artifact_path"]
                row["profile_hash_status"] = "R5D_PROFILE_DEPENDENCY_PENDING_OUTPUT"
                row["profile_reuse_status"] = "R5D_PROFILE_GENERATION_REQUIRED_BEFORE_DOWNSTREAM"
            row["prompt_template_id"] = RANKING_PROMPT_TEMPLATE_ID
            row["parser_contract"] = RANKING_PARSER_CONTRACT
            row["expected_output_schema"] = canonical_json(
                {"ranked_candidate_indices": list(range(1, int(scope.get("expected_k", 10)) + 1))}
            )
            row["prompt_render"] = render_ranking_template(
                str(row["prompt_render"]),
                int(scope.get("expected_k", 10)),
                int(row["candidate_count"]),
            )
        else:
            raise R5DError(f"unexpected R5D task {row['task']}")
        row["prompt_hash"] = sha256_text(str(row["prompt_render"]))
        row["output_destination"] = normalize_rel(
            Path((config.get("outputs") or {})["result_namespace"]) / f"{row['request_id']}.json"
        )
        row["request_key_sha256"] = sha256_text(r5d_request_key(row))
        row["material_context_hash"] = r5d_material_context_hash(row)
        rows.append(row)

    frame = pd.DataFrame(rows).reset_index(drop=True)
    namespace = result_namespace(project_root, config).resolve()
    if not frame["request_id"].is_unique:
        raise R5DError("R5D request IDs are not unique")
    if not frame["output_destination"].is_unique:
        raise R5DError("R5D output destinations are not unique")
    forbidden_prefixes = [
        ORIGINAL_R5_NAMESPACE,
        R5B_CORRECTED_NAMESPACE,
        R5C_ORIGINAL_NAMESPACE,
        R5C_CORRECTED_NAMESPACE,
        R5D_FAILED_V1_NAMESPACE,
    ]
    for destination in frame["output_destination"].astype(str):
        destination_path = project_path(project_root, destination)
        if namespace not in [destination_path.resolve(), *destination_path.resolve().parents]:
            raise R5DError(f"R5D output path outside namespace: {destination}")
        normalized = normalize_rel(destination)
        if any(normalized.startswith(prefix) for prefix in forbidden_prefixes):
            raise R5DError(f"R5D output path overlaps protected namespace: {destination}")

    return manifest, frame, manifest_sha


def profile_text_from_payload(payload: Mapping[str, Any]) -> str:
    profile = payload.get("user_profile")
    if isinstance(profile, str):
        return profile.strip()
    parsed = payload.get("parse")
    if isinstance(parsed, Mapping):
        value = parsed.get("user_profile")
        if isinstance(value, str):
            return value.strip()
    return ""


def profile_hash_from_text(profile: str) -> str:
    return sha256_text(str(profile).strip())


def parse_profile_response(response_text: str) -> dict[str, Any]:
    text = str(response_text or "").strip()
    if not text:
        return {
            "parse_valid": False,
            "parse_status": "empty_output",
            "parse_error": "empty output",
            "validity_class": "PARSE_FAILURE",
            "user_profile": "",
            "profile_hash": "",
            "output_length": 0,
            "expected_output_count": 1,
            "duplicate_count": 0,
            "unknown_item_count": 0,
            "positive_rank": 0,
            "hr_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "metric_ready": False,
        }
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "parse_valid": False,
            "parse_status": "parse_failed",
            "parse_error": str(exc),
            "validity_class": "PARSE_FAILURE",
            "user_profile": "",
            "profile_hash": "",
            "output_length": 0,
            "expected_output_count": 1,
            "duplicate_count": 0,
            "unknown_item_count": 0,
            "positive_rank": 0,
            "hr_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "metric_ready": False,
        }
    if not isinstance(parsed, dict):
        status = "invalid_schema"
        error = "top-level JSON must be an object"
        profile = ""
    elif set(parsed.keys()) != {"user_profile"}:
        status = "invalid_schema"
        error = "profile JSON must contain exactly one key user_profile"
        profile = str(parsed.get("user_profile", "")).strip() if isinstance(parsed.get("user_profile"), str) else ""
    elif not isinstance(parsed.get("user_profile"), str):
        status = "invalid_schema"
        error = "user_profile must be a string"
        profile = ""
    else:
        profile = str(parsed["user_profile"]).strip()
        status = "parsed_valid" if profile else "empty_profile"
        error = "" if profile else "user_profile is empty"
    valid = status == "parsed_valid"
    return {
        "parse_valid": valid,
        "parse_status": status,
        "parse_error": error,
        "validity_class": "VALID" if valid else ("PARSE_FAILURE" if status in {"parse_failed", "empty_output"} else "OTHER"),
        "user_profile": profile if valid else "",
        "profile_hash": profile_hash_from_text(profile) if valid else "",
        "output_length": 1 if valid else 0,
        "expected_output_count": 1,
        "duplicate_count": 0,
        "unknown_item_count": 0,
        "positive_rank": 0,
        "hr_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "metric_ready": False,
    }


def validity_class_from_parse(status: str) -> str:
    if status == "parsed_valid":
        return "VALID"
    if status == "invalid_length":
        return "WRONG_LENGTH"
    if status in {"parse_failed", "empty_output"}:
        return "PARSE_FAILURE"
    if status == "duplicate_indices":
        return "DUPLICATE"
    if status == "unknown_indices":
        return "OUT_OF_RANGE"
    return "OTHER"


def parse_ranking_response(row: Mapping[str, Any], response_text: str) -> dict[str, Any]:
    candidate_ids = candidate_item_ids(row)
    expected_count = row_expected_count(row)
    parsed = parse_strict_index_response(
        response_text,
        candidate_items=candidate_ids,
        expected_count=expected_count,
        require_full_permutation=False,
    )
    rank = positive_rank(row, parsed.output_items) if parsed.parse_valid else 0
    return {
        "parse_valid": bool(parsed.parse_valid),
        "parse_status": parsed.status,
        "parse_error": parsed.error,
        "validity_class": "VALID" if parsed.parse_valid else validity_class_from_parse(parsed.status),
        "ranked_candidate_indices": list(parsed.output_indices),
        "ranked_item_ids": list(parsed.output_items),
        "duplicate_count": int(parsed.duplicate_count),
        "unknown_item_count": int(parsed.unknown_item_count),
        "output_length": int(parsed.output_length),
        "expected_output_count": expected_count,
        "positive_rank": rank,
        "hr_at_5": hr_at_5(rank) if parsed.parse_valid else 0.0,
        "ndcg_at_5": ndcg_at_5(rank) if parsed.parse_valid else 0.0,
        "metric_ready": bool(parsed.parse_valid),
        "profile_hash": str(row.get("materialized_profile_hash", row.get("profile_hash", ""))),
        "user_profile": "",
    }


def incomplete_parse_record(row: Mapping[str, Any], response_text: str) -> dict[str, Any]:
    return {
        "parse_valid": False,
        "parse_status": "backend_incomplete",
        "parse_error": "Ollama generation did not report done=true",
        "validity_class": "BACKEND_INCOMPLETE",
        "ranked_candidate_indices": [],
        "ranked_item_ids": [],
        "duplicate_count": 0,
        "unknown_item_count": 0,
        "output_length": 0,
        "expected_output_count": row_expected_count(row),
        "positive_rank": 0,
        "hr_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "metric_ready": False,
        "profile_hash": str(row.get("materialized_profile_hash", row.get("profile_hash", ""))),
        "user_profile": "",
        "diagnostic_raw_response_length": len(str(response_text or "")),
    }


def parse_output_record(row: Mapping[str, Any], response_text: str) -> dict[str, Any]:
    if str(row.get("task", "")) == "profile_generation":
        return parse_profile_response(response_text)
    return parse_ranking_response(row, response_text)


def materialized_profile_from_dependency(
    project_root: Path,
    row: Mapping[str, Any],
) -> tuple[str, str, str]:
    dep_id = str(row.get("r5d_profile_dependency_request_id", ""))
    dep_path = str(row.get("r5d_profile_artifact_path", ""))
    if not dep_id:
        return str(row.get("profile_hash", "")), str(row.get("profile_hash_status", "")), ""
    path = project_path(project_root, dep_path)
    if not path.exists():
        raise R5DError(f"downstream R5D dependency is missing: {dep_id} at {dep_path}")
    payload = load_json(path)
    if str(payload.get("execution_status", "")) != "SUCCEEDED" or bool(payload.get("parse_valid")) is not True:
        raise R5DError(f"downstream R5D dependency is not valid: {dep_id}")
    profile = profile_text_from_payload(payload)
    if not profile:
        raise R5DError(f"downstream R5D dependency has empty profile: {dep_id}")
    observed_hash = profile_hash_from_text(profile)
    payload_hash = str(payload.get("profile_hash", ""))
    if payload_hash and payload_hash != observed_hash:
        raise R5DError(f"profile hash mismatch for dependency {dep_id}")
    return observed_hash, "R5D_PROFILE_DEPENDENCY_MATERIALIZED", profile


def materialize_execution_row(
    project_root: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    materialized = dict(row)
    materialized["template_prompt_hash"] = str(row.get("prompt_hash", ""))
    if str(row.get("task", "")) == "ranking" and str(row.get("r5d_profile_dependency_request_id", "")):
        profile_hash, status, profile = materialized_profile_from_dependency(project_root, row)
        old_context = f"PENDING_PROFILE_GENERATION:{row.get('r5a_profile_dependency_request_id', '')}"
        new_context = (
            f"Generated profile dependency {row.get('r5d_profile_dependency_request_id', '')}; "
            f"profile_hash={profile_hash}\n{profile}"
        )
        prompt = str(row["prompt_render"])
        if old_context not in prompt:
            raise R5DError(f"cannot locate pending profile placeholder in downstream prompt {row.get('request_id')}")
        prompt = prompt.replace(old_context, new_context, 1)
        materialized["prompt_render"] = prompt
        materialized["prompt_hash"] = sha256_text(prompt)
        materialized["materialized_profile_hash"] = profile_hash
        materialized["profile_hash_status"] = status
        materialized["profile_materialized"] = "true"
    else:
        materialized["profile_materialized"] = "false"
        materialized["materialized_profile_hash"] = str(row.get("profile_hash", ""))
    return materialized


def build_result_payload(
    row: Mapping[str, Any],
    manifest_sha: str,
    protocol_hash: str,
    runtime: r5c.R5COllamaResult,
    started_at: str,
    ended_at: str,
    gate_stage: str,
) -> dict[str, Any]:
    if not runtime.response_received:
        parse_record = incomplete_parse_record(row, "")
        parse_record["validity_class"] = "BACKEND_FAILED"
        parse_record["parse_status"] = "backend_failed"
        status = "BACKEND_FAILED"
        error_category = runtime.error_category or "backend_failed"
    elif not runtime.backend_complete:
        parse_record = incomplete_parse_record(row, runtime.response_text)
        status = "BACKEND_INCOMPLETE"
        error_category = runtime.error_category or "backend_incomplete"
    else:
        parse_record = parse_output_record(row, runtime.response_text)
        status = "SUCCEEDED"
        error_category = "" if parse_record["parse_valid"] else str(parse_record["validity_class"]).lower()

    payload: dict[str, Any] = {
        "phase": PHASE,
        "gate_stage": gate_stage,
        "request_id": str(row["request_id"]),
        "r5a_request_id": str(row["r5a_request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": protocol_hash,
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "model_label": str(row.get("model_label", "")),
        "user_id": str(row["user_id"]),
        "task": str(row["task"]),
        "recovery_family": str(row["recovery_family"]),
        "experimental_condition": str(row["experimental_condition"]),
        "history_length": str(row.get("history_length", "")),
        "history_item_count": str(row.get("history_item_count", "")),
        "history_source": str(row.get("history_source", "")),
        "prompt_template_id": str(row["prompt_template_id"]),
        "template_prompt_hash": str(row.get("template_prompt_hash", row.get("prompt_hash", ""))),
        "prompt_hash": str(row["prompt_hash"]),
        "candidate_list_hash": str(row.get("candidate_list_hash", "")),
        "unordered_candidate_item_set_hash": str(row.get("unordered_candidate_item_set_hash", "")),
        "history_hash": str(row["history_hash"]),
        "material_context_hash": str(row["material_context_hash"]),
        "request_key_sha256": str(row["request_key_sha256"]),
        "positive_item_id": str(row.get("positive_item_id", "")),
        "positive_candidate_position": str(row.get("positive_candidate_position", "")),
        "candidate_count": candidate_count_int(row),
        "parser_contract": str(row["parser_contract"]),
        "expected_output_schema": str(row["expected_output_schema"]),
        "r5a_profile_dependency_request_id": str(row.get("r5a_profile_dependency_request_id", "")),
        "r5d_profile_dependency_request_id": str(row.get("r5d_profile_dependency_request_id", "")),
        "r5d_profile_artifact_path": str(row.get("r5d_profile_artifact_path", "")),
        "profile_hash_status": str(row.get("profile_hash_status", "")),
        "profile_materialized": str(row.get("profile_materialized", "false")),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "runtime_backend": "ollama",
        "runtime_model_id": runtime.runtime_model_id,
        "backend_complete": runtime.backend_complete,
        "ollama_done_reason": runtime.done_reason,
        "temperature": str(row.get("temperature", "0")),
        "seed": str(row.get("seed", "0")),
        "max_new_tokens": str(row.get("max_new_tokens", "512")),
        "retry_count": 0,
        "latency_seconds": round(runtime.latency_seconds, 6),
        "execution_status": status,
        "error_category": error_category,
        "error": runtime.error,
        "raw_response_text": runtime.response_text,
        "ollama_raw_response": runtime.raw_response or {},
        "evidence_class": EVIDENCE_CLASS,
        "global_status": GLOBAL_STATUS,
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
        **parse_record,
    }
    payload["metric_ready"] = bool(payload["execution_status"] == "SUCCEEDED" and payload["parse_valid"] and str(row["task"]) == "ranking")
    payload["result_hash"] = sha256_text(canonical_json({k: v for k, v in payload.items() if k != "result_hash"}))
    return payload


def validate_existing_result(
    source_row: Mapping[str, Any],
    payload: Mapping[str, Any],
    manifest_sha: str,
    protocol_hash: str,
    path: Path,
    *,
    project_root: Path,
) -> None:
    materialized = materialize_execution_row(project_root, source_row)
    required_equal = {
        "phase": PHASE,
        "request_id": str(source_row["request_id"]),
        "r5a_request_id": str(source_row["r5a_request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": protocol_hash,
        "dataset": str(source_row["dataset"]),
        "model_name": str(source_row["model_name"]),
        "user_id": str(source_row["user_id"]),
        "task": str(source_row["task"]),
        "experimental_condition": str(source_row["experimental_condition"]),
        "history_length": str(source_row.get("history_length", "")),
        "template_prompt_hash": str(source_row.get("prompt_hash", "")),
        "prompt_hash": str(materialized.get("prompt_hash", "")),
        "candidate_list_hash": str(source_row.get("candidate_list_hash", "")),
        "history_hash": str(source_row["history_hash"]),
        "material_context_hash": str(source_row["material_context_hash"]),
    }
    for key, expected in required_equal.items():
        observed = str(payload.get(key, ""))
        if observed != str(expected):
            raise R5DError(
                f"existing R5D output identity mismatch at {path}: {key} observed={observed} expected={expected}"
            )
    if str(payload.get("execution_status", "")) not in TERMINAL_STATUSES:
        raise R5DError(f"existing R5D output is non-terminal at {path}")
    if str(payload.get("execution_status", "")) == "SUCCEEDED" and bool(payload.get("backend_complete")) is not True:
        raise R5DError(f"existing R5D output has SUCCEEDED without backend_complete=true at {path}")


def execution_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str, str, str]:
    task_rank = 0 if str(row.get("task", "")) == "profile_generation" else 1
    return (
        task_rank,
        str(row.get("model_name", "")),
        str(row.get("dataset", "")),
        str(row.get("user_id", "")),
        str(row.get("request_id", "")),
    )


def execute_selected_requests(
    project_root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifest_sha: str,
    protocol_hash: str,
    *,
    allow_api_calls: bool,
    allow_local_ollama: bool,
    gate_stage: str,
) -> int:
    if not allow_api_calls:
        raise R5DError("R5D execution requires --allow-api-calls")
    if not allow_local_ollama:
        raise R5DError("R5D execution requires --allow-local-ollama")
    namespace = result_namespace(project_root, config).resolve()
    executed = 0
    for source_row in sorted([dict(row) for row in rows], key=execution_sort_key):
        destination = project_path(project_root, str(source_row["output_destination"]))
        if namespace not in [destination.resolve(), *destination.resolve().parents]:
            raise R5DError(f"R5D output destination outside namespace: {destination}")
        rel_destination = rel_path(project_root, destination).replace("\\", "/")
        if any(
            rel_destination.startswith(prefix)
            for prefix in [
                ORIGINAL_R5_NAMESPACE,
                R5B_CORRECTED_NAMESPACE,
                R5C_ORIGINAL_NAMESPACE,
                R5C_CORRECTED_NAMESPACE,
                R5D_FAILED_V1_NAMESPACE,
            ]
        ):
            raise R5DError(f"refusing to write protected namespace: {destination}")
        if destination.exists():
            validate_existing_result(
                source_row,
                load_json(destination),
                manifest_sha,
                protocol_hash,
                destination,
                project_root=project_root,
            )
            continue
        row = materialize_execution_row(project_root, source_row)
        started_at = r5c.utc_now()
        runtime = r5c.generate_local_ollama(row, config)
        ended_at = r5c.utc_now()
        payload = build_result_payload(row, manifest_sha, protocol_hash, runtime, started_at, ended_at, gate_stage)
        write_json_atomic(payload, destination)
        executed += 1
    return executed


def status_for_missing(row: Mapping[str, Any], manifest_sha: str) -> dict[str, Any]:
    return {
        "request_id": str(row["request_id"]),
        "r5a_request_id": str(row["r5a_request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": "",
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "user_id": str(row["user_id"]),
        "task": str(row["task"]),
        "experimental_condition": str(row["experimental_condition"]),
        "history_length": str(row.get("history_length", "")),
        "history_item_count": str(row.get("history_item_count", "")),
        "output_destination": str(row["output_destination"]),
        "result_file_exists": "false",
        "execution_status": "NOT_EXECUTED",
        "terminal_accounting_status": "NOT_EXECUTED",
        "terminal_status": "false",
        "backend_complete": "false",
        "parse_valid": "false",
        "valid_output": "false",
        "metric_ready": "false",
        "parse_status": "",
        "validity_class": "NOT_EXECUTED",
        "positive_rank": 0,
        "hr_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "output_length": 0,
        "expected_output_count": row_expected_count(row),
        "duplicate_count": 0,
        "unknown_item_count": 0,
        "error_category": "",
        "error": "",
        "retry_count": 0,
        "result_hash": "",
        "result_file_sha256": "",
        "latency_seconds": "",
        "started_at_utc": "",
        "ended_at_utc": "",
        "runtime_model_id": "",
        "gate_stage": "",
        "ollama_done_reason": "",
        "template_prompt_hash": str(row.get("prompt_hash", "")),
        "prompt_hash": "",
        "r5a_profile_dependency_request_id": str(row.get("r5a_profile_dependency_request_id", "")),
        "r5d_profile_dependency_request_id": str(row.get("r5d_profile_dependency_request_id", "")),
        "profile_hash": str(row.get("profile_hash", "")),
        "profile_hash_status": str(row.get("profile_hash_status", "")),
        "profile_materialized": "false",
        "ranked_candidate_indices_json": "[]",
        "ranked_item_ids_json": "[]",
        "values_within_1_to_n": "false",
        "dataset_item_id_substitution_count": 0,
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
    }


def parsed_values_from_payload(payload: Mapping[str, Any]) -> list[int]:
    values = payload.get("ranked_candidate_indices", [])
    if isinstance(values, list) and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return [int(value) for value in values]
    return []


def status_for_payload(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    manifest_sha: str,
    protocol_hash: str,
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    validate_existing_result(row, payload, manifest_sha, protocol_hash, path, project_root=project_root)
    execution_status = str(payload.get("execution_status", ""))
    parse_valid = bool(payload.get("parse_valid", False))
    metric_ready = bool(execution_status == "SUCCEEDED" and parse_valid and str(row["task"]) == "ranking")
    if execution_status == "SUCCEEDED":
        terminal_accounting = "COMPLETED_VALID" if parse_valid else "COMPLETED_INVALID_OUTPUT"
    else:
        terminal_accounting = execution_status
    values = parsed_values_from_payload(payload)
    count = candidate_count_int(row)
    item_ids = {str(value) for value in candidate_item_ids(row)}
    item_id_substitutions = [
        value
        for value in values
        if count and not 1 <= value <= count and str(value) in item_ids
    ]
    return {
        "request_id": str(row["request_id"]),
        "r5a_request_id": str(row["r5a_request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": str(payload.get("protocol_hash", "")),
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "user_id": str(row["user_id"]),
        "task": str(row["task"]),
        "experimental_condition": str(row["experimental_condition"]),
        "history_length": str(row.get("history_length", "")),
        "history_item_count": str(row.get("history_item_count", "")),
        "output_destination": str(row["output_destination"]),
        "result_file_exists": "true",
        "execution_status": execution_status,
        "terminal_accounting_status": terminal_accounting,
        "terminal_status": "true",
        "backend_complete": bool_text(bool(payload.get("backend_complete", False))),
        "parse_valid": bool_text(parse_valid),
        "valid_output": bool_text(parse_valid),
        "metric_ready": bool_text(metric_ready),
        "parse_status": str(payload.get("parse_status", "")),
        "validity_class": str(payload.get("validity_class", "")),
        "positive_rank": int(payload.get("positive_rank", 0) or 0),
        "hr_at_5": float(payload.get("hr_at_5", 0.0) or 0.0) if metric_ready else 0.0,
        "ndcg_at_5": float(payload.get("ndcg_at_5", 0.0) or 0.0) if metric_ready else 0.0,
        "output_length": int(payload.get("output_length", 0) or 0),
        "expected_output_count": int(payload.get("expected_output_count", row_expected_count(row)) or row_expected_count(row)),
        "duplicate_count": int(payload.get("duplicate_count", 0) or 0),
        "unknown_item_count": int(payload.get("unknown_item_count", 0) or 0),
        "error_category": str(payload.get("error_category", "")),
        "error": str(payload.get("error", "")),
        "retry_count": int(payload.get("retry_count", 0) or 0),
        "result_hash": str(payload.get("result_hash", "")),
        "result_file_sha256": sha256_file(path),
        "latency_seconds": payload.get("latency_seconds", ""),
        "started_at_utc": str(payload.get("started_at_utc", "")),
        "ended_at_utc": str(payload.get("ended_at_utc", "")),
        "runtime_model_id": str(payload.get("runtime_model_id", "")),
        "gate_stage": str(payload.get("gate_stage", "")),
        "ollama_done_reason": str(payload.get("ollama_done_reason", "")),
        "template_prompt_hash": str(payload.get("template_prompt_hash", "")),
        "prompt_hash": str(payload.get("prompt_hash", "")),
        "r5a_profile_dependency_request_id": str(row.get("r5a_profile_dependency_request_id", "")),
        "r5d_profile_dependency_request_id": str(row.get("r5d_profile_dependency_request_id", "")),
        "profile_hash": str(payload.get("profile_hash", "")),
        "profile_hash_status": str(payload.get("profile_hash_status", "")),
        "profile_materialized": str(payload.get("profile_materialized", "false")),
        "ranked_candidate_indices_json": canonical_json(values),
        "ranked_item_ids_json": canonical_json(payload.get("ranked_item_ids", [])),
        "values_within_1_to_n": bool_text(bool(values) and all(1 <= value <= count for value in values)) if str(row["task"]) == "ranking" else "not_applicable",
        "dataset_item_id_substitution_count": len(item_id_substitutions),
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
    }


def build_status(
    project_root: Path,
    frame: pd.DataFrame,
    manifest_sha: str,
    protocol_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        path = project_path(project_root, str(row["output_destination"]))
        if path.exists():
            rows.append(
                status_for_payload(
                    row,
                    load_json(path),
                    manifest_sha,
                    protocol_hash,
                    path,
                    project_root=project_root,
                )
            )
        else:
            rows.append(status_for_missing(row, manifest_sha))
    return pd.DataFrame(rows)


def build_dependency_validation(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    profile = frame.loc[frame["task"].eq("profile_generation")]
    profile_by_old = {
        str(row["r5a_request_id"]): row
        for row in profile.to_dict("records")
    }
    profile_by_new = {
        str(row["request_id"]): row
        for row in profile.to_dict("records")
    }
    for row in frame.loc[frame["task"].eq("ranking")].to_dict("records"):
        old_dep = str(row.get("r5a_profile_dependency_request_id", ""))
        new_dep = str(row.get("r5d_profile_dependency_request_id", ""))
        dataset = str(row["dataset"])
        if dataset in {"ML-1M", "LastFM"}:
            valid = bool(old_dep in profile_by_old and new_dep in profile_by_new)
            dep_status = "R5D_GENERATED_PROFILE_REQUIRED"
        else:
            valid = bool(str(row.get("profile_hash_status", "")) == "INPUT_PROFILE_HASH_AVAILABLE" and str(row.get("profile_hash", "")))
            dep_status = "EXISTING_F6L_PROFILE_REUSED"
        rows.append(
            {
                "request_id": row["request_id"],
                "r5a_request_id": row["r5a_request_id"],
                "dataset": dataset,
                "model_name": row["model_name"],
                "user_id": row["user_id"],
                "dependency_status": dep_status,
                "r5a_profile_dependency_request_id": old_dep,
                "r5d_profile_dependency_request_id": new_dep,
                "dependency_valid": bool_text(valid),
            }
        )
    return pd.DataFrame(rows)


def build_scope_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "scope": "purpose",
            "key": "R5D",
            "count": len(frame),
            "notes": "Generation-involved profile recommendation recovery from frozen R5A manifest.",
        }
    ]
    for dataset, group in frame.groupby("dataset", sort=True):
        rows.append({"scope": "dataset", "key": str(dataset), "count": len(group), "notes": ""})
    for model, group in frame.groupby("model_name", sort=True):
        rows.append({"scope": "model", "key": str(model), "count": len(group), "notes": ""})
    for condition, group in frame.groupby("experimental_condition", sort=True):
        rows.append({"scope": "condition", "key": str(condition), "count": len(group), "notes": ""})
    for key, group in frame.groupby(["task", "dataset", "model_name", "experimental_condition"], sort=True):
        rows.append({"scope": "distinct_cell", "key": "|".join(map(str, key)), "count": len(group), "notes": ""})
    return pd.DataFrame(rows)


def protocol_metadata(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    manifest_sha: str,
    dependency_validation: pd.DataFrame,
    runtime_tags: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, pd.DataFrame]:
    module_path = Path(__file__).resolve()
    parser_path = project_root / "src" / "llm" / "canonical_llm_contract.py"
    manifest_builder = project_root / "src" / "analysis" / "reproduction_r5a_controlled_recovery_manifest.py"
    prompt_template_hash = sha256_text(canonical_json(sorted(frame["prompt_hash"].astype(str).unique())))
    dependency_hash = sha256_text(canonical_json(dependency_validation.fillna("").to_dict("records")))
    generation_config = {
        "endpoint": str((config.get("runtime") or {}).get("endpoint", "")),
        "format": str((config.get("runtime") or {}).get("format", "json")),
        "stream": str((config.get("runtime") or {}).get("stream", False)).lower(),
        "temperature": str((config.get("runtime") or {}).get("temperature", 0)),
        "max_retries": str((config.get("runtime") or {}).get("max_retries", 0)),
        "timeout_seconds": str((config.get("runtime") or {}).get("timeout_seconds", 120)),
        "num_predict_values": sorted(frame["max_new_tokens"].astype(str).unique()),
        "seed_values": sorted(frame["seed"].astype(str).unique()),
    }
    models = []
    for model in sorted(set(frame["model_name"].astype(str))):
        details = (runtime_tags or {}).get(model, {})
        models.append(
            {
                "model_name": model,
                "digest": str(details.get("digest", "")),
                "details": details.get("details", {}) if isinstance(details.get("details"), dict) else {},
            }
        )
    payload = {
        "phase": PHASE,
        "manifest_sha256": manifest_sha,
        "selected_rows": len(frame),
        "profile_generation_rows": int(frame["task"].eq("profile_generation").sum()),
        "downstream_recommendation_rows": int(frame["task"].eq("ranking").sum()),
        "expected_ranking_k": expected_k(config),
        "metric_k": metric_k(config),
        "profile_prompt_template_id": PROFILE_PROMPT_TEMPLATE_ID,
        "ranking_prompt_template_id": RANKING_PROMPT_TEMPLATE_ID,
        "prompt_template_family_hash": prompt_template_hash,
        "profile_parser_contract": PROFILE_PARSER_CONTRACT,
        "ranking_parser_contract": RANKING_PARSER_CONTRACT,
        "parser_hash": sha256_file(parser_path),
        "runner_hash": sha256_file(module_path),
        "config_hash": sha256_file(config_path),
        "manifest_builder_hash": sha256_file(manifest_builder),
        "metric_hash": sha256_file(module_path),
        "dependency_validation_hash": dependency_hash,
        "profile_materialization_policy": "ML-1M/LastFM downstream prompts use same-run valid R5D profile raw outputs; Beauty/Sports reuse existing F6L profile source context",
        "generation_config": generation_config,
        "models": models,
    }
    protocol_hash = sha256_text(canonical_json(payload))
    rows = [
        {"component": "manifest", "value": manifest_sha, "sha256": manifest_sha},
        {"component": "prompt_templates", "value": f"{PROFILE_PROMPT_TEMPLATE_ID};{RANKING_PROMPT_TEMPLATE_ID}", "sha256": prompt_template_hash},
        {"component": "parser", "value": f"{PROFILE_PARSER_CONTRACT};{RANKING_PARSER_CONTRACT}", "sha256": payload["parser_hash"]},
        {"component": "runner", "value": normalize_rel(module_path), "sha256": payload["runner_hash"]},
        {"component": "config", "value": normalize_rel(config_path), "sha256": payload["config_hash"]},
        {"component": "manifest_builder", "value": normalize_rel(manifest_builder), "sha256": payload["manifest_builder_hash"]},
        {"component": "dependency_validation", "value": "profile-generation dependency graph", "sha256": dependency_hash},
        {"component": "metrics", "value": "HR@5/NDCG@5 over profile-plus-history ranking rows", "sha256": payload["metric_hash"]},
        {"component": "generation_config", "value": canonical_json(generation_config), "sha256": sha256_text(canonical_json(generation_config))},
        {"component": "protocol", "value": "R5D_PROTOCOL_PREFLIGHT_READY", "sha256": protocol_hash},
    ]
    for model in models:
        rows.append({"component": "model_identifier", "value": model["model_name"], "sha256": str(model.get("digest", ""))})
    return protocol_hash, pd.DataFrame(rows)


def context_length_for_model(
    config: Mapping[str, Any],
    runtime_tags: Mapping[str, Mapping[str, Any]],
    model: str,
) -> int:
    details = runtime_tags.get(model, {}).get("details", {})
    if isinstance(details, dict):
        for key in ["context_length", "num_ctx"]:
            value = details.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return int((config.get("runtime") or {}).get("assumed_context_length", 32768))


def approximate_tokens(text: str) -> int:
    return max(1, (len(str(text)) + 3) // 4)


def prompt_for_context_check(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    prompt = str(row.get("prompt_render", ""))
    if str(row.get("task", "")) == "ranking" and str(row.get("r5d_profile_dependency_request_id", "")):
        max_chars = int(((config.get("profile_materialization") or {}).get("max_profile_chars_for_context_preflight", 4096)))
        synthetic = "X" * max_chars
        old_context = f"PENDING_PROFILE_GENERATION:{row.get('r5a_profile_dependency_request_id', '')}"
        new_context = (
            f"Generated profile dependency {row.get('r5d_profile_dependency_request_id', '')}; "
            f"profile_hash={'0' * 64}\n{synthetic}"
        )
        prompt = prompt.replace(old_context, new_context, 1)
    return prompt


def build_context_preflight(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    runtime_tags: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    runtime = config.get("runtime") or {}
    min_margin = int(runtime.get("min_context_margin_tokens", 1024))
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(["task", "model_name", "dataset", "experimental_condition"], sort=True):
        task, model_name, dataset, condition = key
        records = group.to_dict("records")
        max_prompt = max(records, key=lambda row: len(prompt_for_context_check(row, config)))
        prompt = prompt_for_context_check(max_prompt, config)
        output_budget = int(max_prompt.get("max_new_tokens", 512) or 512)
        context_length = context_length_for_model(config, runtime_tags, str(model_name))
        input_tokens = approximate_tokens(prompt)
        margin = context_length - input_tokens - output_budget
        rows.append(
            {
                "scope": "task_model_dataset_condition",
                "task": task,
                "model_name": model_name,
                "dataset": dataset,
                "experimental_condition": condition,
                "max_prompt_chars": len(prompt),
                "max_prompt_words": len(prompt.split()),
                "approx_input_tokens": input_tokens,
                "output_budget_tokens": output_budget,
                "context_length": context_length,
                "margin_tokens": margin,
                "status": "pass" if margin >= min_margin else "fail",
                "worst_case_request_id": str(max_prompt["request_id"]),
                "notes": "profile dependencies use configured synthetic max profile size before downstream profiles exist",
            }
        )
    return pd.DataFrame(rows)


def protected_artifact_integrity(project_root: Path) -> pd.DataFrame:
    inherited = protected_hash_check(project_root)
    if inherited.empty:
        return inherited
    frame = inherited.copy()
    hash_ok = frame["expected_sha256"].astype(str).str.upper().eq(frame["observed_sha256"].astype(str).str.upper())
    clean_status = frame["git_status"].astype(str).eq("")
    exists = frame["exists"].astype(str).eq("true")
    ok = exists & hash_ok & clean_status
    frame["hash_unchanged"] = [bool_text(value) for value in ok.tolist()]
    frame["protection_status"] = ["PROTECTED_UNCHANGED" if value else "PROTECTED_CONFLICT" for value in ok.tolist()]
    return frame


def build_static_preflight(
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    frame: pd.DataFrame,
    manifest_sha: str,
    dependency_validation: pd.DataFrame,
    context_preflight: pd.DataFrame,
    protocol_hash: str,
) -> pd.DataFrame:
    scope = config.get("scope") or {}
    runtime = config.get("runtime") or {}
    outputs = config.get("outputs") or {}
    namespace = normalize_rel(outputs["result_namespace"]).rstrip("/") + "/"
    new_outputs = set(frame["output_destination"].astype(str).map(normalize_rel))
    old_outputs = set(frame["original_output_destination"].astype(str).map(normalize_rel))
    prompts = frame["prompt_render"].astype(str)
    ranking = frame.loc[frame["task"].eq("ranking")]
    profile = frame.loc[frame["task"].eq("profile_generation")]
    current_head = git_stdout(["rev-parse", "HEAD"], project_root)
    original_repo_status = git_stdout(["-C", "original_repo/EvaLLMasRecommender", "status", "--short"], project_root)
    protected = protected_artifact_integrity(project_root)
    r5c_freeze_path = project_path(project_root, ((config.get("frozen_r5c_corrected") or {}).get("freeze_manifest", "")))
    r5c_freeze = load_json(r5c_freeze_path) if r5c_freeze_path.exists() else {}
    r5c_protocol = str(r5c_freeze.get("protocol_hash", "") or r5c_freeze.get("r5c_corrected_protocol_hash", ""))
    r5c_freeze_ready = bool(r5c_freeze.get("freeze_ready", False))
    cell_counts = frame.groupby(["task", "dataset", "model_name", "experimental_condition"], sort=True).size()
    out_of_scope = manifest.loc[~manifest["recovery_family"].eq("GENERATION_INVOLVED_RECOVERY")]
    ranking_expected = ranking.apply(row_expected_count, axis=1) if not ranking.empty else pd.Series(dtype=int)
    checks: list[tuple[str, str, bool, str, str]] = [
        ("STATIC-001", "manifest SHA", manifest_sha == EXPECTED_R5A_MANIFEST_SHA256, manifest_sha, EXPECTED_R5A_MANIFEST_SHA256),
        ("STATIC-002", "repository HEAD is R5C corrected freeze", current_head == str((config.get("frozen_r5c_corrected") or {}).get("freeze_commit", "")), current_head, str((config.get("frozen_r5c_corrected") or {}).get("freeze_commit", ""))),
        ("STATIC-003", "R5C corrected freeze manifest exists", r5c_freeze_path.exists(), normalize_rel(r5c_freeze_path), "exists"),
        ("STATIC-004", "R5C corrected protocol hash", r5c_protocol == str((config.get("frozen_r5c_corrected") or {}).get("protocol_hash", "")), r5c_protocol, str((config.get("frozen_r5c_corrected") or {}).get("protocol_hash", ""))),
        ("STATIC-005", "R5C corrected freeze ready", r5c_freeze_ready, str(r5c_freeze_ready).lower(), "true"),
        ("STATIC-006", "manifest rows", len(manifest) == int((config.get("frozen_r5a") or {}).get("manifest_rows", 4800)), str(len(manifest)), "4800"),
        ("STATIC-007", "selected generation rows", len(frame) == int(scope.get("selected_rows", 1200)), str(len(frame)), "1200"),
        ("STATIC-008", "out-of-scope rows remain unselected", len(out_of_scope) == 3600, str(len(out_of_scope)), "3600"),
        ("STATIC-009", "profile-generation rows", len(profile) == int(scope.get("profile_generation_rows", 400)), str(len(profile)), str(scope.get("profile_generation_rows", 400))),
        ("STATIC-010", "downstream ranking rows", len(ranking) == int(scope.get("downstream_recommendation_rows", 800)), str(len(ranking)), str(scope.get("downstream_recommendation_rows", 800))),
        ("STATIC-011", "datasets", set(frame["dataset"]) == set(scope.get("datasets") or []), str(sorted(set(frame["dataset"]))), str(sorted(scope.get("datasets") or []))),
        ("STATIC-012", "models", set(frame["model_name"]) == set(scope.get("models") or []), str(sorted(set(frame["model_name"]))), str(sorted(scope.get("models") or []))),
        ("STATIC-013", "distinct cells", len(cell_counts) == int(scope.get("expected_cells", 12)) and cell_counts.eq(100).all(), f"{len(cell_counts)} cells; min={int(cell_counts.min())}; max={int(cell_counts.max())}", "12 cells of 100"),
        ("STATIC-014", "profile datasets", set(profile["dataset"]) == set(scope.get("profile_generation_datasets") or []), str(sorted(set(profile["dataset"]))), str(sorted(scope.get("profile_generation_datasets") or []))),
        ("STATIC-015", "ranking datasets", set(ranking["dataset"]) == set(scope.get("downstream_datasets") or []), str(sorted(set(ranking["dataset"]))), str(sorted(scope.get("downstream_datasets") or []))),
        ("STATIC-016", "ranking expected K is 10", not ranking.empty and ranking_expected.eq(int(scope.get("expected_k", 10))).all(), str(sorted(set(ranking_expected.astype(int)))) if not ranking_expected.empty else "[]", "[10]"),
        ("STATIC-017", "ranking candidate count 20", ranking["candidate_count"].astype(int).eq(20).all(), str(sorted(set(ranking["candidate_count"].astype(str)))), "20"),
        ("STATIC-018", "profile candidate count 0", profile["candidate_count"].astype(int).eq(0).all(), str(sorted(set(profile["candidate_count"].astype(str)))), "0"),
        ("STATIC-019", "profile prompt strict JSON", profile["prompt_render"].astype(str).str.contains("exactly one key user_profile", regex=False).all(), str(len(profile)), str(len(profile))),
        ("STATIC-020", "ranking prompt says EXACTLY 10", ranking["prompt_render"].astype(str).str.contains("EXACTLY 10", regex=False).all(), str(int(ranking["prompt_render"].astype(str).str.contains("EXACTLY 10", regex=False).sum())), str(len(ranking))),
        ("STATIC-021", "ranking prompt states positions 1..20", ranking["prompt_render"].astype(str).str.contains("candidate POSITIONS 1..20", regex=False).all(), str(int(ranking["prompt_render"].astype(str).str.contains("candidate POSITIONS 1..20", regex=False).sum())), str(len(ranking))),
        ("STATIC-022", "ranking prompt rejects item IDs", ranking["prompt_render"].astype(str).str.contains("not dataset item IDs", regex=False).all() and ranking["prompt_render"].astype(str).str.contains("Do not return item IDs", regex=False).all(), "checked", "all ranking prompts"),
        ("STATIC-023", "parser contracts", set(profile["parser_contract"]) == {PROFILE_PARSER_CONTRACT} and set(ranking["parser_contract"]) == {RANKING_PARSER_CONTRACT}, str(sorted(set(frame["parser_contract"]))), f"{PROFILE_PARSER_CONTRACT};{RANKING_PARSER_CONTRACT}"),
        ("STATIC-024", "dependency graph valid", dependency_validation["dependency_valid"].eq("true").all(), f"{int(dependency_validation['dependency_valid'].eq('true').sum())}/{len(dependency_validation)}", f"{len(dependency_validation)}/{len(dependency_validation)}"),
        ("STATIC-025", "output namespace R5D", namespace == normalize_rel(outputs["result_namespace"]).rstrip("/") + "/", namespace, normalize_rel(outputs["result_namespace"]).rstrip("/") + "/"),
        ("STATIC-026", "no output path overlap with R5A destinations", old_outputs.isdisjoint(new_outputs), str(len(old_outputs.intersection(new_outputs))), "0"),
        ("STATIC-027", "no protected namespace overlap", not any(path.startswith((ORIGINAL_R5_NAMESPACE, R5B_CORRECTED_NAMESPACE, R5C_ORIGINAL_NAMESPACE, R5C_CORRECTED_NAMESPACE, R5D_FAILED_V1_NAMESPACE)) for path in new_outputs), "checked", "none"),
        ("STATIC-028", "local endpoint only", str(runtime.get("endpoint", "")).startswith("http://127.0.0.1") or str(runtime.get("endpoint", "")).startswith("http://localhost"), str(runtime.get("endpoint", "")), "localhost Ollama"),
        ("STATIC-029", "format json stream false", str(runtime.get("format", "")) == "json" and bool(runtime.get("stream", False)) is False, f"format={runtime.get('format')};stream={runtime.get('stream')}", "format=json;stream=false"),
        ("STATIC-030", "max_retries remains zero", int(runtime.get("max_retries", 0)) == 0, str(runtime.get("max_retries", 0)), "0"),
        ("STATIC-031", "context envelope", context_preflight["status"].eq("pass").all(), f"{int(context_preflight['status'].eq('pass').sum())}/{len(context_preflight)}", "all pass"),
        ("STATIC-032", "disk free guard", shutil.disk_usage(project_root).free >= int(runtime.get("disk_free_min_bytes", 0)), str(shutil.disk_usage(project_root).free), f">={runtime.get('disk_free_min_bytes', 0)}"),
        ("STATIC-033", "original repo clean", original_repo_status == "", original_repo_status, "empty"),
        ("STATIC-034", "protected artifacts unchanged", protected.empty or protected["protection_status"].eq("PROTECTED_UNCHANGED").all(), f"{int(protected['protection_status'].eq('PROTECTED_UNCHANGED').sum())}/{len(protected)}" if not protected.empty else "0/0", f"{len(protected)}/{len(protected)}" if not protected.empty else "0/0"),
        ("STATIC-035", "protocol hash present", len(protocol_hash) == 64, protocol_hash, "64-char SHA256"),
    ]
    return pd.DataFrame(
        [
            {"check_id": check_id, "check_name": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected}
            for check_id, name, ok, observed, expected in checks
        ]
    )


def first_numeric_item_id_case(frame: pd.DataFrame) -> tuple[dict[str, Any], list[int]]:
    for _, row in frame.loc[frame["task"].eq("ranking")].iterrows():
        record = row.to_dict()
        ids: list[int] = []
        for value in candidate_item_ids(record):
            if str(value).isdigit() and int(value) > int(record["candidate_count"]):
                ids.append(int(value))
            if len(ids) == 10:
                return record, ids
    raise R5DError("cannot find ranking row with ten numeric item IDs outside 1..N")


def build_synthetic_tests(frame: pd.DataFrame) -> pd.DataFrame:
    ranking = frame.loc[frame["task"].eq("ranking")].iloc[0].to_dict()
    id_row, item_ids = first_numeric_item_id_case(frame)
    valid10 = [2, 5, 1, 7, 4, 9, 3, 8, 6, 10]
    ranking_cases = [
        ("ranking_valid_exact10", canonical_json({"ranked_candidate_indices": valid10}), True, "parsed_valid", ranking),
        ("ranking_k_minus_1", canonical_json({"ranked_candidate_indices": valid10[:-1]}), False, "invalid_length", ranking),
        ("ranking_k_plus_1", canonical_json({"ranked_candidate_indices": valid10 + [11]}), False, "invalid_length", ranking),
        ("ranking_numbered_output", "1. 2\n2. 5\n3. 1\n4. 7\n5. 4\n6. 9\n7. 3\n8. 8\n9. 6\n10. 10", False, "parse_failed", ranking),
        ("ranking_duplicate", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5, 6, 7, 8, 8, 10]}), False, "duplicate_indices", ranking),
        ("ranking_out_of_range", canonical_json({"ranked_candidate_indices": [1, 2, 3, 4, 5, 6, 7, 8, 9, 21]}), False, "unknown_indices", ranking),
        ("ranking_item_ids_in_indices_field", canonical_json({"ranked_candidate_indices": item_ids}), False, "unknown_indices", id_row),
        ("ranking_truncated_output", '{"ranked_candidate_indices":[1,2,3', False, "parse_failed", ranking),
        ("ranking_backend_incomplete", '{"ranked_candidate_indices":[1,2,3', False, "backend_incomplete", ranking),
    ]
    profile_cases = [
        ("profile_valid", canonical_json({"user_profile": "likes animated films and family-friendly stories"}), True, "parsed_valid"),
        ("profile_empty", canonical_json({"user_profile": ""}), False, "empty_profile"),
        ("profile_extra_key", canonical_json({"user_profile": "x", "extra": "y"}), False, "invalid_schema"),
        ("profile_plain_text", "likes animated films", False, "parse_failed"),
        ("profile_backend_incomplete", '{"user_profile":"likes animated', False, "backend_incomplete"),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, text, expected_valid, expected_status, case_row in ranking_cases:
        if case_id == "ranking_backend_incomplete":
            observed = incomplete_parse_record(case_row, text)
            observed_valid = bool(observed["parse_valid"])
            observed_status = str(observed["parse_status"])
            length = int(observed["output_length"])
            duplicates = int(observed["duplicate_count"])
            unknown = int(observed["unknown_item_count"])
        else:
            parsed = parse_strict_index_response(
                text,
                candidate_items=candidate_item_ids(case_row),
                expected_count=10,
                require_full_permutation=False,
            )
            observed_valid = bool(parsed.parse_valid)
            observed_status = parsed.status
            length = parsed.output_length
            duplicates = parsed.duplicate_count
            unknown = parsed.unknown_item_count
        rows.append(
            {
                "case_id": case_id,
                "task": "ranking",
                "expected_valid": bool_text(expected_valid),
                "observed_valid": bool_text(observed_valid),
                "expected_status": expected_status,
                "observed_status": observed_status,
                "output_length": length,
                "duplicate_count": duplicates,
                "unknown_item_count": unknown,
                "pass": bool_text(observed_valid == expected_valid and observed_status == expected_status),
                "notes": "actual dataset item IDs must not be accepted as candidate positions" if "item_ids" in case_id else "",
            }
        )
    profile_row = frame.loc[frame["task"].eq("profile_generation")].iloc[0].to_dict()
    for case_id, text, expected_valid, expected_status in profile_cases:
        if case_id == "profile_backend_incomplete":
            observed = incomplete_parse_record(profile_row, text)
        else:
            observed = parse_profile_response(text)
        rows.append(
            {
                "case_id": case_id,
                "task": "profile_generation",
                "expected_valid": bool_text(expected_valid),
                "observed_valid": bool_text(bool(observed["parse_valid"])),
                "expected_status": expected_status,
                "observed_status": str(observed["parse_status"]),
                "output_length": int(observed["output_length"]),
                "duplicate_count": int(observed["duplicate_count"]),
                "unknown_item_count": int(observed["unknown_item_count"]),
                "pass": bool_text(bool(observed["parse_valid"]) == expected_valid and str(observed["parse_status"]) == expected_status),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def stable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return sorted(frame.to_dict("records"), key=execution_sort_key)


def select_canary_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    profile = frame.loc[frame["task"].eq("profile_generation")]
    ranking = frame.loc[frame["task"].eq("ranking")]
    for (dataset, model_name), group in profile.groupby(["dataset", "model_name"], sort=True):
        prof = stable_records(group)[0]
        rows.append(prof)
        used.add(str(prof["request_id"]))
        dep = ranking.loc[
            ranking["r5d_profile_dependency_request_id"].astype(str).eq(str(prof["request_id"]))
            & ranking["dataset"].eq(dataset)
            & ranking["model_name"].eq(model_name)
        ]
        if dep.empty:
            raise R5DError(f"cannot find downstream canary dependency for {prof['request_id']}")
        rank = stable_records(dep)[0]
        rows.append(rank)
        used.add(str(rank["request_id"]))
    for (dataset, model_name), group in ranking.loc[ranking["r5d_profile_dependency_request_id"].astype(str).eq("")].groupby(["dataset", "model_name"], sort=True):
        row = stable_records(group)[0]
        rows.append(row)
        used.add(str(row["request_id"]))
    if len(used) != 12:
        raise R5DError(f"R5D canary selection expected 12 rows, got {len(used)}")
    return sorted(rows, key=execution_sort_key)


def select_early_rows(frame: pd.DataFrame, skip_request_ids: Iterable[str], n: int) -> list[dict[str, Any]]:
    skipped = set(skip_request_ids)
    rows: list[dict[str, Any]] = []
    profile = frame.loc[frame["task"].eq("profile_generation")]
    ranking = frame.loc[frame["task"].eq("ranking")]
    for (dataset, model_name), group in profile.groupby(["dataset", "model_name"], sort=True):
        candidates = [row for row in stable_records(group) if str(row["request_id"]) not in skipped][:n]
        rows.extend(candidates)
        for prof in candidates:
            dep = ranking.loc[
                ranking["r5d_profile_dependency_request_id"].astype(str).eq(str(prof["request_id"]))
                & ranking["dataset"].eq(dataset)
                & ranking["model_name"].eq(model_name)
            ]
            if dep.empty:
                raise R5DError(f"cannot find downstream early dependency for {prof['request_id']}")
            rows.append(stable_records(dep)[0])
    for _, group in ranking.loc[ranking["r5d_profile_dependency_request_id"].astype(str).eq("")].groupby(
        ["dataset", "model_name"],
        sort=True,
    ):
        candidates = [row for row in stable_records(group) if str(row["request_id"]) not in skipped][:n]
        rows.extend(candidates)
    unique = {str(row["request_id"]): row for row in rows if str(row["request_id"]) not in skipped}
    return sorted(unique.values(), key=execution_sort_key)


def select_all_remaining(frame: pd.DataFrame, skip_request_ids: Iterable[str]) -> list[dict[str, Any]]:
    skipped = set(skip_request_ids)
    return [row for row in stable_records(frame) if str(row["request_id"]) not in skipped]


def health_summary(status: pd.DataFrame, *, label: str) -> pd.DataFrame:
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    rows: list[dict[str, Any]] = []

    def add(scope: str, key: str, group: pd.DataFrame) -> None:
        total = len(group)
        valid = int(group["parse_valid"].eq("true").sum())
        rows.append(
            {
                "health_check": label,
                "scope": scope,
                "key": key,
                "accounted": total,
                "valid": valid,
                "invalid": total - valid,
                "validity_rate": float(valid / total) if total else 0.0,
                "wrong_length": int(group["validity_class"].eq("WRONG_LENGTH").sum()),
                "parse_failure": int(group["validity_class"].eq("PARSE_FAILURE").sum()),
                "backend_incomplete": int(group["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "backend_failed": int(group["execution_status"].eq("BACKEND_FAILED").sum()),
                "duplicate": int(group["validity_class"].eq("DUPLICATE").sum()),
                "out_of_range": int(group["validity_class"].eq("OUT_OF_RANGE").sum()),
                "candidate_violation": int(group["unknown_item_count"].astype(int).sum()),
                "item_id_substitution": int(group["dataset_item_id_substitution_count"].astype(int).sum()),
            }
        )

    add("overall", "all", terminal)
    for task, group in terminal.groupby("task", sort=True):
        add("task", str(task), group)
    for model, group in terminal.groupby("model_name", sort=True):
        add("model", str(model), group)
    for dataset, group in terminal.groupby("dataset", sort=True):
        add("dataset", str(dataset), group)
    for condition, group in terminal.groupby("experimental_condition", sort=True):
        add("condition", str(condition), group)
    for key, group in terminal.groupby(["task", "model_name", "dataset", "experimental_condition"], sort=True):
        add("task_model_dataset_condition", "|".join(map(str, key)), group)
    return pd.DataFrame(rows)


def health_pass(health: pd.DataFrame, config: Mapping[str, Any]) -> tuple[bool, str]:
    if health.empty:
        return False, "no terminal outputs"
    overall = health.loc[health["scope"].eq("overall")].iloc[0]
    if int(overall["backend_incomplete"]) > 0:
        return False, "backend incomplete present"
    if int(overall["backend_failed"]) > 0:
        return False, "backend failed present"
    if int(overall["wrong_length"]) > 0:
        return False, "wrong_length present"
    if int(overall["parse_failure"]) > 0:
        return False, "parse_failure present"
    if int(overall["out_of_range"]) > 0:
        return False, "out_of_range present"
    if int(overall["duplicate"]) > 0:
        return False, "duplicate present"
    if int(overall["item_id_substitution"]) > 0:
        return False, "dataset item-ID substitution present"
    if float(overall["validity_rate"]) < 1.0:
        return False, f"overall validity {overall['validity_rate']} < 1.0"
    return True, ""


def canary_pass(status: pd.DataFrame, canary_ids: set[str], config: Mapping[str, Any]) -> tuple[bool, str, pd.DataFrame]:
    canary = status.loc[status["request_id"].isin(canary_ids)].copy()
    table = canary[
        [
            "task",
            "model_name",
            "dataset",
            "experimental_condition",
            "backend_complete",
            "output_length",
            "expected_output_count",
            "parse_valid",
            "validity_class",
            "parse_status",
            "values_within_1_to_n",
            "dataset_item_id_substitution_count",
            "request_id",
            "r5a_request_id",
        ]
    ].rename(
        columns={
            "task": "Task",
            "model_name": "Model",
            "dataset": "Dataset",
            "experimental_condition": "Condition",
            "backend_complete": "Complete",
            "output_length": "ParsedLength",
            "expected_output_count": "ExpectedLength",
            "parse_valid": "Valid",
            "validity_class": "Failure",
            "values_within_1_to_n": "Within1ToN",
            "dataset_item_id_substitution_count": "ItemIdSubstitutionCount",
        }
    )
    expected_cells = int((config.get("scope") or {}).get("expected_cells", 12))
    if len(canary) != expected_cells:
        return False, f"canary accounted {len(canary)} != {expected_cells}", table
    failed = canary.loc[~canary["parse_valid"].eq("true")]
    if not failed.empty:
        return False, f"canary invalid outputs {len(failed)}", table
    ranking = canary.loc[canary["task"].eq("ranking")]
    if not ranking["output_length"].astype(int).eq(int((config.get("scope") or {}).get("expected_k", 10))).all():
        return False, "canary ranking did not return exactly K=10 everywhere", table
    if not ranking["values_within_1_to_n"].eq("true").all():
        return False, "canary ranking has values outside 1..N", table
    if int(ranking["dataset_item_id_substitution_count"].astype(int).sum()) > 0:
        return False, "canary repeated item-ID substitution", table
    health = health_summary(canary, label="canary")
    ok, reason = health_pass(health, config)
    return ok, reason, table


def build_execution_manifest(frame: pd.DataFrame, manifest_sha: str, protocol_hash: str) -> pd.DataFrame:
    keep = [
        "request_id",
        "r5a_request_id",
        "recovery_family",
        "dataset",
        "model_name",
        "model_label",
        "task",
        "user_id",
        "experimental_condition",
        "history_length",
        "history_item_count",
        "history_hash",
        "history_source",
        "candidate_policy",
        "candidate_order_seed",
        "candidate_order_algorithm",
        "candidate_count",
        "candidate_list_hash",
        "unordered_candidate_item_set_hash",
        "positive_item_id",
        "positive_candidate_position",
        "r5a_profile_dependency_request_id",
        "r5d_profile_dependency_request_id",
        "r5d_profile_artifact_path",
        "profile_hash",
        "profile_hash_status",
        "prompt_template_id",
        "prompt_hash",
        "parser_contract",
        "expected_output_schema",
        "temperature",
        "seed",
        "max_new_tokens",
        "output_destination",
        "original_output_destination",
        "request_key_sha256",
        "material_context_hash",
        "gap_ids_served",
        "metrics_served",
    ]
    result = frame[keep].copy()
    result.insert(0, "phase", PHASE)
    result.insert(1, "r5a_manifest_sha256", manifest_sha)
    result.insert(2, "protocol_hash", protocol_hash)
    result["execution_backend"] = "local_ollama"
    result["external_api_allowed"] = "false"
    result["scientific_scope"] = "frozen_r5a_generation_involved_recovery_with_predeclared_profile_dependency_materialization"
    result["evidence_class"] = EVIDENCE_CLASS
    return result


def build_output_validity(status: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "request_id",
        "r5a_request_id",
        "dataset",
        "model_name",
        "user_id",
        "task",
        "experimental_condition",
        "execution_status",
        "backend_complete",
        "parse_valid",
        "metric_ready",
        "parse_status",
        "validity_class",
        "output_length",
        "expected_output_count",
        "duplicate_count",
        "unknown_item_count",
        "values_within_1_to_n",
        "dataset_item_id_substitution_count",
        "positive_rank",
        "hr_at_5",
        "ndcg_at_5",
        "profile_hash",
        "result_hash",
    ]
    return status[keep].copy()


def build_metrics_by_condition(status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ranking = status.loc[status["task"].eq("ranking")]
    for key, group in ranking.groupby(["dataset", "model_name", "experimental_condition"], sort=True):
        dataset, model_name, condition = key
        n_manifest = len(group)
        terminal = group.loc[~group["execution_status"].eq("NOT_EXECUTED")]
        n_valid = int(terminal["parse_valid"].eq("true").sum())
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "experimental_condition": condition,
                "n_manifest_requests": n_manifest,
                "n_manifest": n_manifest,
                "n_terminal": len(terminal),
                "n_valid_outputs": n_valid,
                "n_valid": n_valid,
                "n_invalid_outputs": int(len(terminal) - n_valid),
                "n_invalid": int(len(terminal) - n_valid),
                "n_wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
                "n_out_of_range": int(terminal["validity_class"].eq("OUT_OF_RANGE").sum()),
                "n_parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
                "n_backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "n_backend_failed": int(terminal["execution_status"].eq("BACKEND_FAILED").sum()),
                "n_not_executed": int(group["execution_status"].eq("NOT_EXECUTED").sum()),
                "hr_at_5": float(group["hr_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "HR@5": float(group["hr_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "ndcg_at_5": float(group["ndcg_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "NDCG@5": float(group["ndcg_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
            }
        )
    return pd.DataFrame(rows)


def build_generation_results(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in metrics.to_dict("records"):
        for metric_label, column in [("HR@5", "HR@5"), ("NDCG@5", "NDCG@5")]:
            rows.append(
                {
                    "dataset": record["dataset"],
                    "model_name": record["model_name"],
                    "metric": metric_label,
                    "experimental_condition": record["experimental_condition"],
                    "value": float(record[column]),
                    "n_manifest": int(record["n_manifest"]),
                    "n_valid": int(record["n_valid"]),
                    "n_invalid": int(record["n_invalid"]),
                    "denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
                    "evidence_class": EVIDENCE_CLASS,
                    "paper_exact": "false",
                }
            )
    return pd.DataFrame(rows)


def build_conditional_valid_metrics(status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ranking = status.loc[status["task"].eq("ranking")]
    for key, group in ranking.groupby(["dataset", "model_name", "experimental_condition"], sort=True):
        dataset, model_name, condition = key
        valid = group.loc[group["parse_valid"].eq("true")]
        denom = len(valid)
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "experimental_condition": condition,
                "n_valid_outputs": denom,
                "diagnostic_hr_at_5_valid_only": float(valid["hr_at_5"].astype(float).sum() / denom) if denom else 0.0,
                "diagnostic_ndcg_at_5_valid_only": float(valid["ndcg_at_5"].astype(float).sum() / denom) if denom else 0.0,
                "diagnostic_only": "true",
                "notes": "Conditional-valid metrics separate format compliance from ranking quality.",
            }
        )
    return pd.DataFrame(rows)


def build_dataset_model_summary(status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in status.groupby(["dataset", "model_name"], sort=True):
        dataset, model_name = key
        terminal = group.loc[~group["execution_status"].eq("NOT_EXECUTED")]
        valid = int(terminal["parse_valid"].eq("true").sum())
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "n_manifest_requests": len(group),
                "n_terminal": len(terminal),
                "n_valid_outputs": valid,
                "n_invalid_outputs": int(len(terminal) - valid),
                "n_wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
                "n_out_of_range": int(terminal["validity_class"].eq("OUT_OF_RANGE").sum()),
                "n_parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
                "n_backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "n_backend_failed": int(terminal["execution_status"].eq("BACKEND_FAILED").sum()),
                "n_not_executed": int(group["execution_status"].eq("NOT_EXECUTED").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_gap_resolution(frame: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gap_id, group in frame.groupby("gap_ids_served", sort=True):
        status_group = status.loc[status["request_id"].isin(set(group["request_id"].astype(str)))]
        terminal = status_group.loc[~status_group["execution_status"].eq("NOT_EXECUTED")]
        valid = int(terminal["parse_valid"].eq("true").sum())
        rows.append(
            {
                "gap_id": gap_id,
                "dataset": ",".join(sorted(set(group["dataset"].astype(str)))),
                "metric": ",".join(sorted(set(group["metrics_served"].astype(str)))),
                "requests_expected": len(group),
                "requests_accounted": len(terminal),
                "valid": valid,
                "invalid": int(len(terminal) - valid),
                "wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
                "out_of_range": int(terminal["validity_class"].eq("OUT_OF_RANGE").sum()),
                "parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
                "backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
                "gap_recovered": bool_text(len(terminal) == len(group) and valid == len(group)),
                "evidence_class": EVIDENCE_CLASS,
            }
        )
    return pd.DataFrame(rows)


def build_runtime_provenance(
    project_root: Path,
    config: Mapping[str, Any],
    status: pd.DataFrame,
    runtime_available: bool,
    runtime_error: str,
    runtime_tags: Mapping[str, Mapping[str, Any]],
    protocol_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    version = r5c.ollama_version(config) if runtime_available else "unavailable"
    disk_usage = shutil.disk_usage(project_root)
    for model in sorted(set((config.get("scope") or {}).get("models") or [])):
        details = runtime_tags.get(model, {})
        model_details = details.get("details", {}) if isinstance(details.get("details"), dict) else {}
        model_status = status.loc[status["model_name"].eq(model)] if not status.empty else pd.DataFrame()
        rows.append(
            {
                "phase": PHASE,
                "protocol_hash": protocol_hash,
                "backend": "ollama",
                "ollama_version": version,
                "model_name": model,
                "model_available": bool_text(model in runtime_tags),
                "model_digest": str(details.get("digest", "")),
                "model_parameter_size": str(model_details.get("parameter_size", "")),
                "model_quantization_level": str(model_details.get("quantization_level", "")),
                "context_length": str(context_length_for_model(config, runtime_tags, model)),
                "runtime_error": runtime_error,
                "requests_accounted": int((~model_status["execution_status"].eq("NOT_EXECUTED")).sum()) if not model_status.empty else 0,
                "valid_outputs": int(model_status["parse_valid"].eq("true").sum()) if not model_status.empty else 0,
                "disk_free_bytes": disk_usage.free,
                "max_retries": int((config.get("runtime") or {}).get("max_retries", 0)),
            }
        )
    return pd.DataFrame(rows)


def build_validation(
    static_preflight: pd.DataFrame,
    synthetic_tests: pd.DataFrame,
    context_preflight: pd.DataFrame,
    canary_table: pd.DataFrame,
    early_health: pd.DataFrame,
    status: pd.DataFrame,
    dependency_validation: pd.DataFrame,
    *,
    full_run_authorized: bool,
    decision: str,
) -> pd.DataFrame:
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    ranking_terminal = terminal.loc[terminal["task"].eq("ranking")]
    profile_terminal = terminal.loc[terminal["task"].eq("profile_generation")]
    checks: list[tuple[str, str, bool, str, str]] = [
        ("VAL-001", "static preflight pass", static_preflight["status"].eq("pass").all(), f"{int(static_preflight['status'].eq('pass').sum())}/{len(static_preflight)}", f"{len(static_preflight)}/{len(static_preflight)}"),
        ("VAL-002", "synthetic pass", synthetic_tests["pass"].eq("true").all(), f"{int(synthetic_tests['pass'].eq('true').sum())}/{len(synthetic_tests)}", f"{len(synthetic_tests)}/{len(synthetic_tests)}"),
        ("VAL-003", "context preflight pass", context_preflight["status"].eq("pass").all(), f"{int(context_preflight['status'].eq('pass').sum())}/{len(context_preflight)}", f"{len(context_preflight)}/{len(context_preflight)}"),
        ("VAL-004", "dependency graph pass", dependency_validation["dependency_valid"].eq("true").all(), f"{int(dependency_validation['dependency_valid'].eq('true').sum())}/{len(dependency_validation)}", f"{len(dependency_validation)}/{len(dependency_validation)}"),
        ("VAL-005", "canary table valid when present", canary_table.empty or canary_table["Valid"].eq("true").all(), "checked", "all true"),
        ("VAL-006", "early health pass when present", early_health.empty or health_pass(early_health, {"gates": {}})[0], "checked", "pass"),
        ("VAL-007", "max retries zero", terminal.empty or terminal["retry_count"].astype(int).eq(0).all(), "checked", "0"),
        ("VAL-008", "incomplete is not succeeded", terminal.empty or terminal.loc[terminal["validity_class"].eq("BACKEND_INCOMPLETE"), "execution_status"].ne("SUCCEEDED").all(), "checked", "no incomplete succeeded"),
        ("VAL-009", "ranking exactly K when valid", ranking_terminal.empty or ranking_terminal.loc[ranking_terminal["parse_valid"].eq("true"), "output_length"].astype(int).eq(10).all(), "checked", "10"),
        ("VAL-010", "profile JSON valid when valid", profile_terminal.empty or profile_terminal.loc[profile_terminal["parse_valid"].eq("true"), "output_length"].astype(int).eq(1).all(), "checked", "1"),
        ("VAL-011", "full authorization consistent", (not full_run_authorized) or decision in {DECISION_EARLY_PASS, DECISION_COMPLETE, DECISION_EXECUTION_FAILED, DECISION_SCIENTIFIC_FAILED}, f"authorized={full_run_authorized};decision={decision}", "consistent"),
        ("VAL-012", "complete means all valid", decision != DECISION_COMPLETE or (len(terminal) == len(status) and int(terminal["parse_valid"].eq("true").sum()) == len(status)), f"terminal={len(terminal)};valid={int(terminal['parse_valid'].eq('true').sum())}", "1200/1200"),
    ]
    return pd.DataFrame(
        [
            {"check_id": check_id, "check_name": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected}
            for check_id, name, ok, observed, expected in checks
        ]
    )


def rebuild_frames(
    project_root: Path,
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    manifest_sha: str,
    protocol_hash: str,
    runtime_available: bool,
    runtime_error: str,
    runtime_tags: Mapping[str, Mapping[str, Any]],
    dependency_validation: pd.DataFrame,
    static_preflight: pd.DataFrame,
    synthetic_tests: pd.DataFrame,
    context_preflight: pd.DataFrame,
    protocol_freeze: pd.DataFrame,
    canary_results: pd.DataFrame | None,
    early_health: pd.DataFrame | None,
    scope_summary: pd.DataFrame,
    decision: str,
    full_run_authorized: bool,
) -> dict[str, pd.DataFrame]:
    status = build_status(project_root, frame, manifest_sha, protocol_hash)
    metrics = build_metrics_by_condition(status)
    generation_results = build_generation_results(metrics)
    conditional = build_conditional_valid_metrics(status)
    summary = build_dataset_model_summary(status)
    gaps = build_gap_resolution(frame, status)
    runtime = build_runtime_provenance(project_root, config, status, runtime_available, runtime_error, runtime_tags, protocol_hash)
    canary_frame = canary_results if canary_results is not None else pd.DataFrame()
    early_frame = early_health if early_health is not None else pd.DataFrame()
    validation = build_validation(
        static_preflight,
        synthetic_tests,
        context_preflight,
        canary_frame,
        early_frame,
        status,
        dependency_validation,
        full_run_authorized=full_run_authorized,
        decision=decision,
    )
    return {
        "execution_manifest": build_execution_manifest(frame, manifest_sha, protocol_hash),
        "request_execution_status": status,
        "dependency_validation": dependency_validation,
        "output_validity": build_output_validity(status),
        "metrics_by_condition": metrics,
        "generation_results": generation_results,
        "conditional_valid_metrics": conditional,
        "dataset_model_summary": summary,
        "gap_resolution": gaps,
        "runtime_provenance": runtime,
        "validation": validation,
        "scope_summary": scope_summary,
        "static_preflight": static_preflight,
        "synthetic_tests": synthetic_tests,
        "context_preflight": context_preflight,
        "protocol_freeze": protocol_freeze,
        "canary_results": canary_frame,
        "early_health": early_frame,
    }


def render_markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit).fillna("")
    header = "| " + " | ".join(subset.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in record.values()) + " |"
        for record in subset.to_dict("records")
    ]
    suffix = f"\n\n_Showing {len(subset)} of {len(frame)} rows._" if len(frame) > len(subset) else ""
    return "\n".join([header, divider, *rows]) + suffix


def render_report(
    frames: Mapping[str, pd.DataFrame],
    decision: str,
    manifest_sha: str,
    protocol_hash: str,
    full_run_authorized: bool,
    blocked_reason: str,
    executed_this_run: int,
) -> str:
    status = frames["request_execution_status"]
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    valid = int(terminal["parse_valid"].eq("true").sum())
    gaps = frames["gap_resolution"]
    recovered = int(gaps["gap_recovered"].eq("true").sum()) if not gaps.empty else 0
    return f"""# R5D Generation-Involved Recovery

Decision: `{decision}`

    R5D v2 executes only frozen R5A `GENERATION_INVOLVED_RECOVERY` rows. It preserves R5A/R5B/R5C raw outputs and the failed R5D v1 gate raw outputs, and writes a separate R5D v2 namespace.

## Scope

{render_markdown_table(frames["scope_summary"], ["scope", "key", "count", "notes"])}

## Protocol

- Manifest SHA256: `{manifest_sha}`
- Protocol hash: `{protocol_hash}`
- Profile parser: `{PROFILE_PARSER_CONTRACT}`
- Ranking parser: `{RANKING_PARSER_CONTRACT}`
- Max retries: `0`
- Paper exact: `NO`

{render_markdown_table(frames["protocol_freeze"], ["component", "value", "sha256"])}

## Static Preflight

{render_markdown_table(frames["static_preflight"], ["check_id", "check_name", "status", "observed", "expected"])}

## Synthetic Tests

{render_markdown_table(frames["synthetic_tests"], ["case_id", "task", "expected_status", "observed_status", "pass", "notes"])}

## Canary

{render_markdown_table(frames.get("canary_results", pd.DataFrame()), ["Task", "Model", "Dataset", "Condition", "Complete", "ParsedLength", "ExpectedLength", "Valid", "Failure", "Within1ToN", "ItemIdSubstitutionCount"])}

## Early Health

{render_markdown_table(frames.get("early_health", pd.DataFrame()), ["scope", "key", "accounted", "valid", "invalid", "wrong_length", "parse_failure", "backend_incomplete", "out_of_range", "item_id_substitution"])}

## Request Accounting

- Requests expected: `{len(status)}`
- Requests accounted: `{len(terminal)}`
- Valid: `{valid}`
- Invalid: `{len(terminal) - valid}`
- Executed this invocation: `{executed_this_run}`
- Full run authorized: `{"YES" if full_run_authorized else "NO"}`
- Blocked reason: `{blocked_reason}`

## Metrics

{render_markdown_table(frames["metrics_by_condition"], ["dataset", "model_name", "experimental_condition", "n_manifest", "n_valid", "n_invalid", "HR@5", "NDCG@5"])}

## Gap Resolution

Target gaps recovered: `{recovered}/{len(gaps)}`

{render_markdown_table(gaps, ["gap_id", "dataset", "requests_expected", "requests_accounted", "valid", "invalid", "gap_recovered"])}

## Validation

{render_markdown_table(frames["validation"], ["check_id", "check_name", "status", "observed", "expected"])}

## Scientific Interpretation

R5D is local adapted generation-involved recovery. It is not paper-exact because local open models, strict JSON contracts, and explicit same-run profile materialization differ from paper execution conditions.
"""


def write_freeze_candidate(
    project_root: Path,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    manifest_sha: str,
    protocol_hash: str,
    decision: str,
) -> None:
    status = frames["request_execution_status"]
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    valid = int(terminal["parse_valid"].eq("true").sum())
    namespace = result_namespace(project_root, config)
    raw_paths = sorted(namespace.glob("*.json")) if namespace.exists() else []
    aggregate_hash = sha256_text(canonical_json([sha256_file(path) for path in raw_paths]))
    gaps = frames["gap_resolution"]
    payload = {
        "freeze_metadata_version": "r5d_generation_involved_freeze_candidate_v1",
        "phase": PHASE,
        "decision": decision,
        "created_utc": r5c.utc_now(),
        "r5a_manifest_sha256": manifest_sha,
        "protocol_hash": protocol_hash,
        "r5c_corrected_freeze_commit": str((config.get("frozen_r5c_corrected") or {}).get("freeze_commit", "")),
        "r5c_corrected_protocol_hash": str((config.get("frozen_r5c_corrected") or {}).get("protocol_hash", "")),
        "raw_namespace": normalize_rel((config.get("outputs") or {})["result_namespace"]),
        "raw_count": len(raw_paths),
        "raw_aggregate_sha256": aggregate_hash,
        "status_counts": {
            "requests_expected": len(status),
            "requests_accounted": len(terminal),
            "valid": valid,
            "invalid": int(len(terminal) - valid),
            "wrong_length": int(terminal["validity_class"].eq("WRONG_LENGTH").sum()),
            "out_of_range": int(terminal["validity_class"].eq("OUT_OF_RANGE").sum()),
            "parse_failure": int(terminal["validity_class"].eq("PARSE_FAILURE").sum()),
            "backend_incomplete": int(terminal["validity_class"].eq("BACKEND_INCOMPLETE").sum()),
        },
        "target_gaps_recovered": int(gaps["gap_recovered"].eq("true").sum()) if not gaps.empty else 0,
        "target_gaps_total": len(gaps),
        "paper_exact": False,
        "freeze_ready": bool(decision == DECISION_COMPLETE and len(terminal) == len(status) and valid == len(status)),
    }
    write_json_atomic(payload, project_path(project_root, (config.get("outputs") or {})["freeze_candidate"]))


def write_outputs(
    project_root: Path,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    report: str,
    manifest_sha: str,
    protocol_hash: str,
    decision: str,
) -> None:
    outputs = configured_outputs(config)
    for key in OUTPUT_KEYS:
        write_csv(frames[key], project_root / outputs[key])
    diag = diagnostics_dir(project_root, config)
    diag.mkdir(parents=True, exist_ok=True)
    for key in ["static_preflight", "synthetic_tests", "context_preflight", "protocol_freeze", "canary_results", "early_health"]:
        write_csv(frames.get(key, pd.DataFrame()), diag / f"{key}.csv")
    report_path = project_path(project_root, (config.get("outputs") or {})["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_freeze_candidate(project_root, config, frames, manifest_sha, protocol_hash, decision)


def update_notes(project_root: Path, decision: str, manifest_sha: str, protocol_hash: str, status: pd.DataFrame) -> None:
    notes_path = project_root / NOTES_PATH
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    valid = int(terminal["parse_valid"].eq("true").sum())
    section = f"""

## R5D Generation-Involved recovery

- Decision: `{decision}`.
- Manifest SHA256: `{manifest_sha}`.
- Protocol hash: `{protocol_hash}`.
- Scope: new `R5D_GENERATION_INVOLVED_RECOVERY_V2` identity over 1200 frozen R5A `GENERATION_INVOLVED_RECOVERY` rows.
- Two-stage policy: ML-1M/LastFM profile-generation rows are executed before dependent downstream profile-plus-history ranking rows; Beauty/Sports reuse existing F6L profile context.
- Output contract: profile rows require strict short natural-language `user_profile` JSON; ranking rows require exactly 10 unique 1-based candidate positions, not item IDs.
- Outputs: `results/recovery/r5d_generation_involved_v2/`, `results/diagnostics/r5d_generation_involved_v2/`, `results/tables/reproduction_r5d_generation_involved_v2_*.csv`, and `report/reproduction_r5d_generation_involved_v2_recovery.md`.
- Request accounting: expected `1200`, accounted `{len(terminal)}`, valid `{valid}`, invalid `{int(len(terminal) - valid)}`.
- No R5D successor phase was executed.
"""
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "## R5D Generation-Involved recovery"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + section
    else:
        existing = existing.rstrip() + section
    notes_path.write_text(existing.lstrip() + "\n", encoding="utf-8")


def write_diagnostic_frame(project_root: Path, config: Mapping[str, Any], frame: pd.DataFrame, name: str) -> None:
    write_csv(frame, diagnostics_dir(project_root, config) / name)


def build_r5d_generation_involved(
    project_root: Path,
    config_path: Path | None = None,
    *,
    write: bool = True,
    run_gated: bool = False,
    allow_api_calls: bool = False,
    allow_local_ollama: bool = False,
    stop_after: str = "full",
) -> R5DBuildResult:
    actual_config_path = config_path or DEFAULT_CONFIG_PATH
    config = load_config(actual_config_path)
    manifest, frame, manifest_sha = load_r5d_manifest(project_root, config)
    dependency_validation = build_dependency_validation(frame)
    scope_summary = build_scope_summary(frame)

    runtime_available = False
    runtime_error = ""
    runtime_tags: dict[str, dict[str, Any]] = {}
    if run_gated:
        runtime_available, runtime_error, runtime_tags = r5c.validate_runtime_available(
            config,
            (config.get("scope") or {}).get("models", []),
        )
    context_preflight = build_context_preflight(frame, config, runtime_tags)
    protocol_hash, protocol_freeze = protocol_metadata(
        project_root,
        actual_config_path,
        config,
        frame,
        manifest_sha,
        dependency_validation,
        runtime_tags,
    )
    static_preflight = build_static_preflight(
        project_root,
        actual_config_path,
        config,
        manifest,
        frame,
        manifest_sha,
        dependency_validation,
        context_preflight,
        protocol_hash,
    )
    if run_gated and not runtime_available:
        static_preflight = pd.concat(
            [
                static_preflight,
                pd.DataFrame(
                    [
                        {
                            "check_id": "STATIC-036",
                            "check_name": "runtime models available",
                            "status": "fail",
                            "observed": runtime_error,
                            "expected": "qwen2:7b-instruct and mistral:7b-instruct available",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    synthetic_tests = build_synthetic_tests(frame)

    static_pass = static_preflight["status"].eq("pass").all()
    synthetic_pass = synthetic_tests["pass"].eq("true").all()
    context_pass = context_preflight["status"].eq("pass").all()
    decision = DECISION_PREFLIGHT_FAILED
    blocked_reason = ""
    full_run_authorized = False
    executed_this_run = 0
    canary_table = pd.DataFrame()
    early_health = pd.DataFrame()
    canary_ids: set[str] = set()
    early_ids: set[str] = set()

    if not static_pass:
        blocked_reason = "static preflight failed"
    elif not synthetic_pass:
        blocked_reason = "synthetic parser/validator tests failed"
    elif not context_pass:
        decision = DECISION_CONTEXT_FAILED
        blocked_reason = "context length preflight failed"
    elif run_gated:
        canary_rows = select_canary_rows(frame)
        canary_ids = {str(row["request_id"]) for row in canary_rows}
        executed_this_run += execute_selected_requests(
            project_root,
            config,
            canary_rows,
            manifest_sha,
            protocol_hash,
            allow_api_calls=allow_api_calls,
            allow_local_ollama=allow_local_ollama,
            gate_stage="canary",
        )
        status = build_status(project_root, frame, manifest_sha, protocol_hash)
        canary_ok, canary_reason, canary_table = canary_pass(status, canary_ids, config)
        if not canary_ok:
            decision = DECISION_CANARY_FAILED
            blocked_reason = f"canary failed: {canary_reason}"
        elif stop_after == "canary":
            decision = DECISION_CANARY_PASS
            blocked_reason = "canary pass; early/full run not requested in this invocation"
        else:
            early_rows = select_early_rows(
                frame,
                skip_request_ids=canary_ids,
                n=int((config.get("gates") or {}).get("early_additional_per_cell", 2)),
            )
            executed_this_run += execute_selected_requests(
                project_root,
                config,
                early_rows,
                manifest_sha,
                protocol_hash,
                allow_api_calls=allow_api_calls,
                allow_local_ollama=allow_local_ollama,
                gate_stage="early",
            )
            early_ids = canary_ids.union(str(row["request_id"]) for row in early_rows)
            status = build_status(project_root, frame, manifest_sha, protocol_hash)
            early_health = health_summary(status.loc[status["request_id"].isin(early_ids)], label="early")
            early_ok, early_reason = health_pass(early_health, config)
            if not early_ok:
                decision = DECISION_EARLY_FAILED
                blocked_reason = f"early health failed: {early_reason}"
            else:
                full_run_authorized = True
                if stop_after == "early":
                    decision = DECISION_EARLY_PASS
                    blocked_reason = "early health pass; full run not requested in this invocation"
                else:
                    remaining = select_all_remaining(frame, skip_request_ids=early_ids)
                    checkpoints = [int(value) for value in (config.get("gates") or {}).get("periodic_accounted_checkpoints", [])]
                    next_checkpoint_idx = 0
                    batch: list[dict[str, Any]] = []
                    for row in remaining:
                        batch.append(row)
                        status = build_status(project_root, frame, manifest_sha, protocol_hash)
                        accounted_now = len(status.loc[~status["execution_status"].eq("NOT_EXECUTED")])
                        if next_checkpoint_idx < len(checkpoints) and accounted_now + len(batch) >= checkpoints[next_checkpoint_idx]:
                            checkpoint = checkpoints[next_checkpoint_idx]
                            executed_this_run += execute_selected_requests(
                                project_root,
                                config,
                                batch,
                                manifest_sha,
                                protocol_hash,
                                allow_api_calls=allow_api_calls,
                                allow_local_ollama=allow_local_ollama,
                                gate_stage=f"full_checkpoint_{checkpoint}",
                            )
                            batch = []
                            status = build_status(project_root, frame, manifest_sha, protocol_hash)
                            checkpoint_health = health_summary(status, label=f"checkpoint_{checkpoint}")
                            ok, reason = health_pass(checkpoint_health, config)
                            write_diagnostic_frame(project_root, config, checkpoint_health, f"health_checkpoint_{checkpoint}.csv")
                            if not ok:
                                decision = DECISION_EXECUTION_FAILED
                                blocked_reason = f"CIRCUIT BREAKER TRIGGERED at {checkpoint}: {reason}"
                                full_run_authorized = False
                                break
                            next_checkpoint_idx += 1
                    if decision != DECISION_EXECUTION_FAILED:
                        if batch:
                            executed_this_run += execute_selected_requests(
                                project_root,
                                config,
                                batch,
                                manifest_sha,
                                protocol_hash,
                                allow_api_calls=allow_api_calls,
                                allow_local_ollama=allow_local_ollama,
                                gate_stage="full_final",
                            )
                        status = build_status(project_root, frame, manifest_sha, protocol_hash)
                        terminal_count = int((~status["execution_status"].eq("NOT_EXECUTED")).sum())
                        if terminal_count != len(frame):
                            decision = DECISION_EXECUTION_FAILED
                            blocked_reason = f"full execution incomplete: {terminal_count}/{len(frame)}"
                        else:
                            final_health = health_summary(status, label="full")
                            ok, reason = health_pass(final_health, config)
                            write_diagnostic_frame(project_root, config, final_health, "health_full.csv")
                            if not ok:
                                decision = DECISION_SCIENTIFIC_FAILED
                                blocked_reason = f"final scientific health failed: {reason}"
                                full_run_authorized = False
                            else:
                                decision = DECISION_COMPLETE
    elif static_pass and synthetic_pass and context_pass:
        decision = DECISION_PREFLIGHT_READY
        blocked_reason = "preflight ready but canary/full run not requested"

    frames = rebuild_frames(
        project_root,
        config,
        frame,
        manifest_sha,
        protocol_hash,
        runtime_available,
        runtime_error,
        runtime_tags,
        dependency_validation,
        static_preflight,
        synthetic_tests,
        context_preflight,
        protocol_freeze,
        canary_table,
        early_health,
        scope_summary,
        decision,
        full_run_authorized,
    )
    report = render_report(frames, decision, manifest_sha, protocol_hash, full_run_authorized, blocked_reason, executed_this_run)
    if write:
        write_outputs(project_root, config, frames, report, manifest_sha, protocol_hash, decision)
        update_notes(project_root, decision, manifest_sha, protocol_hash, frames["request_execution_status"])
    return R5DBuildResult(
        decision=decision,
        frames=frames,
        report=report,
        manifest_sha256=manifest_sha,
        protocol_hash=protocol_hash,
        executed_this_run=executed_this_run,
        full_run_authorized=full_run_authorized,
        blocked_reason=blocked_reason,
    )


def print_summary(result: R5DBuildResult) -> None:
    status = result.frames["request_execution_status"]
    terminal = status.loc[~status["execution_status"].eq("NOT_EXECUTED")]
    valid = int(terminal["parse_valid"].eq("true").sum())
    static = "PASS" if result.frames["static_preflight"]["status"].eq("pass").all() else "FAIL"
    synthetic = "PASS" if result.frames["synthetic_tests"]["pass"].eq("true").all() else "FAIL"
    canary = "PASS" if not result.frames["canary_results"].empty and result.frames["canary_results"]["Valid"].eq("true").all() else ("NOT_RUN" if result.frames["canary_results"].empty else "FAIL")
    early = "PASS" if not result.frames["early_health"].empty and health_pass(result.frames["early_health"], {})[0] else ("NOT_RUN" if result.frames["early_health"].empty else "FAIL")
    gaps = result.frames["gap_resolution"]
    print(f"FINAL_DECISION={result.decision}")
    print(f"MANIFEST_SHA_VERIFIED={bool_text(result.manifest_sha256 == EXPECTED_R5A_MANIFEST_SHA256)}")
    print(f"R5D_PROTOCOL_HASH={result.protocol_hash}")
    print(f"STATIC_PREFLIGHT={static}")
    print(f"SYNTHETIC={synthetic}")
    print(f"CANARY={canary}")
    print(f"EARLY_HEALTH={early}")
    print(f"FULL RUN AUTHORIZED: {'YES' if result.full_run_authorized else 'NO'}")
    print("REQUESTS_EXPECTED=1200")
    print(f"REQUESTS_ACCOUNTED={len(terminal)}")
    print(f"VALID={valid}")
    print(f"INVALID={int(len(terminal) - valid)}")
    print(f"WRONG_LENGTH={int(terminal['validity_class'].eq('WRONG_LENGTH').sum())}")
    print(f"OUT_OF_RANGE={int(terminal['validity_class'].eq('OUT_OF_RANGE').sum())}")
    print(f"PARSE_FAILURE={int(terminal['validity_class'].eq('PARSE_FAILURE').sum())}")
    print(f"BACKEND_INCOMPLETE={int(terminal['validity_class'].eq('BACKEND_INCOMPLETE').sum())}")
    print(f"TARGET_GAPS_RECOVERED={int(gaps['gap_recovered'].eq('true').sum()) if not gaps.empty else 0}/{len(gaps)}")
    print(f"EXECUTED_THIS_RUN={result.executed_this_run}")
    if result.blocked_reason:
        print(f"blocked_reason={result.blocked_reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--run-gated", action="store_true", help="Run preflight, canary, early health, and full run only if gates pass.")
    parser.add_argument("--stop-after", choices=["canary", "early", "full"], default="full")
    parser.add_argument("--allow-api-calls", action="store_true", help="Required before any local LLM/Ollama inference call.")
    parser.add_argument("--allow-local-ollama", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = Path(args.project_root or config.get("project_root", ".")).resolve()
    try:
        result = build_r5d_generation_involved(
            project_root,
            config_path=config_path,
            write=not args.no_write,
            run_gated=args.run_gated,
            allow_api_calls=args.allow_api_calls,
            allow_local_ollama=args.allow_local_ollama,
            stop_after=args.stop_after,
        )
    except R5DError as exc:
        print(DECISION_EXECUTION_FAILED)
        print(f"blocked_reason={exc}")
        return 2
    print_summary(result)
    return 0 if result.decision in {DECISION_COMPLETE, DECISION_PREFLIGHT_READY, DECISION_CANARY_PASS, DECISION_EARLY_PASS} else 1


if __name__ == "__main__":
    raise SystemExit(main())

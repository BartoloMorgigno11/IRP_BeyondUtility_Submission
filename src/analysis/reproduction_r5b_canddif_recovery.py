"""Execute and validate the R5B controlled CandDif recovery.

R5B is intentionally narrow: it executes only the frozen CandDif rows from the
R5A manifest through a local Ollama endpoint, persists one atomic JSON artifact
per request, and computes offline HR@5/NDCG@5/CandDif summaries. It never
regenerates prompts, candidates, histories, or request IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from src.analysis.r1_metric_recovery_paths import relocation_record, resolve_r1_metric_recovery_artifact
from src.llm.canonical_llm_contract import parse_strict_index_response


EXPECTED_HEAD = "98c28c5364c40e571765a3b50d4dab8affe80743"
EXPECTED_BRANCH = "master"
EXPECTED_R5A_MANIFEST_SHA256 = "290CCB9E809FF7BDAECDF83AE9C2AC9DF79A3715B6FC9CC14E520521001E0AFD"
DECISION_COMPLETE = "REPRODUCTION_R5B_CANDDIF_RECOVERY_COMPLETE"
DECISION_PARTIAL = "REPRODUCTION_R5B_CANDDIF_RECOVERY_PARTIAL"
DECISION_BLOCKED = "REPRODUCTION_R5B_CANDDIF_RECOVERY_BLOCKED"

DEFAULT_CONFIG_PATH = Path("configs") / "reproduction_r5b_canddif_recovery.yaml"
NOTES_PATH = Path("report") / "notes.md"
R4_PROTECTED_HASH_CHECK = Path("results") / "tables" / "reproduction_r4_protected_artifact_hash_check.csv"
R4_GAP_PRIORITY = Path("results") / "tables" / "reproduction_r4_gap_scientific_priority.csv"
R4_CANDIDATE_POSITION = Path("results") / "tables" / "reproduction_r4_candidate_position_reconciliation.csv"
FAIL_STRICT_DENOMINATOR_POLICY = "FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO"
EVIDENCE_CLASS = "PAPER_ALIGNED_ADAPTED_RECOVERY"
GLOBAL_STATUS = "CRITICAL_EXPERIMENTAL_PATH_PROTOCOL_LIMITED"

P0_EXPECTED = {
    ("Beauty", "CandDif-HR"),
    ("Beauty", "CandDif-NDCG"),
    ("Sports", "CandDif-HR"),
    ("Sports", "CandDif-NDCG"),
    ("LastFM", "CandDif-HR"),
    ("LastFM", "CandDif-NDCG"),
}

PROTECTED_ROOT_ARTIFACTS = {
    "final_metric_paper_comparison_values.md",
    "original_repo_metric_file_inventory.csv",
    "original_repo_metric_values_decoded.csv",
    "original_repo_metric_values_decoded_summary.csv",
    "original_repo_metric_values_decoded_wide.csv",
    "original_repo_requested_metric_values_decoded.csv",
    "requested_metric_candidate_position_bias_values.csv",
    "requested_metric_generation_involved_performance_values.csv",
    "requested_metric_history_length_sensitivity_values.csv",
    "requested_metric_novelty_values.csv",
    "requested_metric_paper_comparison_values.csv",
    "requested_metric_values_summary.csv",
}

OUTPUT_KEYS = (
    "execution_manifest",
    "request_execution_status",
    "pair_validation",
    "output_validity",
    "metrics_by_condition",
    "canddif_results",
    "dataset_model_summary",
    "gap_resolution",
    "official_comparison",
    "runtime_provenance",
    "validation",
)


@dataclass(frozen=True)
class OllamaResult:
    response_received: bool
    response_text: str
    latency_seconds: float
    raw_response: dict[str, Any] | None
    error: str
    error_category: str
    runtime_model_id: str


@dataclass(frozen=True)
class BuildResult:
    decision: str
    frames: dict[str, pd.DataFrame]
    report: str
    manifest_sha256: str
    executed_this_run: int
    blocked_reason: str


class R5BContractError(RuntimeError):
    """Raised when frozen request identity or resume metadata is inconsistent."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def as_project_path(project_root: Path, path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else project_root / raw


def rel_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(frame.fillna("").to_csv(index=False, lineterminator="\n"))


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise R5BContractError(f"R5B result is not a JSON object: {path}")
    return payload


def load_config(path: Path | None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def git_stdout(args: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def git_lines(args: Sequence[str], cwd: Path) -> list[str]:
    output = git_stdout(args, cwd)
    return [line for line in output.splitlines() if line]


def porcelain_path(line: str) -> str:
    if len(line) >= 4 and line[2] == " ":
        path = line[3:]
    elif " " in line:
        path = line.split(maxsplit=1)[1]
    else:
        path = line
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip()


def output_path(config: Mapping[str, Any], key: str) -> Path:
    return Path((config.get("outputs") or {})[key])


def configured_outputs(config: Mapping[str, Any]) -> dict[str, Path]:
    return {key: output_path(config, key) for key in OUTPUT_KEYS}


def configured_report_path(config: Mapping[str, Any]) -> Path:
    return Path((config.get("outputs") or {})["report"])


def r5a_manifest_path(config: Mapping[str, Any]) -> Path:
    return Path((config.get("frozen_r5a") or {})["manifest"])


def expected_manifest_sha(config: Mapping[str, Any]) -> str:
    return str((config.get("frozen_r5a") or {}).get("manifest_sha256", EXPECTED_R5A_MANIFEST_SHA256)).upper()


def candidate_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = json.loads(str(row["candidate_list"]))
    if not isinstance(parsed, list):
        raise R5BContractError(f"candidate_list is not a list for {row['request_id']}")
    return [dict(item) for item in parsed]


def history_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = json.loads(str(row["history_items"]))
    if not isinstance(parsed, list):
        raise R5BContractError(f"history_items is not a list for {row['request_id']}")
    return [dict(item) for item in parsed]


def candidate_item_ids(row: Mapping[str, Any]) -> list[str]:
    return [str(item.get("item_id", "")) for item in candidate_records(row)]


def recompute_prompt_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(str(row["prompt_render"]))


def recompute_candidate_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(str(row["candidate_list"]))


def recompute_history_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(str(row["history_items"]))


def normalize_bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def positive_rank(row: Mapping[str, Any], output_items: Sequence[str]) -> int:
    positive = str(row.get("positive_item_id", ""))
    for idx, item_id in enumerate(output_items, start=1):
        if str(item_id) == positive:
            return idx
    return 0


def hr_at_5(rank: int) -> float:
    return 1.0 if 1 <= rank <= 5 else 0.0


def ndcg_at_5(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if 1 <= rank <= 5 else 0.0


def canddif_value(acc_first: float, acc_random: float) -> tuple[float, str]:
    numerator = 1.0 - float(acc_random)
    denominator = 1.0 - float(acc_first)
    if numerator <= 0 and denominator <= 0:
        return 0.0, "BOTH_ERROR_RATES_ZERO_RETURNED_0"
    if numerator <= 0:
        return math.log(1e-12 / max(denominator, 1e-12)), "NUMERATOR_CLAMPED_EPSILON_1E-12"
    if denominator <= 0:
        return math.log(numerator / 1e-12), "DENOMINATOR_CLAMPED_EPSILON_1E-12"
    return math.log(numerator / denominator), "NONE"


def parse_expected_count(row: Mapping[str, Any]) -> int:
    schema = str(row.get("expected_output_schema", "")).strip()
    try:
        parsed = json.loads(schema)
    except json.JSONDecodeError:
        return 5
    values = parsed.get("ranked_candidate_indices") if isinstance(parsed, dict) else None
    return len(values) if isinstance(values, list) and values else 5


def load_r5a_manifest(project_root: Path, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    path = as_project_path(project_root, r5a_manifest_path(config))
    if not path.exists():
        raise FileNotFoundError(path)
    observed_sha = sha256_file(path)
    expected_sha = expected_manifest_sha(config)
    if observed_sha != expected_sha:
        raise R5BContractError(
            f"R5B_MANIFEST_HASH_MISMATCH observed={observed_sha} expected={expected_sha}"
        )
    manifest = read_csv(path)
    expected_rows = int((config.get("frozen_r5a") or {}).get("manifest_rows", 4800))
    if len(manifest) != expected_rows:
        raise R5BContractError(f"R5A manifest row count changed: {len(manifest)} != {expected_rows}")

    scope = config.get("scope") or {}
    family = str(scope.get("recovery_family", "CANDDIF_RECOVERY"))
    canddif = manifest.loc[manifest["recovery_family"].eq(family)].copy()
    selected_rows = int(scope.get("selected_rows", 1200))
    if len(canddif) != selected_rows:
        raise R5BContractError(f"CandDif subset row count changed: {len(canddif)} != {selected_rows}")
    if not canddif["task"].eq("ranking").all():
        raise R5BContractError("CandDif subset contains non-ranking task rows")
    if canddif["dataset"].eq("ML-1M").any():
        raise R5BContractError("R5B CandDif subset unexpectedly contains ML-1M rows")
    expected_datasets = set(scope.get("datasets") or ["Beauty", "Sports", "LastFM"])
    expected_models = set(scope.get("models") or ["qwen2:7b-instruct", "mistral:7b-instruct"])
    expected_conditions = set(scope.get("conditions") or ["CANDDIF_FIRST", "CANDDIF_RANDOM"])
    if set(canddif["dataset"]) != expected_datasets:
        raise R5BContractError(f"CandDif datasets changed: {sorted(set(canddif['dataset']))}")
    if set(canddif["model_name"]) != expected_models:
        raise R5BContractError(f"CandDif models changed: {sorted(set(canddif['model_name']))}")
    if set(canddif["experimental_condition"]) != expected_conditions:
        raise R5BContractError(f"CandDif conditions changed: {sorted(set(canddif['experimental_condition']))}")
    if not canddif["request_id"].is_unique:
        raise R5BContractError("CandDif request IDs are not unique")
    if not canddif["output_destination"].is_unique:
        raise R5BContractError("CandDif output destinations are not unique")
    return manifest, canddif.reset_index(drop=True), observed_sha


def validate_material_hashes(canddif: pd.DataFrame) -> tuple[int, int, int]:
    prompt_mismatch = 0
    candidate_mismatch = 0
    history_mismatch = 0
    for row in canddif.to_dict("records"):
        prompt_mismatch += int(recompute_prompt_hash(row) != str(row["prompt_hash"]).upper())
        candidate_mismatch += int(recompute_candidate_hash(row) != str(row["candidate_list_hash"]).upper())
        history_mismatch += int(recompute_history_hash(row) != str(row["history_hash"]).upper())
    return prompt_mismatch, candidate_mismatch, history_mismatch


def random_order_matches(first_row: Mapping[str, Any], random_row: Mapping[str, Any]) -> bool:
    seed = str(random_row["candidate_order_seed"])
    dataset = str(random_row["dataset"])
    user_id = str(random_row["user_id"])
    candidates = candidate_records(first_row)
    expected = sorted(
        candidates,
        key=lambda item: sha256_text(
            f"{seed}|{dataset}|{user_id}|{item.get('item_id', '')}|{item.get('title', '')}"
        ),
    )
    if expected and bool(expected[0].get("is_positive")) and len(expected) > 1:
        expected = expected[1:] + expected[:1]
    observed_ids = [str(item["item_id"]) for item in candidate_records(random_row)]
    expected_ids = [str(item["item_id"]) for item in expected]
    return observed_ids == expected_ids


def build_pair_validation(canddif: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = canddif.groupby(["dataset", "model_name", "user_id"], sort=True)
    for idx, ((dataset, model_name, user_id), group) in enumerate(grouped, start=1):
        conditions = set(group["experimental_condition"])
        has_pair = conditions == {"CANDDIF_FIRST", "CANDDIF_RANDOM"} and len(group) == 2
        if has_pair:
            first = group.loc[group["experimental_condition"].eq("CANDDIF_FIRST")].iloc[0].to_dict()
            random = group.loc[group["experimental_condition"].eq("CANDDIF_RANDOM")].iloc[0].to_dict()
            first_ids = candidate_item_ids(first)
            random_ids = candidate_item_ids(random)
            same_set = sorted(first_ids) == sorted(random_ids)
            different_order = first_ids != random_ids
            same_history = str(first["history_hash"]) == str(random["history_hash"])
            same_prompt_family = str(first["prompt_template_id"]) == str(random["prompt_template_id"])
            same_unordered = str(first["unordered_candidate_item_set_hash"]) == str(
                random["unordered_candidate_item_set_hash"]
            )
            first_positive = str(first["positive_candidate_position"])
            random_positive = str(random["positive_candidate_position"])
            random_seed_ok = random_order_matches(first, random)
            pair_valid = (
                same_set
                and different_order
                and same_history
                and same_prompt_family
                and same_unordered
                and first_positive == "1"
                and random_positive != "1"
                and random_seed_ok
            )
            first_request_id = first["request_id"]
            random_request_id = random["request_id"]
            notes = "first/random pair preserves history and unordered candidate set"
        else:
            pair_valid = False
            first_request_id = ""
            random_request_id = ""
            same_set = False
            different_order = False
            same_history = False
            same_prompt_family = False
            first_positive = ""
            random_positive = ""
            random_seed_ok = False
            notes = f"unexpected conditions={sorted(conditions)} row_count={len(group)}"
        rows.append(
            {
                "pair_id": f"R5B-CANDDIF-PAIR-{idx:04d}",
                "dataset": dataset,
                "model_name": model_name,
                "user_id": user_id,
                "first_request_id": first_request_id,
                "random_request_id": random_request_id,
                "same_candidate_items": normalize_bool_text(same_set),
                "different_order_as_expected": normalize_bool_text(different_order),
                "same_history": normalize_bool_text(same_history),
                "same_prompt_family": normalize_bool_text(same_prompt_family),
                "same_unordered_candidate_set_hash": normalize_bool_text(
                    has_pair
                    and str(first["unordered_candidate_item_set_hash"])
                    == str(random["unordered_candidate_item_set_hash"])
                )
                if has_pair
                else "false",
                "first_positive_position": first_positive,
                "random_positive_position": random_positive,
                "random_order_matches_seed": normalize_bool_text(random_seed_ok),
                "pair_valid": normalize_bool_text(pair_valid),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def build_execution_manifest(canddif: pd.DataFrame, manifest_sha: str) -> pd.DataFrame:
    keep = [
        "request_id",
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
        "candidate_policy",
        "candidate_order_seed",
        "candidate_order_algorithm",
        "candidate_count",
        "candidate_list_hash",
        "unordered_candidate_item_set_hash",
        "positive_item_id",
        "positive_candidate_position",
        "prompt_template_id",
        "prompt_hash",
        "parser_contract",
        "expected_output_schema",
        "temperature",
        "seed",
        "max_new_tokens",
        "output_destination",
        "gap_ids_served",
        "metrics_served",
        "request_key_sha256",
        "material_context_hash",
    ]
    frame = canddif[keep].copy()
    frame.insert(0, "r5b_execution_scope", "CANDDIF_ONLY")
    frame.insert(1, "r5a_manifest_sha256", manifest_sha)
    frame.insert(2, "recovery_request_id", frame["request_id"])
    frame["execution_allowed_in_r5b"] = "true"
    frame["execution_backend"] = "local_ollama"
    frame["external_api_allowed"] = "false"
    frame["evidence_class"] = EVIDENCE_CLASS
    frame["global_status"] = GLOBAL_STATUS
    return frame


def classify_parse_failure(status: str) -> str:
    if status == "parsed_valid":
        return ""
    if status == "duplicate_indices":
        return "duplicate_index_or_item"
    if status == "unknown_indices":
        return "out_of_range_index"
    if status == "invalid_length":
        return "wrong_output_length"
    if status in {"parse_failed", "empty_output"}:
        return "parse_failure"
    if status in {"invalid_schema", "missing_permutation_indices"}:
        return "structural_invalidity"
    return "other_invalidity"


def parse_output_record(row: Mapping[str, Any], response_text: str) -> dict[str, Any]:
    candidate_ids = candidate_item_ids(row)
    expected_count = parse_expected_count(row)
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
        "parse_failure_category": classify_parse_failure(parsed.status),
        "ranked_candidate_indices": list(parsed.output_indices),
        "ranked_item_ids": list(parsed.output_items),
        "duplicate_count": int(parsed.duplicate_count),
        "unknown_item_count": int(parsed.unknown_item_count),
        "output_length": int(parsed.output_length),
        "expected_output_count": expected_count,
        "positive_rank": rank,
        "hr_at_5": hr_at_5(rank) if parsed.parse_valid else 0.0,
        "ndcg_at_5": ndcg_at_5(rank) if parsed.parse_valid else 0.0,
        "invalid_output_metric_policy": FAIL_STRICT_DENOMINATOR_POLICY,
    }


def validate_existing_result(row: Mapping[str, Any], payload: Mapping[str, Any], manifest_sha: str, path: Path) -> None:
    required_equal = {
        "request_id": str(row["request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "user_id": str(row["user_id"]),
        "experimental_condition": str(row["experimental_condition"]),
        "prompt_hash": str(row["prompt_hash"]),
        "candidate_list_hash": str(row["candidate_list_hash"]),
        "history_hash": str(row["history_hash"]),
        "material_context_hash": str(row["material_context_hash"]),
    }
    for key, expected in required_equal.items():
        observed = str(payload.get(key, ""))
        if observed != str(expected):
            raise R5BContractError(
                f"existing R5B output identity mismatch at {path}: {key} observed={observed} expected={expected}"
            )
    if str(payload.get("execution_status", "")) not in {"SUCCEEDED", "BACKEND_FAILED"}:
        raise R5BContractError(f"existing R5B output is non-terminal at {path}")


def build_result_payload(
    row: Mapping[str, Any],
    manifest_sha: str,
    runtime: OllamaResult,
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    parse_record = parse_output_record(row, runtime.response_text)
    status = "SUCCEEDED" if runtime.response_received else "BACKEND_FAILED"
    error_category = runtime.error_category or parse_record["parse_failure_category"]
    payload: dict[str, Any] = {
        "phase": "R5B",
        "request_id": str(row["request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "model_label": str(row.get("model_label", "")),
        "user_id": str(row["user_id"]),
        "task": str(row["task"]),
        "recovery_family": str(row["recovery_family"]),
        "experimental_condition": str(row["experimental_condition"]),
        "prompt_hash": str(row["prompt_hash"]),
        "candidate_list_hash": str(row["candidate_list_hash"]),
        "unordered_candidate_item_set_hash": str(row["unordered_candidate_item_set_hash"]),
        "history_hash": str(row["history_hash"]),
        "material_context_hash": str(row["material_context_hash"]),
        "request_key_sha256": str(row["request_key_sha256"]),
        "positive_item_id": str(row["positive_item_id"]),
        "positive_candidate_position": str(row["positive_candidate_position"]),
        "candidate_count": int(row["candidate_count"]),
        "parser_contract": str(row["parser_contract"]),
        "expected_output_schema": str(row["expected_output_schema"]),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "runtime_backend": "ollama",
        "runtime_model_id": runtime.runtime_model_id,
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
        **parse_record,
    }
    result_hash = sha256_text(canonical_json({k: v for k, v in payload.items() if k != "result_hash"}))
    payload["result_hash"] = result_hash
    return payload


def ensure_local_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http":
        raise R5BContractError(f"R5B only permits local HTTP Ollama endpoints, observed {endpoint}")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise R5BContractError(f"R5B only permits localhost Ollama endpoints, observed {endpoint}")


def http_json_get(endpoint: str, timeout_seconds: int) -> dict[str, Any]:
    ensure_local_endpoint(endpoint)
    with urllib.request.urlopen(endpoint, timeout=timeout_seconds) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise R5BContractError(f"local endpoint returned non-object JSON: {endpoint}")
    return parsed


def ollama_version(config: Mapping[str, Any]) -> str:
    endpoint = str((config.get("runtime") or {}).get("version_endpoint", "http://127.0.0.1:11434/api/version"))
    try:
        payload = http_json_get(endpoint, int((config.get("runtime") or {}).get("timeout_seconds", 120)))
    except Exception as exc:
        return f"unavailable:{exc}"
    return str(payload.get("version", "unknown"))


def ollama_tags(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    endpoint = str((config.get("runtime") or {}).get("tags_endpoint", "http://127.0.0.1:11434/api/tags"))
    payload = http_json_get(endpoint, int((config.get("runtime") or {}).get("timeout_seconds", 120)))
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise R5BContractError("Ollama tags endpoint returned invalid models list")
    return {str(item.get("name", "")): dict(item) for item in models if isinstance(item, dict)}


def validate_runtime_available(config: Mapping[str, Any], required_models: Iterable[str]) -> tuple[bool, str, dict[str, dict[str, Any]]]:
    try:
        tags = ollama_tags(config)
    except Exception as exc:
        return False, str(exc), {}
    missing = sorted(set(required_models).difference(tags))
    if missing:
        return False, f"missing Ollama models: {missing}", tags
    return True, "", tags


def generate_local_ollama(row: Mapping[str, Any], config: Mapping[str, Any]) -> OllamaResult:
    runtime_cfg = config.get("runtime") or {}
    endpoint = str(runtime_cfg.get("endpoint", "http://127.0.0.1:11434/api/generate"))
    ensure_local_endpoint(endpoint)
    options = {
        "temperature": float(row.get("temperature", runtime_cfg.get("temperature", 0))),
        "seed": int(row.get("seed", 0) or 0),
        "num_predict": int(row.get("max_new_tokens", runtime_cfg.get("max_new_tokens", 512)) or 512),
    }
    payload = {
        "model": str(row["model_name"]),
        "prompt": str(row["prompt_render"]),
        "stream": False,
        "format": str(runtime_cfg.get("format", "json")),
        "options": options,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=int(runtime_cfg.get("timeout_seconds", 120))) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return OllamaResult(False, "", time.perf_counter() - started, None, f"HTTP {exc.code}: {exc.reason}", "backend_http_error", str(row["model_name"]))
    except urllib.error.URLError as exc:
        return OllamaResult(False, "", time.perf_counter() - started, None, str(exc.reason), "backend_connection_error", str(row["model_name"]))
    except TimeoutError:
        return OllamaResult(False, "", time.perf_counter() - started, None, "timed out", "backend_timeout", str(row["model_name"]))
    except Exception as exc:  # pragma: no cover - depends on local Ollama runtime state.
        return OllamaResult(False, "", time.perf_counter() - started, None, str(exc), "backend_error", str(row["model_name"]))
    latency = time.perf_counter() - started
    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        return OllamaResult(False, "", latency, None, f"invalid Ollama JSON: {exc}", "backend_invalid_json", str(row["model_name"]))
    response_text = parsed.get("response")
    if not isinstance(response_text, str):
        return OllamaResult(False, "", latency, parsed, "Ollama response field missing", "backend_invalid_response", str(row["model_name"]))
    model_id = str(parsed.get("model") or row["model_name"])
    return OllamaResult(True, response_text.strip(), latency, parsed, "", "", model_id)


def execute_requests(
    project_root: Path,
    config: Mapping[str, Any],
    canddif: pd.DataFrame,
    manifest_sha: str,
    *,
    allow_local_ollama: bool,
    max_requests: int | None = None,
    smoke: bool = False,
) -> int:
    if not allow_local_ollama:
        raise R5BContractError("R5B execution requires --allow-local-ollama")
    rows = canddif.to_dict("records")
    if smoke:
        selected_ids: set[str] = set()
        for _, group in canddif.groupby(["model_name"], sort=True):
            first_group = group.groupby(["dataset", "user_id"], sort=True)
            _, pair = next(iter(first_group))
            selected_ids.update(pair["request_id"].astype(str).tolist())
        rows = [row for row in rows if str(row["request_id"]) in selected_ids]
    executed = 0
    for row in rows:
        destination = as_project_path(project_root, str(row["output_destination"]))
        if not rel_path(project_root, destination).startswith("results/recovery/r5/"):
            raise R5BContractError(f"output destination outside R5 namespace: {destination}")
        if destination.exists():
            payload = load_json(destination)
            validate_existing_result(row, payload, manifest_sha, destination)
            continue
        if max_requests is not None and executed >= max_requests:
            break
        started_at = utc_now()
        runtime = generate_local_ollama(row, config)
        ended_at = utc_now()
        payload = build_result_payload(row, manifest_sha, runtime, started_at, ended_at)
        write_json_atomic(payload, destination)
        executed += 1
    return executed


def status_for_missing(row: Mapping[str, Any], manifest_sha: str, path: Path) -> dict[str, Any]:
    return {
        "request_id": str(row["request_id"]),
        "recovery_request_id": str(row["request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "model": str(row["model_name"]),
        "user_id": str(row["user_id"]),
        "user": str(row["user_id"]),
        "experimental_condition": str(row["experimental_condition"]),
        "condition": str(row["experimental_condition"]),
        "output_destination": str(row["output_destination"]),
        "output_path": str(row["output_destination"]),
        "result_file_exists": "false",
        "execution_status": "NOT_EXECUTED",
        "terminal_accounting_status": "NOT_EXECUTED",
        "terminal_status": "false",
        "parse_valid": "false",
        "valid_output": "false",
        "parse_status": "",
        "parser_status": "",
        "parse_failure_category": "",
        "positive_rank": 0,
        "hr_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "error_category": "",
        "error": "",
        "retry_count": 0,
        "result_hash": "",
        "result_file_sha256": "",
        "latency_seconds": "",
        "started_at_utc": "",
        "ended_at_utc": "",
        "runtime_model_id": "",
        "resume_status": "MISSING",
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
    }


def status_for_payload(row: Mapping[str, Any], payload: Mapping[str, Any], manifest_sha: str, path: Path) -> dict[str, Any]:
    validate_existing_result(row, payload, manifest_sha, path)
    terminal_accounting = "BACKEND_FAILED"
    if str(payload.get("execution_status", "")) == "SUCCEEDED":
        terminal_accounting = "COMPLETED_VALID" if bool(payload.get("parse_valid", False)) else "COMPLETED_INVALID_OUTPUT"
    return {
        "request_id": str(row["request_id"]),
        "recovery_request_id": str(row["request_id"]),
        "r5a_manifest_sha256": manifest_sha,
        "dataset": str(row["dataset"]),
        "model_name": str(row["model_name"]),
        "model": str(row["model_name"]),
        "user_id": str(row["user_id"]),
        "user": str(row["user_id"]),
        "experimental_condition": str(row["experimental_condition"]),
        "condition": str(row["experimental_condition"]),
        "output_destination": str(row["output_destination"]),
        "output_path": str(row["output_destination"]),
        "result_file_exists": "true",
        "execution_status": str(payload.get("execution_status", "")),
        "terminal_accounting_status": terminal_accounting,
        "terminal_status": "true",
        "parse_valid": normalize_bool_text(bool(payload.get("parse_valid", False))),
        "valid_output": normalize_bool_text(bool(payload.get("parse_valid", False))),
        "parse_status": str(payload.get("parse_status", "")),
        "parser_status": str(payload.get("parse_status", "")),
        "parse_failure_category": str(payload.get("parse_failure_category", "")),
        "positive_rank": int(payload.get("positive_rank", 0) or 0),
        "hr_at_5": float(payload.get("hr_at_5", 0.0) or 0.0),
        "ndcg_at_5": float(payload.get("ndcg_at_5", 0.0) or 0.0),
        "error_category": str(payload.get("error_category", "")),
        "error": str(payload.get("error", "")),
        "retry_count": int(payload.get("retry_count", 0) or 0),
        "result_hash": str(payload.get("result_hash", "")),
        "result_file_sha256": sha256_file(path),
        "latency_seconds": payload.get("latency_seconds", ""),
        "started_at_utc": str(payload.get("started_at_utc", "")),
        "ended_at_utc": str(payload.get("ended_at_utc", "")),
        "runtime_model_id": str(payload.get("runtime_model_id", "")),
        "resume_status": "REUSED_TERMINAL_OUTPUT",
        "metric_denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
    }


def build_status(canddif: pd.DataFrame, manifest_sha: str, project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in canddif.to_dict("records"):
        path = as_project_path(project_root, str(row["output_destination"]))
        if path.exists():
            rows.append(status_for_payload(row, load_json(path), manifest_sha, path))
        else:
            rows.append(status_for_missing(row, manifest_sha, path))
    return pd.DataFrame(rows)


def build_output_validity(status: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "request_id",
        "dataset",
        "model_name",
        "user_id",
        "experimental_condition",
        "execution_status",
        "parse_valid",
        "parse_status",
        "parse_failure_category",
        "positive_rank",
        "hr_at_5",
        "ndcg_at_5",
        "metric_denominator_policy",
        "result_hash",
    ]
    return status[keep].copy()


def build_metrics_by_condition(status: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = status.groupby(["dataset", "model_name", "experimental_condition"], sort=True)
    for (dataset, model_name, condition), group in grouped:
        n_manifest = len(group)
        n_executed = int((~group["execution_status"].eq("NOT_EXECUTED")).sum())
        n_valid = int(group["parse_valid"].eq("true").sum())
        n_invalid = int(group["execution_status"].eq("SUCCEEDED").sum()) - n_valid
        n_backend_failed = int(group["execution_status"].eq("BACKEND_FAILED").sum())
        n_not_executed = int(group["execution_status"].eq("NOT_EXECUTED").sum())
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "experimental_condition": condition,
                "n_manifest_requests": n_manifest,
                "n_manifest": n_manifest,
                "n_executed_or_terminal": n_executed,
                "n_executed": n_executed,
                "n_valid_outputs": n_valid,
                "n_valid": n_valid,
                "n_invalid_outputs": max(0, n_invalid),
                "n_invalid": max(0, n_invalid),
                "n_backend_failed": n_backend_failed,
                "n_not_executed": n_not_executed,
                "hr_at_5": float(group["hr_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "HR@5": float(group["hr_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "ndcg_at_5": float(group["ndcg_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "NDCG@5": float(group["ndcg_at_5"].astype(float).sum() / n_manifest) if n_manifest else 0.0,
                "denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
                "evidence_class": EVIDENCE_CLASS,
            }
        )
    return pd.DataFrame(rows)


def build_canddif_results(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = metrics.groupby(["dataset", "model_name"], sort=True)
    for (dataset, model_name), group in grouped:
        by_condition = group.set_index("experimental_condition")
        hr_first = float(by_condition.loc["CANDDIF_FIRST", "hr_at_5"]) if "CANDDIF_FIRST" in by_condition.index else 0.0
        hr_random = float(by_condition.loc["CANDDIF_RANDOM", "hr_at_5"]) if "CANDDIF_RANDOM" in by_condition.index else 0.0
        ndcg_first = float(by_condition.loc["CANDDIF_FIRST", "ndcg_at_5"]) if "CANDDIF_FIRST" in by_condition.index else 0.0
        ndcg_random = (
            float(by_condition.loc["CANDDIF_RANDOM", "ndcg_at_5"]) if "CANDDIF_RANDOM" in by_condition.index else 0.0
        )
        canddif_hr, hr_safety = canddif_value(hr_first, hr_random)
        canddif_ndcg, ndcg_safety = canddif_value(ndcg_first, ndcg_random)
        first_n = int(by_condition.loc["CANDDIF_FIRST", "n_manifest_requests"]) if "CANDDIF_FIRST" in by_condition.index else 0
        random_n = (
            int(by_condition.loc["CANDDIF_RANDOM", "n_manifest_requests"]) if "CANDDIF_RANDOM" in by_condition.index else 0
        )
        first_valid = int(by_condition.loc["CANDDIF_FIRST", "n_valid_outputs"]) if "CANDDIF_FIRST" in by_condition.index else 0
        random_valid = (
            int(by_condition.loc["CANDDIF_RANDOM", "n_valid_outputs"]) if "CANDDIF_RANDOM" in by_condition.index else 0
        )
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "cohort": "R5A_CANDDIF_100_USER_CONTROLLED",
                "n_users": min(first_n, random_n),
                "HR_first": hr_first,
                "HR_random": hr_random,
                "CandDif_HR": canddif_hr,
                "NDCG_first": ndcg_first,
                "NDCG_random": ndcg_random,
                "CandDif_NDCG": canddif_ndcg,
                "canddif_formula": "log((1-Acc_random)/(1-Acc_first))",
                "numerical_safety_policy_hr": hr_safety,
                "numerical_safety_policy_ndcg": ndcg_safety,
                "validity_summary": f"FIRST valid={first_valid}/{first_n}; RANDOM valid={random_valid}/{random_n}",
                "denominator_policy": FAIL_STRICT_DENOMINATOR_POLICY,
                "evidence_class": EVIDENCE_CLASS,
                "paper_official_comparison_availability": "SEE_reproduction_r5b_official_comparison.csv",
                "notes": "Paper-aligned adapted local recovery; not paper-exact protocol identity.",
                "global_status": GLOBAL_STATUS,
            }
        )
    return pd.DataFrame(rows)


def build_dataset_model_summary(status: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model_name), group in status.groupby(["dataset", "model_name"], sort=True):
        metric_group = metrics.loc[metrics["dataset"].eq(dataset) & metrics["model_name"].eq(model_name)]
        rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "n_manifest_requests": len(group),
                "n_succeeded": int(group["execution_status"].eq("SUCCEEDED").sum()),
                "n_backend_failed": int(group["execution_status"].eq("BACKEND_FAILED").sum()),
                "n_not_executed": int(group["execution_status"].eq("NOT_EXECUTED").sum()),
                "n_valid_outputs": int(group["parse_valid"].eq("true").sum()),
                "n_invalid_outputs": int(
                    group["execution_status"].eq("SUCCEEDED").sum() - group["parse_valid"].eq("true").sum()
                ),
                "first_hr_at_5": metric_value(metric_group, "CANDDIF_FIRST", "hr_at_5"),
                "random_hr_at_5": metric_value(metric_group, "CANDDIF_RANDOM", "hr_at_5"),
                "first_ndcg_at_5": metric_value(metric_group, "CANDDIF_FIRST", "ndcg_at_5"),
                "random_ndcg_at_5": metric_value(metric_group, "CANDDIF_RANDOM", "ndcg_at_5"),
                "evidence_class": EVIDENCE_CLASS,
            }
        )
    return pd.DataFrame(rows)


def metric_value(metrics: pd.DataFrame, condition: str, column: str) -> float:
    row = metrics.loc[metrics["experimental_condition"].eq(condition)]
    return float(row[column].iloc[0]) if not row.empty else 0.0


def load_p0_scope(project_root: Path) -> pd.DataFrame:
    path = project_root / R4_GAP_PRIORITY
    if not path.exists():
        return pd.DataFrame(columns=["gap_id", "dataset", "metric", "priority"])
    scope = read_csv(path)
    p0 = scope.loc[scope["priority"].eq("P0_MUST_CLOSE")].copy()
    return p0.loc[p0[["dataset", "metric"]].apply(tuple, axis=1).isin(P0_EXPECTED)]


def build_gap_resolution(project_root: Path, status: pd.DataFrame, canddif_results: pd.DataFrame) -> pd.DataFrame:
    p0 = load_p0_scope(project_root)
    rows: list[dict[str, Any]] = []
    status_by_dataset = status.groupby("dataset", sort=True)
    for dataset, metric in sorted(P0_EXPECTED):
        group = status_by_dataset.get_group(dataset) if dataset in status_by_dataset.groups else pd.DataFrame()
        metric_key = "CandDif_HR" if metric.endswith("HR") else "CandDif_NDCG"
        result_rows = canddif_results.loc[canddif_results["dataset"].eq(dataset)]
        metric_available = not result_rows.empty and metric_key in result_rows.columns
        terminal = not group.empty and not group["execution_status"].eq("NOT_EXECUTED").any()
        backend_failed = int(group["execution_status"].eq("BACKEND_FAILED").sum()) if not group.empty else 0
        status_text = (
            "RECOVERED_PAPER_ALIGNED_ADAPTED"
            if terminal and backend_failed == 0 and metric_available
            else "PARTIAL_RECOVERY_INCOMPLETE"
        )
        gap_rows = p0.loc[p0["dataset"].eq(dataset) & p0["metric"].eq(metric)]
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "gap_id": str(gap_rows["gap_id"].iloc[0]) if not gap_rows.empty else "",
                "priority": "P0_MUST_CLOSE",
                "requests_accounted": len(group),
                "not_executed": int(group["execution_status"].eq("NOT_EXECUTED").sum()) if not group.empty else 0,
                "backend_failed": backend_failed,
                "metric_rows": len(result_rows),
                "metric_value_column": metric_key,
                "resolution_status": status_text,
                "evidence_class": EVIDENCE_CLASS,
                "global_status": GLOBAL_STATUS,
            }
        )
    return pd.DataFrame(rows)


def build_official_comparison(project_root: Path, canddif_results: pd.DataFrame) -> pd.DataFrame:
    official_path = project_root / R4_CANDIDATE_POSITION
    official = read_csv(official_path) if official_path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in canddif_results.to_dict("records"):
        dataset = str(row["dataset"])
        official_rows = official.loc[official["dataset"].eq(dataset)] if not official.empty and "dataset" in official.columns else pd.DataFrame()
        official_exact = (
            str(official_rows["official_exact_canddif"].iloc[0])
            if not official_rows.empty and "official_exact_canddif" in official_rows
            else ""
        )
        official_partial = (
            str(official_rows["official_partial_canddif"].iloc[0])
            if not official_rows.empty and "official_partial_canddif" in official_rows
            else ""
        )
        local_diagnostic = (
            str(official_rows["local_canddif_diagnostics"].iloc[0])
            if not official_rows.empty and "local_canddif_diagnostics" in official_rows
            else ""
        )
        for metric, value_key in [("CandDif-HR", "CandDif_HR"), ("CandDif-NDCG", "CandDif_NDCG")]:
            rows.append(
                {
                    "dataset": dataset,
                    "model_name": row["model_name"],
                    "metric": metric,
                    "local_adapted_canddif": row[value_key],
                    "official_evidence_rows": len(official_rows),
                    "official_exact_canddif_cells": official_exact,
                    "official_partial_canddif_cells": official_partial,
                    "previous_local_canddif_diagnostic_cells": local_diagnostic,
                    "comparison_class": (
                        "DIRECTIONAL_COMPARISON_ONLY" if dataset in {"Beauty", "Sports"} else "NO_DIRECT_COMPARISON"
                    ),
                    "directional_agreement": "NOT_EVALUATED_NO_DEFENSIBLE_NUMERIC_BASELINE",
                    "notes": (
                        "R5B uses local adapted open models and strict index JSON output; "
                        "values are not paper-exact replacements."
                    ),
                }
            )
    return pd.DataFrame(rows)


def protected_hash_check(project_root: Path) -> pd.DataFrame:
    protected_path = project_root / R4_PROTECTED_HASH_CHECK
    if not protected_path.exists():
        return pd.DataFrame(
            columns=[
                "path",
                "exists",
                "tracked",
                "git_status",
                "expected_sha256",
                "observed_sha256",
                "hash_unchanged",
                "protection_status",
            ]
        )
    protected = read_csv(protected_path)
    status_lines = git_lines(["status", "--short", "--untracked-files=all"], project_root)
    status_by_path = {porcelain_path(line).replace("\\", "/"): line for line in status_lines}
    rows: list[dict[str, Any]] = []
    for row in protected.to_dict("records"):
        path_text = str(row["path"]).replace("\\", "/")
        path = resolve_r1_metric_recovery_artifact(project_root, path_text)
        record = relocation_record(project_root, path_text)
        exists = path.exists()
        observed = sha256_file(path) if exists else ""
        expected = (record.sha256 if record else str(row["expected_sha256"])).upper()
        current_path_text = record.current_path if record else path_text
        tracked = bool(git_stdout(["ls-files", "--", current_path_text], project_root))
        unchanged = exists and observed == expected and (record is not None or not tracked)
        rows.append(
            {
                "path": path_text,
                "exists": normalize_bool_text(exists),
                "tracked": normalize_bool_text(tracked),
                "git_status": status_by_path.get(current_path_text, ""),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_unchanged": normalize_bool_text(unchanged),
                "protection_status": "PROTECTED_UNCHANGED" if unchanged else "PROTECTED_CONFLICT",
            }
        )
    return pd.DataFrame(rows)


def out_of_scope_recovery_outputs(project_root: Path, canddif: pd.DataFrame, config: Mapping[str, Any]) -> list[str]:
    namespace = as_project_path(project_root, (config.get("outputs") or {}).get("result_namespace", "results/recovery/r5"))
    if not namespace.exists():
        return []
    allowed = {as_project_path(project_root, str(path)).resolve() for path in canddif["output_destination"].astype(str)}
    out: list[str] = []
    for path in namespace.glob("*.json"):
        if path.resolve() not in allowed:
            out.append(rel_path(project_root, path))
    return sorted(out)


def worktree_summary(project_root: Path) -> tuple[str, str, str]:
    branch = git_stdout(["branch", "--show-current"], project_root)
    head = git_stdout(["rev-parse", "HEAD"], project_root)
    staged = git_stdout(["diff", "--cached", "--stat"], project_root)
    return branch, head, staged


def build_runtime_provenance(
    project_root: Path,
    config: Mapping[str, Any],
    status: pd.DataFrame,
    runtime_tags: Mapping[str, Mapping[str, Any]],
    runtime_available: bool,
    runtime_error: str,
) -> pd.DataFrame:
    branch, head, staged = worktree_summary(project_root)
    disk_usage = shutil.disk_usage(project_root)
    rows: list[dict[str, Any]] = []
    version = ollama_version(config)
    required_models = sorted(set((config.get("scope") or {}).get("models") or []))
    for model in required_models:
        details = runtime_tags.get(model, {})
        model_details = details.get("details", {}) if isinstance(details.get("details"), dict) else {}
        model_status = status.loc[status["model_name"].eq(model)] if not status.empty else pd.DataFrame()
        rows.append(
            {
                "phase": "R5B",
                "backend": "ollama",
                "ollama_version": version,
                "model_name": model,
                "model_available": normalize_bool_text(model in runtime_tags),
                "model_digest": str(details.get("digest", "")),
                "model_parameter_size": str(model_details.get("parameter_size", "")),
                "model_quantization_level": str(model_details.get("quantization_level", "")),
                "context_length": str(model_details.get("context_length", "")),
                "runtime_available": normalize_bool_text(runtime_available),
                "runtime_error": runtime_error,
                "request_count": len(model_status),
                "succeeded": int(model_status["execution_status"].eq("SUCCEEDED").sum()) if not model_status.empty else 0,
                "backend_failed": int(model_status["execution_status"].eq("BACKEND_FAILED").sum()) if not model_status.empty else 0,
                "not_executed": int(model_status["execution_status"].eq("NOT_EXECUTED").sum()) if not model_status.empty else 0,
                "git_branch": branch,
                "git_head": head,
                "staged_diff_empty": normalize_bool_text(staged == ""),
                "disk_free_bytes": disk_usage.free,
                "external_api_allowed": "false",
                "downloads_allowed": "false",
                "training_allowed": "false",
            }
        )
    return pd.DataFrame(rows)


def build_validation(
    project_root: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    canddif: pd.DataFrame,
    manifest_sha: str,
    frames: Mapping[str, pd.DataFrame],
    runtime_available: bool,
    runtime_error: str,
    prompt_mismatch: int,
    candidate_mismatch: int,
    history_mismatch: int,
    out_of_scope_outputs: Sequence[str],
) -> tuple[pd.DataFrame, str, str]:
    status = frames["request_execution_status"]
    pair_validation = frames["pair_validation"]
    gap_resolution = frames["gap_resolution"]
    protected = protected_hash_check(project_root)
    branch, head, staged = worktree_summary(project_root)
    original_status = git_stdout(["-C", "original_repo/EvaLLMasRecommender", "status", "--short"], project_root)
    terminal_count = int((~status["execution_status"].eq("NOT_EXECUTED")).sum())
    succeeded = int(status["execution_status"].eq("SUCCEEDED").sum())
    backend_failed = int(status["execution_status"].eq("BACKEND_FAILED").sum())
    not_executed = int(status["execution_status"].eq("NOT_EXECUTED").sum())
    invalid = int(status["execution_status"].eq("SUCCEEDED").sum() - status["parse_valid"].eq("true").sum())
    p0_set = set(gap_resolution[["dataset", "metric"]].apply(tuple, axis=1))
    disk_free_min = int((config.get("runtime") or {}).get("disk_free_min_bytes", 0))
    disk_free = shutil.disk_usage(project_root).free
    checks: list[tuple[str, str, bool, str, str, str]] = [
        ("R5B-VAL-001", "expected branch", branch == EXPECTED_BRANCH, branch, EXPECTED_BRANCH, "R5B executes from the R5A freeze branch."),
        ("R5B-VAL-002", "expected head", head == EXPECTED_HEAD, head, EXPECTED_HEAD, "R5B starts from the frozen R5A commit."),
        ("R5B-VAL-003", "no staged diff", staged == "", staged, "empty", "R5B does not stage or commit artifacts."),
        ("R5B-VAL-004", "original repo clean", original_status == "", original_status, "empty", "R5B must not modify original_repo/EvaLLMasRecommender."),
        (
            "R5B-VAL-005",
            "protected artifacts unchanged",
            len(protected) == 12 and protected["protection_status"].eq("PROTECTED_UNCHANGED").all(),
            f"{int(protected['protection_status'].eq('PROTECTED_UNCHANGED').sum())}/{len(protected)}",
            "12/12",
            "The 12 protected root artifacts remain untracked and SHA256-identical.",
        ),
        ("R5B-VAL-006", "manifest sha exact", manifest_sha == expected_manifest_sha(config), manifest_sha, expected_manifest_sha(config), "R5B uses only the frozen R5A manifest."),
        ("R5B-VAL-007", "manifest row count", len(manifest) == 4800, str(len(manifest)), "4800", "R5A manifest size is unchanged."),
        ("R5B-VAL-008", "CandDif row count", len(canddif) == 1200, str(len(canddif)), "1200", "R5B selects only CANDDIF_RECOVERY rows."),
        (
            "R5B-VAL-009",
            "no History Generation ML1M CandDif",
            not canddif["dataset"].eq("ML-1M").any()
            and canddif["recovery_family"].eq("CANDDIF_RECOVERY").all()
            and canddif["task"].eq("ranking").all(),
            f"families={sorted(canddif['recovery_family'].unique())}; datasets={sorted(canddif['dataset'].unique())}",
            "CandDif ranking only; no ML-1M",
            "No out-of-scope recovery family is selected.",
        ),
        (
            "R5B-VAL-010",
            "pair count and validity",
            len(pair_validation) == 600 and pair_validation["pair_valid"].eq("true").all(),
            f"pairs={len(pair_validation)}; valid={int(pair_validation['pair_valid'].eq('true').sum())}",
            "600 valid",
            "Each user/model has FIRST and RANDOM rows with the frozen candidate relation.",
        ),
        (
            "R5B-VAL-011",
            "material hashes match manifest",
            prompt_mismatch == 0 and candidate_mismatch == 0 and history_mismatch == 0,
            f"prompt={prompt_mismatch};candidate={candidate_mismatch};history={history_mismatch}",
            "0;0;0",
            "R5B did not regenerate or alter prompts, candidates, or histories.",
        ),
        (
            "R5B-VAL-012",
            "runtime models available",
            runtime_available,
            runtime_error,
            "qwen2:7b-instruct and mistral:7b-instruct available",
            "Only local Ollama runtime is allowed.",
        ),
        (
            "R5B-VAL-013",
            "request accounting sums",
            len(status) == 1200 and terminal_count + not_executed == 1200,
            f"rows={len(status)};terminal={terminal_count};not_executed={not_executed}",
            "rows=1200;terminal+not_executed=1200",
            "Every frozen CandDif request has exactly one status row.",
        ),
        (
            "R5B-VAL-014",
            "all requests executed",
            not_executed == 0,
            str(not_executed),
            "0",
            "Completion requires all 1200 CandDif requests to reach a terminal state.",
        ),
        (
            "R5B-VAL-015",
            "no backend failures",
            backend_failed == 0,
            str(backend_failed),
            "0",
            "Runtime failures block a complete R5B freeze.",
        ),
        (
            "R5B-VAL-016",
            "invalid outputs preserved not hallucination",
            invalid >= 0 and status.loc[status["execution_status"].eq("SUCCEEDED"), "parse_status"].astype(str).str.len().gt(0).all(),
            f"invalid_outputs={invalid}",
            "invalid_outputs>=0 with parse status recorded",
            "Invalid outputs are retained and scored as zero, not repaired.",
        ),
        (
            "R5B-VAL-017",
            "P0 gap set accounted",
            p0_set == P0_EXPECTED and len(gap_resolution) == 6,
            canonical_json(sorted(p0_set)),
            canonical_json(sorted(P0_EXPECTED)),
            "Only the six R4 P0 CandDif rows are resolved by R5B.",
        ),
        (
            "R5B-VAL-018",
            "P0 gap resolution complete",
            gap_resolution["resolution_status"].eq("RECOVERED_PAPER_ALIGNED_ADAPTED").all(),
            gap_resolution["resolution_status"].value_counts().to_dict(),
            "all RECOVERED_PAPER_ALIGNED_ADAPTED",
            "Complete only when every P0 CandDif metric has terminal local adapted evidence.",
        ),
        (
            "R5B-VAL-019",
            "no out-of-scope raw outputs",
            len(out_of_scope_outputs) == 0,
            ";".join(out_of_scope_outputs[:5]),
            "none",
            "The R5 output namespace contains no non-CandDif artifacts.",
        ),
        (
            "R5B-VAL-020",
            "disk free guard",
            disk_free >= disk_free_min,
            str(disk_free),
            f">={disk_free_min}",
            "Local runtime has enough disk headroom for atomic raw output writes.",
        ),
    ]
    rows = [
        {
            "check_id": check_id,
            "check_name": name,
            "status": "pass" if ok else "fail",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
        for check_id, name, ok, observed, expected, details in checks
    ]
    validation = pd.DataFrame(rows)
    failed = validation.loc[validation["status"].eq("fail")]
    if failed.empty:
        return validation, DECISION_COMPLETE, ""
    complete_blockers = set(failed["check_id"]).intersection(
        {"R5B-VAL-001", "R5B-VAL-002", "R5B-VAL-003", "R5B-VAL-004", "R5B-VAL-005", "R5B-VAL-006", "R5B-VAL-007", "R5B-VAL-008", "R5B-VAL-009", "R5B-VAL-010", "R5B-VAL-011", "R5B-VAL-012", "R5B-VAL-019", "R5B-VAL-020"}
    )
    if complete_blockers:
        return validation, DECISION_BLOCKED, "; ".join(f"{row.check_id}:{row.check_name}" for row in failed.itertuples())
    return validation, DECISION_PARTIAL, "; ".join(f"{row.check_id}:{row.check_name}" for row in failed.itertuples())


def render_markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    selected = frame.loc[:, list(columns)].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(str(row[column]).replace("\n", " ") for column in columns)
        + " |"
        for row in selected.to_dict("records")
    ]
    return "\n".join([header, separator, *body])


def render_report(frames: Mapping[str, pd.DataFrame], decision: str, manifest_sha: str, blocked_reason: str) -> str:
    status = frames["request_execution_status"]
    validation = frames["validation"]
    summary = frames["dataset_model_summary"]
    canddif = frames["canddif_results"]
    gap = frames["gap_resolution"]
    runtime = frames["runtime_provenance"]
    metrics = frames["metrics_by_condition"]
    official = frames["official_comparison"]
    pass_count = int(validation["status"].eq("pass").sum())
    fail_count = int(validation["status"].eq("fail").sum())
    terminal = int((~status["execution_status"].eq("NOT_EXECUTED")).sum())
    succeeded = int(status["execution_status"].eq("SUCCEEDED").sum())
    invalid = int(status["execution_status"].eq("SUCCEEDED").sum() - status["parse_valid"].eq("true").sum())
    backend_failed = int(status["execution_status"].eq("BACKEND_FAILED").sum())
    not_executed = int(status["execution_status"].eq("NOT_EXECUTED").sum())
    valid = int(status["parse_valid"].eq("true").sum())
    return f"""# R5B Controlled CandDif Recovery

Decision: `{decision}`

## Purpose

Execute and audit only the frozen R5A CandDif recovery subset, preserving one atomic local Ollama JSON result per request and computing offline HR@5, NDCG@5, CandDif-HR, and CandDif-NDCG.

## Frozen R5A Execution Contract

R5B uses `results/tables/reproduction_r5a_recovery_request_manifest.csv` as the only request universe. It does not regenerate prompts, candidates, histories, models, users, request IDs, or output destinations.

## Manifest SHA

Manifest SHA256: `{manifest_sha}`

## Execution Scope

Selected rows: `{len(status)}` `CANDDIF_RECOVERY` requests. History selected: `0`. Generation selected: `0`. ML-1M CandDif selected: `0`.

## Runtime

{render_markdown_table(runtime, ["model_name", "model_available", "model_digest", "model_quantization_level", "request_count", "succeeded", "backend_failed", "not_executed"])}

## Models

R5B used only the frozen model identifiers in the manifest: `mistral:7b-instruct` and `qwen2:7b-instruct`.

## Datasets/Cohorts

Datasets: `Beauty`, `Sports`, and `LastFM`. Each dataset/model condition has the frozen 100-user R5A CandDif cohort.

## FIRST/RANDOM Protocol

Each logical pair has one `CANDDIF_FIRST` row and one `CANDDIF_RANDOM` row. The unordered candidate item set and history are shared; candidate order is the intended controlled difference.

## Execution Completeness

- Frozen CandDif requests: `{len(status)}`
- Terminal outputs: `{terminal}`
- Succeeded backend calls: `{succeeded}`
- Backend failures: `{backend_failed}`
- Not executed: `{not_executed}`

## Output Validity

- Valid strict-parser outputs: `{valid}`
- Invalid strict-parser outputs preserved: `{invalid}`
- Denominator policy: `{FAIL_STRICT_DENOMINATOR_POLICY}`

## HR/NDCG By Condition

{render_markdown_table(metrics, ["dataset", "model_name", "experimental_condition", "n_manifest", "n_valid", "n_invalid", "HR@5", "NDCG@5"])}

## CandDif-HR Results

{render_markdown_table(canddif, ["dataset", "model_name", "n_users", "HR_first", "HR_random", "CandDif_HR", "numerical_safety_policy_hr", "evidence_class"])}

## CandDif-NDCG Results

{render_markdown_table(canddif, ["dataset", "model_name", "n_users", "NDCG_first", "NDCG_random", "CandDif_NDCG", "numerical_safety_policy_ndcg", "evidence_class"])}

## Dataset/Model Interpretation

{render_markdown_table(summary, ["dataset", "model_name", "n_manifest_requests", "n_succeeded", "n_valid_outputs", "n_invalid_outputs", "n_backend_failed", "n_not_executed"])}

Positive CandDif means FIRST improves the corresponding accuracy relative to RANDOM; values near zero indicate little measured difference; negative values indicate RANDOM outperforms FIRST. No statistical significance is claimed.

## Comparison With Official Evidence

{render_markdown_table(official, ["dataset", "model_name", "metric", "local_adapted_canddif", "official_exact_canddif_cells", "official_partial_canddif_cells", "comparison_class"])}

R5B does not force numeric paper/official comparison where model identity, protocol, dataset condition, candidate construction, and output contract are not identical.

## Resolution Of Six P0 Gaps

{render_markdown_table(gap, ["dataset", "metric", "requests_accounted", "not_executed", "backend_failed", "resolution_status"])}

## Remaining Limitations

The recovered evidence is local and adapted: model identity, preprocessing, candidate generation lineage, and strict JSON index output differ from paper-exact conditions.

## Effect On Global Reproduction Status

The global project status remains `{GLOBAL_STATUS}`.

## Protection/Provenance

R5B validation checks the branch, HEAD, staged diff, `original_repo/` cleanliness, protected root hashes, manifest SHA, pair contract, runtime availability, request accounting, and output namespace.

## Validation

Validation result: `{pass_count} PASS / {fail_count} FAIL`.

{render_markdown_table(validation, ["check_id", "check_name", "status", "observed", "expected"])}

## What R5C May Do Next

After R5B review/freeze, R5C may address the frozen History-Length recovery subset. R5B does not execute R5C.

Blocked or partial reason: `{blocked_reason}`
"""


def update_notes(project_root: Path, frames: Mapping[str, pd.DataFrame], decision: str, manifest_sha: str) -> None:
    notes_path = project_root / NOTES_PATH
    status = frames["request_execution_status"]
    validation = frames["validation"]
    section = f"""
## R5B controlled CandDif recovery

- Decision: `{decision}`.
- Scope: executed only frozen R5A `CANDDIF_RECOVERY` requests; no History/Generation recovery, ML-1M CandDif, external APIs, training, downloads, Food.com work, or `original_repo/` modification.
- Manifest SHA256: `{manifest_sha}`.
- Outputs: `results/recovery/r5/*.json`, `results/tables/reproduction_r5b_*.csv`, and `report/reproduction_r5b_canddif_recovery.md`.
- Request accounting: total `{len(status)}`, succeeded `{int(status["execution_status"].eq("SUCCEEDED").sum())}`, backend failed `{int(status["execution_status"].eq("BACKEND_FAILED").sum())}`, not executed `{int(status["execution_status"].eq("NOT_EXECUTED").sum())}`.
- Parser/metric policy: strict JSON index parser; invalid outputs are preserved and score zero under `{FAIL_STRICT_DENOMINATOR_POLICY}`.
- Validation: `{int(validation["status"].eq("pass").sum())} PASS / {int(validation["status"].eq("fail").sum())} FAIL`.
- Evidence class: `{EVIDENCE_CLASS}`; global status remains `{GLOBAL_STATUS}`.
"""
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    marker = "## R5B controlled CandDif recovery"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n" + section
    else:
        existing = existing.rstrip() + "\n" + section
    notes_path.write_text(existing.lstrip(), encoding="utf-8")


def build_r5b(
    project_root: Path,
    config_path: Path | None = None,
    *,
    write: bool = True,
    execute: bool = False,
    smoke: bool = False,
    allow_local_ollama: bool = False,
    max_requests: int | None = None,
) -> BuildResult:
    config = load_config(config_path)
    manifest, canddif, manifest_sha = load_r5a_manifest(project_root, config)
    prompt_mismatch, candidate_mismatch, history_mismatch = validate_material_hashes(canddif)
    if prompt_mismatch or candidate_mismatch or history_mismatch:
        raise R5BContractError(
            f"material hash mismatch prompt={prompt_mismatch} candidate={candidate_mismatch} history={history_mismatch}"
        )
    runtime_available, runtime_error, runtime_tags = validate_runtime_available(
        config, (config.get("scope") or {}).get("models", [])
    )
    executed_this_run = 0
    if execute or smoke:
        if not runtime_available:
            raise R5BContractError(f"R5B local runtime unavailable: {runtime_error}")
        executed_this_run = execute_requests(
            project_root,
            config,
            canddif,
            manifest_sha,
            allow_local_ollama=allow_local_ollama,
            max_requests=max_requests,
            smoke=smoke,
        )

    pair_validation = build_pair_validation(canddif)
    execution_manifest = build_execution_manifest(canddif, manifest_sha)
    status = build_status(canddif, manifest_sha, project_root)
    output_validity = build_output_validity(status)
    metrics = build_metrics_by_condition(status)
    canddif_results = build_canddif_results(metrics)
    summary = build_dataset_model_summary(status, metrics)
    gap_resolution = build_gap_resolution(project_root, status, canddif_results)
    official_comparison = build_official_comparison(project_root, canddif_results)
    out_scope = out_of_scope_recovery_outputs(project_root, canddif, config)
    runtime_provenance = build_runtime_provenance(
        project_root, config, status, runtime_tags, runtime_available, runtime_error
    )
    frames: dict[str, pd.DataFrame] = {
        "execution_manifest": execution_manifest,
        "pair_validation": pair_validation,
        "request_execution_status": status,
        "output_validity": output_validity,
        "metrics_by_condition": metrics,
        "canddif_results": canddif_results,
        "dataset_model_summary": summary,
        "gap_resolution": gap_resolution,
        "official_comparison": official_comparison,
        "runtime_provenance": runtime_provenance,
    }
    validation, decision, blocked_reason = build_validation(
        project_root,
        config,
        manifest,
        canddif,
        manifest_sha,
        frames,
        runtime_available,
        runtime_error,
        prompt_mismatch,
        candidate_mismatch,
        history_mismatch,
        out_scope,
    )
    frames["validation"] = validation
    report = render_report(frames, decision, manifest_sha, blocked_reason)
    if write:
        outputs = configured_outputs(config)
        for key in OUTPUT_KEYS:
            write_csv(frames[key], project_root / outputs[key])
        report_path = project_root / configured_report_path(config)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        update_notes(project_root, frames, decision, manifest_sha)
    return BuildResult(
        decision=decision,
        frames=frames,
        report=report,
        manifest_sha256=manifest_sha,
        executed_this_run=executed_this_run,
        blocked_reason=blocked_reason,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-local-ollama", action="store_true")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    project_root = Path(args.project_root or config.get("project_root", ".")).resolve()
    should_execute = args.execute
    if args.preflight_only or args.analysis_only:
        should_execute = False
    try:
        result = build_r5b(
            project_root,
            config_path=config_path,
            write=not args.no_write,
            execute=should_execute,
            smoke=args.smoke,
            allow_local_ollama=args.allow_local_ollama,
            max_requests=args.max_requests,
        )
    except R5BContractError as exc:
        message = str(exc)
        if "R5B_MANIFEST_HASH_MISMATCH" in message:
            print("R5B_MANIFEST_HASH_MISMATCH")
        else:
            print(DECISION_BLOCKED)
        print(f"blocked_reason={message}")
        return 2

    status = result.frames["request_execution_status"]
    validation = result.frames["validation"]
    print(f"decision={result.decision}")
    print(f"manifest_sha256={result.manifest_sha256}")
    print(f"selected_rows={len(status)}")
    print(f"executed_this_run={result.executed_this_run}")
    print(f"succeeded={int(status['execution_status'].eq('SUCCEEDED').sum())}")
    print(f"backend_failed={int(status['execution_status'].eq('BACKEND_FAILED').sum())}")
    print(f"not_executed={int(status['execution_status'].eq('NOT_EXECUTED').sum())}")
    print(f"validation_pass={int(validation['status'].eq('pass').sum())}")
    print(f"validation_fail={int(validation['status'].eq('fail').sum())}")
    if result.blocked_reason:
        print(f"blocked_reason={result.blocked_reason}")
    return 0 if result.decision == DECISION_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())

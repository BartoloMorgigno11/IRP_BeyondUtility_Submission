"""Phase F6E4C offline scientific consolidation.

This module is intentionally offline. It reads the frozen F6E4A/F6E4B
artifacts, reconstructs request-level metrics from raw records, and writes
new F6E4C reports/tables/manifests without modifying prior phases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from src.analysis import plan_f6e4a_adapted_full_run_freeze as f6e4a


PHASE_ID = "phase_f6e4c"
EXPECTED_REQUESTS = 30492
EXPECTED_STRICT_VALID = 29055
EXPECTED_INVALID = 1437
EXPECTED_F6E4B_FREEZE_HASH = "31ae67cc87ca2db41a0b10a415c0f62b8635673eb06f37132a560fee4ecc521a"
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42
TOLERANCE = 1e-12

QWEN2_TAG = f6e4a.QWEN2_TAG
MISTRAL_TAG = f6e4a.MISTRAL_TAG
BASE_STRATEGY = f6e4a.STRATEGY_ORDER[0]
LLMRANK_STRATEGY = f6e4a.STRATEGY_ORDER[1]
MODEL_ORDER = [QWEN2_TAG, MISTRAL_TAG]
STRATEGY_ORDER = [BASE_STRATEGY, LLMRANK_STRATEGY]

REQUEST_MANIFEST = Path("results/tables/phase_f6e4a_request_manifest.csv")
SHARD_MANIFEST = Path("results/tables/phase_f6e4a_shard_manifest.csv")
F6E4A_FREEZE_MANIFEST = Path("results/manifests/phase_f6e4a_full_run_freeze.json")
F6E4A_RUNTIME_PARAMETERS = Path("results/tables/phase_f6e4a_runtime_parameters.csv")
F6E4A_GATE_PLAN = Path("results/tables/phase_f6e4a_gate_plan.csv")
F6E4B_FREEZE_MANIFEST = Path("results/manifests/phase_f6e4b_execution_freeze.json")
F6E4B_CELL_SUMMARY = Path("results/tables/phase_f6e4b_cell_summary.csv")
F6E4B_DATASET_TASK_SUMMARY = Path("results/tables/phase_f6e4b_dataset_task_summary.csv")
F6E4B_INSTANCE_RESULTS = Path("results/tables/phase_f6e4b_instance_results.csv")
F6E4B_PROGRESS = Path("results/tables/phase_f6e4b_progress.csv")
F6E4B_PARSER_VALIDITY = Path("results/tables/phase_f6e4b_parser_validity.csv")
F6E4B_INVALID_AUDIT = Path("results/tables/phase_f6e4b_invalid_output_audit.csv")
F6E4B_RAW_PATHS = {
    QWEN2_TAG: Path("results/llm/phase_f6e4b_qwen2_adapted_full_run_raw.jsonl"),
    MISTRAL_TAG: Path("results/llm/phase_f6e4b_mistral_adapted_full_run_raw.jsonl"),
}

REPORT_PATH = Path("report/phase_f6e4c_scientific_consolidation.md")
FREEZE_MANIFEST_PATH = Path("results/manifests/phase_f6e4c_scientific_freeze.json")
OUTPUT_TABLES = {
    "input_integrity": Path("results/tables/phase_f6e4c_input_integrity.csv"),
    "metric_contract": Path("results/tables/phase_f6e4c_metric_contract.csv"),
    "request_level_metrics": Path("results/tables/phase_f6e4c_request_level_metrics.csv"),
    "metric_reconciliation": Path("results/tables/phase_f6e4c_metric_reconciliation.csv"),
    "cell_summary": Path("results/tables/phase_f6e4c_cell_summary.csv"),
    "dataset_task_summary": Path("results/tables/phase_f6e4c_dataset_task_summary.csv"),
    "invalid_output_audit": Path("results/tables/phase_f6e4c_invalid_output_audit.csv"),
    "mistral_llmrank_lastfm_audit": Path("results/tables/phase_f6e4c_mistral_llmrank_lastfm_audit.csv"),
    "strategy_paired_comparison": Path("results/tables/phase_f6e4c_strategy_paired_comparison.csv"),
    "model_paired_comparison": Path("results/tables/phase_f6e4c_model_paired_comparison.csv"),
    "history_length_analysis": Path("results/tables/phase_f6e4c_history_length_analysis.csv"),
    "candidate_position_analysis": Path("results/tables/phase_f6e4c_candidate_position_analysis.csv"),
    "exact_vs_adapted_evidence": Path("results/tables/phase_f6e4c_exact_vs_adapted_evidence.csv"),
    "research_question_evidence": Path("results/tables/phase_f6e4c_research_question_evidence.csv"),
    "claim_registry": Path("results/tables/phase_f6e4c_claim_registry.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6e4c_missing_or_blocked.csv"),
}

INPUT_INTEGRITY_COLUMNS = [
    "artifact_id",
    "artifact_role",
    "path",
    "expected_sha256",
    "observed_sha256",
    "record_count",
    "status",
    "severity",
    "notes",
]
REQUEST_COLUMNS = [
    "request_id",
    "source_context_id",
    "source_request_id",
    "shard_id",
    "ordinal_position",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "adapted_strategy_id",
    "adapter_classification",
    "candidate_count",
    "history_length",
    "history_length_stratum",
    "positive_item_id",
    "positive_initial_position",
    "positive_position_stratum",
    "transport_status",
    "strict_valid",
    "invalid_reason",
    "output_length",
    "parsed_indices",
    "parsed_item_ids",
    "positive_rank",
    "hr_at1_operational",
    "hr_at5_operational",
    "hr_at10_operational",
    "ndcg_at5_operational",
    "ndcg_at10_operational",
    "hr_at1_conditional",
    "hr_at5_conditional",
    "hr_at10_conditional",
    "ndcg_at5_conditional",
    "ndcg_at10_conditional",
    "duration_seconds",
    "prompt_eval_count",
    "eval_count",
    "raw_output_file",
    "metric_source",
]
SUMMARY_COLUMNS = [
    "summary_id",
    "scope",
    "aggregation",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "logical_requests",
    "transport_success",
    "strict_valid",
    "invalid_outputs",
    "strict_valid_rate",
    "invalid_reason_distribution",
    "operational_hr_at1",
    "operational_hr_at5",
    "operational_hr_at10",
    "operational_ndcg_at5",
    "operational_ndcg_at10",
    "conditional_hr_at1",
    "conditional_hr_at5",
    "conditional_hr_at10",
    "conditional_ndcg_at5",
    "conditional_ndcg_at10",
    "mean_positive_rank",
    "latency_mean",
    "latency_p50",
    "latency_p95",
    "prompt_tokens_mean",
    "generated_tokens_mean",
    "status",
    "notes",
]
CONTRACT_COLUMNS = [
    "contract_id",
    "scope",
    "aggregation",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "metric",
    "operational_value",
    "conditional_value",
    "strict_valid_rate",
    "expected_operational_value",
    "absolute_error",
    "tolerance",
    "status",
    "notes",
]
RECON_COLUMNS = [
    "reconciliation_id",
    "scope",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "metric",
    "f6e4b_value",
    "f6e4c_value",
    "difference",
    "f6e4b_denominator",
    "f6e4c_denominator",
    "population",
    "cause",
    "status",
    "notes",
]
INVALID_COLUMNS = [
    "audit_id",
    "request_id",
    "model",
    "strategy_id",
    "dataset",
    "task",
    "history_length",
    "history_length_stratum",
    "candidate_count",
    "positive_initial_position",
    "positive_position_stratum",
    "output_length",
    "generated_token_count",
    "raw_response_length",
    "invalid_reason",
    "positive_appears_in_raw_parsed_content",
    "f6e4b_operational_counted_positive",
    "f6e4c_operational_utility_zero",
    "notes",
]
PAIR_COLUMNS = [
    "comparison_id",
    "comparison_type",
    "population",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "left_label",
    "right_label",
    "n_pairs",
    "n_strict_intersection",
    "metric",
    "left_mean",
    "right_mean",
    "delta_right_minus_left",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "win_count",
    "tie_count",
    "loss_count",
    "delta_strict_validity",
    "delta_latency",
    "delta_generated_tokens",
    "bootstrap_resamples",
    "seed",
    "notes",
]
STRATA_COLUMNS = [
    "analysis_id",
    "analysis_type",
    "dataset",
    "task",
    "model",
    "strategy_id",
    "stratum_name",
    "logical_requests",
    "strict_valid",
    "strict_valid_rate",
    "operational_hr_at5",
    "conditional_hr_at5",
    "operational_ndcg_at5",
    "conditional_ndcg_at5",
    "mean_positive_rank",
    "latency_mean",
    "generated_tokens_mean",
    "classification",
    "notes",
]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f"{prefix}-{sha256_text(stable_json(parts))[:length]}"


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, str):
        return value
    number = float(value)
    if math.isnan(number):
        return ""
    text = f"{number:.15f}".rstrip("0").rstrip(".")
    return text or "0"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def file_sha256_or_missing(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def command_output(args: Sequence[str], root: Path, timeout: int = 20) -> str:
    try:
        result = subprocess.run(args, cwd=root, text=True, capture_output=True, check=False, timeout=timeout)
    except Exception as exc:  # pragma: no cover - defensive metadata only
        return f"unavailable:{type(exc).__name__}"
    return (result.stdout + result.stderr).strip()


def ndcg_at_rank(rank: int, cutoff: int) -> float:
    if rank <= 0 or rank > cutoff:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def percentile(values: Sequence[float], quantile: float) -> float:
    clean = sorted(float(value) for value in values if value != "")
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[int(position)]
    fraction = position - lower
    return clean[lower] * (1 - fraction) + clean[upper] * fraction


def table_hash(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    payload = "\n".join(stable_json({column: row.get(column, "") for column in columns}) for row in rows)
    return sha256_text(payload)


def output_file_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in OUTPUT_TABLES.items():
        hashes[name] = file_sha256_or_missing(path)
    hashes["report"] = file_sha256_or_missing(REPORT_PATH)
    return hashes


def protected_summary(root: Path) -> dict[str, int]:
    f2_rows = read_csv_rows(root / f6e4a.F2_PROTECTED_VALIDATION)
    f6d4a_rows = read_csv_rows(root / f6e4a.F6D4A_PROTECTED_VERIFICATION)
    changed = sum(1 for row in f2_rows if row.get("status") != "unchanged")
    missing = sum(1 for row in f2_rows if row.get("exists_after") == "false")
    checked = len(f2_rows)
    for row in f6d4a_rows:
        checked += parse_int(row.get("checked"))
        changed += parse_int(row.get("changed"))
        missing += parse_int(row.get("missing"))
    return {"checked": checked, "changed": changed, "missing": missing}


def compact_repository_state(root: Path) -> dict[str, str]:
    return {
        "directory": str(root),
        "branch": command_output(["git", "branch", "--show-current"], root),
        "head": command_output(["git", "rev-parse", "HEAD"], root),
        "last_commit": command_output(["git", "log", "-1", "--oneline"], root),
        "staged_files": command_output(["git", "diff", "--name-only", "--cached"], root),
        "python_version": platform.python_version(),
        "python_executable": os.sys.executable,
        "timestamp_policy": "not_recorded_in_deterministic_freeze",
        "worktree_policy": "full git status recorded in final operator response",
    }


def expected_input_hashes(root: Path) -> dict[Path, str]:
    freeze = json.loads((root / F6E4B_FREEZE_MANIFEST).read_text(encoding="utf-8"))
    expected: dict[Path, str] = {
        F6E4B_FREEZE_MANIFEST: EXPECTED_F6E4B_FREEZE_HASH,
        REQUEST_MANIFEST: str(freeze.get("f6e4a_request_manifest_hash", "")),
        SHARD_MANIFEST: str(freeze.get("f6e4a_shard_manifest_hash", "")),
        F6E4A_FREEZE_MANIFEST: str(freeze.get("f6e4a_freeze_hash", "")),
        F6E4A_RUNTIME_PARAMETERS: "",
        F6E4A_GATE_PLAN: "",
        F6E4B_PROGRESS: "",
        F6E4B_INSTANCE_RESULTS: "",
        F6E4B_CELL_SUMMARY: "",
        F6E4B_DATASET_TASK_SUMMARY: "",
        F6E4B_PARSER_VALIDITY: "",
        F6E4B_INVALID_AUDIT: "",
    }
    raw_hashes = freeze.get("raw_hashes", {})
    for path_text, digest in raw_hashes.items():
        expected[Path(path_text)] = str(digest)
    for path_text, digest in {**f6e4a.F6D4A_RAW_HASHES, **f6e4a.F6E3_RAW_HASHES, **f6e4a.F6E3B_RAW_HASHES}.items():
        expected[Path(path_text)] = digest
    return expected


def verify_inputs(root: Path) -> list[dict[str, str]]:
    expected = expected_input_hashes(root)
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for path, expected_hash in sorted(expected.items(), key=lambda item: item[0].as_posix()):
        observed = file_sha256_or_missing(root / path)
        status = "pass" if observed != "missing" and (not expected_hash or observed == expected_hash) else "fail"
        count = ""
        if path.suffix.lower() in {".csv", ".jsonl"} and observed != "missing":
            count = str(line_count(root / path) - (1 if path.suffix.lower() == ".csv" else 0))
        rows.append(
            {
                "artifact_id": stable_id("F6E4C-IN", path.as_posix()),
                "artifact_role": "frozen_input",
                "path": path.as_posix(),
                "expected_sha256": expected_hash,
                "observed_sha256": observed,
                "record_count": count,
                "status": status,
                "severity": "critical",
                "notes": "Hash checked without modifying the artifact.",
            }
        )
        if status != "pass":
            failures.append(path.as_posix())
    protected = protected_summary(root)
    rows.append(
        {
            "artifact_id": "F6E4C-PROTECTED-SUMMARY",
            "artifact_role": "protected_input_summary",
            "path": "F2/F6D4A protected input inventories",
            "expected_sha256": "changed=0;missing=0",
            "observed_sha256": stable_json(protected),
            "record_count": str(protected["checked"]),
            "status": "pass" if protected["changed"] == 0 and protected["missing"] == 0 else "fail",
            "severity": "critical",
            "notes": "Protected input verification reused existing frozen validation tables.",
        }
    )
    if protected["changed"] or protected["missing"]:
        failures.append("protected_inputs")
    if failures:
        raise RuntimeError("f6e4c_blocked_by_raw_manifest_inconsistency:" + ",".join(failures))
    return rows


def load_manifest_rows(root: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(root / REQUEST_MANIFEST)
    if len(rows) != EXPECTED_REQUESTS:
        raise RuntimeError(f"f6e4c_blocked_by_raw_manifest_inconsistency:request_count={len(rows)}")
    ids = [row["request_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("f6e4c_blocked_by_raw_manifest_inconsistency:duplicate_manifest_request_id")
    return rows


def load_raw_by_request(root: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for model, path in F6E4B_RAW_PATHS.items():
        rows = read_jsonl(root / path)
        for row in rows:
            request_id = str(row.get("request_id", ""))
            if request_id in by_id:
                raise RuntimeError(f"f6e4c_blocked_by_raw_manifest_inconsistency:duplicate_raw_request_id:{request_id}")
            row["_raw_output_file"] = path.as_posix()
            if row.get("model") != model:
                raise RuntimeError(f"f6e4c_blocked_by_raw_manifest_inconsistency:model_path_mismatch:{request_id}")
            by_id[request_id] = row
    if len(by_id) != EXPECTED_REQUESTS:
        raise RuntimeError(f"f6e4c_blocked_by_raw_manifest_inconsistency:raw_count={len(by_id)}")
    return by_id


def reconstruct_request_metrics(root: Path) -> list[dict[str, str]]:
    manifest_rows = load_manifest_rows(root)
    raw_by_id = load_raw_by_request(root)
    rows: list[dict[str, str]] = []
    missing = sorted(set(row["request_id"] for row in manifest_rows) - set(raw_by_id))
    unexpected = sorted(set(raw_by_id) - set(row["request_id"] for row in manifest_rows))
    if missing or unexpected:
        raise RuntimeError(f"f6e4c_blocked_by_raw_manifest_inconsistency:missing={len(missing)}:unexpected={len(unexpected)}")
    for manifest in manifest_rows:
        raw = raw_by_id[manifest["request_id"]]
        strict = parse_bool(raw.get("strict_valid"))
        transport = str(raw.get("transport_status", "missing"))
        parsed_ids = [str(value) for value in raw.get("parsed_item_ids", []) if str(value)]
        parsed_indices = [str(value) for value in raw.get("parsed_indices", [])]
        positive = manifest["positive_item_id"]
        rank = parsed_ids.index(positive) + 1 if positive in parsed_ids and strict else 0
        operational = {
            "hr_at1": 1.0 if strict and 0 < rank <= 1 else 0.0,
            "hr_at5": 1.0 if strict and 0 < rank <= 5 else 0.0,
            "hr_at10": 1.0 if strict and 0 < rank <= 10 else 0.0,
            "ndcg_at5": ndcg_at_rank(rank, 5) if strict else 0.0,
            "ndcg_at10": ndcg_at_rank(rank, 10) if strict else 0.0,
        }
        conditional = operational if strict else {key: "" for key in operational}
        invalid_reason = str(raw.get("invalid_reason", ""))
        if not strict and not invalid_reason:
            invalid_reason = str(raw.get("parse_status", "non_strict")) or "non_strict"
        rows.append(
            {
                "request_id": manifest["request_id"],
                "source_context_id": manifest.get("source_context_id", ""),
                "source_request_id": manifest["source_request_id"],
                "shard_id": manifest["shard_id"],
                "ordinal_position": manifest["ordinal_position"],
                "dataset": manifest["dataset"],
                "task": manifest["task"],
                "model": manifest["model"],
                "strategy_id": manifest["core_strategy_id"],
                "adapted_strategy_id": manifest["adapted_strategy_id"],
                "adapter_classification": manifest["adapter_classification"],
                "candidate_count": manifest["candidate_count"],
                "history_length": manifest.get("history_length", ""),
                "history_length_stratum": manifest.get("history_length_stratum", history_stratum(parse_int(manifest.get("history_length")))),
                "positive_item_id": positive,
                "positive_initial_position": manifest["positive_initial_position"],
                "positive_position_stratum": manifest.get("positive_position_stratum", positive_position_stratum(parse_int(manifest["positive_initial_position"]))),
                "transport_status": transport,
                "strict_valid": str(strict).lower(),
                "invalid_reason": "" if strict else invalid_reason,
                "output_length": str(len(parsed_indices)),
                "parsed_indices": "|".join(parsed_indices),
                "parsed_item_ids": "|".join(parsed_ids),
                "positive_rank": str(rank) if rank else "",
                "hr_at1_operational": fmt(operational["hr_at1"]),
                "hr_at5_operational": fmt(operational["hr_at5"]),
                "hr_at10_operational": fmt(operational["hr_at10"]),
                "ndcg_at5_operational": fmt(operational["ndcg_at5"]),
                "ndcg_at10_operational": fmt(operational["ndcg_at10"]),
                "hr_at1_conditional": fmt(conditional["hr_at1"]) if strict else "",
                "hr_at5_conditional": fmt(conditional["hr_at5"]) if strict else "",
                "hr_at10_conditional": fmt(conditional["hr_at10"]) if strict else "",
                "ndcg_at5_conditional": fmt(conditional["ndcg_at5"]) if strict else "",
                "ndcg_at10_conditional": fmt(conditional["ndcg_at10"]) if strict else "",
                "duration_seconds": fmt(parse_float(raw.get("duration_seconds"))),
                "prompt_eval_count": str(parse_int(raw.get("prompt_eval_count"))),
                "eval_count": str(parse_int(raw.get("eval_count"))),
                "raw_output_file": str(raw.get("_raw_output_file", "")),
                "metric_source": "recomputed_from_f6e4a_manifest_and_f6e4b_raw",
            }
        )
    strict_count = sum(1 for row in rows if row["strict_valid"] == "true")
    invalid_count = len(rows) - strict_count
    if strict_count != EXPECTED_STRICT_VALID or invalid_count != EXPECTED_INVALID:
        raise RuntimeError(f"f6e4c_blocked_by_raw_manifest_inconsistency:strict={strict_count}:invalid={invalid_count}")
    return rows


def history_stratum(length: int) -> str:
    if length <= 5:
        return "history_le_5"
    if length <= 10:
        return "history_6_10"
    if length <= 20:
        return "history_11_20"
    return "history_gt_20"


def positive_position_stratum(position: int) -> str:
    if position <= 5:
        return "positive_top5"
    if position <= 10:
        return "positive_6_10"
    if position <= 20:
        return "positive_11_20"
    return "positive_out_of_range"


def group_rows(rows: Sequence[Mapping[str, str]], keys: Sequence[str]) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def metric_mean(rows: Sequence[Mapping[str, str]], column: str) -> str:
    values = [parse_float(row[column]) for row in rows if row.get(column, "") != ""]
    return fmt(mean(values)) if values else ""


def summarise_group(group: Sequence[Mapping[str, str]], scope: str, aggregation: str, dataset: str, task: str, model: str, strategy: str) -> dict[str, str]:
    strict = [row for row in group if row["strict_valid"] == "true"]
    invalid = [row for row in group if row["strict_valid"] != "true"]
    durations = [parse_float(row["duration_seconds"]) for row in group if row.get("duration_seconds")]
    prompt_tokens = [parse_float(row["prompt_eval_count"]) for row in group if row.get("prompt_eval_count")]
    eval_tokens = [parse_float(row["eval_count"]) for row in group if row.get("eval_count")]
    positive_ranks = [parse_float(row["positive_rank"]) for row in strict if row.get("positive_rank")]
    invalid_dist = Counter(row["invalid_reason"] for row in invalid)
    return {
        "summary_id": stable_id("F6E4C-SUM", scope, aggregation, dataset, task, model, strategy),
        "scope": scope,
        "aggregation": aggregation,
        "dataset": dataset,
        "task": task,
        "model": model,
        "strategy_id": strategy,
        "logical_requests": str(len(group)),
        "transport_success": str(sum(1 for row in group if row["transport_status"] == "success")),
        "strict_valid": str(len(strict)),
        "invalid_outputs": str(len(invalid)),
        "strict_valid_rate": fmt(len(strict) / len(group)) if group else "0",
        "invalid_reason_distribution": stable_json(dict(sorted(invalid_dist.items()))),
        "operational_hr_at1": metric_mean(group, "hr_at1_operational"),
        "operational_hr_at5": metric_mean(group, "hr_at5_operational"),
        "operational_hr_at10": metric_mean(group, "hr_at10_operational"),
        "operational_ndcg_at5": metric_mean(group, "ndcg_at5_operational"),
        "operational_ndcg_at10": metric_mean(group, "ndcg_at10_operational"),
        "conditional_hr_at1": metric_mean(strict, "hr_at1_conditional"),
        "conditional_hr_at5": metric_mean(strict, "hr_at5_conditional"),
        "conditional_hr_at10": metric_mean(strict, "hr_at10_conditional"),
        "conditional_ndcg_at5": metric_mean(strict, "ndcg_at5_conditional"),
        "conditional_ndcg_at10": metric_mean(strict, "ndcg_at10_conditional"),
        "mean_positive_rank": fmt(mean(positive_ranks)) if positive_ranks else "",
        "latency_mean": fmt(mean(durations)) if durations else "",
        "latency_p50": fmt(percentile(durations, 0.5)) if durations else "",
        "latency_p95": fmt(percentile(durations, 0.95)) if durations else "",
        "prompt_tokens_mean": fmt(mean(prompt_tokens)) if prompt_tokens else "",
        "generated_tokens_mean": fmt(mean(eval_tokens)) if eval_tokens else "",
        "status": "complete" if len(group) > 0 else "missing",
        "notes": "F6E4C authoritative recomputation; micro aggregation unless stated otherwise.",
    }


def build_dataset_task_summary(request_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (dataset, task, model, strategy), group in sorted(group_rows(request_rows, ["dataset", "task", "model", "strategy_id"]).items()):
        rows.append(summarise_group(group, "dataset_task_cell", "micro", dataset, task, model, strategy))
    return rows


def build_cell_summary(request_rows: Sequence[Mapping[str, str]], dataset_task_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (model, strategy), group in sorted(group_rows(request_rows, ["model", "strategy_id"]).items()):
        rows.append(summarise_group(group, "cell", "micro", "ALL", "ALL", model, strategy))
        component_rows = [row for row in dataset_task_rows if row["model"] == model and row["strategy_id"] == strategy]
        rows.append(macro_summary(component_rows, "cell", model, strategy))
    return rows


def macro_summary(component_rows: Sequence[Mapping[str, str]], scope: str, model: str, strategy: str) -> dict[str, str]:
    def avg(column: str) -> str:
        values = [parse_float(row[column]) for row in component_rows if row.get(column, "") != ""]
        return fmt(mean(values)) if values else ""

    logical = sum(parse_int(row["logical_requests"]) for row in component_rows)
    strict = sum(parse_int(row["strict_valid"]) for row in component_rows)
    invalid = sum(parse_int(row["invalid_outputs"]) for row in component_rows)
    return {
        "summary_id": stable_id("F6E4C-SUM", scope, "macro_dataset_task", model, strategy),
        "scope": scope,
        "aggregation": "macro_dataset_task",
        "dataset": "ALL",
        "task": "ALL",
        "model": model,
        "strategy_id": strategy,
        "logical_requests": str(logical),
        "transport_success": str(sum(parse_int(row["transport_success"]) for row in component_rows)),
        "strict_valid": str(strict),
        "invalid_outputs": str(invalid),
        "strict_valid_rate": fmt(strict / logical) if logical else "0",
        "invalid_reason_distribution": "macro_not_additive",
        "operational_hr_at1": avg("operational_hr_at1"),
        "operational_hr_at5": avg("operational_hr_at5"),
        "operational_hr_at10": avg("operational_hr_at10"),
        "operational_ndcg_at5": avg("operational_ndcg_at5"),
        "operational_ndcg_at10": avg("operational_ndcg_at10"),
        "conditional_hr_at1": avg("conditional_hr_at1"),
        "conditional_hr_at5": avg("conditional_hr_at5"),
        "conditional_hr_at10": avg("conditional_hr_at10"),
        "conditional_ndcg_at5": avg("conditional_ndcg_at5"),
        "conditional_ndcg_at10": avg("conditional_ndcg_at10"),
        "mean_positive_rank": avg("mean_positive_rank"),
        "latency_mean": avg("latency_mean"),
        "latency_p50": avg("latency_p50"),
        "latency_p95": avg("latency_p95"),
        "prompt_tokens_mean": avg("prompt_tokens_mean"),
        "generated_tokens_mean": avg("generated_tokens_mean"),
        "status": "complete",
        "notes": "Macro average across dataset/task cells; separated from authoritative micro values.",
    }


def build_metric_contract(summary_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "contract_id": "F6E4C-CONTRACT-REQUEST-NONSTRICT",
            "scope": "request_level",
            "aggregation": "not_applicable",
            "dataset": "ALL",
            "task": "ALL",
            "model": "ALL",
            "strategy_id": "ALL",
            "metric": "non_strict_operational_zero_conditional_missing",
            "operational_value": "0",
            "conditional_value": "",
            "strict_valid_rate": "",
            "expected_operational_value": "0",
            "absolute_error": "0",
            "tolerance": fmt(TOLERANCE),
            "status": "pass",
            "notes": "Validated while constructing request-level metrics.",
        }
    ]
    metrics = [
        ("hr_at1", "operational_hr_at1", "conditional_hr_at1"),
        ("hr_at5", "operational_hr_at5", "conditional_hr_at5"),
        ("hr_at10", "operational_hr_at10", "conditional_hr_at10"),
        ("ndcg_at5", "operational_ndcg_at5", "conditional_ndcg_at5"),
        ("ndcg_at10", "operational_ndcg_at10", "conditional_ndcg_at10"),
    ]
    for summary in summary_rows:
        if summary["aggregation"] != "micro":
            continue
        rate = parse_float(summary["strict_valid_rate"])
        for metric, op_col, cond_col in metrics:
            operational = parse_float(summary[op_col])
            conditional = parse_float(summary[cond_col]) if summary.get(cond_col) else 0.0
            expected = conditional * rate
            error = abs(operational - expected)
            status = "pass" if error <= TOLERANCE and operational <= conditional + TOLERANCE else "fail"
            rows.append(
                {
                    "contract_id": stable_id("F6E4C-CON", summary["summary_id"], metric),
                    "scope": summary["scope"],
                    "aggregation": summary["aggregation"],
                    "dataset": summary["dataset"],
                    "task": summary["task"],
                    "model": summary["model"],
                    "strategy_id": summary["strategy_id"],
                    "metric": metric,
                    "operational_value": fmt(operational),
                    "conditional_value": fmt(conditional),
                    "strict_valid_rate": summary["strict_valid_rate"],
                    "expected_operational_value": fmt(expected),
                    "absolute_error": fmt(error),
                    "tolerance": fmt(TOLERANCE),
                    "status": status,
                    "notes": "Operational utility denominator is logical requests; conditional denominator is strict-valid requests.",
                }
            )
    return rows


def build_metric_reconciliation(root: Path, f6e4c_cell_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    f6e4b_rows = [row for row in read_csv_rows(root / F6E4B_CELL_SUMMARY) if row.get("scope") == "cell"]
    f6e4c_by_key = {(row["model"], row["strategy_id"]): row for row in f6e4c_cell_rows if row["aggregation"] == "micro"}
    metrics = [
        ("operational_hr_at5", "operational_hr_at5", "operational_hr_at5", "logical"),
        ("operational_ndcg_at5", "operational_ndcg_at5", "operational_ndcg_at5", "logical"),
        ("conditional_hr_at5", "conditional_hr_at5", "conditional_hr_at5", "strict_valid"),
        ("conditional_ndcg_at5", "conditional_ndcg_at5", "conditional_ndcg_at5", "strict_valid"),
    ]
    rows: list[dict[str, str]] = []
    for old in f6e4b_rows:
        new = f6e4c_by_key[(old["model"], old["strategy_id"])]
        old_rate = parse_float(old["strict_valid_rate"])
        for metric, old_col, new_col, population in metrics:
            old_value = parse_float(old[old_col])
            new_value = parse_float(new[new_col])
            difference = new_value - old_value
            f6e4b_contract_expected = parse_float(old.get(metric.replace("operational", "conditional"), "0")) * old_rate if metric.startswith("operational") else old_value
            violates_contract = metric.startswith("operational") and abs(old_value - f6e4b_contract_expected) > 1e-6
            changed = abs(difference) > 1e-6
            cause = "rounding_only"
            if violates_contract:
                cause = "invalid_output_counted_in_operational_utility"
            elif changed:
                cause = "implementation_bug"
            rows.append(
                {
                    "reconciliation_id": stable_id("F6E4C-REC", old["model"], old["strategy_id"], metric),
                    "scope": "cell",
                    "dataset": "ALL",
                    "task": "ALL",
                    "model": old["model"],
                    "strategy_id": old["strategy_id"],
                    "metric": metric,
                    "f6e4b_value": old[old_col],
                    "f6e4c_value": fmt(new_value),
                    "difference": fmt(difference),
                    "f6e4b_denominator": old["expected_requests"] if population == "logical" else old["strict_valid"],
                    "f6e4c_denominator": new["logical_requests"] if population == "logical" else new["strict_valid"],
                    "population": population,
                    "cause": cause,
                    "status": "superseded" if changed or violates_contract else "consistent",
                    "notes": "F6E4B aggregate preserved as preliminary execution summary; F6E4C recomputation is authoritative.",
                }
            )
    return rows


def build_invalid_output_audit(root: Path, request_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    f6e4b_by_id = {row["request_id"]: row for row in read_csv_rows(root / F6E4B_INSTANCE_RESULTS)}
    raw_by_id = load_raw_by_request(root)
    rows: list[dict[str, str]] = []
    for row in request_rows:
        if row["strict_valid"] == "true":
            continue
        raw = raw_by_id[row["request_id"]]
        parsed_ids = [str(value) for value in raw.get("parsed_item_ids", [])]
        old = f6e4b_by_id.get(row["request_id"], {})
        old_counted = parse_float(old.get("hr_at5_operational")) > 0 or parse_float(old.get("ndcg_at5_operational")) > 0
        rows.append(
            {
                "audit_id": stable_id("F6E4C-INV", row["request_id"]),
                "request_id": row["request_id"],
                "model": row["model"],
                "strategy_id": row["strategy_id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "history_length": row["history_length"],
                "history_length_stratum": row["history_length_stratum"],
                "candidate_count": row["candidate_count"],
                "positive_initial_position": row["positive_initial_position"],
                "positive_position_stratum": row["positive_position_stratum"],
                "output_length": row["output_length"],
                "generated_token_count": row["eval_count"],
                "raw_response_length": str(len(str(raw.get("raw_response", "")))),
                "invalid_reason": row["invalid_reason"],
                "positive_appears_in_raw_parsed_content": str(row["positive_item_id"] in parsed_ids).lower(),
                "f6e4b_operational_counted_positive": str(old_counted).lower(),
                "f6e4c_operational_utility_zero": "true",
                "notes": "Invalid outputs are preserved but never repaired or promoted to strict utility.",
            }
        )
    reason_counts = Counter(row["invalid_reason"] for row in rows)
    expected = {"invalid_length": 1271, "extra_text_or_code_fence": 111, "duplicate_indices": 50, "json_parse_failed": 5}
    if dict(reason_counts) != expected:
        raise RuntimeError(f"f6e4c_blocked_by_raw_manifest_inconsistency:invalid_reasons={stable_json(dict(reason_counts))}")
    return rows


def bootstrap_ci(deltas: Sequence[float], *, seed_offset: int = 0) -> tuple[str, str]:
    if not deltas:
        return "", ""
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 1:
        value = fmt(float(arr[0]))
        return value, value
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means: list[np.ndarray] = []
    done = 0
    chunk = max(1, min(256, 5_000_000 // max(1, arr.size)))
    while done < BOOTSTRAP_RESAMPLES:
        current = min(chunk, BOOTSTRAP_RESAMPLES - done)
        idx = rng.integers(0, arr.size, size=(current, arr.size), dtype=np.int32)
        means.append(arr[idx].mean(axis=1))
        done += current
    samples = np.concatenate(means)
    return fmt(float(np.quantile(samples, 0.025))), fmt(float(np.quantile(samples, 0.975)))


def paired_metric_rows(
    left_rows: Sequence[Mapping[str, str]],
    right_rows: Sequence[Mapping[str, str]],
    *,
    comparison_type: str,
    dataset: str,
    task: str,
    model: str,
    strategy: str,
    left_label: str,
    right_label: str,
    seed_offset: int,
) -> list[dict[str, str]]:
    left_by_key = {pair_key(row): row for row in left_rows}
    right_by_key = {pair_key(row): row for row in right_rows}
    keys = sorted(set(left_by_key) & set(right_by_key))
    pairs = [(left_by_key[key], right_by_key[key]) for key in keys]
    strict_pairs = [(left, right) for left, right in pairs if left["strict_valid"] == "true" and right["strict_valid"] == "true"]
    rows: list[dict[str, str]] = []
    for population, metric_col, metric_name, source_pairs in [
        ("operational", "hr_at5_operational", "hr_at5", pairs),
        ("operational", "ndcg_at5_operational", "ndcg_at5", pairs),
        ("conditional_intersection", "hr_at5_conditional", "hr_at5", strict_pairs),
        ("conditional_intersection", "ndcg_at5_conditional", "ndcg_at5", strict_pairs),
    ]:
        left_values = [parse_float(left[metric_col]) for left, _ in source_pairs if left.get(metric_col) != ""]
        right_values = [parse_float(right[metric_col]) for _, right in source_pairs if right.get(metric_col) != ""]
        deltas = [right - left for left, right in zip(left_values, right_values)]
        ci_low, ci_high = bootstrap_ci(deltas, seed_offset=seed_offset + len(rows))
        wins = sum(1 for delta in deltas if delta > 0)
        losses = sum(1 for delta in deltas if delta < 0)
        ties = len(deltas) - wins - losses
        delta_strict = mean([1.0 if right["strict_valid"] == "true" else 0.0 for _, right in pairs]) - mean([1.0 if left["strict_valid"] == "true" else 0.0 for left, _ in pairs]) if pairs else 0.0
        delta_latency = mean([parse_float(right["duration_seconds"]) - parse_float(left["duration_seconds"]) for left, right in pairs]) if pairs else 0.0
        delta_tokens = mean([parse_float(right["eval_count"]) - parse_float(left["eval_count"]) for left, right in pairs]) if pairs else 0.0
        rows.append(
            {
                "comparison_id": stable_id("F6E4C-PAIR", comparison_type, population, dataset, task, model, strategy, metric_name),
                "comparison_type": comparison_type,
                "population": population,
                "dataset": dataset,
                "task": task,
                "model": model,
                "strategy_id": strategy,
                "left_label": left_label,
                "right_label": right_label,
                "n_pairs": str(len(pairs)),
                "n_strict_intersection": str(len(strict_pairs)),
                "metric": metric_name,
                "left_mean": fmt(mean(left_values)) if left_values else "",
                "right_mean": fmt(mean(right_values)) if right_values else "",
                "delta_right_minus_left": fmt(mean(deltas)) if deltas else "",
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "win_count": str(wins),
                "tie_count": str(ties),
                "loss_count": str(losses),
                "delta_strict_validity": fmt(delta_strict),
                "delta_latency": fmt(delta_latency),
                "delta_generated_tokens": fmt(delta_tokens),
                "bootstrap_resamples": str(BOOTSTRAP_RESAMPLES),
                "seed": str(BOOTSTRAP_SEED),
                "notes": "Paired comparison on shared source_request_id/dataset/task; invalid outputs are zero only for operational population.",
            }
        )
    return rows


def pair_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (row["dataset"], row["task"], row["source_request_id"])


def build_strategy_paired_comparison(request_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        for dataset, task in dataset_task_scopes(request_rows):
            base = [row for row in request_rows if row["model"] == model and row["strategy_id"] == BASE_STRATEGY and (dataset == "ALL" or (row["dataset"], row["task"]) == (dataset, task))]
            llmrank = [row for row in request_rows if row["model"] == model and row["strategy_id"] == LLMRANK_STRATEGY and (dataset == "ALL" or (row["dataset"], row["task"]) == (dataset, task))]
            rows.extend(
                paired_metric_rows(
                    base,
                    llmrank,
                    comparison_type="base_vs_llmrank",
                    dataset=dataset,
                    task=task,
                    model=model,
                    strategy="ALL",
                    left_label="Base",
                    right_label="LLMRank",
                    seed_offset=len(rows) * 10,
                )
            )
    return rows


def build_model_paired_comparison(request_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for strategy in STRATEGY_ORDER:
        for dataset, task in dataset_task_scopes(request_rows):
            qwen = [row for row in request_rows if row["model"] == QWEN2_TAG and row["strategy_id"] == strategy and (dataset == "ALL" or (row["dataset"], row["task"]) == (dataset, task))]
            mistral = [row for row in request_rows if row["model"] == MISTRAL_TAG and row["strategy_id"] == strategy and (dataset == "ALL" or (row["dataset"], row["task"]) == (dataset, task))]
            rows.extend(
                paired_metric_rows(
                    qwen,
                    mistral,
                    comparison_type="qwen2_vs_mistral",
                    dataset=dataset,
                    task=task,
                    model="ALL",
                    strategy=strategy,
                    left_label="Qwen2",
                    right_label="Mistral",
                    seed_offset=1000 + len(rows) * 10,
                )
            )
    return rows


def dataset_task_scopes(request_rows: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    scopes = [("ALL", "ALL")]
    scopes.extend(sorted({(row["dataset"], row["task"]) for row in request_rows}))
    return scopes


def build_strata_analysis(request_rows: Sequence[Mapping[str, str]], key: str, analysis_type: str, classification: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (dataset, task, model, strategy, stratum), group in sorted(group_rows(request_rows, ["dataset", "task", "model", "strategy_id", key]).items()):
        summary = summarise_group(group, analysis_type, "micro", dataset, task, model, strategy)
        rows.append(
            {
                "analysis_id": stable_id("F6E4C-STRATA", analysis_type, dataset, task, model, strategy, stratum),
                "analysis_type": analysis_type,
                "dataset": dataset,
                "task": task,
                "model": model,
                "strategy_id": strategy,
                "stratum_name": stratum,
                "logical_requests": summary["logical_requests"],
                "strict_valid": summary["strict_valid"],
                "strict_valid_rate": summary["strict_valid_rate"],
                "operational_hr_at5": summary["operational_hr_at5"],
                "conditional_hr_at5": summary["conditional_hr_at5"],
                "operational_ndcg_at5": summary["operational_ndcg_at5"],
                "conditional_ndcg_at5": summary["conditional_ndcg_at5"],
                "mean_positive_rank": summary["mean_positive_rank"],
                "latency_mean": summary["latency_mean"],
                "generated_tokens_mean": summary["generated_tokens_mean"],
                "classification": classification,
                "notes": "Deterministic strata from frozen F6E4A metadata.",
            }
        )
    return rows


def build_mistral_llmrank_lastfm_audit(request_rows: Sequence[Mapping[str, str]], invalid_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    group = [
        row
        for row in request_rows
        if row["model"] == MISTRAL_TAG and row["strategy_id"] == LLMRANK_STRATEGY and row["dataset"] == "LastFM" and row["task"] == "ranking"
    ]
    invalid = [row for row in invalid_rows if row["model"] == MISTRAL_TAG and row["strategy_id"] == LLMRANK_STRATEGY and row["dataset"] == "LastFM" and row["task"] == "ranking"]
    length_counts = Counter(row["output_length"] for row in invalid)
    reason_counts = Counter(row["invalid_reason"] for row in invalid)
    strict = [row for row in group if row["strict_valid"] == "true"]
    generated = [parse_float(row["eval_count"]) for row in group]
    durations = [parse_float(row["duration_seconds"]) for row in group]
    pct_20 = length_counts.get("20", 0) / len(group) if group else 0.0
    pct_10 = (sum(1 for row in group if row["output_length"] == "10")) / len(group) if group else 0.0
    cause = "core_adapter_instruction_conflict" if pct_20 > 0.25 else "mixed_causes"
    summary = summarise_group(group, "mistral_llmrank_lastfm", "micro", "LastFM", "ranking", MISTRAL_TAG, LLMRANK_STRATEGY)
    return [
        {
            "audit_id": "F6E4C-MISTRAL-LLMRANK-LASTFM",
            "logical_requests": str(len(group)),
            "strict_valid": str(len(strict)),
            "invalid": str(len(invalid)),
            "invalid_rate": fmt(len(invalid) / len(group)) if group else "0",
            "invalid_reason_distribution": stable_json(dict(sorted(reason_counts.items()))),
            "output_length_distribution": stable_json(dict(sorted(length_counts.items()))),
            "percent_output_length_20": fmt(pct_20),
            "percent_output_length_10": fmt(pct_10),
            "latency_mean": fmt(mean(durations)) if durations else "",
            "generated_tokens_mean": fmt(mean(generated)) if generated else "",
            "history_length_distribution": stable_json(dict(sorted(Counter(row["history_length_stratum"] for row in group).items()))),
            "candidate_position_distribution": stable_json(dict(sorted(Counter(row["positive_position_stratum"] for row in group).items()))),
            "operational_hr_at5": summary["operational_hr_at5"],
            "conditional_hr_at5": summary["conditional_hr_at5"],
            "operational_ndcg_at5": summary["operational_ndcg_at5"],
            "conditional_ndcg_at5": summary["conditional_ndcg_at5"],
            "prompt_adapter_conflict_evidence": "Many invalid outputs have length 20 while the adapted JSON contract requires top-10 indices only.",
            "classified_cause": cause,
            "notes": "No prompt is modified in F6E4C; this is a diagnostic classification only.",
        }
    ]


def build_exact_vs_adapted(root: Path, request_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    f6e3 = read_csv_rows(root / "results/tables/phase_f6e3_parser_validity.csv")
    f6e3a = read_csv_rows(root / "results/tables/phase_f6e3a_parser_ladder_results.csv")
    f6e3a_root = read_csv_rows(root / "results/tables/phase_f6e3a_root_cause.csv")
    f6e3b = read_csv_rows(root / "results/tables/phase_f6e3b_parser_validity.csv")
    f6e4b_strict = sum(1 for row in request_rows if row["strict_valid"] == "true")
    root_causes = Counter(row.get("primary_cause", "") for row in f6e3a_root)
    return [
        evidence_row("F6E3", "exact_parametric_textual_prompt_pilot", len(f6e3), sum(parse_bool(row.get("strict_valid")) for row in f6e3), "exact-prompt local pilot failed structural validity; no utility claim allowed."),
        evidence_row("F6E3A", "offline_parser_recovery_audit", 32, 0, f"Parser ladder did not recover strict outputs; root_causes={stable_json(dict(sorted(root_causes.items())))}."),
        evidence_row("F6E3B", "adapted_indexed_output_contract_pilot", len(f6e3b), sum(parse_bool(row.get("strict_valid")) for row in f6e3b), "Adapted JSON/index pilot improved structural reliability; still not paper-exact."),
        evidence_row("F6E4B", "adapted_indexed_output_contract_full_run", len(request_rows), f6e4b_strict, "Full adapted run provides authoritative local adapted evidence after F6E4C metric contract correction."),
    ]


def evidence_row(phase: str, scope: str, total: int, strict: int, claim: str) -> dict[str, str]:
    return {
        "evidence_id": stable_id("F6E4C-EVID", phase, scope),
        "phase": phase,
        "evidence_scope": scope,
        "sample_size": str(total),
        "strict_valid": str(strict),
        "strict_valid_rate": fmt(strict / total) if total else "0",
        "output_contract": "adapted_indexed_output_contract" if phase in {"F6E3B", "F6E4B"} else "exact_parametric_textual_prompt",
        "exactness_classification": "adapted_not_paper_exact" if phase in {"F6E3B", "F6E4B"} else "exact_prompt_local_pilot",
        "allowed_claim": claim,
        "prohibited_claim": "The adapted prompt is paper-exact.",
    }


def build_research_question_evidence() -> list[dict[str, str]]:
    items = [
        ("RQ1", "LLM utility rispetto alle baseline tradizionali", "not_tested", "F6E4C does not recompute traditional baseline comparisons.", "No direct baseline superiority claim from F6E4C."),
        ("RQ2", "Domain-knowledge effect", "not_tested", "No controlled domain-knowledge intervention in F6E4B.", "Domain knowledge remains outside this adapted run."),
        ("RQ3", "Long-tail/novelty behavior", "not_tested", "No long-tail metric is recomputed in F6E4C.", "Long-tail claims require separate metric phase."),
        ("RQ4", "Short-history/cold-start behavior", "partially_supported", "History-length strata are observational.", "In the adapted setting, performance can be described by history stratum only."),
        ("RQ5", "Candidate-position bias", "partially_supported", "Candidate-position strata are adapted diagnostics.", "Position effects are diagnostic, not CandDif-exact."),
        ("RQ6", "Profile-generation effectiveness", "not_tested", "F6E4B is ranking/reranking only.", "Profile generation is not evaluated."),
        ("RQ7", "Hallucination and output reliability", "supported_in_adapted_setting", "F6E3/F6E3B/F6E4B structural-validity evidence.", "The adapted indexed contract improved local structural reliability."),
        ("RQ8", "Ranking versus re-ranking", "partially_supported", "F6E4C paired comparisons cover available adapted ranking/reranking cells.", "Only adapted local cells are covered."),
        ("RQ9", "Base versus LLMRank", "partially_supported", "F6E4C paired strategy comparisons.", "Direction is setting-specific and must include reliability effects."),
        ("RQ10", "Effect of structured output adaptation", "supported_in_adapted_setting", "F6E3 0/32 strict, F6E3B 31/32 strict, F6E4B 29055/30492 strict.", "Structured indexed output greatly improved local compliance, not paper exactness."),
    ]
    return [
        {
            "rq_id": rq_id,
            "paper_question_or_claim": claim,
            "classification": classification,
            "local_evidence": evidence,
            "datasets": "Beauty;Sports;ML-1M;LastFM where available",
            "tasks": "ranking;reranking except LastFM reranking",
            "models": "qwen2:7b-instruct;mistral:7b-instruct",
            "metric": "HR@5;NDCG@5;strict_validity as applicable",
            "result_direction": classification,
            "exact_or_adapted_status": "adapted_local_setting",
            "limitation": "Not paper-exact model/version/prompt/protocol.",
            "allowed_conclusion": allowed,
        }
        for rq_id, claim, classification, evidence, allowed in items
    ]


def build_claim_registry() -> list[dict[str, str]]:
    claims = [
        ("CLAIM-F6E4C-001", "In our adapted local setting, all 30492 planned requests were executed.", "phase_f6e4c_request_level_metrics.csv", "high", "adapted_full_run", "In our adapted local setting, the full matrix is complete.", "We exactly reproduced the paper.", "Completeness is execution coverage, not paper-exactness."),
        ("CLAIM-F6E4C-002", "F6E4C metrics supersede preliminary F6E4B aggregate utility where metric-contract violations are found.", "phase_f6e4c_metric_reconciliation.csv", "high", "derived_metric_correction", "F6E4C recomputed utility is authoritative for this run.", "F6E4B raw are invalid.", "Raw are preserved and valid; only derived summaries may be superseded."),
        ("CLAIM-F6E4C-003", "The adapted indexed output contract improved structural reliability relative to the exact-parametric local pilot.", "phase_f6e4c_exact_vs_adapted_evidence.csv", "high", "adapted_contract_evidence", "The adapted contract greatly improved local structural reliability.", "The adapted prompt is paper-exact.", "Evidence compares local pilots and full adapted run."),
        ("CLAIM-F6E4C-004", "Mistral LLMRank on LastFM has a systematic reliability weakness in this adapted run.", "phase_f6e4c_mistral_llmrank_lastfm_audit.csv", "medium", "adapted_diagnostic", "Mistral LLMRank LastFM needs contract repair before stronger claims.", "Mistral is always worse.", "Finding is dataset/task/contract specific."),
        ("CLAIM-F6E4C-005", "Base-vs-LLMRank effects must be reported as paired operational and conditional effects.", "phase_f6e4c_strategy_paired_comparison.csv", "high", "adapted_paired_analysis", "The paired comparison separates utility and reliability effects.", "LLMRank is universally worse.", "No universal claim across models/protocols."),
        ("CLAIM-F6E4C-006", "Qwen2-vs-Mistral effects are local adapted comparisons, not exact model-version conclusions.", "phase_f6e4c_model_paired_comparison.csv", "high", "exact_family_version_uncertain", "Mistral/Qwen2 differences are local adapted evidence.", "Mistral is always superior.", "Ollama tags are not exact paper versions."),
    ]
    return [
        {
            "claim_id": claim_id,
            "claim_text": text,
            "evidence_table": table,
            "evidence_level": level,
            "exactness_classification": exactness,
            "allowed_wording": allowed,
            "prohibited_wording": prohibited,
            "limitation": limitation,
        }
        for claim_id, text, table, level, exactness, allowed, prohibited, limitation in claims
    ]


def build_missing_or_blocked(reconciliation_rows: Sequence[Mapping[str, str]], invalid_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    superseded = sum(1 for row in reconciliation_rows if row["status"] == "superseded")
    rows = [
        {
            "block_id": "F6E4C-PAPER-EXACTNESS",
            "category": "scientific_scope",
            "status": "not_paper_exact",
            "severity": "medium",
            "evidence": "phase_f6e4c_claim_registry.csv",
            "safe_next_action": "keep_current_matrix_frozen",
            "notes": "F6E4C is an adapted local consolidation, not exact paper replication.",
        },
        {
            "block_id": "F6E4C-INVALID-OUTPUTS",
            "category": "output_validity",
            "status": "invalid_outputs_preserved",
            "severity": "medium",
            "evidence": "phase_f6e4c_invalid_output_audit.csv",
            "safe_next_action": "run_targeted_mistral_llmrank_contract_repair_pilot",
            "notes": f"invalid={len(invalid_rows)}",
        },
    ]
    if superseded:
        rows.append(
            {
                "block_id": "F6E4C-F6E4B-DERIVED-METRICS-SUPERSEDED",
                "category": "metric_contract",
                "status": "corrected_in_f6e4c",
                "severity": "low",
                "evidence": "phase_f6e4c_metric_reconciliation.csv",
                "safe_next_action": "keep_current_matrix_frozen",
                "notes": f"superseded_metric_rows={superseded}",
            }
        )
    return rows


def build_all_tables(root: Path) -> dict[str, list[dict[str, str]]]:
    input_integrity = verify_inputs(root)
    request_metrics = reconstruct_request_metrics(root)
    dataset_summary = build_dataset_task_summary(request_metrics)
    cell_summary = build_cell_summary(request_metrics, dataset_summary)
    metric_contract = build_metric_contract([*dataset_summary, *cell_summary])
    if any(row["status"] != "pass" for row in metric_contract):
        raise RuntimeError("f6e4c_blocked_by_unresolved_metric_contract")
    reconciliation = build_metric_reconciliation(root, cell_summary)
    invalid_audit = build_invalid_output_audit(root, request_metrics)
    mistral_lastfm = build_mistral_llmrank_lastfm_audit(request_metrics, invalid_audit)
    strategy_pairs = build_strategy_paired_comparison(request_metrics)
    model_pairs = build_model_paired_comparison(request_metrics)
    history = build_strata_analysis(request_metrics, "history_length_stratum", "history_length", "observational_history_length_analysis")
    candidate = build_strata_analysis(request_metrics, "positive_position_stratum", "candidate_position", "adapted_candidate_position_analysis")
    exact_vs_adapted = build_exact_vs_adapted(root, request_metrics)
    rq = build_research_question_evidence()
    claims = build_claim_registry()
    missing = build_missing_or_blocked(reconciliation, invalid_audit)
    return {
        "input_integrity": input_integrity,
        "metric_contract": metric_contract,
        "request_level_metrics": request_metrics,
        "metric_reconciliation": reconciliation,
        "cell_summary": cell_summary,
        "dataset_task_summary": dataset_summary,
        "invalid_output_audit": invalid_audit,
        "mistral_llmrank_lastfm_audit": mistral_lastfm,
        "strategy_paired_comparison": strategy_pairs,
        "model_paired_comparison": model_pairs,
        "history_length_analysis": history,
        "candidate_position_analysis": candidate,
        "exact_vs_adapted_evidence": exact_vs_adapted,
        "research_question_evidence": rq,
        "claim_registry": claims,
        "missing_or_blocked": missing,
    }


def table_columns(name: str) -> list[str]:
    mapping = {
        "input_integrity": INPUT_INTEGRITY_COLUMNS,
        "metric_contract": CONTRACT_COLUMNS,
        "request_level_metrics": REQUEST_COLUMNS,
        "metric_reconciliation": RECON_COLUMNS,
        "cell_summary": SUMMARY_COLUMNS,
        "dataset_task_summary": SUMMARY_COLUMNS,
        "invalid_output_audit": INVALID_COLUMNS,
        "mistral_llmrank_lastfm_audit": [
            "audit_id",
            "logical_requests",
            "strict_valid",
            "invalid",
            "invalid_rate",
            "invalid_reason_distribution",
            "output_length_distribution",
            "percent_output_length_20",
            "percent_output_length_10",
            "latency_mean",
            "generated_tokens_mean",
            "history_length_distribution",
            "candidate_position_distribution",
            "operational_hr_at5",
            "conditional_hr_at5",
            "operational_ndcg_at5",
            "conditional_ndcg_at5",
            "prompt_adapter_conflict_evidence",
            "classified_cause",
            "notes",
        ],
        "strategy_paired_comparison": PAIR_COLUMNS,
        "model_paired_comparison": PAIR_COLUMNS,
        "history_length_analysis": STRATA_COLUMNS,
        "candidate_position_analysis": STRATA_COLUMNS,
        "exact_vs_adapted_evidence": [
            "evidence_id",
            "phase",
            "evidence_scope",
            "sample_size",
            "strict_valid",
            "strict_valid_rate",
            "output_contract",
            "exactness_classification",
            "allowed_claim",
            "prohibited_claim",
        ],
        "research_question_evidence": [
            "rq_id",
            "paper_question_or_claim",
            "classification",
            "local_evidence",
            "datasets",
            "tasks",
            "models",
            "metric",
            "result_direction",
            "exact_or_adapted_status",
            "limitation",
            "allowed_conclusion",
        ],
        "claim_registry": [
            "claim_id",
            "claim_text",
            "evidence_table",
            "evidence_level",
            "exactness_classification",
            "allowed_wording",
            "prohibited_wording",
            "limitation",
        ],
        "missing_or_blocked": [
            "block_id",
            "category",
            "status",
            "severity",
            "evidence",
            "safe_next_action",
            "notes",
        ],
    }
    return mapping[name]


def final_decision(reconciliation_rows: Sequence[Mapping[str, str]]) -> str:
    if any(row["status"] == "superseded" for row in reconciliation_rows):
        return "f6e4c_scientific_consolidation_complete_with_corrected_derived_metrics"
    return "f6e4c_scientific_consolidation_complete"


def recommended_next_action(decision: str, missing_rows: Sequence[Mapping[str, str]]) -> str:
    if decision.endswith("corrected_derived_metrics") and any(row["block_id"] == "F6E4C-INVALID-OUTPUTS" for row in missing_rows):
        return "keep_current_matrix_frozen"
    return "keep_current_matrix_frozen"


def build_manifest(root: Path, tables: Mapping[str, Sequence[Mapping[str, str]]], decision: str) -> dict[str, Any]:
    table_hashes = {name: table_hash(rows, table_columns(name)) for name, rows in sorted(tables.items())}
    scientific_payload = stable_json(
        {
            "request_level": table_hashes["request_level_metrics"],
            "cell_summary": table_hashes["cell_summary"],
            "dataset_task_summary": table_hashes["dataset_task_summary"],
            "strategy_pairs": table_hashes["strategy_paired_comparison"],
            "model_pairs": table_hashes["model_paired_comparison"],
            "claims": table_hashes["claim_registry"],
        }
    )
    return {
        "phase": PHASE_ID,
        "decision": decision,
        "recommended_next_action": recommended_next_action(decision, tables["missing_or_blocked"]),
        "expected_requests": EXPECTED_REQUESTS,
        "request_level_rows": len(tables["request_level_metrics"]),
        "strict_valid": sum(1 for row in tables["request_level_metrics"] if row["strict_valid"] == "true"),
        "invalid_outputs": sum(1 for row in tables["request_level_metrics"] if row["strict_valid"] != "true"),
        "classification": "adapted_indexed_output_contract",
        "model_equivalence": "exact_family_version_uncertain",
        "input_hashes": {row["path"]: row["observed_sha256"] for row in tables["input_integrity"] if row["path"].startswith("results/")},
        "raw_f6e4b_hashes": {path.as_posix(): file_sha256_or_missing(root / path) for path in F6E4B_RAW_PATHS.values()},
        "request_manifest_hash": file_sha256_or_missing(root / REQUEST_MANIFEST),
        "metric_contract_hash": table_hashes["metric_contract"],
        "request_level_metric_hash": table_hashes["request_level_metrics"],
        "summary_table_hashes": table_hashes,
        "bootstrap_configuration": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "scientific_content_hash": sha256_text(scientific_payload),
        "protected_summary": protected_summary(root),
        "runtime_metadata": compact_repository_state(root),
        "superseded_preliminary_metric_rows": [row for row in tables["metric_reconciliation"] if row["status"] == "superseded"],
    }


def build_report(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    progress = {
        "requests": manifest["request_level_rows"],
        "strict_valid": manifest["strict_valid"],
        "invalid_outputs": manifest["invalid_outputs"],
        "decision": manifest["decision"],
    }
    sections = [
        ("Initial repository state", stable_json(manifest["runtime_metadata"])),
        ("Frozen-input verification", f"Input rows: {len(tables['input_integrity'])}; protected={stable_json(manifest['protected_summary'])}."),
        ("Raw F6E4B hashes and counts", stable_json(manifest["raw_f6e4b_hashes"])),
        ("Request-level reconstruction", stable_json(progress)),
        ("Metric-contract audit", f"Rows: {len(tables['metric_contract'])}; all checks pass."),
        ("Operational/conditional reconciliation", "Operational metrics use all logical requests with invalid outputs set to zero; conditional metrics use strict-valid rows only."),
        ("Superseded F6E4B metrics", f"Superseded rows: {len(manifest['superseded_preliminary_metric_rows'])}."),
        ("Corrected authoritative metrics", "F6E4C request-level and summary tables are authoritative for the adapted full run."),
        ("Invalid-output audit", f"Rows: {len(tables['invalid_output_audit'])}."),
        ("Mistral LLMRank LastFM diagnosis", stable_json(tables["mistral_llmrank_lastfm_audit"][0])),
        ("Cell summaries", f"Rows: {len(tables['cell_summary'])}."),
        ("Dataset/task summaries", f"Rows: {len(tables['dataset_task_summary'])}."),
        ("Base-vs-LLMRank paired comparison", f"Rows: {len(tables['strategy_paired_comparison'])}; bootstrap={BOOTSTRAP_RESAMPLES}; seed={BOOTSTRAP_SEED}."),
        ("Qwen2-vs-Mistral paired comparison", f"Rows: {len(tables['model_paired_comparison'])}; bootstrap={BOOTSTRAP_RESAMPLES}; seed={BOOTSTRAP_SEED}."),
        ("History-length analysis", f"Rows: {len(tables['history_length_analysis'])}; classification observational_history_length_analysis."),
        ("Candidate-position analysis", f"Rows: {len(tables['candidate_position_analysis'])}; classification adapted_candidate_position_analysis."),
        ("Exact-vs-adapted evidence", f"Rows: {len(tables['exact_vs_adapted_evidence'])}."),
        ("Research-question evidence map", f"Rows: {len(tables['research_question_evidence'])}."),
        ("Claim registry", f"Rows: {len(tables['claim_registry'])}."),
        ("Implementation files", "src/analysis/plan_f6e4c_scientific_consolidation.py and CLI builder."),
        ("Tests added", "tests/test_plan_f6e4c_scientific_consolidation.py."),
        ("Targeted regression results", "Recorded in final operator response after execution."),
        ("Excluded unsafe tests", "Inference, training CUDA, download, URL/API tests are excluded."),
        ("Determinism", "Builder output is byte-deterministic when run twice on the same frozen inputs."),
        ("Protected-input verification", stable_json(manifest["protected_summary"])),
        ("Scientific freeze manifest", FREEZE_MANIFEST_PATH.as_posix()),
        ("Missing or blocked evidence", f"Rows: {len(tables['missing_or_blocked'])}."),
        ("Final decision", manifest["decision"]),
        ("Recommended next action", manifest["recommended_next_action"]),
        ("Final git state", "Recorded in final operator response; no staging or commit is performed."),
        ("Files created or modified", "Only F6E4C source, CLI, tests, report, tables and manifest."),
        ("Remaining limitations", "Adapted local run; not paper-exact; LastFM reranking remains excluded."),
    ]
    lines = ["# Phase F6E4C Scientific Consolidation", ""]
    for index, (title, body) in enumerate(sections, start=1):
        lines.extend([f"## {index}. {title}", "", str(body), ""])
    return "\n".join(lines)


def write_outputs(root: Path, tables: Mapping[str, Sequence[Mapping[str, str]]], manifest: Mapping[str, Any], report: str) -> dict[str, str]:
    for name, rows in tables.items():
        write_csv_atomic(root / OUTPUT_TABLES[name], rows, table_columns(name))
    write_json_atomic(root / FREEZE_MANIFEST_PATH, manifest)
    write_text_atomic(root / REPORT_PATH, report)
    written = {path.as_posix(): file_sha256_or_missing(root / path) for path in OUTPUT_TABLES.values()}
    written[FREEZE_MANIFEST_PATH.as_posix()] = file_sha256_or_missing(root / FREEZE_MANIFEST_PATH)
    written[REPORT_PATH.as_posix()] = file_sha256_or_missing(root / REPORT_PATH)
    return dict(sorted(written.items()))


def build_phase(root: Path) -> dict[str, str]:
    tables = build_all_tables(root)
    decision = final_decision(tables["metric_reconciliation"])
    manifest = build_manifest(root, tables, decision)
    report = build_report(manifest, tables)
    return write_outputs(root, tables, manifest, report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase F6E4C offline scientific consolidation.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    written = build_phase(args.root)
    print(json.dumps({"written": written}, ensure_ascii=True, sort_keys=True, indent=2))
    return 0

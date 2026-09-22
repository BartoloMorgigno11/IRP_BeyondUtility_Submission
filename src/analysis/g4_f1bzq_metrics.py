"""Compute G4 F1BZQ reranking metrics from the frozen analytic universe.

This phase is offline-only. It reads the frozen G4 analytic universe and the
frozen G4 metric/denominator contract, computes only metrics derivable from
that universe, and records every unavailable or excluded metric explicitly.
It does not call Ollama/API, run statistics, or modify frozen evidence,
F1BZQ pools, protocol, runner, or universe artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.utils.config import find_project_root  # noqa: E402


EXPECTED_HEAD = "d8b045314b4bb0baeee569e6a7453de712020065"
PHASE_ID = "g4_f1bzq_metrics_freeze"
DECISION_FROZEN = "G4_METRICS_FROZEN"
DECISION_BLOCKED = "G4_METRICS_BLOCKED"

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = ("BASE", "LLMRANK")
EXPECTED_USERS = {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}
EXPECTED_POSITIVE_PRESENT = {"beauty": 49, "sports": 28, "ml1m": 193, "lastfm": 469, "foodcom": 4}
EXPECTED_LOGICAL_ROWS = 29292
EXPECTED_TOTAL_USERS = 4882
EXPECTED_CELLS = 30
EXPECTED_STRICT_VALID = 28986
EXPECTED_INVALID = 306
EXPECTED_INVALID_LENGTH = 255
EXPECTED_INVALID_DUPLICATE = 51
EXPECTED_BACKEND_ERRORS = 0
TOP_K = 5
OUTPUT_RANKING_LENGTH = 10

PATHS = {
    "analytic_universe": Path("results/tables/g4_f1bzq_analytic_universe.csv"),
    "analytic_cell_counts": Path("results/tables/g4_f1bzq_analytic_cell_counts.csv"),
    "analytic_denominator_counts": Path("results/tables/g4_f1bzq_analytic_denominator_counts.csv"),
    "analytic_denominator_contract": Path("results/tables/g4_f1bzq_analytic_denominator_contract.csv"),
    "analytic_validation": Path("results/tables/g4_f1bzq_analytic_validation.csv"),
    "analytic_manifest": Path("results/manifests/g4_f1bzq_analytic_universe_freeze.json"),
    "metric_contract": Path("results/tables/g4_f1bzq_metric_denominator_contract.csv"),
    "protocol_manifest": Path("results/manifests/g4_f1bzq_reranking_protocol_freeze.json"),
    "request_manifest": Path("results/manifests/g4_f1bzq_request_universe_manifest.json"),
    "retention_manifest": Path("results/manifests/g4_f1bzq_execution_retention_manifest.json"),
    "request_universe": Path("results/tables/g4_f1bzq_reranking_request_universe.csv"),
    "execution_order": Path("results/tables/g4_f1bzq_reranking_execution_order.csv"),
    "candidate_hashes": Path("results/tables/g4_f1bzq_candidate_hashes.csv"),
    "candidate_recall": Path("results/tables/g4_f1bzq_candidate_recall_at_20.csv"),
    "model_digests": Path("results/tables/g4_f1bzq_model_digest_freeze.csv"),
    "evidence_inventory": Path("results/tables/g4_f1bzq_execution_evidence_inventory.csv"),
    "source": Path("src/analysis/g4_f1bzq_metrics.py"),
    "notes": Path("report/notes.md"),
}

OUTPUTS = {
    "metric_results": Path("results/tables/g4_f1bzq_metric_results.csv"),
    "metric_denominators": Path("results/tables/g4_f1bzq_metric_denominators.csv"),
    "candidate_conditional": Path("results/tables/g4_f1bzq_candidate_conditional_metrics.csv"),
    "reliability": Path("results/tables/g4_f1bzq_reliability.csv"),
    "undefined_blocked": Path("results/tables/g4_f1bzq_metric_undefined_blocked.csv"),
    "validation": Path("results/tables/g4_f1bzq_metric_validation.csv"),
    "manifest": Path("results/manifests/g4_f1bzq_metrics_freeze.json"),
    "report": Path("report/g4_f1bzq_metrics_validation.md"),
}

ALLOWED_DIRTY_PATHS = {
    PATHS["source"],
    PATHS["notes"],
    *OUTPUTS.values(),
}

PRIMARY_UTILITY_METRICS = ("hr_at_5", "ndcg_at_5", "mrr")
RERANKING_DIAGNOSTICS = ("candidate_recall_at_20", "conditional_hr_at_5", "conditional_ndcg_at_5")
BEYOND_COMPUTED = ("item_coverage_at_5", "oic_at_5")
RELIABILITY_COMPUTED = (
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
)
RELIABILITY_UNDEFINED = ("token_counts", "hardware_runtime_provenance")
COMPUTED_METRICS = set(PRIMARY_UTILITY_METRICS + RERANKING_DIAGNOSTICS + BEYOND_COMPUTED + RELIABILITY_COMPUTED)

UNDEFINED_REASONS = {
    "self_information_at_5": "requires train item popularity and total train interactions, absent from the frozen analytic universe",
    "serendipity_at_5": "requires aligned MostPop hit primitive, absent from the frozen analytic universe",
    "arp_at_5": "requires train item popularity, absent from the frozen analytic universe",
    "aplt_at_5": "requires train item popularity and frozen long-tail threshold inputs, absent from the frozen analytic universe",
    "catalog_coverage_at_5": "requires the full catalog denominator, absent from the frozen analytic universe",
    "shannon_category_entropy_at_5": "requires item category metadata, absent from the frozen analytic universe",
    "gini_at_5": "requires a full catalog exposure vector including zero-exposure items; no candidate-union substitution is allowed",
    "jain_index_at_5": "requires a full catalog exposure vector including zero-exposure items; no candidate-union substitution is allowed",
    "ild_at_5": "requires item feature/category metadata, absent from the frozen analytic universe",
    "calibration_at_5": "requires item feature metadata and history item distributions, absent from the frozen analytic universe",
    "token_counts": "token usage fields are absent from the frozen analytic universe",
    "hardware_runtime_provenance": "hardware runtime provenance is not a per-cell metric field in the frozen analytic universe",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute frozen G4 F1BZQ metrics from the analytic universe.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Compute and validate without writing artifacts.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator and expected metric outputs to be dirty while frozen inputs remain unchanged.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any) -> str:
    payload = stable_json([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32].upper()}"


def read_csv(root: Path, path: Path) -> pd.DataFrame:
    return pd.read_csv(root / path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8-sig"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def int_value(value: Any, default: int = 0) -> int:
    text = str(value).strip()
    return default if text == "" else int(text)


def float_value(value: Any) -> float:
    text = str(value).strip()
    return float("nan") if text == "" else float(text)


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite metric value: {value}")
    return f"{float(value):.17g}"


def parse_json_array(value: Any) -> list[str]:
    if str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("expected JSON array")
    return [str(item) for item in parsed]


def parse_int_json_array(value: Any) -> list[int]:
    return [int(item) for item in parse_json_array(value)]


def git_output(root: Path, args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def dirty_path_from_status_line(line: str) -> str:
    if line.startswith("?? "):
        path_text = line[3:].strip()
    elif len(line) >= 3 and line[2] == " ":
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


def metric_contract_lookup(contract: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {str(row.metric_id): {key: str(getattr(row, key)) for key in contract.columns} for row in contract.itertuples()}


def metric_row(
    *,
    dataset: str,
    model: str,
    prompt: str,
    metric_id: str,
    metric_family: str,
    value: float | None,
    status: str,
    numerator: float | int | str,
    denominator: float | int | str,
    denominator_unit: str,
    denominator_scope: str,
    invalid_terminal_policy: str,
    planned_users: int,
    positive_present_users: int,
    positive_absent_users: int,
    strict_valid_users: int,
    invalid_terminal_users: int,
    backend_error_users: int,
    value_json: Mapping[str, Any] | list[Any] | str | None = None,
    item_denominator_count: int | str = "",
    pair_denominator_count: int | str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "result_id": stable_id("G4MET", dataset, model, prompt, metric_id),
        "phase_id": PHASE_ID,
        "dataset": dataset,
        "model": model,
        "prompt_strategy": prompt,
        "metric_id": metric_id,
        "metric_family": metric_family,
        "k": TOP_K if metric_id.endswith("_at_5") or metric_id in {"hr_at_5", "ndcg_at_5"} else "",
        "value": "" if value is None else format_float(float(value)),
        "value_json": "" if value_json is None else (value_json if isinstance(value_json, str) else stable_json(value_json)),
        "status": status,
        "numerator": str(numerator),
        "denominator": str(denominator),
        "denominator_unit": denominator_unit,
        "denominator_scope": denominator_scope,
        "invalid_terminal_policy": invalid_terminal_policy,
        "planned_users": planned_users,
        "positive_present_users": positive_present_users,
        "positive_absent_users": positive_absent_users,
        "strict_valid_users": strict_valid_users,
        "invalid_terminal_users": invalid_terminal_users,
        "backend_error_users": backend_error_users,
        "item_denominator_count": item_denominator_count,
        "pair_denominator_count": pair_denominator_count,
        "source_table": PATHS["analytic_universe"].as_posix(),
        "notes": notes,
    }


def undefined_row(
    *,
    dataset: str,
    model: str,
    prompt: str,
    contract_row: Mapping[str, str],
    planned_users: int,
    positive_present_users: int,
    positive_absent_users: int,
    strict_valid_users: int,
    invalid_terminal_users: int,
    backend_error_users: int,
) -> dict[str, Any]:
    metric_id = contract_row["metric_id"]
    included = bool_value(contract_row["included_in_g4"])
    if not included:
        status = "EXCLUDED_BY_FROZEN_G4_CONTRACT"
        reason = contract_row["reason"]
        denominator_value = ""
        denominator_status = "NOT_APPLICABLE_TO_G4"
    else:
        status = "UNDEFINED_FROZEN_ANALYTIC_UNIVERSE_INPUT_MISSING"
        reason = UNDEFINED_REASONS.get(metric_id, "not computable from the frozen analytic universe without adding inputs")
        denominator_value = strict_valid_users if contract_row["metric_family"] == "beyond_utility_g1_g2_compatible" else planned_users
        denominator_status = "USER_DENOMINATOR_AVAILABLE_REQUIRED_METRIC_INPUT_MISSING"
    return {
        "result_id": stable_id("G4UNDEF", dataset, model, prompt, metric_id),
        "phase_id": PHASE_ID,
        "dataset": dataset,
        "model": model,
        "prompt_strategy": prompt,
        "metric_id": metric_id,
        "metric_family": contract_row["metric_family"],
        "included_in_g4": contract_row["included_in_g4"],
        "contract_status": contract_row["status"],
        "status": status,
        "reason": reason,
        "denominator_contract": contract_row["denominator"],
        "denominator_value": denominator_value,
        "denominator_status": denominator_status,
        "planned_users": planned_users,
        "positive_present_users": positive_present_users,
        "positive_absent_users": positive_absent_users,
        "strict_valid_users": strict_valid_users,
        "invalid_terminal_users": invalid_terminal_users,
        "backend_error_users": backend_error_users,
        "source_table": PATHS["analytic_universe"].as_posix(),
        "no_denominator_substitution": "true",
    }


def denominator_row_from_result(row: Mapping[str, Any], contract_row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "denominator_id": stable_id("G4DEN", row["dataset"], row["model"], row["prompt_strategy"], row["metric_id"]),
        "phase_id": PHASE_ID,
        "dataset": row["dataset"],
        "model": row["model"],
        "prompt_strategy": row["prompt_strategy"],
        "metric_id": row["metric_id"],
        "metric_family": row["metric_family"],
        "metric_status": row["status"],
        "contract_denominator": contract_row["denominator"],
        "denominator_value": row["denominator"],
        "denominator_unit": row["denominator_unit"],
        "denominator_scope": row["denominator_scope"],
        "invalid_terminal_policy": row["invalid_terminal_policy"],
        "planned_users": row["planned_users"],
        "positive_present_users": row["positive_present_users"],
        "positive_absent_users": row["positive_absent_users"],
        "strict_valid_users": row["strict_valid_users"],
        "invalid_terminal_users": row["invalid_terminal_users"],
        "backend_error_users": row["backend_error_users"],
        "item_denominator_count": row.get("item_denominator_count", ""),
        "pair_denominator_count": row.get("pair_denominator_count", ""),
        "source_table": row["source_table"],
        "no_silent_denominator_substitution": "true",
        "notes": row.get("notes", ""),
    }


def denominator_row_from_undefined(row: Mapping[str, Any], contract_row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "denominator_id": stable_id("G4DEN", row["dataset"], row["model"], row["prompt_strategy"], row["metric_id"]),
        "phase_id": PHASE_ID,
        "dataset": row["dataset"],
        "model": row["model"],
        "prompt_strategy": row["prompt_strategy"],
        "metric_id": row["metric_id"],
        "metric_family": row["metric_family"],
        "metric_status": row["status"],
        "contract_denominator": contract_row["denominator"],
        "denominator_value": row["denominator_value"],
        "denominator_unit": contract_row["denominator"],
        "denominator_scope": row["denominator_status"],
        "invalid_terminal_policy": contract_row["invalid_terminal_policy"],
        "planned_users": row["planned_users"],
        "positive_present_users": row["positive_present_users"],
        "positive_absent_users": row["positive_absent_users"],
        "strict_valid_users": row["strict_valid_users"],
        "invalid_terminal_users": row["invalid_terminal_users"],
        "backend_error_users": row["backend_error_users"],
        "item_denominator_count": "",
        "pair_denominator_count": "",
        "source_table": row["source_table"],
        "no_silent_denominator_substitution": row["no_denominator_substitution"],
        "notes": row["reason"],
    }


def compute_row_contributions(universe: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, list[str]], list[str]]:
    work = universe.copy()
    work["_strict_valid"] = work["strict_parse_valid"].map(bool_value)
    work["_positive_present"] = work["positive_present_in_pool"].map(bool_value)
    work["_backend_success"] = work["backend_success"].map(bool_value)
    work["_hr_at_5_contribution"] = 0.0
    work["_ndcg_at_5_contribution"] = 0.0
    work["_mrr_contribution"] = 0.0
    work["_positive_rank_zero_based"] = ""
    work["_positive_absent_zero_contribution"] = "true"
    top5_items_by_index: dict[int, list[str]] = {}
    defects: list[str] = []

    for idx, row in work.iterrows():
        strict_valid = bool(row["_strict_valid"])
        positive_present = bool(row["_positive_present"])
        if not strict_valid:
            continue

        try:
            ranked_indices = parse_int_json_array(row["parsed_top10_candidate_indices_if_strict_valid"])
            ranked_items = parse_json_array(row["parsed_top10_item_ids_if_strict_valid"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            defects.append(f"{row['canonical_request_id']}: strict-valid row has unparsable top-10 output: {exc}")
            continue

        if len(ranked_indices) != OUTPUT_RANKING_LENGTH or len(set(ranked_indices)) != OUTPUT_RANKING_LENGTH:
            defects.append(f"{row['canonical_request_id']}: strict-valid row violates top-10 unique index contract")
        if len(ranked_items) != OUTPUT_RANKING_LENGTH:
            defects.append(f"{row['canonical_request_id']}: strict-valid row violates top-10 item contract")
        if any(index < 1 or index > int_value(row["candidate_count"]) for index in ranked_indices):
            defects.append(f"{row['canonical_request_id']}: strict-valid row references an out-of-range candidate index")

        top5_items_by_index[idx] = ranked_items[:TOP_K]

        if not positive_present:
            continue

        positive_position = int_value(row["positive_input_position"], default=-1)
        if positive_position < 1:
            defects.append(f"{row['canonical_request_id']}: positive-present row has no positive input position")
            continue
        try:
            rank_zero_based = ranked_indices.index(positive_position)
        except ValueError:
            rank_zero_based = None

        if rank_zero_based is not None:
            work.at[idx, "_positive_rank_zero_based"] = str(rank_zero_based)
            work.at[idx, "_mrr_contribution"] = 1.0 / float(rank_zero_based + 1)
            if rank_zero_based < TOP_K:
                work.at[idx, "_hr_at_5_contribution"] = 1.0
                work.at[idx, "_ndcg_at_5_contribution"] = 1.0 / math.log2(rank_zero_based + 2)

    return work, top5_items_by_index, defects


def candidate_union_for_rows(rows: pd.DataFrame) -> set[str]:
    candidate_union: set[str] = set()
    for value in rows["candidate_items"]:
        candidate_union.update(parse_json_array(value))
    return candidate_union


def oic_at_5(top5_lists: list[list[str]]) -> tuple[float | None, int, int]:
    user_count = len(top5_lists)
    if user_count < 2:
        return None, 0, 0
    exposure_counts: Counter[str] = Counter()
    for items in top5_lists:
        exposure_counts.update(set(items[:TOP_K]))
    overlap_pairs = sum(count * (count - 1) // 2 for count in exposure_counts.values())
    pair_count = user_count * (user_count - 1) // 2
    pair_position_denominator = pair_count * TOP_K
    return overlap_pairs / float(pair_position_denominator), overlap_pairs, pair_position_denominator


def latency_summary(group: pd.DataFrame) -> dict[str, float | int | str]:
    runtimes = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna().astype(float)
    if runtimes.empty:
        raise ValueError("latency metric requested but runtime_seconds has no numeric values")
    started = pd.to_datetime(group["request_started_at_utc"], utc=True, errors="coerce")
    completed = pd.to_datetime(group["request_completed_at_utc"], utc=True, errors="coerce")
    min_started = started.min()
    max_completed = completed.max()
    wall_seconds = float((max_completed - min_started).total_seconds()) if pd.notna(min_started) and pd.notna(max_completed) else float("nan")
    throughput_per_hour = float(len(group) / (wall_seconds / 3600.0)) if wall_seconds > 0 else float("nan")
    values = runtimes.to_numpy(dtype=float)
    return {
        "runtime_observations": int(len(values)),
        "mean_latency_seconds": float(np.mean(values)),
        "median_latency_seconds": float(np.median(values)),
        "p90_latency_seconds": float(np.quantile(values, 0.90)),
        "p95_latency_seconds": float(np.quantile(values, 0.95)),
        "p99_latency_seconds": float(np.quantile(values, 0.99)),
        "min_request_started_at_utc": "" if pd.isna(min_started) else min_started.isoformat().replace("+00:00", "Z"),
        "max_request_completed_at_utc": "" if pd.isna(max_completed) else max_completed.isoformat().replace("+00:00", "Z"),
        "wall_seconds": wall_seconds,
        "throughput_requests_per_hour": throughput_per_hour,
    }


def compute_metrics(
    universe: pd.DataFrame,
    contract: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    work, top5_items_by_index, contribution_defects = compute_row_contributions(universe)
    contract_by_metric = metric_contract_lookup(contract)
    metric_results: list[dict[str, Any]] = []
    undefined_blocked: list[dict[str, Any]] = []
    denominator_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for model in MODELS:
            for prompt in PROMPTS:
                group = work[
                    (work["dataset"] == dataset)
                    & (work["model"] == model)
                    & (work["prompt_strategy"] == prompt)
                ]
                planned_users = int(len(group))
                positive_present_users = int(group["_positive_present"].sum())
                positive_absent_users = planned_users - positive_present_users
                strict_valid_users = int(group["_strict_valid"].sum())
                invalid_terminal_users = planned_users - strict_valid_users
                backend_error_users = int((~group["_backend_success"]).sum())
                invalid_length = int((group["terminal_state"] == "INVALID_LENGTH").sum())
                invalid_duplicate = int((group["terminal_state"] == "INVALID_DUPLICATE").sum())
                hallucination_mask = (
                    pd.to_numeric(group["parsed_unknown_index_count"], errors="coerce").fillna(0).gt(0)
                    | group["terminal_state"].isin(
                        ["HALLUCINATION", "HALLUCINATION_OR_OUT_OF_CANDIDATE", "INVALID_CANDIDATE_REFERENCE"]
                    )
                )
                hallucination_rows = int(hallucination_mask.sum())

                utility_denominator_unit = "planned user-request rows within dataset/model/prompt cell"
                utility_policy = "invalid terminal response contributes zero; positive-absent users contribute zero"
                utility_scope = "all planned users"
                for metric_id, contribution_col in (
                    ("hr_at_5", "_hr_at_5_contribution"),
                    ("ndcg_at_5", "_ndcg_at_5_contribution"),
                    ("mrr", "_mrr_contribution"),
                ):
                    numerator = float(group[contribution_col].sum())
                    result = metric_row(
                        dataset=dataset,
                        model=model,
                        prompt=prompt,
                        metric_id=metric_id,
                        metric_family="primary_unconditional_utility",
                        value=numerator / float(planned_users),
                        status="COMPUTED_FROM_FROZEN_ANALYTIC_UNIVERSE",
                        numerator=format_float(numerator),
                        denominator=planned_users,
                        denominator_unit=utility_denominator_unit,
                        denominator_scope=utility_scope,
                        invalid_terminal_policy=utility_policy,
                        planned_users=planned_users,
                        positive_present_users=positive_present_users,
                        positive_absent_users=positive_absent_users,
                        strict_valid_users=strict_valid_users,
                        invalid_terminal_users=invalid_terminal_users,
                        backend_error_users=backend_error_users,
                    )
                    metric_results.append(result)
                    denominator_rows.append(denominator_row_from_result(result, contract_by_metric[metric_id]))

                candidate_recall = positive_present_users / float(planned_users)
                result = metric_row(
                    dataset=dataset,
                    model=model,
                    prompt=prompt,
                    metric_id="candidate_recall_at_20",
                    metric_family="reranking_diagnostic",
                    value=candidate_recall,
                    status="COMPUTED_FROM_FROZEN_F1BZQ_POOL_IDENTITY",
                    numerator=positive_present_users,
                    denominator=planned_users,
                    denominator_unit="unique dataset/user candidate pools represented in this cell",
                    denominator_scope="all users",
                    invalid_terminal_policy="not applicable to terminal response validity",
                    planned_users=planned_users,
                    positive_present_users=positive_present_users,
                    positive_absent_users=positive_absent_users,
                    strict_valid_users=strict_valid_users,
                    invalid_terminal_users=invalid_terminal_users,
                    backend_error_users=backend_error_users,
                )
                metric_results.append(result)
                denominator_rows.append(denominator_row_from_result(result, contract_by_metric["candidate_recall_at_20"]))

                positive_group = group[group["_positive_present"]]
                conditional_scope = "positive-present users only"
                conditional_policy = "invalid terminal response contributes zero within positive-present subset"
                for metric_id, contribution_col in (
                    ("conditional_hr_at_5", "_hr_at_5_contribution"),
                    ("conditional_ndcg_at_5", "_ndcg_at_5_contribution"),
                ):
                    numerator = float(positive_group[contribution_col].sum())
                    result = metric_row(
                        dataset=dataset,
                        model=model,
                        prompt=prompt,
                        metric_id=metric_id,
                        metric_family="reranking_diagnostic",
                        value=numerator / float(positive_present_users),
                        status="COMPUTED_FROM_FROZEN_ANALYTIC_UNIVERSE",
                        numerator=format_float(numerator),
                        denominator=positive_present_users,
                        denominator_unit="positive-present user-request rows within dataset/model/prompt cell",
                        denominator_scope=conditional_scope,
                        invalid_terminal_policy=conditional_policy,
                        planned_users=planned_users,
                        positive_present_users=positive_present_users,
                        positive_absent_users=positive_absent_users,
                        strict_valid_users=strict_valid_users,
                        invalid_terminal_users=invalid_terminal_users,
                        backend_error_users=backend_error_users,
                        notes="Food.com conditional denominator is 4 users" if dataset == "foodcom" else "",
                    )
                    metric_results.append(result)
                    denominator_rows.append(denominator_row_from_result(result, contract_by_metric[metric_id]))

                valid_group = group[group["_strict_valid"]]
                valid_top5_lists = [top5_items_by_index[idx] for idx in valid_group.index if idx in top5_items_by_index]
                exposed_top5_items = {item for items in valid_top5_lists for item in items[:TOP_K]}
                candidate_union = candidate_union_for_rows(valid_group)
                item_coverage_value = len(exposed_top5_items) / float(len(candidate_union))
                result = metric_row(
                    dataset=dataset,
                    model=model,
                    prompt=prompt,
                    metric_id="item_coverage_at_5",
                    metric_family="beyond_utility_g1_g2_compatible",
                    value=item_coverage_value,
                    status="COMPUTED_FROM_STRICT_VALID_FROZEN_ANALYTIC_UNIVERSE_ROWS",
                    numerator=len(exposed_top5_items),
                    denominator=strict_valid_users,
                    denominator_unit="strict-valid user-request rows; item coverage denominator is strict-valid candidate union",
                    denominator_scope="strict-valid users",
                    invalid_terminal_policy="invalid terminal responses excluded from beyond-utility denominator",
                    planned_users=planned_users,
                    positive_present_users=positive_present_users,
                    positive_absent_users=positive_absent_users,
                    strict_valid_users=strict_valid_users,
                    invalid_terminal_users=invalid_terminal_users,
                    backend_error_users=backend_error_users,
                    item_denominator_count=len(candidate_union),
                    notes="unique strict-valid top-5 items divided by strict-valid candidate union; no full-catalog substitution",
                )
                metric_results.append(result)
                denominator_rows.append(denominator_row_from_result(result, contract_by_metric["item_coverage_at_5"]))

                oic_value, overlap_pairs, pair_denominator = oic_at_5(valid_top5_lists)
                if oic_value is None:
                    raise RuntimeError(f"OIC undefined for {dataset}/{model}/{prompt}: fewer than two strict-valid rows")
                result = metric_row(
                    dataset=dataset,
                    model=model,
                    prompt=prompt,
                    metric_id="oic_at_5",
                    metric_family="beyond_utility_g1_g2_compatible",
                    value=oic_value,
                    status="COMPUTED_FROM_STRICT_VALID_FROZEN_ANALYTIC_UNIVERSE_ROWS",
                    numerator=overlap_pairs,
                    denominator=strict_valid_users,
                    denominator_unit="strict-valid user-request rows; pair denominator records unordered user pairs times K",
                    denominator_scope="strict-valid users",
                    invalid_terminal_policy="invalid terminal responses excluded from beyond-utility denominator",
                    planned_users=planned_users,
                    positive_present_users=positive_present_users,
                    positive_absent_users=positive_absent_users,
                    strict_valid_users=strict_valid_users,
                    invalid_terminal_users=invalid_terminal_users,
                    backend_error_users=backend_error_users,
                    pair_denominator_count=pair_denominator,
                    notes="mean pairwise top-5 overlap over strict-valid users",
                )
                metric_results.append(result)
                denominator_rows.append(denominator_row_from_result(result, contract_by_metric["oic_at_5"]))

                latency = latency_summary(group)
                invalid_json = {
                    "counts": {
                        "invalid_length": invalid_length,
                        "duplicate_indices": invalid_duplicate,
                        "hallucination_or_unknown_index": hallucination_rows,
                        "backend_errors": backend_error_users,
                    },
                    "rates_per_planned_request": {
                        "invalid_length": invalid_length / float(planned_users),
                        "duplicate_indices": invalid_duplicate / float(planned_users),
                        "hallucination_or_unknown_index": hallucination_rows / float(planned_users),
                        "backend_errors": backend_error_users / float(planned_users),
                    },
                }
                reliability_row = {
                    "phase_id": PHASE_ID,
                    "dataset": dataset,
                    "model": model,
                    "prompt_strategy": prompt,
                    "planned_requests": planned_users,
                    "strict_valid": strict_valid_users,
                    "structural_invalid": invalid_terminal_users,
                    "invalid_length": invalid_length,
                    "invalid_duplicate": invalid_duplicate,
                    "hallucination_or_unknown_index": hallucination_rows,
                    "backend_errors": backend_error_users,
                    "strict_valid_rate": format_float(strict_valid_users / float(planned_users)),
                    "structural_invalid_rate": format_float(invalid_terminal_users / float(planned_users)),
                    "invalid_length_rate": format_float(invalid_length / float(planned_users)),
                    "invalid_duplicate_rate": format_float(invalid_duplicate / float(planned_users)),
                    "hallucination_rate": format_float(hallucination_rows / float(planned_users)),
                    "backend_failure_rate": format_float(backend_error_users / float(planned_users)),
                    "mean_latency_seconds": format_float(float(latency["mean_latency_seconds"])),
                    "median_latency_seconds": format_float(float(latency["median_latency_seconds"])),
                    "p90_latency_seconds": format_float(float(latency["p90_latency_seconds"])),
                    "p95_latency_seconds": format_float(float(latency["p95_latency_seconds"])),
                    "p99_latency_seconds": format_float(float(latency["p99_latency_seconds"])),
                    "min_request_started_at_utc": latency["min_request_started_at_utc"],
                    "max_request_completed_at_utc": latency["max_request_completed_at_utc"],
                    "wall_seconds": format_float(float(latency["wall_seconds"])),
                    "throughput_requests_per_hour": format_float(float(latency["throughput_requests_per_hour"])),
                    "invalid_reason_distribution_json": stable_json(invalid_json),
                    "token_counts_status": "UNDEFINED_FROZEN_ANALYTIC_UNIVERSE_INPUT_MISSING",
                    "hardware_runtime_provenance_status": "UNDEFINED_FROZEN_ANALYTIC_UNIVERSE_INPUT_MISSING",
                }
                reliability_rows.append(reliability_row)

                reliability_values: list[tuple[str, float | None, str | None, float | int | str, Mapping[str, Any] | None]] = [
                    ("strict_parse_validity", strict_valid_users / float(planned_users), None, strict_valid_users, None),
                    ("structural_invalidity_rate", invalid_terminal_users / float(planned_users), None, invalid_terminal_users, None),
                    ("invalid_reason_distribution", None, "COMPUTED_JSON_FROM_FROZEN_ANALYTIC_UNIVERSE", invalid_terminal_users, invalid_json),
                    ("duplicate_rate", invalid_duplicate / float(planned_users), None, invalid_duplicate, None),
                    ("hallucination", hallucination_rows / float(planned_users), None, hallucination_rows, None),
                    ("backend_failure_rate", backend_error_users / float(planned_users), None, backend_error_users, None),
                    ("latency", float(latency["mean_latency_seconds"]), None, latency["runtime_observations"], latency),
                    ("mean_latency", float(latency["mean_latency_seconds"]), None, latency["runtime_observations"], None),
                    ("median_latency", float(latency["median_latency_seconds"]), None, latency["runtime_observations"], None),
                    ("p90_latency", float(latency["p90_latency_seconds"]), None, latency["runtime_observations"], None),
                    ("p95_latency", float(latency["p95_latency_seconds"]), None, latency["runtime_observations"], None),
                    ("p99_latency", float(latency["p99_latency_seconds"]), None, latency["runtime_observations"], None),
                    ("throughput", float(latency["throughput_requests_per_hour"]), None, planned_users, {"wall_seconds": latency["wall_seconds"]}),
                ]
                for metric_id, value, status_override, numerator, value_json in reliability_values:
                    result = metric_row(
                        dataset=dataset,
                        model=model,
                        prompt=prompt,
                        metric_id=metric_id,
                        metric_family="reliability_efficiency",
                        value=value,
                        status=status_override or "COMPUTED_FROM_FROZEN_ANALYTIC_UNIVERSE",
                        numerator=numerator,
                        denominator=planned_users,
                        denominator_unit="planned requests",
                        denominator_scope="all planned requests",
                        invalid_terminal_policy="invalid terminal response remains terminal scientific evidence and is counted",
                        planned_users=planned_users,
                        positive_present_users=positive_present_users,
                        positive_absent_users=positive_absent_users,
                        strict_valid_users=strict_valid_users,
                        invalid_terminal_users=invalid_terminal_users,
                        backend_error_users=backend_error_users,
                        value_json=value_json,
                    )
                    metric_results.append(result)
                    denominator_rows.append(denominator_row_from_result(result, contract_by_metric[metric_id]))

                for contract_row in contract.to_dict("records"):
                    metric_id = str(contract_row["metric_id"])
                    if metric_id in COMPUTED_METRICS:
                        continue
                    row = undefined_row(
                        dataset=dataset,
                        model=model,
                        prompt=prompt,
                        contract_row=contract_row,
                        planned_users=planned_users,
                        positive_present_users=positive_present_users,
                        positive_absent_users=positive_absent_users,
                        strict_valid_users=strict_valid_users,
                        invalid_terminal_users=invalid_terminal_users,
                        backend_error_users=backend_error_users,
                    )
                    undefined_blocked.append(row)
                    denominator_rows.append(denominator_row_from_undefined(row, contract_by_metric[metric_id]))

    metric_frame = pd.DataFrame(metric_results).sort_values(["dataset", "model", "prompt_strategy", "metric_family", "metric_id"])
    denominator_frame = pd.DataFrame(denominator_rows).sort_values(["dataset", "model", "prompt_strategy", "metric_family", "metric_id"])
    candidate_conditional_frame = metric_frame[
        metric_frame["metric_id"].isin(RERANKING_DIAGNOSTICS)
    ].copy()
    reliability_frame = pd.DataFrame(reliability_rows).sort_values(["dataset", "model", "prompt_strategy"])
    undefined_frame = pd.DataFrame(undefined_blocked).sort_values(["dataset", "model", "prompt_strategy", "metric_family", "metric_id"])
    return metric_frame, denominator_frame, candidate_conditional_frame, reliability_frame, undefined_frame, contribution_defects


def build_validation(
    *,
    root: Path,
    head_before: str,
    status_before: str,
    universe: pd.DataFrame,
    contract: pd.DataFrame,
    metric_results: pd.DataFrame,
    denominator_frame: pd.DataFrame,
    candidate_conditional: pd.DataFrame,
    reliability: pd.DataFrame,
    undefined_blocked: pd.DataFrame,
    contribution_defects: list[str],
    allow_generator_dirty: bool,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check_id: str, passed: bool, observed: Any, expected: Any, details: str = "") -> None:
        rows.append(
            {
                "phase_id": PHASE_ID,
                "check_id": check_id,
                "status": "PASS" if passed else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
                "details": details,
            }
        )

    analytic_manifest = read_json(root, PATHS["analytic_manifest"])
    add("head_matches_precondition", head_before == EXPECTED_HEAD, head_before, EXPECTED_HEAD)
    unexpected_status = unexpected_git_status_lines(status_before)
    git_ok = status_before.strip() == "" or (allow_generator_dirty and not unexpected_status)
    add("git_status_clean_or_expected_metric_dirty", git_ok, status_before.replace("\n", " | "), "clean or expected G4 metric outputs only")

    frozen_hash_expectations: dict[str, str] = {}
    frozen_hash_expectations.update(analytic_manifest.get("input_hashes", {}))
    frozen_hash_expectations.update(analytic_manifest.get("output_hashes", {}))
    frozen_hash_expectations.update(
        {
            PATHS["analytic_manifest"].as_posix(): sha256_file(root / PATHS["analytic_manifest"]),
            PATHS["metric_contract"].as_posix(): sha256_file(root / PATHS["metric_contract"]),
        }
    )
    for path_text, expected_hash in sorted(frozen_hash_expectations.items()):
        path = root / Path(path_text)
        if not path.exists():
            add(f"frozen_input_exists::{path_text}", False, "missing", "exists")
            continue
        observed_hash = sha256_file(path)
        add(f"frozen_input_hash::{path_text}", observed_hash == expected_hash, observed_hash, expected_hash)

    add("analytic_manifest_decision", analytic_manifest.get("decision_token") == "G4_ANALYTIC_UNIVERSE_FROZEN", analytic_manifest.get("decision_token"), "G4_ANALYTIC_UNIVERSE_FROZEN")
    add("analytic_manifest_validation_passed", bool(analytic_manifest.get("validation_passed")), analytic_manifest.get("validation_passed"), True)

    add("logical_rows", len(universe) == EXPECTED_LOGICAL_ROWS, len(universe), EXPECTED_LOGICAL_ROWS)
    unique_users = universe[["dataset", "user_id"]].drop_duplicates().shape[0]
    add("unique_dataset_users", unique_users == EXPECTED_TOTAL_USERS, unique_users, EXPECTED_TOTAL_USERS)
    cell_count = universe[["dataset", "model", "prompt_strategy"]].drop_duplicates().shape[0]
    add("cell_count", cell_count == EXPECTED_CELLS, cell_count, EXPECTED_CELLS)
    add("contract_metric_count", len(contract) == 38, len(contract), 38)

    for dataset in DATASETS:
        dataset_rows = universe[universe["dataset"] == dataset]
        dataset_users = dataset_rows[["dataset", "user_id"]].drop_duplicates().shape[0]
        dataset_positive = dataset_rows[["dataset", "user_id", "positive_present_in_pool"]].drop_duplicates()
        positive_present = int(dataset_positive["positive_present_in_pool"].map(bool_value).sum())
        add(f"dataset_users::{dataset}", dataset_users == EXPECTED_USERS[dataset], dataset_users, EXPECTED_USERS[dataset])
        add(
            f"positive_present_users::{dataset}",
            positive_present == EXPECTED_POSITIVE_PRESENT[dataset],
            positive_present,
            EXPECTED_POSITIVE_PRESENT[dataset],
        )

    strict_valid = int(universe["strict_parse_valid"].map(bool_value).sum())
    invalid = len(universe) - strict_valid
    invalid_length = int((universe["terminal_state"] == "INVALID_LENGTH").sum())
    invalid_duplicate = int((universe["terminal_state"] == "INVALID_DUPLICATE").sum())
    backend_errors = int((~universe["backend_success"].map(bool_value)).sum())
    add("strict_valid_total", strict_valid == EXPECTED_STRICT_VALID, strict_valid, EXPECTED_STRICT_VALID)
    add("invalid_total", invalid == EXPECTED_INVALID, invalid, EXPECTED_INVALID)
    add("invalid_length_total", invalid_length == EXPECTED_INVALID_LENGTH, invalid_length, EXPECTED_INVALID_LENGTH)
    add("invalid_duplicate_total", invalid_duplicate == EXPECTED_INVALID_DUPLICATE, invalid_duplicate, EXPECTED_INVALID_DUPLICATE)
    add("backend_errors_total", backend_errors == EXPECTED_BACKEND_ERRORS, backend_errors, EXPECTED_BACKEND_ERRORS)

    expected_keys = set()
    for dataset in DATASETS:
        for model in MODELS:
            for prompt in PROMPTS:
                for metric_id in contract["metric_id"].tolist():
                    expected_keys.add((dataset, model, prompt, metric_id))
    result_keys = [tuple(row) for row in metric_results[["dataset", "model", "prompt_strategy", "metric_id"]].to_numpy()]
    undefined_keys = [tuple(row) for row in undefined_blocked[["dataset", "model", "prompt_strategy", "metric_id"]].to_numpy()]
    denominator_keys = [tuple(row) for row in denominator_frame[["dataset", "model", "prompt_strategy", "metric_id"]].to_numpy()]
    combined_key_counts = Counter(result_keys + undefined_keys)
    denominator_key_counts = Counter(denominator_keys)
    add("computed_metric_rows", len(metric_results) == 630, len(metric_results), 630)
    add("undefined_or_blocked_rows", len(undefined_blocked) == 510, len(undefined_blocked), 510)
    add("denominator_rows", len(denominator_frame) == len(expected_keys), len(denominator_frame), len(expected_keys))
    add(
        "metric_cell_universe_exact_once",
        set(combined_key_counts) == expected_keys and all(count == 1 for count in combined_key_counts.values()),
        f"keys={len(combined_key_counts)} max_count={max(combined_key_counts.values()) if combined_key_counts else 0}",
        f"keys={len(expected_keys)} count=1",
    )
    add(
        "denominator_cell_universe_exact_once",
        set(denominator_key_counts) == expected_keys and all(count == 1 for count in denominator_key_counts.values()),
        f"keys={len(denominator_key_counts)} max_count={max(denominator_key_counts.values()) if denominator_key_counts else 0}",
        f"keys={len(expected_keys)} count=1",
    )

    for dataset in DATASETS:
        subset = metric_results[
            (metric_results["dataset"] == dataset) & (metric_results["metric_id"] == "candidate_recall_at_20")
        ]
        values = sorted(set(subset["value"].tolist()))
        expected_recall = EXPECTED_POSITIVE_PRESENT[dataset] / float(EXPECTED_USERS[dataset])
        passed = len(values) == 1 and math.isclose(float(values[0]), expected_recall, rel_tol=0.0, abs_tol=1e-15)
        add(f"candidate_recall_identical::{dataset}", passed, values, format_float(expected_recall))

    work, _, _ = compute_row_contributions(universe)
    positive_absent_rows = work[~work["_positive_present"]]
    positive_absent_utility_sum = float(
        positive_absent_rows[["_hr_at_5_contribution", "_ndcg_at_5_contribution", "_mrr_contribution"]].sum().sum()
    )
    add("positive_absent_users_zero_unconditional_utility", positive_absent_utility_sum == 0.0, positive_absent_utility_sum, 0.0)

    conditional = candidate_conditional[candidate_conditional["metric_id"].isin(["conditional_hr_at_5", "conditional_ndcg_at_5"])]
    conditional_denominators_ok = all(
        int(row.denominator) == EXPECTED_POSITIVE_PRESENT[str(row.dataset)] for row in conditional.itertuples()
    )
    add("conditional_denominators_positive_present_only", conditional_denominators_ok, "checked", "denominator equals positive-present users")
    foodcom_conditional = conditional[conditional["dataset"] == "foodcom"]
    foodcom_ok = len(foodcom_conditional) == 12 and set(foodcom_conditional["denominator"].astype(str)) == {"4"}
    add("foodcom_conditional_denominator_marked_4", foodcom_ok, f"rows={len(foodcom_conditional)} denominators={sorted(set(foodcom_conditional['denominator'].astype(str)))}", "12 rows denominator 4")

    beyond_computed = metric_results[metric_results["metric_family"] == "beyond_utility_g1_g2_compatible"]
    add("beyond_computed_rows", len(beyond_computed) == 60, len(beyond_computed), 60)
    beyond_blocked = undefined_blocked[undefined_blocked["metric_family"] == "beyond_utility_g1_g2_compatible"]
    add("beyond_undefined_rows", len(beyond_blocked) == 300, len(beyond_blocked), 300)
    reliability_metric_rows = metric_results[metric_results["metric_family"] == "reliability_efficiency"]
    add("reliability_computed_rows", len(reliability_metric_rows) == 390, len(reliability_metric_rows), 390)
    reliability_blocked = undefined_blocked[undefined_blocked["metric_family"] == "reliability_efficiency"]
    add("reliability_undefined_rows", len(reliability_blocked) == 60, len(reliability_blocked), 60)
    excluded_rows = undefined_blocked[undefined_blocked["status"] == "EXCLUDED_BY_FROZEN_G4_CONTRACT"]
    add("excluded_contract_rows", len(excluded_rows) == 150, len(excluded_rows), 150)
    no_silent_substitution = (
        undefined_blocked["no_denominator_substitution"].eq("true").all()
        and not set(undefined_blocked["metric_id"]).intersection(set(metric_results["metric_id"]))
    )
    add("blocked_or_undefined_no_denominator_substitution", bool(no_silent_substitution), "checked", "all blocked/undefined metrics absent from computed results")

    reliability_totals = reliability[["planned_requests", "strict_valid", "structural_invalid", "invalid_length", "invalid_duplicate", "backend_errors"]].astype(int).sum()
    add("reliability_planned_total", int(reliability_totals["planned_requests"]) == EXPECTED_LOGICAL_ROWS, int(reliability_totals["planned_requests"]), EXPECTED_LOGICAL_ROWS)
    add("reliability_strict_valid_total", int(reliability_totals["strict_valid"]) == EXPECTED_STRICT_VALID, int(reliability_totals["strict_valid"]), EXPECTED_STRICT_VALID)
    add("reliability_invalid_total", int(reliability_totals["structural_invalid"]) == EXPECTED_INVALID, int(reliability_totals["structural_invalid"]), EXPECTED_INVALID)
    add("reliability_invalid_length_total", int(reliability_totals["invalid_length"]) == EXPECTED_INVALID_LENGTH, int(reliability_totals["invalid_length"]), EXPECTED_INVALID_LENGTH)
    add("reliability_invalid_duplicate_total", int(reliability_totals["invalid_duplicate"]) == EXPECTED_INVALID_DUPLICATE, int(reliability_totals["invalid_duplicate"]), EXPECTED_INVALID_DUPLICATE)
    add("reliability_backend_errors_total", int(reliability_totals["backend_errors"]) == EXPECTED_BACKEND_ERRORS, int(reliability_totals["backend_errors"]), EXPECTED_BACKEND_ERRORS)

    add("strict_valid_output_parse_defects", not contribution_defects, len(contribution_defects), 0, " | ".join(contribution_defects[:5]))
    metric_value_nonempty = metric_results["value"].ne("").sum()
    add("computed_scalar_or_json_values_present", metric_value_nonempty == 600, metric_value_nonempty, 600, "30 invalid_reason_distribution rows are JSON-only")

    return pd.DataFrame(rows)


def build_report(
    *,
    validation: pd.DataFrame,
    metric_results: pd.DataFrame,
    candidate_conditional: pd.DataFrame,
    reliability: pd.DataFrame,
    undefined_blocked: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> str:
    validation_passes = int((validation["status"] == "PASS").sum())
    validation_failures = int((validation["status"] == "FAIL").sum())
    candidate_rows = candidate_conditional[candidate_conditional["metric_id"] == "candidate_recall_at_20"]
    recall_lines = []
    for dataset in DATASETS:
        value = sorted(set(candidate_rows[candidate_rows["dataset"] == dataset]["value"].tolist()))[0]
        recall_lines.append(f"- {dataset}: {value} ({EXPECTED_POSITIVE_PRESENT[dataset]}/{EXPECTED_USERS[dataset]})")
    reliability_totals = reliability[["planned_requests", "strict_valid", "structural_invalid", "invalid_length", "invalid_duplicate", "backend_errors"]].astype(int).sum()
    blocked_counts = undefined_blocked.groupby(["metric_family", "status"]).size().reset_index(name="rows")
    blocked_lines = [
        f"- {row.metric_family} / {row.status}: {int(row.rows)}"
        for row in blocked_counts.sort_values(["metric_family", "status"]).itertuples()
    ]
    return "\n".join(
        [
            "# G4 F1BZQ Metrics Freeze",
            "",
            f"- Phase: {PHASE_ID}",
            f"- Decision token: {manifest['decision_token']}",
            "- Source: frozen G4 analytic universe only",
            "- Ollama/API calls: 0",
            "- Statistics/bootstrap: not run",
            "",
            "## Scope",
            "",
            f"- Logical rows: {manifest['summary']['logical_rows']}",
            f"- Users: {manifest['summary']['unique_users']}",
            f"- Cells: {manifest['summary']['cells']}",
            f"- Computed metric rows: {len(metric_results)}",
            f"- Undefined/blocked metric rows: {len(undefined_blocked)}",
            "",
            "## Candidate Recall",
            "",
            *recall_lines,
            "",
            "## Reliability Accounting",
            "",
            f"- Planned/backend calls represented: {int(reliability_totals['planned_requests'])}",
            f"- Strict-valid: {int(reliability_totals['strict_valid'])}",
            f"- Structural invalid: {int(reliability_totals['structural_invalid'])}",
            f"- invalid_length: {int(reliability_totals['invalid_length'])}",
            f"- duplicate_indices: {int(reliability_totals['invalid_duplicate'])}",
            f"- Backend errors: {int(reliability_totals['backend_errors'])}",
            "",
            "## Undefined Or Blocked",
            "",
            *blocked_lines,
            "",
            "## Validation",
            "",
            f"- PASS: {validation_passes}",
            f"- FAIL: {validation_failures}",
            f"- Status: {'PASS' if validation_failures == 0 else 'FAIL'}",
            "",
            "No scientific synthesis, ranking-vs-reranking interpretation, or statistical testing is included in this freeze.",
            "",
        ]
    )


def append_notes(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = "\n".join(
        [
            f"## {PHASE_ID} - {manifest['created_at_utc']}",
            "",
            "- Computed G4 F1BZQ metrics from the frozen analytic universe only.",
            "- No Ollama/API calls; no bootstrap/statistical tests; no frozen evidence/pool/protocol/runner/universe edits.",
            f"- Metric result rows: {manifest['summary']['metric_result_rows']}; undefined/blocked rows: {manifest['summary']['undefined_blocked_rows']}.",
            f"- Candidate recall by dataset: {stable_json(manifest['summary']['candidate_recall_by_dataset'])}.",
            f"- Validation passed: {bool_text(manifest['validation_passed'])}.",
            f"- Next safe step: {manifest['next_safe_step']}.",
            "",
        ]
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_lines = existing.splitlines()
    marker = f"## {PHASE_ID} - "
    for index, line in enumerate(existing_lines):
        if line.startswith(marker):
            existing_lines = existing_lines[:index]
            break
    prefix = "\n".join(existing_lines).rstrip()
    output = f"{prefix}\n\n{entry}" if prefix else f"{entry}"
    path.write_text(output, encoding="utf-8", newline="\n")

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_project_root(Path(args.output_root).resolve())
    head_before = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short"], check=False)

    universe = read_csv(root, PATHS["analytic_universe"])
    contract = read_csv(root, PATHS["metric_contract"])
    metric_results, denominators, candidate_conditional, reliability, undefined_blocked, contribution_defects = compute_metrics(
        universe, contract
    )
    validation = build_validation(
        root=root,
        head_before=head_before,
        status_before=status_before,
        universe=universe,
        contract=contract,
        metric_results=metric_results,
        denominator_frame=denominators,
        candidate_conditional=candidate_conditional,
        reliability=reliability,
        undefined_blocked=undefined_blocked,
        contribution_defects=contribution_defects,
        allow_generator_dirty=args.allow_generator_dirty,
    )
    validation_passed = bool((validation["status"] == "FAIL").sum() == 0)

    candidate_recall_by_dataset: dict[str, str] = {}
    recall_rows = candidate_conditional[candidate_conditional["metric_id"] == "candidate_recall_at_20"]
    for dataset in DATASETS:
        values = sorted(set(recall_rows[recall_rows["dataset"] == dataset]["value"].tolist()))
        candidate_recall_by_dataset[dataset] = values[0] if values else ""

    reliability_totals = reliability[
        ["planned_requests", "strict_valid", "structural_invalid", "invalid_length", "invalid_duplicate", "backend_errors"]
    ].astype(int).sum()
    input_hashes = {path.as_posix(): sha256_file(root / path) for path in PATHS.values() if path != PATHS["notes"] and (root / path).exists()}
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "phase_id": PHASE_ID,
        "decision_token": DECISION_FROZEN if validation_passed else DECISION_BLOCKED,
        "created_at_utc": utc_now(),
        "head_before_generation": head_before,
        "precondition_status": "G4_ANALYTIC_UNIVERSE_FROZEN",
        "source": "frozen_g4_analytic_universe_only",
        "zero_inference": True,
        "ollama_or_api_calls_executed": 0,
        "statistical_analysis_executed": False,
        "frozen_evidence_modified": False,
        "f1bzq_pools_modified": False,
        "protocol_config_runner_modified": False,
        "analytic_universe_modified": False,
        "input_hashes": input_hashes,
        "files_created": [path.as_posix() for path in OUTPUTS.values()] + [PATHS["notes"].as_posix()],
        "summary": {
            "logical_rows": int(len(universe)),
            "unique_users": int(universe[["dataset", "user_id"]].drop_duplicates().shape[0]),
            "cells": int(universe[["dataset", "model", "prompt_strategy"]].drop_duplicates().shape[0]),
            "metric_result_rows": int(len(metric_results)),
            "candidate_conditional_rows": int(len(candidate_conditional)),
            "reliability_rows": int(len(reliability)),
            "undefined_blocked_rows": int(len(undefined_blocked)),
            "denominator_rows": int(len(denominators)),
            "validation_passes": int((validation["status"] == "PASS").sum()),
            "validation_failures": int((validation["status"] == "FAIL").sum()),
            "candidate_recall_by_dataset": candidate_recall_by_dataset,
            "strict_valid": int(reliability_totals["strict_valid"]),
            "structural_invalid": int(reliability_totals["structural_invalid"]),
            "invalid_length": int(reliability_totals["invalid_length"]),
            "invalid_duplicate": int(reliability_totals["invalid_duplicate"]),
            "backend_errors": int(reliability_totals["backend_errors"]),
            "computed_metric_ids": sorted(metric_results["metric_id"].unique().tolist()),
            "undefined_or_blocked_metric_ids": sorted(undefined_blocked["metric_id"].unique().tolist()),
        },
        "validation_passed": validation_passed,
        "validation_failures": validation[validation["status"] == "FAIL"].to_dict("records"),
        "next_safe_step": "freeze G4 bootstrap/statistical comparison protocol and run statistics from frozen metric artifacts only",
    }

    report = build_report(
        validation=validation,
        metric_results=metric_results,
        candidate_conditional=candidate_conditional,
        reliability=reliability,
        undefined_blocked=undefined_blocked,
        manifest=manifest,
    )

    if not args.no_write:
        write_csv(root / OUTPUTS["metric_results"], metric_results)
        write_csv(root / OUTPUTS["metric_denominators"], denominators)
        write_csv(root / OUTPUTS["candidate_conditional"], candidate_conditional)
        write_csv(root / OUTPUTS["reliability"], reliability)
        write_csv(root / OUTPUTS["undefined_blocked"], undefined_blocked)
        write_csv(root / OUTPUTS["validation"], validation)
        (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
        (root / OUTPUTS["report"]).write_text(report, encoding="utf-8", newline="\n")
        output_hashes = {path.as_posix(): sha256_file(root / path) for key, path in OUTPUTS.items() if key != "manifest"}
        manifest["output_hashes"] = output_hashes
        write_json(root / OUTPUTS["manifest"], manifest)
        append_notes(root / PATHS["notes"], manifest)

    if not validation_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

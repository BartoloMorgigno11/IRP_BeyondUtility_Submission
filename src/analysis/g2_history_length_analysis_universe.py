"""Build the frozen G2 history-length analytic universe.

This phase consumes retained G1/G2 evidence only. It does not call Ollama,
does not compute metric values, and does not alter the frozen G2 protocol,
execution runner, or raw evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_HEAD = "7d9cb21d21f8c3782bf0a823229417ddc766894a"
PHASE_ID = "g2_history_length_analysis_universe_freeze"
DECISION_FROZEN = "G2_ANALYTIC_UNIVERSE_FROZEN"
DECISION_BLOCKED = "G2_ANALYTIC_UNIVERSE_BLOCKED"

PRIMARY_VIEW = "PRIMARY_B1_GE10"
LONG_VIEW = "LONG_HISTORY_B2_GE20"
SECONDARY_VIEW = "SECONDARY_FULL_CURVE_GE20"
TRUE_ZERO_VIEW = "FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC"
REPORTING_VIEWS = [PRIMARY_VIEW, LONG_VIEW, SECONDARY_VIEW, TRUE_ZERO_VIEW]
HISTORY_BY_VIEW = {
    PRIMARY_VIEW: [0, 1, 2, 5, 10],
    LONG_VIEW: [10, 20],
    SECONDARY_VIEW: [0, 1, 2, 5, 10, 20],
    TRUE_ZERO_VIEW: [0],
}
ALL_HISTORY_LEVELS = [0, 1, 2, 5, 10, 20]

G1_REUSE_ACTION = "REUSE_G1_EXACT_TERMINAL_EVIDENCE"
G2_NEW_ACTION = "NEW_G2_OLLAMA_CALL_REQUIRED"
G1_EVIDENCE = "G1_EXACT_REUSE_TERMINAL_EVIDENCE"
G2_EVIDENCE = "G2_TERMINAL_EVIDENCE"

PATHS = {
    "protocol_manifest": Path("results/manifests/g2_history_length_protocol_freeze.json"),
    "execution_manifest": Path("results/manifests/g2_history_length_execution_freeze.json"),
    "retention_manifest": Path("results/manifests/g2_history_length_execution_retention_manifest.json"),
    "request_universe": Path("results/tables/g2_history_length_request_universe.csv"),
    "execution_order": Path("results/tables/g2_history_length_execution_order.csv"),
    "g1_reuse": Path("results/tables/g2_history_length_g1_execution_reuse.csv"),
    "g1_execution_state": Path("results/tables/g1_primary_ranking_execution_state.csv"),
    "g1_strict_parsing": Path("results/tables/g1_primary_ranking_strict_parsing.csv"),
    "applicability": Path("results/tables/g2_history_length_applicability.csv"),
    "statistical_plan": Path("results/tables/g2_history_length_statistical_plan.csv"),
    "g2_responses": Path("results/llm/extension_canonical/g2_history_length_sensitivity/responses.jsonl"),
    "g2_requests": Path("results/llm/extension_canonical/g2_history_length_sensitivity/requests.jsonl"),
    "g2_response_dir": Path("results/llm/extension_canonical/g2_history_length_sensitivity/responses"),
    "notes": Path("report/notes.md"),
}
OUTPUTS = {
    "analysis_universe": Path("results/tables/g2_history_length_analysis_universe.csv"),
    "membership_universe": Path("results/tables/g2_history_length_analysis_membership_universe.csv"),
    "cell_counts": Path("results/tables/g2_history_length_analysis_cell_counts.csv"),
    "denominator_contract": Path("results/tables/g2_history_length_denominator_contract.csv"),
    "manifest": Path("results/manifests/g2_history_length_analysis_universe_freeze.json"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen G2 analytic-universe artifacts.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Validate and build in memory without writing outputs.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator and expected phase outputs to be dirty while preserving frozen artifact checks.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def read_csv(root: Path, path: Path) -> pd.DataFrame:
    return pd.read_csv(root / path, dtype=str, keep_default_na=False, low_memory=False)


def git_output(root: Path, args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def bool_text(value: Any) -> str:
    return "true" if str(value).strip().lower() == "true" else "false"


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def membership_list(value: str) -> list[str]:
    loaded = json.loads(str(value))
    if not isinstance(loaded, list):
        raise ValueError(f"membership field is not a list: {value}")
    return [str(item) for item in loaded]


def verify_manifest_hashes(
    root: Path, manifest: dict[str, Any], *, skip_gate_paths: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    skipped = skip_gate_paths or set()
    for rel_path, expected in manifest.get("artifact_hashes", {}).items():
        path = root / rel_path
        actual = sha256_file(path) if path.is_file() else ""
        checks[rel_path] = {
            "expected": expected,
            "actual": actual,
            "match": actual == expected,
            "skipped_from_freeze_gate": rel_path in skipped,
        }
    return checks


def load_g2_response_ledger(root: Path) -> pd.DataFrame:
    path = root / PATHS["g2_responses"]
    rows: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    parse_errors = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            request_id = str(record.get("canonical_request_id") or record.get("request_id") or "")
            seen[request_id] += 1
            rows.append(
                {
                    "canonical_request_id": request_id,
                    "evidence_source": G2_EVIDENCE,
                    "source_canonical_request_id": request_id,
                    "terminal_state": str(record.get("terminal_state", "")),
                    "parse_status": str(record.get("parse_status", "")),
                    "strict_parse_valid": bool_text(record.get("strict_parse_valid", False)),
                    "backend_success": bool_text(record.get("backend_success", False)),
                    "backend_status": str(record.get("backend_status", "")),
                    "backend_attempt_count": str(record.get("backend_attempt_count", "")),
                    "retry_count": str(record.get("retry_count", "")),
                    "runtime_seconds": str(record.get("runtime_seconds", "")),
                    "output_length": str(record.get("output_length", "")),
                    "duplicate_count": str(record.get("duplicate_count", "")),
                    "hallucination_count": str(record.get("hallucination_count", "")),
                    "ranked_candidate_indices": str(record.get("ranked_candidate_indices", "")),
                    "ranked_item_ids": str(record.get("ranked_item_ids", "")),
                    "raw_response_hash": str(record.get("raw_response_hash", "")),
                    "raw_record_hash": str(record.get("raw_record_hash", "")),
                    "terminal_record_hash": str(record.get("terminal_record_hash", "")),
                    "request_started_at_utc": str(record.get("request_started_at_utc", "")),
                    "request_completed_at_utc": str(record.get("request_completed_at_utc", "")),
                    "model_digest_observed": str(record.get("model_digest", "")),
                    "generation_config_hash_observed": str(record.get("generation_config_hash", "")),
                    "parser_contract_hash_observed": str(record.get("parser_contract_hash", "")),
                    "history_hash_observed": str(record.get("history_hash", "")),
                    "candidate_hash_observed": str(record.get("candidate_hash", "")),
                    "jsonl_line": line_no,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["parse_errors"] = parse_errors
    frame.attrs["duplicate_ids"] = sum(1 for count in seen.values() if count > 1)
    return frame


def count_request_ledger(root: Path) -> dict[str, Any]:
    path = root / PATHS["g2_requests"]
    counts: Counter[str] = Counter()
    line_map: dict[str, list[int]] = {}
    payload_hashes: dict[str, set[str]] = {}
    parse_errors = 0
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            request_id = str(record.get("canonical_request_id") or record.get("request_id") or "")
            counts[request_id] += 1
            line_map.setdefault(request_id, []).append(line_no)
            payload_hashes.setdefault(request_id, set()).add(str(record.get("payload_hash", "")))
    duplicate_details = [
        {
            "canonical_request_id": request_id,
            "request_ledger_rows": count,
            "request_ledger_lines": line_map.get(request_id, []),
            "payload_hash_unique_count": len(payload_hashes.get(request_id, set())),
        }
        for request_id, count in sorted(counts.items())
        if count > 1
    ]
    return {
        "rows": rows,
        "unique_ids": len(counts),
        "parse_errors": parse_errors,
        "duplicate_ids": len(duplicate_details),
        "duplicate_extra_rows": sum(count - 1 for count in counts.values() if count > 1),
        "duplicate_details": duplicate_details,
    }


def build_g1_evidence(g1_reuse: pd.DataFrame, g1_parse: pd.DataFrame, g1_execution: pd.DataFrame) -> pd.DataFrame:
    parse_cols = [
        "canonical_request_id",
        "parse_status",
        "strict_parse_valid",
        "output_length",
        "ranked_candidate_indices",
        "ranked_item_ids",
        "duplicate_count",
        "hallucination_count",
        "parse_error",
    ]
    execution_cols = [
        "canonical_request_id",
        "backend_success",
        "backend_status",
        "backend_attempt_count",
        "retry_count",
        "runtime_seconds",
        "request_started_at_utc",
        "request_completed_at_utc",
    ]
    merged = g1_reuse.merge(
        g1_parse[parse_cols].rename(columns={"canonical_request_id": "g1_canonical_request_id"}),
        on="g1_canonical_request_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        g1_execution[execution_cols].rename(columns={"canonical_request_id": "g1_canonical_request_id"}),
        on="g1_canonical_request_id",
        how="left",
        validate="one_to_one",
    )
    out = pd.DataFrame(
        {
            "canonical_request_id": merged["g2_canonical_request_id"],
            "evidence_source": G1_EVIDENCE,
            "source_canonical_request_id": merged["g1_canonical_request_id"],
            "source_raw_response_path": merged["g1_raw_response_path"],
            "terminal_state": merged["g1_terminal_state"],
            "parse_status": merged["parse_status"],
            "strict_parse_valid": merged["strict_parse_valid"].map(bool_text),
            "backend_success": merged["backend_success"].map(bool_text),
            "backend_status": merged["backend_status"],
            "backend_attempt_count": merged["backend_attempt_count"],
            "retry_count": merged["retry_count"],
            "runtime_seconds": merged["runtime_seconds"],
            "output_length": merged["output_length"],
            "duplicate_count": merged["duplicate_count"],
            "hallucination_count": merged["hallucination_count"],
            "ranked_candidate_indices": merged["ranked_candidate_indices"],
            "ranked_item_ids": merged["ranked_item_ids"],
            "raw_response_hash": merged["g1_raw_response_hash"],
            "raw_record_hash": "",
            "terminal_record_hash": "",
            "request_started_at_utc": merged["request_started_at_utc"],
            "request_completed_at_utc": merged["request_completed_at_utc"],
            "model_digest_observed": merged["model_digest"],
            "generation_config_hash_observed": merged["generation_config_hash"],
            "parser_contract_hash_observed": merged["parser_contract_hash"],
            "history_hash_observed": merged["history_hash"],
            "candidate_hash_observed": merged["candidate_hash"],
            "jsonl_line": "",
        }
    )
    return out.fillna("")


def invalid_class(row: pd.Series) -> str:
    if str(row.get("terminal_state", "")) == "STRICT_VALID" and str(row.get("strict_parse_valid", "")).lower() == "true":
        return "NONE"
    terminal_state = str(row.get("terminal_state", ""))
    return terminal_state if terminal_state else "MISSING_EVIDENCE"


def status_is_clean_or_expected_phase_paths(status: str, allow_phase_dirty: bool) -> bool:
    if not status:
        return True
    if not allow_phase_dirty:
        return False
    allowed_paths = {
        "report/notes.md",
        "results/manifests/g2_history_length_analysis_universe_freeze.json",
        "results/tables/g2_history_length_analysis_cell_counts.csv",
        "results/tables/g2_history_length_analysis_membership_universe.csv",
        "results/tables/g2_history_length_analysis_universe.csv",
        "results/tables/g2_history_length_denominator_contract.csv",
        "src/analysis/g2_history_length_analysis_universe.py",
    }
    normalized = {line[3:].replace("\\", "/") for line in status.splitlines() if len(line) >= 4}
    return normalized.issubset(allowed_paths)


def input_artifact_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in PATHS.items():
        if name in {"g2_response_dir", "notes"}:
            continue
        full_path = root / path
        if full_path.is_file():
            hashes[path.as_posix()] = sha256_file(full_path)
    return hashes


def build_denominator_contract() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for view in REPORTING_VIEWS:
        rows.append(
            {
                "analysis_membership": view,
                "metric_family": "utility",
                "metrics": "hr_at_5;ndcg_at_5;mrr",
                "history_lengths": stable_json(HISTORY_BY_VIEW[view]),
                "paired_unit": "user_id",
                "denominator_basis": "planned_applicable_users",
                "invalid_response_policy": "count_as_zero_utility_when_terminal_state_is_not_STRICT_VALID",
                "backend_failure_policy": "count_as_zero_utility_if_present_after_backend_retry_exhaustion",
                "metric_specific_subset_policy": "not_applicable_for_utility_phase1",
                "not_applicable_policy": "no logical request and no imputation for frozen not-applicable cells",
                "source_contract": "g2_history_length_statistical_plan.fail_strict_all_planned_users_never_collapse_b1_b2_denominators",
            }
        )
        rows.append(
            {
                "analysis_membership": view,
                "metric_family": "beyond_utility",
                "metrics": "strict_valid_output_consumers_only",
                "history_lengths": stable_json(HISTORY_BY_VIEW[view]),
                "paired_unit": "user_id",
                "denominator_basis": "strict_valid_users",
                "invalid_response_policy": "excluded_from_beyond_utility_denominator_except_invalidity_reliability_metrics",
                "backend_failure_policy": "excluded_from_strict_valid_denominator_if_present",
                "metric_specific_subset_policy": "metric artifacts must declare additional defined-user subsets before aggregation",
                "not_applicable_policy": "no logical request and no imputation for frozen not-applicable cells",
                "source_contract": "configs/g2_history_length_sensitivity.yaml beyond_utility_policy reuse_frozen_g1_definitions_where_applicable",
            }
        )
    rows.append(
        {
            "analysis_membership": "ALL",
            "metric_family": "reliability",
            "metrics": "backend_success_rate;strict_parse_valid_rate;invalid_length_rate;duplicate_index_rate;invalid_candidate_reference_rate",
            "history_lengths": stable_json(ALL_HISTORY_LEVELS),
            "paired_unit": "canonical_request_id",
            "denominator_basis": "planned_applicable_requests",
            "invalid_response_policy": "retained_as_terminal_scientific_evidence",
            "backend_failure_policy": "retained_as_terminal_backend_failure_if_present",
            "metric_specific_subset_policy": "not_applicable_for_reliability_counts",
            "not_applicable_policy": "not-applicable cells excluded from request denominator and reported separately",
            "source_contract": "configs/g2_history_length_sensitivity.yaml reliability metrics",
        }
    )
    return pd.DataFrame(rows)


def build_membership_universe(universe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in universe.to_dict("records"):
        for membership in membership_list(str(row["reporting_view_memberships"])):
            rows.append(
                {
                    "analysis_membership": membership,
                    "canonical_request_id": row["canonical_request_id"],
                    "dataset": row["dataset"],
                    "user_id": row["user_id"],
                    "model": row["model"],
                    "prompt_strategy": row["prompt_strategy"],
                    "history_length": int(row["history_length"]),
                    "history_condition_label": row["history_condition_label"],
                    "evidence_source": row["evidence_source"],
                    "terminal_state": row["terminal_state"],
                    "strict_parse_valid": row["strict_parse_valid"],
                    "invalid_class": row["invalid_class"],
                    "utility_denominator_included": "true",
                    "beyond_utility_denominator_included": row["beyond_utility_denominator_included"],
                }
            )
    return pd.DataFrame(rows)


def build_cell_counts(applicability: pd.DataFrame, membership: pd.DataFrame, protocol_manifest: dict[str, Any]) -> pd.DataFrame:
    working = membership.copy()
    working["is_g1_reuse"] = working["evidence_source"].eq(G1_EVIDENCE)
    working["is_g2_terminal"] = working["evidence_source"].eq(G2_EVIDENCE)
    working["is_strict_valid"] = bool_series(working["strict_parse_valid"])
    working["is_invalid"] = ~working["is_strict_valid"]
    working["is_backend_error"] = working["terminal_state"].eq("BACKEND_FAILURE")
    for state in ["INVALID_LENGTH", "INVALID_DUPLICATE", "HALLUCINATION_OR_OUT_OF_CANDIDATE", "INVALID_FORMAT", "INVALID_CANDIDATE"]:
        working[f"is_{state.lower()}"] = working["terminal_state"].eq(state)

    observed = (
        working.groupby(["analysis_membership", "dataset", "prompt_strategy", "model", "history_length"], sort=False)
        .agg(
            logical_request_rows=("canonical_request_id", "size"),
            unique_users=("user_id", "nunique"),
            g1_reuse_rows=("is_g1_reuse", "sum"),
            g2_terminal_rows=("is_g2_terminal", "sum"),
            strict_valid_rows=("is_strict_valid", "sum"),
            invalid_rows=("is_invalid", "sum"),
            backend_error_rows=("is_backend_error", "sum"),
            invalid_length_rows=("is_invalid_length", "sum"),
            invalid_duplicate_rows=("is_invalid_duplicate", "sum"),
            hallucination_or_out_of_candidate_rows=("is_hallucination_or_out_of_candidate", "sum"),
            invalid_format_rows=("is_invalid_format", "sum"),
            invalid_candidate_rows=("is_invalid_candidate", "sum"),
        )
        .reset_index()
        .rename(columns={"analysis_membership": "reporting_view"})
    )
    observed["history_length"] = observed["history_length"].astype(str)
    cell_counts = applicability.merge(
        observed,
        on=["reporting_view", "dataset", "prompt_strategy", "model", "history_length"],
        how="left",
        validate="one_to_one",
    )
    count_cols = [
        "logical_request_rows",
        "unique_users",
        "g1_reuse_rows",
        "g2_terminal_rows",
        "strict_valid_rows",
        "invalid_rows",
        "backend_error_rows",
        "invalid_length_rows",
        "invalid_duplicate_rows",
        "hallucination_or_out_of_candidate_rows",
        "invalid_format_rows",
        "invalid_candidate_rows",
    ]
    for col in count_cols:
        cell_counts[col] = pd.to_numeric(cell_counts[col], errors="coerce").fillna(0).astype(int)
    cell_counts["planned_applicable_users"] = pd.to_numeric(
        cell_counts["applicable_user_conditions"], errors="coerce"
    ).fillna(0).astype(int)
    cell_counts["utility_denominator_users"] = cell_counts["planned_applicable_users"]
    cell_counts["beyond_utility_strict_valid_users"] = cell_counts["strict_valid_rows"]
    cell_counts["missing_evidence_rows"] = cell_counts["planned_applicable_users"] - cell_counts["logical_request_rows"]
    cell_counts["ambiguous_evidence_rows"] = 0
    cell_counts["duplicate_terminal_evidence_rows"] = 0

    expected_users: dict[tuple[str, str], int] = {}
    for dataset, users in protocol_manifest["b1_primary_paired_users_by_dataset"].items():
        expected_users[(PRIMARY_VIEW, str(dataset))] = int(users)
    for dataset, users in protocol_manifest["b2_long_history_paired_users_by_dataset"].items():
        expected_users[(LONG_VIEW, str(dataset))] = int(users)
        expected_users[(SECONDARY_VIEW, str(dataset))] = int(users)
    expected_users[(TRUE_ZERO_VIEW, "foodcom")] = int(protocol_manifest["foodcom_true_zero_history_users"])
    cell_counts["expected_paired_users_for_view_dataset"] = cell_counts.apply(
        lambda row: expected_users.get((str(row["reporting_view"]), str(row["dataset"])), 0),
        axis=1,
    )
    cell_counts["cell_status"] = cell_counts.apply(
        lambda row: "NOT_APPLICABLE"
        if int(row["planned_applicable_users"]) == 0
        else ("COMPLETE" if int(row["missing_evidence_rows"]) == 0 else "BLOCKED_MISSING_EVIDENCE"),
        axis=1,
    )
    cell_counts["_history_sort"] = pd.to_numeric(cell_counts["history_length"], errors="coerce")
    return cell_counts.sort_values(
        ["reporting_view", "dataset", "prompt_strategy", "model", "_history_sort"], kind="mergesort"
    ).drop(columns=["_history_sort"])


def tuple_counts_to_nested(frame: pd.DataFrame, keys: list[str]) -> dict[str, int]:
    grouped = frame.groupby(keys, sort=False).size().astype(int)
    return {"|".join(str(part) for part in key if str(part)): int(value) for key, value in grouped.items()}


def build_artifacts(root: Path, *, allow_generator_dirty: bool) -> dict[str, Any]:
    failures: list[str] = []
    head_before = git_output(root, ["rev-parse", "HEAD"])
    status_result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"], cwd=root, text=True, capture_output=True
    )
    if status_result.returncode != 0:
        raise RuntimeError(status_result.stderr.strip() or status_result.stdout.strip())
    status_before = status_result.stdout.rstrip()
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True)
    if head_before != EXPECTED_HEAD:
        failures.append(f"HEAD_MISMATCH:{head_before}")
    if not status_is_clean_or_expected_phase_paths(status_before, allow_generator_dirty):
        failures.append("GIT_STATUS_NOT_CLEAN_BEFORE_ANALYTIC_UNIVERSE")
    if diff_check.returncode != 0:
        failures.append("GIT_DIFF_CHECK_FAILED_BEFORE_ANALYTIC_UNIVERSE")

    protocol_manifest = read_json(root, PATHS["protocol_manifest"])
    execution_manifest = read_json(root, PATHS["execution_manifest"])
    retention_manifest = read_json(root, PATHS["retention_manifest"])
    if protocol_manifest.get("decision_token") != "G2_NESTED_HISTORY_PROTOCOL_FROZEN":
        failures.append("PROTOCOL_DECISION_NOT_FROZEN")
    if execution_manifest.get("decision_token") != "G2_EXECUTION_RUNNER_FROZEN":
        failures.append("EXECUTION_RUNNER_NOT_FROZEN")
    if retention_manifest.get("decision_token") != "G2_EXECUTION_RETENTION_FROZEN":
        failures.append("RETENTION_NOT_FROZEN")
    if not retention_manifest.get("validation_passed", False):
        failures.append("RETENTION_MANIFEST_NOT_VALIDATED")

    protocol_hash_checks = verify_manifest_hashes(root, protocol_manifest)
    execution_hash_checks = verify_manifest_hashes(root, execution_manifest)
    retention_hash_checks_before_notes = verify_manifest_hashes(
        root, retention_manifest, skip_gate_paths={"report/notes.md"}
    )
    retention_hash_gate_passed = all(
        row["match"] or row["skipped_from_freeze_gate"] for row in retention_hash_checks_before_notes.values()
    )
    if not all(row["match"] for row in protocol_hash_checks.values()):
        failures.append("PROTOCOL_FROZEN_ARTIFACT_HASH_MISMATCH")
    if not all(row["match"] for row in execution_hash_checks.values()):
        failures.append("EXECUTION_FROZEN_ARTIFACT_HASH_MISMATCH")
    if not retention_hash_gate_passed:
        failures.append("RETENTION_ARTIFACT_HASH_MISMATCH_BEFORE_NOTES_UPDATE")

    archive_path = root / retention_manifest["external_archive"]["path"]
    archive_sha = sha256_file(archive_path) if archive_path.is_file() else ""
    archive_entries = 0
    archive_testzip = "ARCHIVE_MISSING"
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            archive_testzip = archive.testzip()
            archive_entries = len(archive.infolist())
    if archive_sha != retention_manifest["external_archive"]["sha256"]:
        failures.append("G2_EXTERNAL_ARCHIVE_HASH_MISMATCH")
    if archive_testzip is not None:
        failures.append(f"G2_EXTERNAL_ARCHIVE_CORRUPT:{archive_testzip}")
    if archive_entries != int(retention_manifest["external_archive"]["entry_count"]):
        failures.append("G2_EXTERNAL_ARCHIVE_ENTRY_COUNT_MISMATCH")

    request_universe = read_csv(root, PATHS["request_universe"])
    execution_order = read_csv(root, PATHS["execution_order"])
    g1_reuse = read_csv(root, PATHS["g1_reuse"])
    g1_execution = read_csv(root, PATHS["g1_execution_state"])
    g1_parse = read_csv(root, PATHS["g1_strict_parsing"])
    applicability = read_csv(root, PATHS["applicability"])
    statistical_plan = read_csv(root, PATHS["statistical_plan"])

    if len(request_universe) != 88050 or request_universe["canonical_request_id"].nunique() != 88050:
        failures.append("REQUEST_UNIVERSE_COUNT_OR_UNIQUENESS_FAILED")
    if len(execution_order) != 88050 or execution_order["canonical_request_id"].nunique() != 88050:
        failures.append("EXECUTION_ORDER_COUNT_OR_UNIQUENESS_FAILED")
    if set(request_universe["canonical_request_id"]) != set(execution_order["canonical_request_id"]):
        failures.append("REQUEST_UNIVERSE_EXECUTION_ORDER_ID_SET_MISMATCH")
    if len(g1_reuse) != 18660 or g1_reuse["g2_canonical_request_id"].nunique() != 18660:
        failures.append("G1_REUSE_COUNT_OR_UNIQUENESS_FAILED")
    expected_policy = "fail_strict_all_planned_users_never_collapse_b1_b2_denominators"
    if not statistical_plan["denominator_policy"].eq(expected_policy).all():
        failures.append("STATISTICAL_PLAN_DENOMINATOR_POLICY_CHANGED")

    g2_terminal = load_g2_response_ledger(root)
    request_ledger = count_request_ledger(root)
    raw_files = list((root / PATHS["g2_response_dir"]).glob("*.json"))
    raw_ids = {path.stem for path in raw_files}
    if g2_terminal.attrs.get("parse_errors"):
        failures.append("G2_RESPONSE_JSONL_PARSE_ERRORS")
    if len(g2_terminal) != 69390 or g2_terminal["canonical_request_id"].nunique() != 69390:
        failures.append("G2_TERMINAL_COUNT_OR_UNIQUENESS_FAILED")
    if int(g2_terminal.attrs.get("duplicate_ids", 0)) != 0:
        failures.append("G2_TERMINAL_DUPLICATE_IDS")
    if len(raw_files) != 69390 or len(raw_ids) != 69390:
        failures.append("G2_RAW_FILE_COUNT_OR_UNIQUENESS_FAILED")
    if request_ledger["rows"] != 69392 or request_ledger["unique_ids"] != 69390 or request_ledger["duplicate_ids"] != 2:
        failures.append("G2_REQUEST_LEDGER_PROVENANCE_COUNT_MISMATCH")
    if request_ledger["parse_errors"] != 0:
        failures.append("G2_REQUEST_LEDGER_PARSE_ERRORS")

    planned_new_ids = set(request_universe.loc[request_universe["planned_action"].eq(G2_NEW_ACTION), "canonical_request_id"])
    planned_reuse_ids = set(
        request_universe.loc[request_universe["planned_action"].eq(G1_REUSE_ACTION), "canonical_request_id"]
    )
    g2_ids = set(g2_terminal["canonical_request_id"])
    g1_ids = set(g1_reuse["g2_canonical_request_id"])
    if planned_new_ids != g2_ids:
        failures.append("PLANNED_NEW_NOT_EQUAL_G2_TERMINAL_IDS")
    if planned_reuse_ids != g1_ids:
        failures.append("PLANNED_REUSE_NOT_EQUAL_G1_REUSE_IDS")
    if planned_new_ids & planned_reuse_ids or g2_ids & g1_ids:
        failures.append("AMBIGUOUS_G1_G2_EVIDENCE_OVERLAP")
    if not g2_ids.issubset(raw_ids):
        failures.append("G2_TERMINAL_WITHOUT_RAW_FILE")

    g1_evidence = build_g1_evidence(g1_reuse, g1_parse, g1_execution)
    g2_evidence = g2_terminal.copy()
    g2_evidence["source_raw_response_path"] = g2_evidence["canonical_request_id"].map(
        request_universe.set_index("canonical_request_id")["g2_future_raw_response_path"]
    )
    evidence = pd.concat([g1_evidence, g2_evidence], ignore_index=True, sort=False).fillna("")
    if len(evidence) != 88050 or evidence["canonical_request_id"].nunique() != 88050:
        failures.append("RESOLVED_EVIDENCE_COUNT_OR_UNIQUENESS_FAILED")

    base_cols = [
        "canonical_request_id",
        "request_order",
        "dataset",
        "user_id",
        "model",
        "prompt_strategy",
        "reporting_view_memberships",
        "primary_reporting_view",
        "history_condition_label",
        "history_length",
        "full_history_length",
        "positive_item_id",
        "metric_k",
        "llmrank_h0_status",
        "request_identity_hash",
        "planned_action",
    ]
    universe = request_universe[base_cols].merge(evidence, on="canonical_request_id", how="left", validate="one_to_one")
    universe["strict_parse_valid"] = universe["strict_parse_valid"].map(bool_text)
    universe["invalid_class"] = universe.apply(invalid_class, axis=1)
    universe["utility_denominator_included"] = "true"
    universe["utility_invalid_response_policy"] = "ZERO_UTILITY_IF_NOT_STRICT_VALID"
    universe["beyond_utility_denominator_included"] = universe["strict_parse_valid"]
    universe["beyond_utility_denominator_policy"] = "STRICT_VALID_ONLY_WITH_METRIC_SPECIFIC_DEFINED_SUBSETS"
    universe["missing_evidence"] = universe["evidence_source"].eq("").map(lambda value: "true" if value else "false")
    universe["ambiguous_evidence"] = "false"
    universe["duplicate_terminal_evidence"] = "false"
    universe["analysis_ready"] = universe["missing_evidence"].eq("false").map(lambda value: "true" if value else "false")
    for col in ["history_length", "request_order", "full_history_length", "metric_k"]:
        universe[col] = pd.to_numeric(universe[col], errors="coerce").astype("Int64")

    g2_rows = universe.loc[universe["evidence_source"].eq(G2_EVIDENCE)]
    g2_identity = g2_terminal.merge(
        request_universe[
            [
                "canonical_request_id",
                "model_digest",
                "generation_config_hash",
                "parser_contract_hash",
                "history_hash",
                "candidate_hash",
            ]
        ],
        on="canonical_request_id",
        how="left",
        validate="one_to_one",
    )
    for observed, expected in [
        ("model_digest_observed", "model_digest"),
        ("generation_config_hash_observed", "generation_config_hash"),
        ("parser_contract_hash_observed", "parser_contract_hash"),
        ("history_hash_observed", "history_hash"),
        ("candidate_hash_observed", "candidate_hash"),
    ]:
        if not g2_identity[observed].eq(g2_identity[expected]).all():
            failures.append(f"G2_IDENTITY_FIELD_MISMATCH:{observed}")

    if universe["missing_evidence"].ne("false").any():
        failures.append("MISSING_RESOLVED_EVIDENCE")
    if not universe["analysis_ready"].eq("true").all():
        failures.append("ANALYSIS_READY_FALSE_ROWS")

    membership = build_membership_universe(universe)
    levels = {
        view: sorted(
            membership.loc[membership["analysis_membership"].eq(view), "history_length"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )
        for view in REPORTING_VIEWS
    }
    for view, expected_levels in HISTORY_BY_VIEW.items():
        if levels.get(view, []) != expected_levels:
            failures.append(f"HISTORY_LEVELS_MISMATCH:{view}:{levels.get(view, [])}")

    cell_counts = build_cell_counts(applicability, membership, protocol_manifest)
    applicable_bad = cell_counts.loc[
        cell_counts["planned_applicable_users"].gt(0)
        & (
            cell_counts["planned_applicable_users"].ne(cell_counts["logical_request_rows"])
            | cell_counts["unique_users"].ne(cell_counts["expected_paired_users_for_view_dataset"])
            | cell_counts["missing_evidence_rows"].ne(0)
        )
    ]
    if not applicable_bad.empty:
        failures.append(f"APPLICABLE_CELL_DENOMINATOR_MISMATCH:{len(applicable_bad)}")
    true_zero_bad = cell_counts.loc[
        cell_counts["reporting_view"].eq(TRUE_ZERO_VIEW)
        & cell_counts["history_length"].astype(str).isin(["1", "2", "5", "10", "20"])
        & (cell_counts["planned_applicable_users"].ne(0) | cell_counts["logical_request_rows"].ne(0))
    ]
    if not true_zero_bad.empty:
        failures.append("FOODCOM_TRUE_ZERO_H_POSITIVE_NOT_APPLICABLE_MISMATCH")

    g2_backend_accounting = {
        "backend_calls": int(len(g2_rows)),
        "strict_valid": int(g2_rows["terminal_state"].eq("STRICT_VALID").sum()),
        "invalid": int(g2_rows["terminal_state"].ne("STRICT_VALID").sum()),
        "backend_errors": int(g2_rows["terminal_state"].eq("BACKEND_FAILURE").sum()),
    }
    canonical_counts = {
        "logical_requests": int(len(universe)),
        "g1_exact_reuse": int(universe["evidence_source"].eq(G1_EVIDENCE).sum()),
        "g2_terminal_evidence": int(universe["evidence_source"].eq(G2_EVIDENCE).sum()),
        "strict_valid": int(bool_series(universe["strict_parse_valid"]).sum()),
        "invalid": int((~bool_series(universe["strict_parse_valid"])).sum()),
        "backend_errors": int(universe["terminal_state"].eq("BACKEND_FAILURE").sum()),
        "missing": int(universe["missing_evidence"].eq("true").sum()),
        "duplicate_terminal": 0,
        "ambiguous": 0,
    }
    retention_accounting = retention_manifest["accounting"]
    if canonical_counts["g1_exact_reuse"] != int(retention_accounting["g1_exact_reuse"]):
        failures.append("G1_REUSE_ACCOUNTING_MISMATCH")
    if g2_backend_accounting["backend_calls"] != int(retention_accounting["g2_backend_calls"]):
        failures.append("G2_BACKEND_CALL_ACCOUNTING_MISMATCH")
    if g2_backend_accounting["strict_valid"] != int(retention_accounting["strict_valid"]):
        failures.append("G2_STRICT_VALID_ACCOUNTING_MISMATCH")
    if g2_backend_accounting["invalid"] != int(retention_accounting["invalid_terminal_scientific_evidence"]):
        failures.append("G2_INVALID_ACCOUNTING_MISMATCH")

    membership_work = membership.copy()
    membership_work["is_g1_reuse"] = membership_work["evidence_source"].eq(G1_EVIDENCE)
    membership_work["is_g2_terminal"] = membership_work["evidence_source"].eq(G2_EVIDENCE)
    membership_work["is_strict_valid"] = bool_series(membership_work["strict_parse_valid"])
    membership_work["is_invalid"] = ~membership_work["is_strict_valid"]
    membership_summary = []
    for view in REPORTING_VIEWS:
        view_rows = membership_work.loc[membership_work["analysis_membership"].eq(view)]
        membership_summary.append(
            {
                "analysis_membership": view,
                "logical_membership_rows": int(len(view_rows)),
                "unique_canonical_request_ids": int(view_rows["canonical_request_id"].nunique()),
                "cells_total_including_not_applicable": int(cell_counts["reporting_view"].eq(view).sum()),
                "applicable_cells": int(
                    (cell_counts["reporting_view"].eq(view) & cell_counts["planned_applicable_users"].gt(0)).sum()
                ),
                "history_lengths_available": stable_json(levels.get(view, [])),
                "g1_exact_reuse_membership_rows": int(view_rows["is_g1_reuse"].sum()),
                "g2_terminal_membership_rows": int(view_rows["is_g2_terminal"].sum()),
                "strict_valid_membership_rows": int(view_rows["is_strict_valid"].sum()),
                "invalid_membership_rows": int(view_rows["is_invalid"].sum()),
            }
        )
    membership_summary_df = pd.DataFrame(membership_summary)
    denominator = build_denominator_contract()

    universe_cols = base_cols + [
        "evidence_source",
        "source_canonical_request_id",
        "source_raw_response_path",
        "terminal_state",
        "parse_status",
        "strict_parse_valid",
        "invalid_class",
        "backend_success",
        "backend_status",
        "output_length",
        "duplicate_count",
        "hallucination_count",
        "raw_response_hash",
        "utility_denominator_included",
        "utility_invalid_response_policy",
        "beyond_utility_denominator_included",
        "beyond_utility_denominator_policy",
        "missing_evidence",
        "ambiguous_evidence",
        "duplicate_terminal_evidence",
        "analysis_ready",
    ]
    universe = universe[universe_cols].sort_values("request_order", kind="mergesort")
    membership = membership.sort_values(
        ["analysis_membership", "dataset", "prompt_strategy", "model", "history_length", "canonical_request_id"],
        kind="mergesort",
    )

    validation_checks = {
        "head_before_matches_precondition": head_before == EXPECTED_HEAD,
        "git_clean_or_expected_phase_paths_before_generation": status_is_clean_or_expected_phase_paths(
            status_before, allow_generator_dirty
        ),
        "git_diff_check_clean_before_generation": diff_check.returncode == 0,
        "protocol_manifest_frozen": protocol_manifest.get("decision_token") == "G2_NESTED_HISTORY_PROTOCOL_FROZEN",
        "execution_manifest_frozen": execution_manifest.get("decision_token") == "G2_EXECUTION_RUNNER_FROZEN",
        "retention_manifest_frozen": retention_manifest.get("decision_token") == "G2_EXECUTION_RETENTION_FROZEN",
        "protocol_artifact_hashes_match": all(row["match"] for row in protocol_hash_checks.values()),
        "execution_artifact_hashes_match": all(row["match"] for row in execution_hash_checks.values()),
        "retention_artifact_hashes_match_or_mutable_notes_skipped": retention_hash_gate_passed,
        "external_archive_hash_match": archive_sha == retention_manifest["external_archive"]["sha256"],
        "external_archive_zip_test_passed": archive_testzip is None,
        "canonical_logical_rows_match": canonical_counts["logical_requests"] == 88050,
        "g1_reuse_rows_match": canonical_counts["g1_exact_reuse"] == 18660,
        "g2_terminal_rows_match": canonical_counts["g2_terminal_evidence"] == 69390,
        "strict_valid_total_match": canonical_counts["strict_valid"] == 86768,
        "invalid_total_match": canonical_counts["invalid"] == 1282,
        "backend_errors_zero": canonical_counts["backend_errors"] == 0,
        "missing_evidence_zero": canonical_counts["missing"] == 0,
        "duplicates_zero_except_preserved_presend_requests": request_ledger["duplicate_ids"] == 2
        and g2_terminal.attrs.get("duplicate_ids") == 0,
        "ambiguous_zero": canonical_counts["ambiguous"] == 0,
        "all_applicable_cells_complete": bool(
            cell_counts.loc[cell_counts["planned_applicable_users"].gt(0), "cell_status"].eq("COMPLETE").all()
        ),
        "foodcom_true_zero_h_positive_not_applicable": true_zero_bad.empty,
        "no_metric_values_computed": True,
        "no_ollama_calls_executed": True,
        "g3_not_started": True,
    }
    for name, passed in validation_checks.items():
        if not passed:
            failures.append(f"VALIDATION_CHECK_FALSE:{name}")

    return {
        "decision_token": DECISION_FROZEN if not failures else DECISION_BLOCKED,
        "failures": failures,
        "head_before": head_before,
        "status_before": status_before,
        "tables": {
            "analysis_universe": universe,
            "membership_universe": membership,
            "cell_counts": cell_counts,
            "denominator_contract": denominator,
        },
        "manifest_parts": {
            "protocol_hash_checks": protocol_hash_checks,
            "execution_hash_checks": execution_hash_checks,
            "retention_hash_checks_before_notes": retention_hash_checks_before_notes,
            "archive_sha": archive_sha,
            "archive_entries": archive_entries,
            "archive_testzip": archive_testzip,
            "canonical_counts": canonical_counts,
            "g2_backend_accounting": g2_backend_accounting,
            "request_ledger": request_ledger,
            "raw_file_count": len(raw_files),
            "raw_unique_ids": len(raw_ids),
            "raw_file_set_covers_g2_terminal_ids": g2_ids.issubset(raw_ids),
            "levels": levels,
            "membership_summary": membership_summary_df.to_dict("records"),
            "validation_checks": validation_checks,
            "evidence_source_counts": universe["evidence_source"].value_counts().sort_index().astype(int).to_dict(),
            "terminal_state_counts": universe["terminal_state"].value_counts().sort_index().astype(int).to_dict(),
            "terminal_state_counts_by_source": tuple_counts_to_nested(universe, ["evidence_source", "terminal_state"]),
            "dataset_counts": universe["dataset"].value_counts().sort_index().astype(int).to_dict(),
            "model_counts": universe["model"].value_counts().sort_index().astype(int).to_dict(),
            "prompt_counts": universe["prompt_strategy"].value_counts().sort_index().astype(int).to_dict(),
            "history_length_counts": universe["history_length"].astype(int).value_counts().sort_index().astype(int).to_dict(),
            "input_artifact_hashes": input_artifact_hashes(root),
            "generator_sha256": sha256_file(root / Path("src/analysis/g2_history_length_analysis_universe.py")),
        },
        "retention_manifest": retention_manifest,
    }


def write_outputs(root: Path, artifacts: dict[str, Any]) -> dict[str, str]:
    tables = artifacts["tables"]
    for path in OUTPUTS.values():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
    tables["analysis_universe"].to_csv(root / OUTPUTS["analysis_universe"], index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    tables["membership_universe"].to_csv(root / OUTPUTS["membership_universe"], index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    tables["cell_counts"].to_csv(root / OUTPUTS["cell_counts"], index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    tables["denominator_contract"].to_csv(root / OUTPUTS["denominator_contract"], index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)

    generated_at = utc_now()
    notes = root / PATHS["notes"]
    existing_notes = notes.read_text(encoding="utf-8") if notes.exists() else ""
    if PHASE_ID not in existing_notes:
        note = (
            f"\n## {generated_at} - G2 analytic universe freeze\n\n"
            "- Reconstructed `88050` frozen logical G2 requests from retained G1/G2 evidence without Ollama calls or G3.\n"
            "- Canonical split: G1 exact reuse `18660`; G2 terminal evidence `69390`; strict-valid total `86768`; invalid total `1282`; backend errors `0`.\n"
            "- Denominator contract frozen for phase 1: utility uses planned/applicable users with invalid terminal responses counted as zero utility; beyond-utility uses strict-valid users plus metric-specific defined subsets.\n"
            f"- Outputs: `{OUTPUTS['analysis_universe'].as_posix()}`, `{OUTPUTS['membership_universe'].as_posix()}`, `{OUTPUTS['cell_counts'].as_posix()}`, `{OUTPUTS['denominator_contract'].as_posix()}`, `{OUTPUTS['manifest'].as_posix()}`.\n"
            "- Next safe step: compute G2 metrics from the frozen analytic universe only; do not start G3.\n"
            f"\n{PHASE_ID}\n"
        )
        notes.write_text(existing_notes + note, encoding="utf-8", newline="\n")

    output_hashes = {
        path.as_posix(): sha256_file(root / path)
        for name, path in OUTPUTS.items()
        if name != "manifest"
    }
    output_hashes[PATHS["notes"].as_posix()] = sha256_file(root / PATHS["notes"])
    manifest_parts = artifacts["manifest_parts"]
    manifest = {
        "schema_version": "g2_history_length_analysis_universe_freeze_v1",
        "phase_id": PHASE_ID,
        "decision_token": artifacts["decision_token"],
        "generated_at_utc": generated_at,
        "head_before": artifacts["head_before"],
        "precondition": {
            "required_head": EXPECTED_HEAD,
            "required_prior_token": "G2_EXECUTION_RETENTION_FROZEN",
            "status": "satisfied",
        },
        "analysis_memberships": REPORTING_VIEWS,
        "history_lengths_available_by_membership": manifest_parts["levels"],
        "input_artifact_hashes": manifest_parts["input_artifact_hashes"],
        "generator": {
            "path": "src/analysis/g2_history_length_analysis_universe.py",
            "sha256": manifest_parts["generator_sha256"],
        },
        "canonical_accounting": manifest_parts["canonical_counts"],
        "g2_backend_terminal_accounting": manifest_parts["g2_backend_accounting"],
        "evidence_source_counts": manifest_parts["evidence_source_counts"],
        "terminal_state_counts": manifest_parts["terminal_state_counts"],
        "terminal_state_counts_by_source": manifest_parts["terminal_state_counts_by_source"],
        "dataset_counts": manifest_parts["dataset_counts"],
        "model_counts": manifest_parts["model_counts"],
        "prompt_counts": manifest_parts["prompt_counts"],
        "history_length_counts": manifest_parts["history_length_counts"],
        "membership_summary": manifest_parts["membership_summary"],
        "cell_count_rows_including_not_applicable": int(len(tables["cell_counts"])),
        "applicable_cell_count": int(tables["cell_counts"]["planned_applicable_users"].gt(0).sum()),
        "not_applicable_cell_count": int(tables["cell_counts"]["planned_applicable_users"].eq(0).sum()),
        "denominator_contract": {
            "utility": "all planned/applicable users in the frozen paired cell; non-strict terminal responses contribute zero utility",
            "beyond_utility": "strict-valid users only; metric-specific defined-user subsets must be declared by the metric artifact",
            "utility_invalid_responses_count_zero": True,
            "beyond_utility_invalid_responses_excluded_except_reliability": True,
            "not_applicable_cells_have_no_request_and_no_imputation": True,
        },
        "duplicate_presend_resume_records": manifest_parts["request_ledger"],
        "raw_response_retention": {
            "raw_file_count": manifest_parts["raw_file_count"],
            "raw_unique_ids": manifest_parts["raw_unique_ids"],
            "raw_file_set_covers_g2_terminal_ids": manifest_parts["raw_file_set_covers_g2_terminal_ids"],
        },
        "external_archive_check": {
            "path": artifacts["retention_manifest"]["external_archive"]["path"],
            "expected_sha256": artifacts["retention_manifest"]["external_archive"]["sha256"],
            "actual_sha256": manifest_parts["archive_sha"],
            "entry_count": manifest_parts["archive_entries"],
            "zip_testzip_result": manifest_parts["archive_testzip"],
        },
        "frozen_artifact_hash_checks": {
            "protocol": manifest_parts["protocol_hash_checks"],
            "execution": manifest_parts["execution_hash_checks"],
            "retention_before_notes_update": manifest_parts["retention_hash_checks_before_notes"],
        },
        "validation_checks": manifest_parts["validation_checks"],
        "validation_failures": artifacts["failures"],
        "validation_passed": artifacts["decision_token"] == DECISION_FROZEN,
        "outputs": {name: path.as_posix() for name, path in OUTPUTS.items()},
        "output_artifact_hashes": output_hashes,
        "no_inference_guards": {
            "ollama_generation_calls_executed": 0,
            "ollama_metadata_calls_executed": 0,
            "external_api_calls_executed": 0,
            "g3_requests_executed": 0,
            "metrics_computed": False,
            "scientific_result_interpretation_performed": False,
        },
        "next_safe_step": "G2 metric computation from frozen analytic universe; do not start G3.",
    }
    (root / OUTPUTS["manifest"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    output_hashes[OUTPUTS["manifest"].as_posix()] = sha256_file(root / OUTPUTS["manifest"])
    return output_hashes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    artifacts = build_artifacts(root, allow_generator_dirty=bool(args.allow_generator_dirty))
    if artifacts["failures"]:
        print(f"decision={DECISION_BLOCKED}")
        for failure in artifacts["failures"]:
            print(f"failure={failure}")
        return 2
    if not args.no_write:
        write_outputs(root, artifacts)
    counts = artifacts["manifest_parts"]["canonical_counts"]
    print(f"decision={artifacts['decision_token']}")
    print(f"canonical_rows={counts['logical_requests']}")
    print(f"g1_reuse={counts['g1_exact_reuse']}")
    print(f"g2_terminal={counts['g2_terminal_evidence']}")
    print(f"strict_valid_total={counts['strict_valid']}")
    print(f"invalid_total={counts['invalid']}")
    print(f"cell_rows={len(artifacts['tables']['cell_counts'])}")
    print(f"applicable_cells={int(artifacts['tables']['cell_counts']['planned_applicable_users'].gt(0).sum())}")
    print(f"manifest={OUTPUTS['manifest'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

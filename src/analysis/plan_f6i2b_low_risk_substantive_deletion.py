"""Phase F6I2B low-risk substantive deletion.

This phase executes only the eight exact zero-byte log deletions approved by
the Phase F6I2A low-risk deletion bundle. It preserves pre-deletion hashes,
fails closed on changed candidates, and is restart-safe after successful
deletion.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - project requirements include PyYAML
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2b"
PHASE_NAME = "phase_f6i2b_low_risk_substantive_deletion"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
EXPECTED_CANDIDATES = 8

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
PRE_DELETION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_pre_deletion_manifest.json"
DELETION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_deletion_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"
RUNTIME_LOG_PATH = PROJECT_ROOT / "results" / "logs" / f"{PHASE_ID}_runtime_log.jsonl"

F6I2A_INPUTS = {
    "low_risk_bundle": TABLES_DIR / "phase_f6i2a_low_risk_delete_bundle.csv",
    "retention_decisions": TABLES_DIR / "phase_f6i2a_retention_decisions.csv",
    "future_dependency_protection": TABLES_DIR / "phase_f6i2a_future_dependency_protection.csv",
    "source_file_audit": TABLES_DIR / "phase_f6i2a_source_file_audit.csv",
    "results_manifest_audit": TABLES_DIR / "phase_f6i2a_results_manifest_audit.csv",
    "manual_review": TABLES_DIR / "phase_f6i2a_manual_review.csv",
    "validation": TABLES_DIR / "phase_f6i2a_validation.csv",
    "manifest": MANIFESTS_DIR / "phase_f6i2a_consolidation_design_manifest.json",
    "protected_hashes": MANIFESTS_DIR / "phase_f6i2a_protected_input_hashes.json",
    "report": PROJECT_ROOT / "report" / "phase_f6i2a_substantive_repository_consolidation_design.md",
}
F6I1_INPUTS = {
    "inventory": TABLES_DIR / "phase_f6i1_relevant_file_inventory.csv",
    "edges": TABLES_DIR / "phase_f6i1_dependency_edges.csv",
    "retention": TABLES_DIR / "phase_f6i1_retention_classification.csv",
    "delete_candidates": TABLES_DIR / "phase_f6i1_delete_now_candidates.csv",
    "manual_review": TABLES_DIR / "phase_f6i1_manual_review.csv",
    "future_map": TABLES_DIR / "phase_f6i1_future_dependency_map.csv",
    "duplicates": TABLES_DIR / "phase_f6i1_duplicate_groups.csv",
    "validation": TABLES_DIR / "phase_f6i1_validation.csv",
    "initial_capture": MANIFESTS_DIR / "phase_f6i1_initial_capture.json",
    "report": PROJECT_ROOT / "report" / "phase_f6i1_repository_retention_audit_and_safe_cleanup_batch1.md",
}

OUTPUT_TABLES = {
    "candidate_preflight": TABLES_DIR / f"{PHASE_ID}_candidate_preflight.csv",
    "deletion_actions": TABLES_DIR / f"{PHASE_ID}_deletion_actions.csv",
    "rejected_candidates": TABLES_DIR / f"{PHASE_ID}_rejected_candidates.csv",
    "pre_post_dependency_comparison": TABLES_DIR / f"{PHASE_ID}_pre_post_dependency_comparison.csv",
    "future_dependency_validation": TABLES_DIR / f"{PHASE_ID}_future_dependency_validation.csv",
    "canonical_producer_validation": TABLES_DIR / f"{PHASE_ID}_canonical_producer_validation.csv",
    "reference_validation": TABLES_DIR / f"{PHASE_ID}_reference_validation.csv",
    "test_results": TABLES_DIR / f"{PHASE_ID}_test_results.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PROTECTED_PREFIXES = (
    "original_repo/",
    "paper/",
    "data/raw/",
    "results/llm/",
    "results/predictions/",
    "results/checkpoints/",
    "results/models/",
)
ALLOWED_CURRENT_STATE_UPDATES = {
    "report/future_operations_context_handoff.md",
    "results/manifests/current_project_context_snapshot.json",
}
F6I2B_OUTPUT_PATHS = {
    *(path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUT_TABLES.values()),
    PRE_DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
    DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
    PROTECTED_HASHES_PATH.relative_to(PROJECT_ROOT).as_posix(),
    REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
    RUNTIME_LOG_PATH.relative_to(PROJECT_ROOT).as_posix(),
}
PATH_RE = re.compile(
    r"(?P<path>(?:configs|data|docs|original_repo|paper|report|results|src|tests)[/\\][A-Za-z0-9_. /\\'-]+"
    r"\.(?:csv|json|jsonl|log|md|py|txt|ya?ml|pt|pth|pkl|png|pdf|zip|gz))",
    re.IGNORECASE,
)

GATE_COLUMNS = [
    "gate_01_not_original_repo",
    "gate_02_not_paper",
    "gate_03_not_raw_data",
    "gate_04_not_raw_request",
    "gate_05_not_raw_response",
    "gate_06_not_runner_event",
    "gate_07_not_model_prediction",
    "gate_08_not_checkpoint",
    "gate_09_not_candidate_set",
    "gate_10_not_history_or_cohort",
    "gate_11_not_prompt",
    "gate_12_not_parser",
    "gate_13_not_metric_implementation",
    "gate_14_not_current_config",
    "gate_15_not_current_report",
    "gate_16_not_current_manifest",
    "gate_17_not_imported_or_invoked_by_canonical_code",
    "gate_18_not_loaded_by_retained_configuration",
    "gate_19_not_required_by_retained_test",
    "gate_20_not_active_future_task_dependency",
    "gate_21_not_only_producer_of_retained_evidence",
    "gate_22_not_only_record_of_scientific_decision",
    "gate_23_replacement_or_regeneration_path",
    "gate_24_rollback_source",
    "gate_25_post_deletion_validation_covered",
]

CANDIDATE_PREFLIGHT_COLUMNS = [
    "bundle_id",
    "path",
    "requested_action",
    "exists_preflight",
    "recorded_sha256",
    "actual_sha256",
    "recorded_size_bytes",
    "actual_size_bytes",
    "f6i1_tracked_state",
    "git_state_preflight",
    "f6i2a_reason",
    "canonical_replacement",
    "rollback_source",
    "future_dependency_verdict",
    "dependency_edge_count",
    "inbound_reference_count",
    "active_inbound_reference_count",
    "outbound_reference_count",
    "future_task_mappings",
    *GATE_COLUMNS,
    "preflight_status",
    "preflight_reason",
]
DELETION_ACTION_COLUMNS = [
    "bundle_id",
    "path",
    "requested_action",
    "preflight_status",
    "action_executed",
    "original_sha256",
    "original_size_bytes",
    "canonical_replacement",
    "rollback_source",
    "final_exists",
    "final_status",
    "validation_status",
]
REJECTED_COLUMNS = [
    "bundle_id",
    "path",
    "rejection_class",
    "reason",
    "recorded_sha256",
    "actual_sha256",
    "recorded_size_bytes",
    "actual_size_bytes",
]
DEPENDENCY_COMPARISON_COLUMNS = [
    "check_id",
    "scope",
    "pre_nodes",
    "post_nodes",
    "pre_edges",
    "post_edges",
    "removed_nodes",
    "retained_inbound_consumers",
    "status",
    "evidence",
]
FUTURE_VALIDATION_COLUMNS = [
    "path",
    "future_task_group",
    "f6i2a_retention_requirement",
    "delete_allowed_without_replacement",
    "post_deletion_mapping",
    "status",
    "evidence",
]
PRODUCER_VALIDATION_COLUMNS = [
    "check_id",
    "artifact_scope",
    "expected",
    "observed",
    "status",
    "evidence",
]
REFERENCE_VALIDATION_COLUMNS = [
    "path",
    "referencer",
    "reference_kind",
    "historical_allowed",
    "status",
    "evidence",
]
TEST_RESULTS_COLUMNS = [
    "command_id",
    "validation_type",
    "command",
    "returncode",
    "status",
    "observed",
    "evidence",
]
VALIDATION_COLUMNS = [
    "validation_id",
    "check_name",
    "expected",
    "observed",
    "status",
    "severity",
    "evidence",
]
DETERMINISM_COLUMNS = [
    "determinism_id",
    "check_name",
    "first_digest",
    "second_digest",
    "status",
    "evidence",
]
NEXT_ACTIONS_COLUMNS = [
    "action_id",
    "next_phase",
    "action",
    "blocked_by",
    "requires_approval",
    "evidence",
]


@dataclass(frozen=True)
class PhaseInputs:
    f6i2a_bundle: list[dict[str, str]]
    f6i2a_retention: dict[str, dict[str, str]]
    f6i2a_future: dict[str, list[dict[str, str]]]
    f6i2a_results_manifest: dict[str, dict[str, str]]
    f6i2a_manual_review: dict[str, list[dict[str, str]]]
    f6i2a_validation: list[dict[str, str]]
    f6i2a_manifest: dict[str, Any]
    f6i2a_protected_hashes: dict[str, Any]
    f6i1_inventory: dict[str, dict[str, str]]
    f6i1_edges: list[dict[str, str]]
    f6i1_validation: list[dict[str, str]]
    current_snapshot: dict[str, Any]
    handoff_text: str


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    pre_deletion_manifest: dict[str, Any]
    deletion_manifest: dict[str, Any]
    protected_hashes: dict[str, Any]
    report: str
    decision_token: str


def normalize_path(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", value):
        try:
            value = Path(value).resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return value
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def project_path(path: str | Path) -> Path:
    return PROJECT_ROOT / normalize_path(path)


def is_under(path: str, prefix: str) -> bool:
    normalized = normalize_path(path)
    prefix = normalize_path(prefix).rstrip("/")
    return normalized == prefix or normalized.startswith(prefix + "/")


def run_command(args: Sequence[str], timeout: int = 120) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(args),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def git_stdout(args: Sequence[str]) -> str:
    code, stdout, stderr = run_command(["git", *args], timeout=120)
    if code != 0:
        return f"ERROR: {stderr}"
    return stdout


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = sorted(
        [dict(row) for row in rows],
        key=lambda row: tuple(str(row.get(column, "")) for column in columns),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_json_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()


def split_semicolon(value: str) -> list[str]:
    return [normalize_path(part.strip()) for part in str(value).split(";") if part.strip()]


def index_rows(rows: Sequence[Mapping[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {normalize_path(row.get(key, "")): dict(row) for row in rows if row.get(key, "")}


def group_rows(rows: Sequence[Mapping[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get(key, ""):
            grouped[normalize_path(row[key])].append(dict(row))
    return dict(grouped)


def load_inputs() -> PhaseInputs:
    for name, path in {**F6I2A_INPUTS, **F6I1_INPUTS}.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing authoritative input {name}: {path}")
    if not HANDOFF_PATH.exists():
        raise FileNotFoundError(HANDOFF_PATH)
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(SNAPSHOT_PATH)
    f6i2a_bundle = read_csv(F6I2A_INPUTS["low_risk_bundle"])
    return PhaseInputs(
        f6i2a_bundle=f6i2a_bundle,
        f6i2a_retention=index_rows(read_csv(F6I2A_INPUTS["retention_decisions"]), "path"),
        f6i2a_future=group_rows(read_csv(F6I2A_INPUTS["future_dependency_protection"]), "path"),
        f6i2a_results_manifest=index_rows(read_csv(F6I2A_INPUTS["results_manifest_audit"]), "path"),
        f6i2a_manual_review=group_rows(read_csv(F6I2A_INPUTS["manual_review"]), "path"),
        f6i2a_validation=read_csv(F6I2A_INPUTS["validation"]),
        f6i2a_manifest=read_json(F6I2A_INPUTS["manifest"]),
        f6i2a_protected_hashes=read_json(F6I2A_INPUTS["protected_hashes"]),
        f6i1_inventory=index_rows(read_csv(F6I1_INPUTS["inventory"]), "path"),
        f6i1_edges=read_csv(F6I1_INPUTS["edges"]),
        f6i1_validation=read_csv(F6I1_INPUTS["validation"]),
        current_snapshot=read_json(SNAPSHOT_PATH),
        handoff_text=HANDOFF_PATH.read_text(encoding="utf-8", errors="replace"),
    )


def selected_candidate_paths(inputs: PhaseInputs) -> list[str]:
    return [normalize_path(row.get("path", "")) for row in inputs.f6i2a_bundle]


def validate_bundle_shape(inputs: PhaseInputs) -> None:
    paths = selected_candidate_paths(inputs)
    if len(paths) != EXPECTED_CANDIDATES or len(set(paths)) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"F6I2A bundle mismatch: expected {EXPECTED_CANDIDATES} unique paths, "
            f"observed rows={len(paths)} unique={len(set(paths))}"
        )


def git_state_for_path(path: str) -> str:
    stdout = git_stdout(["status", "--short", "--", path])
    return stdout.replace("\n", " | ") if stdout else "clean_or_untracked_parent_collapsed"


def dependency_edges_for(path: str, edges: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    normalized = normalize_path(path)
    rows = []
    for row in edges:
        source = normalize_path(row.get("source_path", ""))
        target = normalize_path(row.get("target_path", ""))
        if source == normalized or target == normalized:
            rows.append(dict(row))
    return sorted(rows, key=lambda row: row.get("edge_id", ""))


def source_or_target_path(row: Mapping[str, str], candidate: str) -> str:
    source = normalize_path(row.get("source_path", ""))
    target = normalize_path(row.get("target_path", ""))
    return source if target == candidate else target


def is_text_path(path: str, size_text: str = "") -> bool:
    suffix = Path(path).suffix.lower()
    try:
        size = int(size_text or "0")
    except ValueError:
        size = 0
    return suffix in TEXT_EXTENSIONS and size <= 5_000_000


def reference_kind_for(referencer: str) -> str:
    normalized = normalize_path(referencer)
    if normalized.startswith("configs/"):
        return "retained_config"
    if normalized.startswith("src/"):
        return "canonical_code"
    if normalized.startswith("tests/"):
        return "retained_test"
    if normalized.startswith("report/") or normalized.startswith("results/tables/") or normalized.startswith("results/manifests/"):
        return "historical_or_manifest_reference"
    return "other_reference"


def historical_reference_allowed(referencer: str) -> bool:
    normalized = normalize_path(referencer)
    if normalized in ALLOWED_CURRENT_STATE_UPDATES:
        return True
    historical_prefixes = (
        "report/phase_f6i1_",
        "report/phase_f6i2a_",
        "report/phase_f6i2b_",
        "results/tables/phase_f6i1_",
        "results/tables/phase_f6i2a_",
        "results/tables/phase_f6i2b_",
        "results/manifests/phase_f6i1_",
        "results/manifests/phase_f6i2a_",
        "results/manifests/phase_f6i2b_",
        "results/tables/workspace_",
    )
    return normalized.startswith(historical_prefixes)


def scan_references(paths: Sequence[str], inventory: Mapping[str, Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    wanted = [normalize_path(path) for path in paths]
    refs: dict[str, list[dict[str, str]]] = {path: [] for path in wanted}
    extra_files = {
        CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        PRE_DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        PROTECTED_HASHES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        HANDOFF_PATH.relative_to(PROJECT_ROOT).as_posix(),
        SNAPSHOT_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    scan_paths = set(inventory) | extra_files
    for referencer in sorted(scan_paths):
        absolute = project_path(referencer)
        row = inventory.get(referencer, {})
        if not absolute.is_file() or not is_text_path(referencer, row.get("size_bytes", str(absolute.stat().st_size))):
            continue
        try:
            text = absolute.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for candidate in wanted:
            if normalize_path(referencer) == candidate:
                continue
            if candidate in text or candidate.replace("/", "\\") in text:
                refs[candidate].append(
                    {
                        "path": candidate,
                        "referencer": normalize_path(referencer),
                        "reference_kind": reference_kind_for(referencer),
                        "historical_allowed": str(historical_reference_allowed(referencer)),
                    }
                )
    return refs


def outbound_references_for(path: str) -> list[str]:
    absolute = project_path(path)
    if not absolute.is_file() or absolute.stat().st_size > 5_000_000 or Path(path).suffix.lower() not in TEXT_EXTENSIONS:
        return []
    try:
        text = absolute.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return sorted({normalize_path(match.group("path")) for match in PATH_RE.finditer(text)})


def gate_results(
    path: str,
    bundle_row: Mapping[str, str],
    inventory_row: Mapping[str, str],
    retention_row: Mapping[str, str],
    result_manifest_row: Mapping[str, str],
    future_rows: Sequence[Mapping[str, str]],
    edges: Sequence[Mapping[str, str]],
    inbound_refs: Sequence[Mapping[str, str]],
    outbound_refs: Sequence[str],
    exists: bool,
    hash_matches: bool,
    size_matches: bool,
) -> dict[str, str]:
    normalized = normalize_path(path)
    lower = normalized.lower()
    active_refs = [row for row in inbound_refs if row["historical_allowed"] != "True"]
    config_refs = [row for row in inbound_refs if row["reference_kind"] == "retained_config"]
    test_refs = [row for row in inbound_refs if row["reference_kind"] == "retained_test"]
    code_refs = [row for row in inbound_refs if row["reference_kind"] == "canonical_code"]
    zero_or_empty = str(inventory_row.get("size_bytes", "")) == "0"
    unique_info = retention_row.get("unique_information_status", "") or inventory_row.get("unique_scientific_information", "")
    result_consumers = result_manifest_row.get("current_consumers", "")
    future_supported_by_manifest = all(
        row.get("evidence", "") == "supporting_provenance"
        and "f6i2b" in row.get("retention_requirement", "").lower()
        for row in future_rows
    )
    replacement = bundle_row.get("replacement", "")
    rollback = bundle_row.get("rollback", "")
    validation = bundle_row.get("validation", "")
    checks = {
        "gate_01_not_original_repo": not is_under(normalized, "original_repo"),
        "gate_02_not_paper": not is_under(normalized, "paper"),
        "gate_03_not_raw_data": not is_under(normalized, "data/raw"),
        "gate_04_not_raw_request": not (is_under(normalized, "results/llm") and "request" in lower),
        "gate_05_not_raw_response": not (is_under(normalized, "results/llm") and "response" in lower),
        "gate_06_not_runner_event": "runner_event" not in lower and "run_event" not in lower,
        "gate_07_not_model_prediction": not is_under(normalized, "results/predictions"),
        "gate_08_not_checkpoint": not is_under(normalized, "results/checkpoints") and "checkpoint" not in lower,
        "gate_09_not_candidate_set": "candidate" not in lower,
        "gate_10_not_history_or_cohort": "history" not in lower and "cohort" not in lower,
        "gate_11_not_prompt": "prompt" not in lower,
        "gate_12_not_parser": "parser" not in lower,
        "gate_13_not_metric_implementation": not is_under(normalized, "src/metrics") and "metric" not in lower,
        "gate_14_not_current_config": not is_under(normalized, "configs"),
        "gate_15_not_current_report": not is_under(normalized, "report"),
        "gate_16_not_current_manifest": not is_under(normalized, "results/manifests"),
        "gate_17_not_imported_or_invoked_by_canonical_code": not code_refs and not edges,
        "gate_18_not_loaded_by_retained_configuration": not config_refs,
        "gate_19_not_required_by_retained_test": not test_refs,
        "gate_20_not_active_future_task_dependency": future_supported_by_manifest,
        "gate_21_not_only_producer_of_retained_evidence": zero_or_empty and not result_consumers,
        "gate_22_not_only_record_of_scientific_decision": zero_or_empty and unique_info.startswith("no:"),
        "gate_23_replacement_or_regeneration_path": bool(replacement),
        "gate_24_rollback_source": bool(rollback),
        "gate_25_post_deletion_validation_covered": bool(validation),
    }
    rows = {name: "pass" if checks[name] else "fail" for name in GATE_COLUMNS}
    if not exists:
        rows["gate_25_post_deletion_validation_covered"] = rows["gate_25_post_deletion_validation_covered"]
    if not hash_matches or not size_matches:
        rows["gate_21_not_only_producer_of_retained_evidence"] = "fail"
    if active_refs:
        rows["gate_17_not_imported_or_invoked_by_canonical_code"] = "fail"
    return rows


def protected_path_set(inputs: PhaseInputs) -> list[str]:
    paths: set[str] = set()
    for row in inputs.f6i2a_protected_hashes.get("rows", []):
        path = normalize_path(row.get("path", ""))
        if path:
            paths.add(path)
    for path, row in inputs.f6i1_inventory.items():
        category = row.get("category", "")
        lower = path.lower()
        if category in {
            "official_repository",
            "paper",
            "raw_data",
            "raw_llm_output",
            "prediction",
            "model_checkpoint",
            "model_artifact",
        }:
            paths.add(path)
        if "candidate" in lower or "history" in lower or "cohort" in lower:
            paths.add(path)
        if "prompt" in lower or "parser" in lower:
            paths.add(path)
        if category == "source_code" and ("metric" in lower or path.startswith("src/metrics/")):
            paths.add(path)
        if path.startswith("report/canonical_") or path.startswith("report/finalization_") or path.startswith("report/phase_"):
            paths.add(path)
        if path.startswith("results/manifests/phase_") or path.startswith("results/manifests/finalization_"):
            paths.add(path)
    for path in [
        *[item.relative_to(PROJECT_ROOT).as_posix() for item in F6I2A_INPUTS.values()],
        *[item.relative_to(PROJECT_ROOT).as_posix() for item in F6I1_INPUTS.values()],
        "README.md",
        "AGENTS.md",
        "PROJECT_SPEC.md",
        "CODEX_PROMPTS.md",
        "requirements.txt",
        "environment.yml",
    ]:
        paths.add(path)
    paths.difference_update(F6I2B_OUTPUT_PATHS)
    paths.difference_update(ALLOWED_CURRENT_STATE_UPDATES)
    for candidate in selected_candidate_paths(inputs):
        paths.discard(candidate)
    return sorted(paths)


def hash_path_rows(paths: Sequence[str]) -> dict[str, Any]:
    rows = []
    for path in sorted({normalize_path(path) for path in paths}):
        absolute = project_path(path)
        if absolute.exists() and absolute.is_file():
            rows.append(
                {
                    "path": path,
                    "status": "present",
                    "size_bytes": str(absolute.stat().st_size),
                    "sha256": sha256_file(absolute),
                }
            )
        elif absolute.exists() and absolute.is_dir():
            rows.append({"path": path, "status": "directory_unit", "size_bytes": "", "sha256": ""})
        else:
            rows.append({"path": path, "status": "missing", "size_bytes": "", "sha256": ""})
    aggregate = stable_json_digest(rows)
    return {"checked_count": len(rows), "rows": rows, "aggregate_hash": aggregate}


def compare_protected(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_by_path = {row["path"]: row for row in before.get("rows", [])}
    after_by_path = {row["path"]: row for row in after.get("rows", [])}
    changed = []
    missing = []
    unchanged = 0
    for path, before_row in before_by_path.items():
        after_row = after_by_path.get(path)
        if after_row is None or after_row.get("status") == "missing":
            missing.append(path)
        elif before_row == after_row:
            unchanged += 1
        else:
            changed.append(path)
    return {
        "checked_count": len(before_by_path),
        "unchanged_count": unchanged,
        "changed_paths": changed,
        "missing_paths": missing,
        "aggregate_hash_before": before.get("aggregate_hash", ""),
        "aggregate_hash_after": after.get("aggregate_hash", ""),
        "all_unchanged": not changed and not missing and before.get("aggregate_hash", "") == after.get("aggregate_hash", ""),
    }


def build_preflight_rows(inputs: PhaseInputs, protected_before: Mapping[str, Any]) -> list[dict[str, str]]:
    validate_bundle_shape(inputs)
    paths = selected_candidate_paths(inputs)
    references = scan_references(paths, inputs.f6i1_inventory)
    rows = []
    for bundle_row in inputs.f6i2a_bundle:
        path = normalize_path(bundle_row["path"])
        absolute = project_path(path)
        inventory_row = inputs.f6i1_inventory.get(path, {})
        retention_row = inputs.f6i2a_retention.get(path, {})
        result_manifest_row = inputs.f6i2a_results_manifest.get(path, {})
        future_rows = inputs.f6i2a_future.get(path, [])
        edges = dependency_edges_for(path, inputs.f6i1_edges)
        inbound_refs = references.get(path, [])
        outbound_refs = outbound_references_for(path)
        exists = absolute.exists() and absolute.is_file()
        actual_sha = sha256_file(absolute) if exists else ""
        actual_size = str(absolute.stat().st_size) if exists else ""
        recorded_sha = inventory_row.get("sha256", "")
        recorded_size = inventory_row.get("size_bytes", "")
        hash_matches = bool(recorded_sha and actual_sha == recorded_sha)
        size_matches = bool(recorded_size and actual_size == recorded_size)
        gates = gate_results(
            path,
            bundle_row,
            inventory_row,
            retention_row,
            result_manifest_row,
            future_rows,
            edges,
            inbound_refs,
            outbound_refs,
            exists,
            hash_matches,
            size_matches,
        )
        reasons = []
        if not exists:
            reasons.append("candidate_missing_at_preflight")
        if not hash_matches:
            reasons.append("sha256_mismatch")
        if not size_matches:
            reasons.append("size_mismatch")
        failed_gates = [name for name, value in gates.items() if value != "pass"]
        if failed_gates:
            reasons.append("failed_gates:" + ";".join(failed_gates))
        status = "passed" if exists and hash_matches and size_matches and not failed_gates else "preflight_rejected_changed_since_f6i2a"
        row = {
            "bundle_id": bundle_row.get("bundle_id", ""),
            "path": path,
            "requested_action": bundle_row.get("action", ""),
            "exists_preflight": str(exists),
            "recorded_sha256": recorded_sha,
            "actual_sha256": actual_sha,
            "recorded_size_bytes": recorded_size,
            "actual_size_bytes": actual_size,
            "f6i1_tracked_state": inventory_row.get("tracked_state", ""),
            "git_state_preflight": git_state_for_path(path),
            "f6i2a_reason": bundle_row.get("reason", ""),
            "canonical_replacement": bundle_row.get("replacement", ""),
            "rollback_source": bundle_row.get("rollback", ""),
            "future_dependency_verdict": stable_json_digest(future_rows),
            "dependency_edge_count": str(len(edges)),
            "inbound_reference_count": str(len(inbound_refs)),
            "active_inbound_reference_count": str(sum(1 for ref in inbound_refs if ref["historical_allowed"] != "True")),
            "outbound_reference_count": str(len(outbound_refs)),
            "future_task_mappings": ";".join(sorted(row.get("future_task_group", "") for row in future_rows)),
            **gates,
            "preflight_status": status,
            "preflight_reason": "pass" if status == "passed" else "|".join(reasons),
        }
        rows.append(row)
    return rows


def build_pre_deletion_manifest(inputs: PhaseInputs) -> dict[str, Any]:
    protected_before = hash_path_rows(protected_path_set(inputs))
    preflight_rows = build_preflight_rows(inputs, protected_before)
    candidates = []
    for row in preflight_rows:
        path = row["path"]
        edges = dependency_edges_for(path, inputs.f6i1_edges)
        inbound = scan_references([path], inputs.f6i1_inventory).get(path, [])
        outbound = outbound_references_for(path)
        candidates.append(
            {
                "bundle_id": row["bundle_id"],
                "path": path,
                "sha256": row["recorded_sha256"],
                "size_bytes": row["recorded_size_bytes"],
                "git_state": row["git_state_preflight"],
                "canonical_replacement": row["canonical_replacement"],
                "rollback_source": row["rollback_source"],
                "future_dependency_verdict": row["future_dependency_verdict"],
                "dependency_edges": edges,
                "inbound_references": inbound,
                "outbound_references": outbound,
                "future_task_mappings": inputs.f6i2a_future.get(path, []),
                "preflight_status": row["preflight_status"],
                "preflight_reason": row["preflight_reason"],
            }
        )
    branch = git_stdout(["branch", "--show-current"])
    head = git_stdout(["rev-parse", "HEAD"])
    initial_git_status = git_stdout(["status", "--short", "--branch"])
    candidate_paths = selected_candidate_paths(inputs)
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "exact_original_runtime_reproduction": False,
        "branch": branch,
        "head": head,
        "complete_git_status": initial_git_status,
        "relevant_file_count": len(inputs.f6i1_inventory),
        "candidate_count": len(candidate_paths),
        "candidate_paths": candidate_paths,
        "protected_inputs_before": protected_before,
        "protected_input_aggregate_hash": protected_before["aggregate_hash"],
        "dependency_graph_before": {
            "node_count": len(inputs.f6i1_inventory),
            "edge_count": len(inputs.f6i1_edges),
            "candidate_edge_count": sum(int(row["dependency_edge_count"]) for row in preflight_rows),
        },
        "candidate_preflight_rows": preflight_rows,
        "candidates": candidates,
        "f6i2a_authoritative_inputs": {
            name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in F6I2A_INPUTS.items()
        },
        "f6i1_inputs": {name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in F6I1_INPUTS.items()},
    }


def load_or_create_pre_deletion_manifest(inputs: PhaseInputs, force_refresh: bool = False) -> dict[str, Any]:
    if PRE_DELETION_MANIFEST_PATH.exists() and not force_refresh:
        manifest = read_json(PRE_DELETION_MANIFEST_PATH)
        if manifest.get("phase_id") != PHASE_ID:
            raise RuntimeError("Existing pre-deletion manifest has unexpected phase id")
        return manifest
    return build_pre_deletion_manifest(inputs)


def append_runtime_log(row: Mapping[str, str]) -> None:
    RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase_id": PHASE_ID,
        "path": row.get("path", ""),
        "action": row.get("action_executed", ""),
        "status": row.get("final_status", ""),
    }
    with RUNTIME_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def execute_candidate_deletions(pre_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    existing = read_json_if_exists(DELETION_MANIFEST_PATH)
    if existing.get("phase_id") == PHASE_ID and any(row.get("final_status") == "deleted" for row in existing.get("actions", [])):
        return [dict(row) for row in existing["actions"]]

    actions: list[dict[str, str]] = []
    for row in pre_manifest.get("candidate_preflight_rows", []):
        path = normalize_path(row["path"])
        absolute = project_path(path)
        preflight_status = row.get("preflight_status", "")
        action_executed = "not_deleted"
        final_status = "rejected"
        if preflight_status == "passed":
            if absolute.exists() and absolute.is_file():
                actual_sha = sha256_file(absolute)
                actual_size = str(absolute.stat().st_size)
                if actual_sha == row["recorded_sha256"] and actual_size == row["recorded_size_bytes"]:
                    absolute.unlink()
                    action_executed = "deleted"
                    final_status = "deleted"
                else:
                    final_status = "preflight_rejected_changed_since_f6i2a"
            elif not absolute.exists():
                final_status = "already_missing_without_f6i2b_manifest"
        action = {
            "bundle_id": row.get("bundle_id", ""),
            "path": path,
            "requested_action": row.get("requested_action", ""),
            "preflight_status": preflight_status,
            "action_executed": action_executed,
            "original_sha256": row.get("recorded_sha256", ""),
            "original_size_bytes": row.get("recorded_size_bytes", ""),
            "canonical_replacement": row.get("canonical_replacement", ""),
            "rollback_source": row.get("rollback_source", ""),
            "final_exists": str(absolute.exists()),
            "final_status": final_status,
            "validation_status": "pending_post_deletion_validation",
        }
        append_runtime_log(action)
        actions.append(action)
    return actions


def deletion_actions_for_state(pre_manifest: Mapping[str, Any], execute_deletion: bool) -> list[dict[str, str]]:
    existing = read_json_if_exists(DELETION_MANIFEST_PATH)
    if existing.get("phase_id") == PHASE_ID and any(row.get("final_status") == "deleted" for row in existing.get("actions", [])):
        return [dict(row) for row in existing["actions"]]
    if execute_deletion:
        return execute_candidate_deletions(pre_manifest)
    return [
        {
            "bundle_id": row.get("bundle_id", ""),
            "path": normalize_path(row["path"]),
            "requested_action": row.get("requested_action", ""),
            "preflight_status": row.get("preflight_status", ""),
            "action_executed": "not_executed_preflight_only",
            "original_sha256": row.get("recorded_sha256", ""),
            "original_size_bytes": row.get("recorded_size_bytes", ""),
            "canonical_replacement": row.get("canonical_replacement", ""),
            "rollback_source": row.get("rollback_source", ""),
            "final_exists": str(project_path(row["path"]).exists()),
            "final_status": "pending_execute_deletion",
            "validation_status": "not_run_preflight_only",
        }
        for row in pre_manifest.get("candidate_preflight_rows", [])
    ]


def rejected_candidates(preflight_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in preflight_rows:
        if row.get("preflight_status") == "passed":
            continue
        rows.append(
            {
                "bundle_id": row.get("bundle_id", ""),
                "path": row.get("path", ""),
                "rejection_class": "preflight_rejected_changed_since_f6i2a",
                "reason": row.get("preflight_reason", ""),
                "recorded_sha256": row.get("recorded_sha256", ""),
                "actual_sha256": row.get("actual_sha256", ""),
                "recorded_size_bytes": row.get("recorded_size_bytes", ""),
                "actual_size_bytes": row.get("actual_size_bytes", ""),
            }
        )
    return rows


def build_dependency_comparison(inputs: PhaseInputs, pre_manifest: Mapping[str, Any], actions: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    deleted_paths = sorted(row["path"] for row in actions if row.get("final_status") == "deleted")
    post_nodes = len(inputs.f6i1_inventory) - len(deleted_paths)
    pre_edges = len(inputs.f6i1_edges)
    post_edges = len(
        [
            edge
            for edge in inputs.f6i1_edges
            if normalize_path(edge.get("source_path", "")) not in deleted_paths
            and normalize_path(edge.get("target_path", "")) not in deleted_paths
        ]
    )
    refs = scan_references(deleted_paths, inputs.f6i1_inventory) if deleted_paths else {}
    active_refs = [
        ref
        for path_refs in refs.values()
        for ref in path_refs
        if ref.get("historical_allowed") != "True"
    ]
    candidate_edges = int(pre_manifest.get("dependency_graph_before", {}).get("candidate_edge_count", 0))
    return [
        {
            "check_id": "DEP-F6I2B-001",
            "scope": "static_dependency_graph",
            "pre_nodes": str(pre_manifest.get("dependency_graph_before", {}).get("node_count", len(inputs.f6i1_inventory))),
            "post_nodes": str(post_nodes),
            "pre_edges": str(pre_edges),
            "post_edges": str(post_edges),
            "removed_nodes": ";".join(deleted_paths),
            "retained_inbound_consumers": str(len(active_refs)),
            "status": "pass" if candidate_edges == 0 and not active_refs else "fail",
            "evidence": f"candidate_edges_before={candidate_edges}; active_refs_after={len(active_refs)}",
        }
    ]


def build_reference_validation(inputs: PhaseInputs, actions: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    deleted_paths = sorted(row["path"] for row in actions if row.get("final_status") == "deleted")
    refs = scan_references(deleted_paths, inputs.f6i1_inventory) if deleted_paths else {}
    rows = []
    for path in deleted_paths:
        path_refs = refs.get(path, [])
        if not path_refs:
            rows.append(
                {
                    "path": path,
                    "referencer": "",
                    "reference_kind": "none",
                    "historical_allowed": "True",
                    "status": "pass",
                    "evidence": "no references to deleted path",
                }
            )
            continue
        for ref in path_refs:
            historical = ref.get("historical_allowed") == "True"
            rows.append(
                {
                    "path": path,
                    "referencer": ref.get("referencer", ""),
                    "reference_kind": ref.get("reference_kind", ""),
                    "historical_allowed": str(historical),
                    "status": "pass" if historical else "fail",
                    "evidence": "historical manifest/report reference" if historical else "active retained reference",
                }
            )
    return rows


def build_future_dependency_validation(
    inputs: PhaseInputs,
    actions: Sequence[Mapping[str, str]],
    manifest_available: bool,
) -> list[dict[str, str]]:
    rows = []
    for action in actions:
        path = action["path"]
        if action.get("final_status") != "deleted":
            continue
        future_rows = inputs.f6i2a_future.get(path, [])
        if not future_rows:
            rows.append(
                {
                    "path": path,
                    "future_task_group": "",
                    "f6i2a_retention_requirement": "",
                    "delete_allowed_without_replacement": "",
                    "post_deletion_mapping": DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
                    "status": "pass",
                    "evidence": "no F6I2A future task rows",
                }
            )
        for future in future_rows:
            supported = (
                future.get("evidence", "") == "supporting_provenance"
                and "f6i2b" in future.get("retention_requirement", "").lower()
                and manifest_available
            )
            rows.append(
                {
                    "path": path,
                    "future_task_group": future.get("future_task_group", ""),
                    "f6i2a_retention_requirement": future.get("retention_requirement", ""),
                    "delete_allowed_without_replacement": future.get("delete_allowed_without_replacement", ""),
                    "post_deletion_mapping": DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
                    "status": "pass" if supported else "fail",
                    "evidence": "manifest preserves original path/hash for F9 provenance" if supported else "unresolved future dependency",
                }
            )
    return rows


def build_canonical_producer_validation(
    inputs: PhaseInputs,
    actions: Sequence[Mapping[str, str]],
    manifest_available: bool,
) -> list[dict[str, str]]:
    deleted = [row for row in actions if row.get("final_status") == "deleted"]
    deleted_paths = {row["path"] for row in deleted}
    deleted_manifest_rows = [inputs.f6i2a_results_manifest.get(path, {}) for path in deleted_paths]
    deleted_current_consumers = [row for row in deleted_manifest_rows if row.get("current_consumers", "")]
    deleted_known_producers = [row for row in deleted_manifest_rows if row.get("producer", "") not in {"", "unknown_or_prior_phase"}]
    retained_manifest_rows = [
        row
        for path, row in inputs.f6i2a_results_manifest.items()
        if path not in deleted_paths and row.get("producer") and row.get("producer") != "unknown_or_prior_phase"
    ]
    retained_with_available_producers = [
        row
        for row in retained_manifest_rows
        if row.get("retained_producer_available", "") in {"True", "true", "offline_builder", "frozen_or_unknown"}
    ]
    return [
        {
            "check_id": "PROD-F6I2B-001",
            "artifact_scope": "deleted_log_replacement",
            "expected": "phase_f6i2b_deletion_manifest_exists",
            "observed": str(manifest_available),
            "status": "pass" if manifest_available or not deleted else "fail",
            "evidence": "deletion manifest preserves path/hash/reason/rollback",
        },
        {
            "check_id": "PROD-F6I2B-002",
            "artifact_scope": "deleted_log_current_consumers",
            "expected": "0 current consumers and 0 unique producers",
            "observed": f"consumers={len(deleted_current_consumers)};known_producers={len(deleted_known_producers)}",
            "status": "pass" if not deleted_current_consumers and not deleted_known_producers else "fail",
            "evidence": "F6I2A manifest classifies these logs as unknown/prior result_other rows",
        },
        {
            "check_id": "PROD-F6I2B-003",
            "artifact_scope": "retained_f6i2a_known_producer_rows",
            "expected": "informational retained producer inventory",
            "observed": f"{len(retained_with_available_producers)}/{len(retained_manifest_rows)}",
            "status": "pass",
            "evidence": "F6I2B does not alter retained producer files; unresolved historical producer availability remains F6I2A context",
        },
    ]


def decision_token_for(actions: Sequence[Mapping[str, str]], protected_compare: Mapping[str, Any]) -> str:
    if not protected_compare.get("all_unchanged", False):
        return "f6i2b_failed_protected_input_change"
    deleted = [row for row in actions if row.get("final_status") == "deleted"]
    rejected = [row for row in actions if row.get("preflight_status") != "passed" or row.get("final_status") not in {"deleted", "pending_execute_deletion"}]
    if len(deleted) == EXPECTED_CANDIDATES and not rejected:
        return "f6i2b_low_risk_deletion_complete"
    if deleted:
        return "f6i2b_partial_deletion_some_candidates_rejected"
    return "f6i2b_blocked_no_candidate_passed_preflight"


def action_summary(actions: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    deleted = [row for row in actions if row.get("final_status") == "deleted"]
    rejected = [row for row in actions if row.get("preflight_status") != "passed" or row.get("final_status") not in {"deleted", "pending_execute_deletion"}]
    return {
        "requested_count": len(actions),
        "deleted_count": len(deleted),
        "rejected_count": len(rejected),
        "deleted_paths": [row["path"] for row in deleted],
        "bytes_removed": sum(int(row.get("original_size_bytes") or "0") for row in deleted),
    }


def build_deletion_manifest(
    inputs: PhaseInputs,
    pre_manifest: Mapping[str, Any],
    actions: Sequence[Mapping[str, str]],
    protected_compare: Mapping[str, Any],
    decision_token: str,
) -> dict[str, Any]:
    summary = action_summary(actions)
    action_by_path = {row["path"]: row for row in actions}
    rows = []
    for candidate in pre_manifest.get("candidates", []):
        path = candidate["path"]
        action = action_by_path.get(path, {})
        rows.append(
            {
                "bundle_id": candidate.get("bundle_id", ""),
                "path": path,
                "original_sha256": candidate.get("sha256", ""),
                "original_size_bytes": candidate.get("size_bytes", ""),
                "f6i2a_reason": next(
                    (row.get("reason", "") for row in inputs.f6i2a_bundle if normalize_path(row.get("path", "")) == path),
                    "",
                ),
                "replacement_or_regeneration_source": candidate.get("canonical_replacement", ""),
                "rollback_source": candidate.get("rollback_source", ""),
                "executed_status": action.get("final_status", ""),
                "validation_status": action.get("validation_status", ""),
            }
        )
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "exact_original_runtime_reproduction": False,
        "decision_token": decision_token,
        "pre_deletion_manifest": PRE_DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "f6i2a_low_risk_bundle": F6I2A_INPUTS["low_risk_bundle"].relative_to(PROJECT_ROOT).as_posix(),
        "summary": summary,
        "protected_input_result": protected_compare,
        "actions": [dict(row) for row in actions],
        "deleted_artifacts": rows,
        "forbidden_actions": {
            "non_approved_deletion": False,
            "movement": False,
            "archive_migration": False,
            "merge": False,
            "training": False,
            "inference": False,
            "ollama_or_api_call": False,
            "download": False,
            "package_change": False,
            "stage_or_commit": False,
        },
    }


def parse_pytest_observed(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}"
    match = re.search(r"(?P<passed>\d+) passed", text)
    warnings = re.search(r"(?P<warnings>\d+) warnings?", text)
    parts = []
    if match:
        parts.append(f"passed={match.group('passed')}")
    if warnings:
        parts.append(f"warnings={warnings.group('warnings')}")
    if not parts:
        parts.append("summary_unparsed")
    return ";".join(parts)


def syntax_parse_retained_sources() -> tuple[bool, str]:
    failures = []
    count = 0
    for base in [PROJECT_ROOT / "src", PROJECT_ROOT / "tests"]:
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if "__pycache__" in rel:
                continue
            count += 1
            try:
                ast.parse(path.read_bytes().decode("utf-8-sig", errors="replace"), filename=rel)
            except SyntaxError as exc:
                failures.append(f"{rel}:{exc.lineno}:{exc.msg}")
    return not failures, f"parsed={count};failures={';'.join(failures[:10])}"


def import_smoke_modules() -> tuple[bool, str]:
    modules = [
        "src.analysis.phase_f6i1_repository_retention_audit",
        "src.analysis.plan_f6i2a_substantive_repository_consolidation_design",
        "src.analysis.plan_f6i2b_low_risk_substantive_deletion",
        "src.experiments.build_phase_f6i2b_low_risk_substantive_deletion",
    ]
    code = "import importlib; " + "; ".join(f"importlib.import_module({module!r})" for module in modules)
    rc, stdout, stderr = run_command([sys.executable, "-B", "-c", code], timeout=120)
    evidence = f"modules={len(modules)};stdout_sha={hashlib.sha256(stdout.encode()).hexdigest().upper()};stderr_sha={hashlib.sha256(stderr.encode()).hexdigest().upper()}"
    return rc == 0, evidence


def schema_validate_current_manifests() -> tuple[bool, str]:
    errors = []
    for path in [SNAPSHOT_PATH, PRE_DELETION_MANIFEST_PATH, DELETION_MANIFEST_PATH, PROTECTED_HASHES_PATH]:
        try:
            payload = read_json(path)
        except Exception as exc:  # pragma: no cover - defensive reporting
            errors.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}:{exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}:not_object")
    snapshot = read_json_if_exists(SNAPSHOT_PATH)
    if "phase_f6i2b" not in snapshot:
        errors.append("current_snapshot_missing_phase_f6i2b")
    return not errors, "errors=" + ";".join(errors[:10]) if errors else "schema_objects_present"


def run_focused_validation_commands() -> list[dict[str, str]]:
    commands = [
        (
            "TEST-F6I2B-001",
            "focused_pytest",
            [sys.executable, "-B", "-m", "pytest", "tests/test_phase_f6i1_repository_retention_audit.py", "-q"],
        ),
        (
            "TEST-F6I2B-002",
            "focused_pytest",
            [sys.executable, "-B", "-m", "pytest", "tests/test_phase_f6i2a_substantive_repository_consolidation_design.py", "-q"],
        ),
        (
            "TEST-F6I2B-003",
            "focused_pytest",
            [sys.executable, "-B", "-m", "pytest", "tests/test_phase_f6i2b_low_risk_substantive_deletion.py", "-q"],
        ),
    ]
    rows = []
    for command_id, validation_type, command in commands:
        rc, stdout, stderr = run_command(command, timeout=240)
        rows.append(
            {
                "command_id": command_id,
                "validation_type": validation_type,
                "command": " ".join(command),
                "returncode": str(rc),
                "status": "pass" if rc == 0 else "fail",
                "observed": parse_pytest_observed(stdout, stderr),
                "evidence": f"stdout_sha={hashlib.sha256(stdout.encode()).hexdigest().upper()};stderr_sha={hashlib.sha256(stderr.encode()).hexdigest().upper()}",
            }
        )
    ok, evidence = syntax_parse_retained_sources()
    rows.append(
        {
            "command_id": "TEST-F6I2B-004",
            "validation_type": "ast_syntax_parse",
            "command": "ast.parse retained src/**/*.py and tests/**/*.py",
            "returncode": "0" if ok else "1",
            "status": "pass" if ok else "fail",
            "observed": evidence,
            "evidence": "offline AST parsing",
        }
    )
    ok, evidence = import_smoke_modules()
    rows.append(
        {
            "command_id": "TEST-F6I2B-005",
            "validation_type": "import_smoke",
            "command": "import selected no-side-effect canonical modules",
            "returncode": "0" if ok else "1",
            "status": "pass" if ok else "fail",
            "observed": evidence,
            "evidence": "offline import smoke",
        }
    )
    ok, evidence = schema_validate_current_manifests()
    rows.append(
        {
            "command_id": "TEST-F6I2B-006",
            "validation_type": "manifest_schema",
            "command": "validate current context and F6I2B manifest JSON objects",
            "returncode": "0" if ok else "1",
            "status": "pass" if ok else "fail",
            "observed": evidence,
            "evidence": "offline JSON schema smoke",
        }
    )
    first = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False, run_focused_validation=False)
    second = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False, run_focused_validation=False)
    ok = semantic_digest(first) == semantic_digest(second)
    rows.append(
        {
            "command_id": "TEST-F6I2B-007",
            "validation_type": "deterministic_builder",
            "command": "build twice in memory without writing",
            "returncode": "0" if ok else "1",
            "status": "pass" if ok else "fail",
            "observed": f"first={semantic_digest(first)};second={semantic_digest(second)}",
            "evidence": "runtime logs excluded",
        }
    )
    return rows


def existing_test_results_or_placeholder() -> list[dict[str, str]]:
    rows = read_csv_if_exists(OUTPUT_TABLES["test_results"])
    if rows:
        return rows
    return [
        {
            "command_id": "TEST-F6I2B-000",
            "validation_type": "not_run",
            "command": "focused validation pending",
            "returncode": "",
            "status": "pending",
            "observed": "",
            "evidence": "run with --run-focused-validation after deletion",
        }
    ]


def build_next_actions(decision_token: str) -> list[dict[str, str]]:
    if decision_token == "f6i2b_low_risk_deletion_complete":
        return [
            {
                "action_id": "NEXT-F6I2B-001",
                "next_phase": "phase_f6i2c_or_next_unoccupied_f6i_family_identifier",
                "action": "review archive-only bundle without moving files",
                "blocked_by": "requires explicit approval and historical-reference plan",
                "requires_approval": "true",
                "evidence": "F6I2B intentionally skipped archive migration and all merge bundles",
            }
        ]
    return [
        {
            "action_id": "NEXT-F6I2B-001",
            "next_phase": PHASE_ID,
            "action": "resolve rejected low-risk deletion preflight rows",
            "blocked_by": decision_token,
            "requires_approval": "true",
            "evidence": "no further consolidation should run until rejected candidates are handled",
        }
    ]


def validate_outputs(
    inputs: PhaseInputs,
    pre_manifest: Mapping[str, Any],
    actions: Sequence[Mapping[str, str]],
    dependency_rows: Sequence[Mapping[str, str]],
    future_rows: Sequence[Mapping[str, str]],
    producer_rows: Sequence[Mapping[str, str]],
    reference_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
    protected_compare: Mapping[str, Any],
    decision_token: str,
) -> list[dict[str, str]]:
    rows = []

    def add(name: str, expected: str, observed: Any, passed: bool, severity: str = "blocking", evidence: str = "") -> None:
        rows.append(
            {
                "validation_id": f"VAL-F6I2B-{len(rows) + 1:03d}",
                "check_name": name,
                "expected": expected,
                "observed": str(observed),
                "status": "pass" if passed else "fail",
                "severity": severity,
                "evidence": evidence,
            }
        )

    branch = git_stdout(["branch", "--show-current"])
    head = git_stdout(["rev-parse", "HEAD"])
    requested_paths = selected_candidate_paths(inputs)
    preflight_rows = pre_manifest.get("candidate_preflight_rows", [])
    passing = [row for row in preflight_rows if row.get("preflight_status") == "passed"]
    deleted = [row for row in actions if row.get("final_status") == "deleted"]
    pending_only = bool(actions) and all(row.get("final_status") == "pending_execute_deletion" for row in actions)
    rejected = rejected_candidates(preflight_rows)
    active_refs = [row for row in reference_rows if row.get("status") != "pass"]
    test_failures = [row for row in test_rows if row.get("status") == "fail"]
    add("selected_phase_identifier", PHASE_ID, PHASE_ID, True)
    add("branch_matches_expected", EXPECTED_BRANCH, branch, branch == EXPECTED_BRANCH)
    add("head_matches_expected", EXPECTED_HEAD, head, head == EXPECTED_HEAD)
    add("f6i2a_bundle_unique_count", str(EXPECTED_CANDIDATES), f"{len(requested_paths)}/{len(set(requested_paths))}", len(requested_paths) == EXPECTED_CANDIDATES and len(set(requested_paths)) == EXPECTED_CANDIDATES)
    add("f6i2a_validation_all_pass", "all pass", Counter(row.get("status", "") for row in inputs.f6i2a_validation), all(row.get("status") == "pass" for row in inputs.f6i2a_validation))
    add("f6i1_validation_all_pass", "all pass", Counter(row.get("status", "") for row in inputs.f6i1_validation), all(row.get("status") == "pass" for row in inputs.f6i1_validation))
    add("candidate_paths_exclusive_to_f6i2a_bundle", "actions subset of bundle", len(set(row["path"] for row in actions) - set(requested_paths)), not (set(row["path"] for row in actions) - set(requested_paths)))
    add("candidate_preflight_passed", str(EXPECTED_CANDIDATES), len(passing), len(passing) == EXPECTED_CANDIDATES)
    add("candidate_rejections", "0", len(rejected), len(rejected) == 0)
    add("deleted_count", str(EXPECTED_CANDIDATES), len(deleted), len(deleted) == EXPECTED_CANDIDATES or pending_only, evidence="pending_execute_deletion accepted only for read-only preflight")
    add("bytes_removed", "0", action_summary(actions)["bytes_removed"], action_summary(actions)["bytes_removed"] == 0 or pending_only)
    add("dependency_graph_validation", "pass", Counter(row.get("status", "") for row in dependency_rows), all(row.get("status") == "pass" for row in dependency_rows))
    add("future_dependency_validation", "pass", Counter(row.get("status", "") for row in future_rows), all(row.get("status") == "pass" for row in future_rows))
    add("canonical_producer_validation", "pass", Counter(row.get("status", "") for row in producer_rows), all(row.get("status") == "pass" for row in producer_rows))
    add("reference_validation", "0 active unresolved references", len(active_refs), not active_refs)
    add("protected_inputs_unchanged", "true", protected_compare.get("all_unchanged"), bool(protected_compare.get("all_unchanged")))
    add("decision_token_allowed", "allowed final token", decision_token, decision_token in {
        "f6i2b_low_risk_deletion_complete",
        "f6i2b_partial_deletion_some_candidates_rejected",
        "f6i2b_blocked_no_candidate_passed_preflight",
        "f6i2b_failed_protected_input_change",
    })
    add("focused_validation_results", "no failures once run", len(test_failures), not test_failures and bool(test_rows), "blocking" if any(row.get("status") == "fail" for row in test_rows) else "nonblocking")
    return rows


def semantic_payload(result: BuildResult) -> dict[str, Any]:
    return {
        "tables": {
            name: sorted([dict(row) for row in rows], key=lambda row: json.dumps(row, sort_keys=True))
            for name, rows in result.tables.items()
            if name not in {"determinism", "test_results", "validation"}
        },
        "pre_deletion_manifest": result.pre_deletion_manifest,
        "deletion_manifest": result.deletion_manifest,
        "protected_hashes": result.protected_hashes,
        "decision_token": result.decision_token,
    }


def semantic_digest(result: BuildResult) -> str:
    return stable_json_digest(semantic_payload(result))


def build_determinism_row(base_result: BuildResult | None = None) -> list[dict[str, str]]:
    if base_result is None:
        first = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False, run_focused_validation=False, skip_determinism=True)
    else:
        first = base_result
    second = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False, run_focused_validation=False, skip_determinism=True)
    first_digest = semantic_digest(first)
    second_digest = semantic_digest(second)
    return [
        {
            "determinism_id": "DET-F6I2B-001",
            "check_name": "restart_safe_two_pass_core_tables_manifests",
            "first_digest": first_digest,
            "second_digest": second_digest,
            "status": "pass" if first_digest == second_digest else "fail",
            "evidence": "runtime log timestamps excluded",
        }
    ]


def update_action_validation_status(actions: Sequence[Mapping[str, str]], validation_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    post_ok = all(row.get("status") == "pass" or row.get("severity") == "nonblocking" for row in validation_rows)
    updated = []
    for row in actions:
        copy = dict(row)
        if copy.get("final_status") == "deleted":
            copy["validation_status"] = "post_deletion_validation_passed" if post_ok else "post_deletion_validation_failed"
        updated.append(copy)
    return updated


def update_handoff_and_snapshot(deletion_manifest: Mapping[str, Any]) -> None:
    summary = deletion_manifest.get("summary", {})
    block = f"""<!-- phase_f6i2b:start -->
## Phase F6I2B Low-Risk Substantive Deletion

- Phase id: `{PHASE_ID}`.
- Runtime classification: `{RUNTIME_CLASSIFICATION}`.
- Decision token: `{deletion_manifest.get('decision_token', '')}`.
- F6I2A requested candidates: `{summary.get('requested_count', 0)}`.
- Files deleted: `{summary.get('deleted_count', 0)}`.
- Candidates rejected: `{summary.get('rejected_count', 0)}`.
- Bytes removed: `{summary.get('bytes_removed', 0)}`.
- Deletion manifest: `results/manifests/phase_f6i2b_deletion_manifest.json`.
- Pre-deletion manifest: `results/manifests/phase_f6i2b_pre_deletion_manifest.json`.
- Scope note: only the F6I2A low-risk deletion bundle was executed; archive, merge, source, YAML, test, and documentation consolidation remain future work.
<!-- phase_f6i2b:end -->"""
    old = HANDOFF_PATH.read_text(encoding="utf-8", errors="replace") if HANDOFF_PATH.exists() else "# Future Operations Context Handoff\n"
    start = "<!-- phase_f6i2b:start -->"
    end = "<!-- phase_f6i2b:end -->"
    if start in old and end in old:
        prefix = old.split(start)[0].rstrip()
        suffix = old.split(end, 1)[1].lstrip()
        new_text = f"{prefix}\n\n{block}\n\n{suffix}".rstrip() + "\n"
    else:
        new_text = old.rstrip() + "\n\n" + block + "\n"
    HANDOFF_PATH.write_text(new_text, encoding="utf-8")

    snapshot = read_json_if_exists(SNAPSHOT_PATH)
    snapshot[PHASE_ID] = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "decision_token": deletion_manifest.get("decision_token", ""),
        "report": REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "pre_deletion_manifest": PRE_DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "deletion_manifest": DELETION_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "protected_hashes": PROTECTED_HASHES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "summary": summary,
        "previous_phase": "phase_f6i2a",
        "historical_f6i2a_counts_preserved": True,
    }
    write_json(SNAPSHOT_PATH, snapshot)


def generate_report(
    pre_manifest: Mapping[str, Any],
    deletion_manifest: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    protected_compare: Mapping[str, Any],
) -> str:
    summary = deletion_manifest.get("summary", {})
    validation_counts = Counter(row.get("status", "") for row in tables["validation"])
    test_counts = Counter(row.get("status", "") for row in tables["test_results"])
    deleted_paths = summary.get("deleted_paths", [])
    unresolved_refs = [row for row in tables["reference_validation"] if row.get("status") != "pass"]
    return f"""# Phase F6I2B Low-Risk Substantive Deletion

Decision token: `{deletion_manifest.get('decision_token', '')}`.

Runtime classification: `{RUNTIME_CLASSIFICATION}`. This is not exact original executable-environment reproduction.

## Scope

- Selected phase identifier: `{PHASE_ID}`.
- Branch: `{pre_manifest.get('branch', '')}`.
- HEAD: `{pre_manifest.get('head', '')}`.
- Authoritative deletion set: `results/tables/phase_f6i2a_low_risk_delete_bundle.csv`.
- F6I2A candidates requested: `{summary.get('requested_count', 0)}`.
- Candidates deleted: `{summary.get('deleted_count', 0)}`.
- Candidates rejected: `{summary.get('rejected_count', 0)}`.
- Bytes removed: `{summary.get('bytes_removed', 0)}`.

## Deleted Paths

{chr(10).join(f'- `{path}`' for path in deleted_paths) if deleted_paths else '- none'}

## Dependency And Reference Audit

- Dependency graph before: `{pre_manifest.get('dependency_graph_before', {}).get('node_count', 0)}` nodes, `{pre_manifest.get('dependency_graph_before', {}).get('edge_count', 0)}` edges.
- Candidate dependency edges before deletion: `{pre_manifest.get('dependency_graph_before', {}).get('candidate_edge_count', 0)}`.
- Dependency comparison table: `results/tables/phase_f6i2b_pre_post_dependency_comparison.csv`.
- Unresolved active references: `{len(unresolved_refs)}`.

## Protected Inputs

- Checked count: `{protected_compare.get('checked_count', 0)}`.
- Unchanged count: `{protected_compare.get('unchanged_count', 0)}`.
- Changed paths: `{len(protected_compare.get('changed_paths', []))}`.
- Missing paths: `{len(protected_compare.get('missing_paths', []))}`.
- Aggregate hash before: `{protected_compare.get('aggregate_hash_before', '')}`.
- Aggregate hash after: `{protected_compare.get('aggregate_hash_after', '')}`.

## Validation

- Validation status counts: `{dict(sorted(validation_counts.items()))}`.
- Focused-test status counts: `{dict(sorted(test_counts.items()))}`.
- Determinism table: `results/tables/phase_f6i2b_determinism.csv`.

## Forbidden Actions

No non-approved deletion, movement, archive migration, source consolidation, YAML consolidation, test consolidation, documentation consolidation, training, inference, Ollama/API call, download, package change, staging, or commit is part of this phase.
"""


def write_outputs(result: BuildResult, update_current_state: bool) -> None:
    for name, path in OUTPUT_TABLES.items():
        rows = result.tables[name]
        columns = {
            "candidate_preflight": CANDIDATE_PREFLIGHT_COLUMNS,
            "deletion_actions": DELETION_ACTION_COLUMNS,
            "rejected_candidates": REJECTED_COLUMNS,
            "pre_post_dependency_comparison": DEPENDENCY_COMPARISON_COLUMNS,
            "future_dependency_validation": FUTURE_VALIDATION_COLUMNS,
            "canonical_producer_validation": PRODUCER_VALIDATION_COLUMNS,
            "reference_validation": REFERENCE_VALIDATION_COLUMNS,
            "test_results": TEST_RESULTS_COLUMNS,
            "validation": VALIDATION_COLUMNS,
            "determinism": DETERMINISM_COLUMNS,
            "next_actions": NEXT_ACTIONS_COLUMNS,
        }[name]
        write_csv(path, rows, columns)
    write_json(PRE_DELETION_MANIFEST_PATH, result.pre_deletion_manifest)
    write_json(DELETION_MANIFEST_PATH, result.deletion_manifest)
    write_json(PROTECTED_HASHES_PATH, result.protected_hashes)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(result.report, encoding="utf-8")
    blocking_failures = [row for row in result.tables["validation"] if row.get("status") != "pass" and row.get("severity") == "blocking"]
    if update_current_state and not blocking_failures:
        update_handoff_and_snapshot(result.deletion_manifest)


def build(
    execute_deletion: bool = False,
    write_outputs_enabled: bool = True,
    update_current_state: bool = True,
    run_focused_validation: bool = False,
    skip_determinism: bool = False,
    force_refresh_preflight: bool = False,
) -> BuildResult:
    inputs = load_inputs()
    pre_manifest = load_or_create_pre_deletion_manifest(inputs, force_refresh=force_refresh_preflight)
    preflight_rows = [dict(row) for row in pre_manifest.get("candidate_preflight_rows", [])]
    actions = deletion_actions_for_state(pre_manifest, execute_deletion)
    protected_after = hash_path_rows(protected_path_set(inputs))
    protected_compare = compare_protected(pre_manifest.get("protected_inputs_before", {}), protected_after)
    decision_token = decision_token_for(actions, protected_compare)
    deletion_manifest = build_deletion_manifest(inputs, pre_manifest, actions, protected_compare, decision_token)
    manifest_available = bool(DELETION_MANIFEST_PATH.exists() or execute_deletion)
    dependency_rows = build_dependency_comparison(inputs, pre_manifest, actions)
    reference_rows = build_reference_validation(inputs, actions)
    future_rows = build_future_dependency_validation(inputs, actions, manifest_available)
    producer_rows = build_canonical_producer_validation(inputs, actions, manifest_available)
    test_rows = run_focused_validation_commands() if run_focused_validation else existing_test_results_or_placeholder()
    validation_rows = validate_outputs(
        inputs,
        pre_manifest,
        actions,
        dependency_rows,
        future_rows,
        producer_rows,
        reference_rows,
        test_rows,
        protected_compare,
        decision_token,
    )
    actions = update_action_validation_status(actions, validation_rows)
    deletion_manifest = build_deletion_manifest(inputs, pre_manifest, actions, protected_compare, decision_token)
    tables = {
        "candidate_preflight": preflight_rows,
        "deletion_actions": actions,
        "rejected_candidates": rejected_candidates(preflight_rows),
        "pre_post_dependency_comparison": dependency_rows,
        "future_dependency_validation": future_rows,
        "canonical_producer_validation": producer_rows,
        "reference_validation": reference_rows,
        "test_results": test_rows,
        "validation": validation_rows,
        "determinism": [],
        "next_actions": build_next_actions(decision_token),
    }
    protected_hashes = {
        "phase_id": PHASE_ID,
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "before": pre_manifest.get("protected_inputs_before", {}),
        "after": protected_after,
        "comparison": protected_compare,
    }
    report = generate_report(pre_manifest, deletion_manifest, tables, protected_compare)
    result = BuildResult(
        tables=tables,
        pre_deletion_manifest=pre_manifest,
        deletion_manifest=deletion_manifest,
        protected_hashes=protected_hashes,
        report=report,
        decision_token=decision_token,
    )
    if not skip_determinism:
        tables["determinism"] = build_determinism_row(result)
        result = BuildResult(
            tables=tables,
            pre_deletion_manifest=pre_manifest,
            deletion_manifest=deletion_manifest,
            protected_hashes=protected_hashes,
            report=report,
            decision_token=decision_token,
        )
    if write_outputs_enabled:
        write_outputs(result, update_current_state=update_current_state)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH), help="F6I2B config path.")
    parser.add_argument("--execute-deletion", action="store_true", help="Delete only candidates passing F6I2B preflight.")
    parser.add_argument("--run-focused-validation", action="store_true", help="Run focused offline tests and smoke validations.")
    parser.add_argument("--skip-current-state-updates", action="store_true", help="Do not update allowed handoff/snapshot files.")
    parser.add_argument("--force-refresh-preflight", action="store_true", help="Rebuild pre-deletion manifest; invalid after deletion.")
    args = parser.parse_args(argv)

    if yaml is None:
        raise RuntimeError("PyYAML is required")
    config_path = project_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.force_refresh_preflight and DELETION_MANIFEST_PATH.exists():
        existing_deletion = read_json_if_exists(DELETION_MANIFEST_PATH)
        if any(row.get("final_status") == "deleted" for row in existing_deletion.get("actions", [])):
            raise RuntimeError("Cannot refresh preflight after an executed deletion manifest exists")

    result = build(
        execute_deletion=args.execute_deletion,
        write_outputs_enabled=True,
        update_current_state=not args.skip_current_state_updates,
        run_focused_validation=args.run_focused_validation,
        force_refresh_preflight=args.force_refresh_preflight,
    )
    blocking_failures = [row for row in result.tables["validation"] if row.get("status") != "pass" and row.get("severity") == "blocking"]
    return 1 if blocking_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

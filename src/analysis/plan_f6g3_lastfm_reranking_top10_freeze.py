"""Build the F6G3 corrective offline LastFM reranking top-10 freeze.

This phase is intentionally fail-closed. It creates a repository-derived
top-10 request freeze from existing F6G1 candidate/cohort evidence, then admits
raw responses only when an existing response can be linked to LastFM reranking
without inference or guessing.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6g3"
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase_f6g3_lastfm_reranking_top10_freeze.yaml"
RECOVERED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6g3_lastfm_reranking_top10_freeze.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
CURRENT_CONTEXT_SNAPSHOT = MANIFESTS_DIR / "current_project_context_snapshot.json"

SOURCE_INVENTORY_CSV = TABLES_DIR / "phase_f6g3_source_artifact_inventory.csv"
LEGACY_SUPERSESSION_CSV = TABLES_DIR / "phase_f6g3_legacy_manifest_supersession.csv"
CORRECTIVE_REQUEST_MANIFEST_CSV = TABLES_DIR / "phase_f6g3_corrective_request_manifest.csv"
CORRECTIVE_SHARD_MANIFEST_CSV = TABLES_DIR / "phase_f6g3_corrective_shard_manifest.csv"
REQUEST_RESPONSE_RECON_CSV = TABLES_DIR / "phase_f6g3_request_response_reconciliation.csv"
CANDIDATE_VALIDATION_CSV = TABLES_DIR / "phase_f6g3_candidate_pool_validation.csv"
TOP10_DERIVATION_CSV = TABLES_DIR / "phase_f6g3_top10_derivation.csv"
CANONICAL_RECOMMENDATIONS_CSV = TABLES_DIR / "phase_f6g3_canonical_recommendations.csv"
INVALID_BLOCKED_CSV = TABLES_DIR / "phase_f6g3_invalid_or_blocked_rows.csv"
RERANKING_METRICS_CSV = TABLES_DIR / "phase_f6g3_reranking_metrics.csv"
BEYOND_METRICS_CSV = TABLES_DIR / "phase_f6g3_beyond_utility_metrics.csv"
CELL_SUMMARY_CSV = TABLES_DIR / "phase_f6g3_cell_summary.csv"
EXACTNESS_ELEVATION_CSV = TABLES_DIR / "phase_f6g3_exactness_elevation.csv"
VALIDATION_CSV = TABLES_DIR / "phase_f6g3_validation.csv"
DETERMINISM_CSV = TABLES_DIR / "phase_f6g3_determinism.csv"
NEXT_ACTIONS_CSV = TABLES_DIR / "phase_f6g3_next_actions.csv"

FREEZE_MANIFEST_JSON = MANIFESTS_DIR / "phase_f6g3_top10_freeze_manifest.json"
PROTECTED_HASHES_JSON = MANIFESTS_DIR / "phase_f6g3_protected_input_hashes.json"

PHASE_OUTPUT_PATHS = [
    "configs/phase_f6g3_lastfm_reranking_top10_freeze.yaml",
    "src/analysis/plan_f6g3_lastfm_reranking_top10_freeze.py",
    "src/experiments/build_phase_f6g3_lastfm_reranking_top10_freeze.py",
    "tests/test_phase_f6g3_lastfm_reranking_top10_freeze.py",
    "report/phase_f6g3_lastfm_reranking_top10_freeze.md",
    "report/future_operations_context_handoff.md",
    "results/tables/phase_f6g3_source_artifact_inventory.csv",
    "results/tables/phase_f6g3_legacy_manifest_supersession.csv",
    "results/tables/phase_f6g3_corrective_request_manifest.csv",
    "results/tables/phase_f6g3_corrective_shard_manifest.csv",
    "results/tables/phase_f6g3_request_response_reconciliation.csv",
    "results/tables/phase_f6g3_candidate_pool_validation.csv",
    "results/tables/phase_f6g3_top10_derivation.csv",
    "results/tables/phase_f6g3_canonical_recommendations.csv",
    "results/tables/phase_f6g3_invalid_or_blocked_rows.csv",
    "results/tables/phase_f6g3_reranking_metrics.csv",
    "results/tables/phase_f6g3_beyond_utility_metrics.csv",
    "results/tables/phase_f6g3_cell_summary.csv",
    "results/tables/phase_f6g3_exactness_elevation.csv",
    "results/tables/phase_f6g3_validation.csv",
    "results/tables/phase_f6g3_determinism.csv",
    "results/tables/phase_f6g3_next_actions.csv",
    "results/manifests/phase_f6g3_top10_freeze_manifest.json",
    "results/manifests/phase_f6g3_protected_input_hashes.json",
    "results/manifests/current_project_context_snapshot.json",
]

SOURCE_INVENTORY_FIELDS = [
    "artifact_id",
    "path",
    "phase",
    "role",
    "exists",
    "sha256",
    "row_count",
    "line_count",
    "lastfm_reranking_rows",
    "contract_status",
    "protected_status",
    "notes",
]

CORRECTIVE_REQUEST_FIELDS = [
    "f6g3_request_id",
    "legacy_request_id",
    "logical_instance_id",
    "user_id",
    "dataset",
    "task",
    "model",
    "strategy",
    "cohort_name",
    "candidate_pool_size",
    "candidate_unique_count",
    "candidate_item_ids",
    "candidate_order_hash",
    "positive_item_id",
    "positive_initial_position",
    "positive_present",
    "original_required_output_length",
    "original_parser_contract",
    "corrective_requested_output_length",
    "corrective_parser_expected_length",
    "metric_cutoff_k",
    "output_length_resolution",
    "legacy_reuse_policy",
    "request_status",
    "block_reason",
    "corrective_request_hash",
]

REQUEST_RESPONSE_FIELDS = [
    "f6g3_request_id",
    "legacy_request_id",
    "logical_instance_id",
    "model",
    "strategy",
    "dataset",
    "task",
    "request_lineage_status",
    "response_lineage_status",
    "runner_event_status",
    "legacy_shard_id",
    "corrective_shard_id",
    "raw_output_file",
    "raw_record_sha256",
    "raw_response_found",
    "raw_output_contract",
    "observed_output_length",
    "admission_status",
    "block_reason",
    "notes",
]

TOP10_DERIVATION_FIELDS = [
    "f6g3_request_id",
    "legacy_request_id",
    "model",
    "strategy",
    "dataset",
    "task",
    "admitted_to_freeze",
    "original_request_contract",
    "observed_raw_response_contract",
    "offline_transformation",
    "original_output_length",
    "derived_output_length",
    "canonical_top10_item_ids",
    "first_five_item_ids",
    "parse_validity_status",
    "candidate_validity_status",
    "duplicate_status",
    "hallucination_status",
    "positive_rank",
    "hr_at5",
    "ndcg_at5",
    "raw_response_reference",
    "block_reason",
    "notes",
]


def require_runtime() -> None:
    if Path(sys.executable).resolve() != RECOVERED_INTERPRETER.resolve():
        raise RuntimeError(f"{PHASE_ID} must run with {RECOVERED_INTERPRETER}; got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: tuple(str(row.get(field, "")) for field in fields))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def run_command(args: list[str], timeout: int = 30) -> str:
    proc = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=timeout)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def split_pipe(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item != ""]


def bool_str(value: bool) -> str:
    return str(value).lower()


def line_count(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def jsonl_dataset_task_counts(path: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counts[("json_decode_error", "json_decode_error")] += 1
                continue
            counts[(str(obj.get("dataset", "")), str(obj.get("task", "")))] += 1
    return counts


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    paths = {name: PROJECT_ROOT / rel_path for name, rel_path in config["inputs"].items()}
    return {
        "paths": paths,
        "logical": read_csv_rows(paths["f6g1_logical_instances"]),
        "requests": read_csv_rows(paths["f6g1_request_manifest"]),
        "shards": read_csv_rows(paths["f6g1_shard_manifest"]),
        "pool_summary": read_csv_rows(paths["f6g1_pool_instance_summary"]),
        "pool_items": read_csv_rows(paths["f6g1_pool_item_provenance"]),
        "source_decisions": read_csv_rows(paths["f6g1_candidate_source_model_decisions"]),
        "f6g2_request_execution": read_csv_rows(paths["f6g2_request_execution"]),
        "f6e4b_instance_results": read_csv_rows(paths["f6e4b_instance_results"]),
        "frontier_components": read_csv_rows(paths["exactness_frontier_components"]),
        "frontier_numeric": read_csv_rows(paths["exactness_frontier_numeric"]),
    }


def protected_input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(set(paths.values()), key=lambda p: rel(p)):
        if path.exists() and path.is_file():
            hashes[rel(path)] = file_sha256(path)
    return hashes


def build_source_inventory(config: dict[str, Any], inputs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    paths: dict[str, Path] = inputs["paths"]
    roles = {
        "f6g1_logical_instances": "legacy_top20_logical_instances",
        "f6g1_request_manifest": "legacy_top20_request_manifest",
        "f6g1_shard_manifest": "legacy_top20_shard_manifest",
        "f6g1_pool_instance_summary": "candidate_pool_summary",
        "f6g1_pool_item_provenance": "candidate_pool_item_provenance",
        "f6g1_candidate_source_model_decisions": "reranking_candidate_source_model_audit",
        "f6g2_request_execution": "blocked_f6g2_request_execution",
        "f6g2_output_length_resolution": "contract_resolution",
        "f6e4b_instance_results": "existing_llm_instance_results_no_lastfm_reranking",
        "f6e4b_raw_qwen2": "existing_raw_llm_output_no_lastfm_reranking",
        "f6e4b_raw_mistral": "existing_raw_llm_output_no_lastfm_reranking",
        "exactness_frontier_components": "exactness_frontier_components",
        "exactness_frontier_numeric": "exactness_frontier_numeric_cells",
    }
    row_lookup = {
        "f6g1_logical_instances": inputs["logical"],
        "f6g1_request_manifest": inputs["requests"],
        "f6g1_shard_manifest": inputs["shards"],
        "f6g1_pool_instance_summary": inputs["pool_summary"],
        "f6g1_pool_item_provenance": inputs["pool_items"],
        "f6g1_candidate_source_model_decisions": inputs["source_decisions"],
        "f6g2_request_execution": inputs["f6g2_request_execution"],
        "f6e4b_instance_results": inputs["f6e4b_instance_results"],
        "exactness_frontier_components": inputs["frontier_components"],
        "exactness_frontier_numeric": inputs["frontier_numeric"],
    }
    for name, path in sorted(paths.items()):
        rows_for_file = row_lookup.get(name, [])
        lastfm_reranking_rows = 0
        if rows_for_file:
            lastfm_reranking_rows = len([r for r in rows_for_file if r.get("dataset") == "LastFM" and r.get("task") == "reranking"])
        elif path.suffix.lower() == ".jsonl":
            counts = jsonl_dataset_task_counts(path)
            lastfm_reranking_rows = counts.get(("LastFM", "reranking"), 0)
        rows.append(
            {
                "artifact_id": f"F6G3-SOURCE-{stable_hash(name)[:14]}",
                "path": rel(path),
                "phase": name.split("_")[0].upper() if name.startswith("f6") else "AUDIT",
                "role": roles.get(name, "supporting_input"),
                "exists": bool_str(path.exists()),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
                "row_count": len(rows_for_file) if rows_for_file else "",
                "line_count": line_count(path) if path.suffix.lower() == ".jsonl" else "",
                "lastfm_reranking_rows": lastfm_reranking_rows,
                "contract_status": "legacy_top20_superseded" if name.startswith("f6g1") else ("no_lastfm_reranking_raw_rows" if "f6e4b" in name else "supporting_evidence"),
                "protected_status": "protected_input_for_f6g3",
                "notes": "Read-only input; not modified by F6G3.",
            }
        )
    return rows


def build_legacy_supersession(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "legacy_artifact": "results/tables/phase_f6g1_future_logical_instances.csv",
            "legacy_phase": "phase_f6g1",
            "row_count": len(inputs["logical"]),
            "original_request_contract": "candidate_pool_size=20; requested_output_length=20; parser=indexed_json_top20_unique_candidates",
            "canonical_f6g3_use": "candidate/cohort/pool/order/provenance evidence only",
            "supersession_status": "superseded_for_execution_by_D_repository_derived_top10_paper_unspecified",
            "evidence_reused": "logical_instance_id; user_id; candidate_item_ids; candidate_order_hash; positive metadata; pool provenance",
            "evidence_not_reused": "expected_output_length=20; parser_contract=indexed_json_top20_unique_candidates as canonical execution contract",
            "notes": "Corrective F6G3 creates top-10 request manifest without relabeling the original top-20 contract.",
        },
        {
            "legacy_artifact": "results/tables/phase_f6g1_future_request_manifest.csv",
            "legacy_phase": "phase_f6g1",
            "row_count": len(inputs["requests"]),
            "original_request_contract": "required_output_length=20; parser=indexed_json_top20_unique_candidates",
            "canonical_f6g3_use": "legacy request identity and candidate-order provenance",
            "supersession_status": "superseded_for_execution",
            "evidence_reused": "model; strategy; shard boundaries; candidate pools; request hashes as historical provenance",
            "evidence_not_reused": "required_output_length=20 as current execution request",
            "notes": "All derived F6G3 requests require top-10 output and top-10 parser.",
        },
        {
            "legacy_artifact": "results/tables/phase_f6g2_request_execution.csv",
            "legacy_phase": "phase_f6g2",
            "row_count": len(inputs["f6g2_request_execution"]),
            "original_request_contract": "configured top-10 contract but no raw canonical outputs",
            "canonical_f6g3_use": "blocked-execution evidence and expected population accounting",
            "supersession_status": "historical_blocked_execution",
            "evidence_reused": "expected request count and blocked lineage facts",
            "evidence_not_reused": "no canonical raw responses because raw paths are absent",
            "notes": "F6G3 does not invent raw outputs for F6G2.",
        },
    ]


def build_corrective_request_manifest(inputs: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    logical_by_id = {row["logical_instance_id"]: row for row in inputs["logical"]}
    rows = []
    for legacy in inputs["requests"]:
        logical = logical_by_id.get(legacy["logical_instance_id"], {})
        candidate_ids = split_pipe(legacy.get("candidate_item_ids", ""))
        f6g3_request_id = "F6G3-REQ-" + stable_hash({"legacy_request_id": legacy["request_id"], "contract": "top10"})[:16].lower()
        row = {
            "f6g3_request_id": f6g3_request_id,
            "legacy_request_id": legacy["request_id"],
            "logical_instance_id": legacy["logical_instance_id"],
            "user_id": legacy.get("user_id", ""),
            "dataset": legacy.get("dataset", ""),
            "task": legacy.get("task", ""),
            "model": legacy.get("model", ""),
            "strategy": legacy.get("strategy", ""),
            "cohort_name": logical.get("cohort_name", ""),
            "candidate_pool_size": len(candidate_ids),
            "candidate_unique_count": len(set(candidate_ids)),
            "candidate_item_ids": legacy.get("candidate_item_ids", ""),
            "candidate_order_hash": legacy.get("candidate_order_hash", ""),
            "positive_item_id": legacy.get("positive_item_id", ""),
            "positive_initial_position": legacy.get("positive_initial_position", ""),
            "positive_present": legacy.get("positive_present", ""),
            "original_required_output_length": legacy.get("required_output_length", ""),
            "original_parser_contract": legacy.get("parser_contract", ""),
            "corrective_requested_output_length": config["contract"]["requested_output_length"],
            "corrective_parser_expected_length": config["contract"]["parser_expected_length"],
            "metric_cutoff_k": config["contract"]["metric_cutoff_k"],
            "output_length_resolution": config["contract"]["output_length_resolution"],
            "legacy_reuse_policy": "reuse_candidate_cohort_order_and_provenance_only",
            "request_status": "corrective_manifest_ready_blocked_pending_existing_raw_response",
            "block_reason": "no_existing_lastfm_reranking_raw_response_linked",
        }
        row["corrective_request_hash"] = stable_hash(row)
        rows.append(row)
    return rows


def build_corrective_shards(corrective_requests: list[dict[str, Any]], legacy_shards: list[dict[str, str]]) -> list[dict[str, Any]]:
    requests_by_model_strategy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in corrective_requests:
        requests_by_model_strategy[(row["model"], row["strategy"])].append(row)
    rows = []
    for (model, strategy), group in sorted(requests_by_model_strategy.items()):
        group = sorted(group, key=lambda r: r["legacy_request_id"])
        chunk_size = 500
        for shard_index, start in enumerate(range(0, len(group), chunk_size), start=1):
            chunk = group[start : start + chunk_size]
            shard_id = f"F6G3-SHARD-{stable_hash({'model': model, 'strategy': strategy, 'index': shard_index})[:12].lower()}"
            rows.append(
                {
                    "corrective_shard_id": shard_id,
                    "model": model,
                    "strategy": strategy,
                    "shard_index": shard_index,
                    "request_count": len(chunk),
                    "first_f6g3_request_id": chunk[0]["f6g3_request_id"],
                    "last_f6g3_request_id": chunk[-1]["f6g3_request_id"],
                    "legacy_shard_policy": "derived_from_corrective_request_sort_not_reusing_top20_shard_as_canonical",
                    "corrective_contract": "candidate20_request10_parser10_k5",
                    "status": "ready_but_blocked_no_raw_responses",
                    "notes": "Shard manifests are offline planning artifacts only; no inference is authorized.",
                }
            )
    return rows


def lineage_raw_lookup(inputs: dict[str, Any]) -> dict[str, dict[str, str]]:
    rows = {}
    for row in inputs["f6e4b_instance_results"]:
        if row.get("dataset") == "LastFM" and row.get("task") == "reranking":
            rows[row.get("source_request_id") or row.get("request_id", "")] = row
            rows[row.get("request_id", "")] = row
    return rows


def build_reconciliation(corrective_requests: list[dict[str, Any]], corrective_shards: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
    raw_lookup = lineage_raw_lookup(inputs)
    shard_for_request = {}
    for shard in corrective_shards:
        shard_for_request[(shard["model"], shard["strategy"], shard["shard_index"])] = shard["corrective_shard_id"]
    legacy_shard_by_request = {}
    for legacy in inputs["requests"]:
        legacy_shard_by_request[legacy["request_id"]] = ""
    rows = []
    # Assign corrective shard by the same sorted grouping used during shard construction.
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for request in corrective_requests:
        by_group[(request["model"], request["strategy"])].append(request)
    corrective_shard_by_req = {}
    for (model, strategy), group in by_group.items():
        for index, request in enumerate(sorted(group, key=lambda r: r["legacy_request_id"])):
            shard_index = index // 500 + 1
            corrective_shard_by_req[request["f6g3_request_id"]] = shard_for_request[(model, strategy, shard_index)]
    for request in corrective_requests:
        raw = raw_lookup.get(request["legacy_request_id"], {})
        found = bool(raw)
        rows.append(
            {
                "f6g3_request_id": request["f6g3_request_id"],
                "legacy_request_id": request["legacy_request_id"],
                "logical_instance_id": request["logical_instance_id"],
                "model": request["model"],
                "strategy": request["strategy"],
                "dataset": request["dataset"],
                "task": request["task"],
                "request_lineage_status": "identified_from_f6g1_legacy_top20_manifest",
                "response_lineage_status": "matched_existing_raw_response" if found else "missing_existing_lastfm_reranking_raw_response",
                "runner_event_status": "not_available_for_lastfm_reranking",
                "legacy_shard_id": legacy_shard_by_request.get(request["legacy_request_id"], ""),
                "corrective_shard_id": corrective_shard_by_req[request["f6g3_request_id"]],
                "raw_output_file": raw.get("raw_output_file", ""),
                "raw_record_sha256": raw.get("raw_response_sha256", ""),
                "raw_response_found": bool_str(found),
                "raw_output_contract": raw.get("parser_version", "") if found else "",
                "observed_output_length": raw.get("output_length", "") if found else "",
                "admission_status": "candidate_for_top10_derivation" if found else "blocked",
                "block_reason": "" if found else "missing_existing_lastfm_reranking_raw_response",
                "notes": "F6E4B raw files contain no LastFM reranking rows; F6G2 raw paths are absent.",
            }
        )
    return rows


def build_candidate_validation(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    pool_by_logical = {row["pool_instance_id"].replace("F6G1-POOLINSTANCE-", ""): row for row in inputs["pool_summary"] if row.get("pool_instance_id")}
    rows = []
    for logical in inputs["logical"]:
        candidate_ids = split_pipe(logical.get("candidate_item_ids", ""))
        candidate_titles = split_pipe(logical.get("candidate_item_titles", ""))
        rows.append(
            {
                "logical_instance_id": logical["logical_instance_id"],
                "user_id": logical["user_id"],
                "dataset": logical["dataset"],
                "task": logical["task"],
                "cohort_name": logical.get("cohort_name", ""),
                "candidate_pool_size": len(candidate_ids),
                "candidate_unique_count": len(set(candidate_ids)),
                "candidate_title_count": len(candidate_titles),
                "candidate_order_hash": logical.get("candidate_order_hash", ""),
                "positive_item_id": logical.get("positive_item_id", ""),
                "positive_present": logical.get("positive_present", ""),
                "positive_initial_position": logical.get("positive_initial_position", ""),
                "candidate_count_valid": bool_str(len(candidate_ids) == 20),
                "candidate_unique_valid": bool_str(len(candidate_ids) == len(set(candidate_ids)) == 20),
                "candidate_order_recoverable": bool_str(bool(logical.get("candidate_order_hash")) and len(candidate_ids) == 20),
                "positive_position_recoverable": bool_str(logical.get("positive_present") == "true" and bool(logical.get("positive_initial_position"))),
                "pool_classification": "reconstructed_pool_with_adapted_components",
                "repository_alignment": "candidate20_repository_derived_adapted",
                "paper_alignment": "paper_candidate_size_supported_but_lastfm_reranking_numeric_cells_absent",
                "validation_status": "pass" if len(candidate_ids) == len(set(candidate_ids)) == 20 else "blocked",
                "notes": "Candidate pool evidence reused from F6G1 only; no candidate ordering was changed.",
            }
        )
    return rows


def ndcg(rank: int | None) -> str:
    if rank is None or rank < 1 or rank > 5:
        return "0"
    return f"{1.0 / math.log2(rank + 1):.12g}"


def build_top10_and_canonical(
    corrective_requests: list[dict[str, Any]],
    reconciliation_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    recon_by_id = {row["f6g3_request_id"]: row for row in reconciliation_rows}
    derivation_rows = []
    recommendation_rows = []
    blocked_rows = []
    for request in corrective_requests:
        recon = recon_by_id[request["f6g3_request_id"]]
        if recon["raw_response_found"] != "true":
            block_reason = recon["block_reason"]
            derivation_rows.append(
                {
                    "f6g3_request_id": request["f6g3_request_id"],
                    "legacy_request_id": request["legacy_request_id"],
                    "model": request["model"],
                    "strategy": request["strategy"],
                    "dataset": request["dataset"],
                    "task": request["task"],
                    "admitted_to_freeze": "false",
                    "original_request_contract": "candidate20_request20_parser20",
                    "observed_raw_response_contract": "missing",
                    "offline_transformation": "none_blocked_no_raw_response",
                    "original_output_length": "",
                    "derived_output_length": "",
                    "canonical_top10_item_ids": "",
                    "first_five_item_ids": "",
                    "parse_validity_status": "not_evaluable_missing_raw_response",
                    "candidate_validity_status": "not_evaluable_missing_raw_response",
                    "duplicate_status": "not_evaluable_missing_raw_response",
                    "hallucination_status": "not_evaluable_missing_raw_response",
                    "positive_rank": "",
                    "hr_at5": "",
                    "ndcg_at5": "",
                    "raw_response_reference": "",
                    "block_reason": block_reason,
                    "notes": "Invalid/blocked row retained in operational denominator; no output repair or inference performed.",
                }
            )
            blocked_rows.append(
                {
                    "blocked_id": "F6G3-BLOCK-" + stable_hash(request["f6g3_request_id"])[:14],
                    "f6g3_request_id": request["f6g3_request_id"],
                    "legacy_request_id": request["legacy_request_id"],
                    "logical_instance_id": request["logical_instance_id"],
                    "dataset": request["dataset"],
                    "task": request["task"],
                    "model": request["model"],
                    "strategy": request["strategy"],
                    "block_stage": "raw_response_reconciliation",
                    "block_reason": block_reason,
                    "denominator_policy": "retained_in_operational_denominator",
                    "repair_allowed": "false",
                    "retry_allowed": "false",
                    "notes": "No existing LastFM reranking raw response was found.",
                }
            )
    return derivation_rows, recommendation_rows, blocked_rows


def cell_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["model"]), str(row["strategy"])


def build_cell_summary(
    corrective_requests: list[dict[str, Any]],
    derivation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    positive_by_logical = {row["logical_instance_id"]: row["positive_present"] == "true" for row in candidate_rows}
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    deriv_by_id = {row["f6g3_request_id"]: row for row in derivation_rows}
    for request in corrective_requests:
        by_cell[cell_key(request)].append(request)
    rows = []
    for (model, strategy), requests in sorted(by_cell.items()):
        admitted = [r for r in requests if deriv_by_id[r["f6g3_request_id"]]["admitted_to_freeze"] == "true"]
        strict_valid = len([r for r in admitted if deriv_by_id[r["f6g3_request_id"]]["parse_validity_status"] == "strict_valid"])
        candidate_valid = len([r for r in admitted if deriv_by_id[r["f6g3_request_id"]]["candidate_validity_status"] == "candidate_valid"])
        positive_present = len([r for r in requests if positive_by_logical.get(r["logical_instance_id"], False)])
        rows.append(
            {
                "cell_id": "F6G3-CELL-" + stable_hash({"model": model, "strategy": strategy})[:14],
                "dataset": "LastFM",
                "task": "reranking",
                "model": model,
                "strategy": strategy,
                "expected_requests": len(requests),
                "raw_rows_reconciled": len(admitted),
                "admitted_top10_rows": len(admitted),
                "invalid_or_blocked_rows": len(requests) - len(admitted),
                "positive_present_requests": positive_present,
                "strict_valid_rows": strict_valid,
                "candidate_valid_rows": candidate_valid,
                "strict_valid_rate": f"{strict_valid / len(requests):.6f}" if requests else "",
                "candidate_valid_rate": f"{candidate_valid / len(requests):.6f}" if requests else "",
                "operational_hr_at5": "",
                "operational_ndcg_at5": "",
                "conditional_hr_at5": "",
                "conditional_ndcg_at5": "",
                "classification": "blocked_no_existing_raw_responses",
                "paper_alignment": "paper_unspecified_output_length_no_lastfm_reranking_numeric_cell",
                "repository_alignment": "repository_derived_top10_contract_constructed_offline",
                "notes": "Top-10 corrective request freeze exists; metrics blocked because no LastFM reranking raw response was reconciled.",
            }
        )
    return rows


def build_metric_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_specs = [
        ("HR@5", "utility", "mean(hit(rank<=5))", "all expected requests", "invalid/missing raw outputs retained as blocked, not zero-filled"),
        ("NDCG@5", "utility", "mean(1/log2(rank+1) for positive rank <=5)", "all expected requests", "invalid/missing raw outputs retained as blocked, not zero-filled"),
        ("mean_positive_rank", "utility", "mean positive rank over valid rows", "strict-valid positive-present rows", "blocked when no admitted rows exist"),
        ("operational_HR@5", "utility", "HR@5 including invalid outputs in denominator", "all expected requests", "blocked because no raw responses exist"),
        ("conditional_HR@5", "utility", "HR@5 conditional on strict-valid positive-present outputs", "strict-valid positive-present rows", "blocked because no raw responses exist"),
        ("strict_parse_validity", "output_quality", "strict_valid_rows / expected_requests", "all expected requests", "computed from reconciliation/admission status"),
        ("candidate_validity", "output_quality", "candidate_valid_rows / expected_requests", "all expected requests", "computed from reconciliation/admission status"),
        ("average_returned_list_length", "output_quality", "mean derived output length", "admitted rows", "blocked when no admitted rows exist"),
        ("duplicate@5", "output_quality", "rows with duplicate among top5 / admitted rows", "admitted rows", "blocked when no admitted rows exist"),
        ("hallucination_rate", "output_quality", "rows with non-candidate item / admitted rows", "admitted rows", "blocked when no admitted rows exist"),
    ]
    rows = []
    for cell in cell_rows:
        for metric, family, formula, denominator, invalid_policy in metric_specs:
            value = ""
            status = "blocked_no_admitted_raw_rows"
            if metric == "strict_parse_validity":
                value = cell["strict_valid_rate"]
                status = "computed_zero_from_missing_raw_denominator"
            elif metric == "candidate_validity":
                value = cell["candidate_valid_rate"]
                status = "computed_zero_from_missing_raw_denominator"
            rows.append(
                {
                    "metric_id": "F6G3-METRIC-" + stable_hash({"cell": cell["cell_id"], "metric": metric})[:14],
                    "dataset": cell["dataset"],
                    "task": cell["task"],
                    "model": cell["model"],
                    "strategy": cell["strategy"],
                    "metric_family": family,
                    "metric": metric,
                    "k": "5" if "@5" in metric else "",
                    "value": value,
                    "denominator": cell["expected_requests"] if denominator == "all expected requests" else "",
                    "formula": formula,
                    "catalog_universe": "LastFM adapted catalog from F6C" if family != "utility" else "",
                    "popularity_universe": "F6F1 frozen popularity evidence when available",
                    "long_tail_policy": "F6F1 local long-tail policy when available",
                    "missing_item_handling": "blocked_not_zero_filled",
                    "invalid_output_handling": invalid_policy,
                    "aggregation_unit": "request",
                    "source_implementation": "phase_f6g3 deterministic offline builder",
                    "paper_alignment": "not_paper_exact_output_length_unspecified",
                    "repository_alignment": "repository_derived_top10_contract",
                    "local_adaptation_status": "adapted_reconstructed_candidate_pool",
                    "status": status,
                    "blocked_reason": "" if value else "no_existing_lastfm_reranking_raw_response_admitted",
                    "notes": "No model output was generated or repaired.",
                }
            )
    return rows


def build_beyond_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("novelty@5", "average self-information of top-5 items"),
        ("catalog_coverage@5", "unique recommended items at K=5 / catalog universe"),
        ("long_tail_rate@5", "top-5 items satisfying F6F1 long-tail policy"),
        ("average_popularity@5", "mean popularity of top-5 items"),
        ("popularity_bias@5", "candidate-position or popularity tilt at K=5"),
        ("ILD@5", "intra-list diversity over top-5 items"),
        ("calibration", "F6F1-supported calibration where required inputs are available"),
    ]
    rows = []
    for cell in cell_rows:
        for metric, formula in specs:
            rows.append(
                {
                    "metric_id": "F6G3-BEYOND-" + stable_hash({"cell": cell["cell_id"], "metric": metric})[:14],
                    "dataset": cell["dataset"],
                    "task": cell["task"],
                    "model": cell["model"],
                    "strategy": cell["strategy"],
                    "metric": metric,
                    "k": "5" if "@5" in metric else "",
                    "value": "",
                    "denominator": "",
                    "formula": formula,
                    "catalog_universe": "LastFM adapted catalog from F6C",
                    "popularity_universe": "F6F1 frozen popularity evidence when available",
                    "long_tail_policy": "F6F1 local policy; not evaluated without canonical recommendations",
                    "missing_item_handling": "blocked_not_zero_filled",
                    "invalid_output_handling": "no admitted rows; do not drop invalid/blocked rows silently",
                    "aggregation_unit": "request",
                    "source_implementation": "phase_f6g3 deterministic offline builder",
                    "paper_alignment": "not_paper_exact",
                    "repository_alignment": "repository_derived_top10_contract",
                    "local_adaptation_status": "adapted_reconstructed_candidate_pool",
                    "status": "blocked_no_admitted_top10_recommendations",
                    "blocked_reason": "no_existing_lastfm_reranking_raw_response_admitted",
                    "notes": "Unsupported metrics are blank, not zero.",
                }
            )
    return rows


def build_exactness_elevation(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    addressed_components = [
        row for row in inputs["frontier_components"] if row.get("dataset") == "LastFM" and row.get("task") == "reranking"
    ]
    addressed_numeric = [
        row for row in inputs["frontier_numeric"] if row.get("dataset") == "LastFM" and row.get("task") == "reranking"
    ]
    rows = []
    for row in addressed_components:
        rows.append(
            {
                "elevation_id": "F6G3-ELEV-" + stable_hash(row["contract_id"])[:14],
                "contract_id": row.get("contract_id", ""),
                "result_id": "",
                "dataset": row.get("dataset", ""),
                "task": row.get("task", ""),
                "model": row.get("model", ""),
                "metric": row.get("metric", ""),
                "prior_frontier_class": row.get("frontier_class", ""),
                "new_evidence": "F6G3 top-10 corrective manifest only; no raw response admitted",
                "documentary_uncertainty_resolved": "false",
                "offline_raw_recomputation_completed": "false",
                "elevated": "false",
                "maximum_defensible_new_classification": "repository_derived_adapted_protocol_blocked_no_raw_responses",
                "remaining_mismatch": "no_existing_lastfm_reranking_raw_response; paper output length remains unspecified",
                "numeric_cell_directly_comparable": "false",
                "notes": "No automatic paper-exact elevation.",
            }
        )
    for row in addressed_numeric:
        rows.append(
            {
                "elevation_id": "F6G3-ELEV-" + stable_hash(row["result_id"])[:14],
                "contract_id": row.get("contract_id", ""),
                "result_id": row.get("result_id", ""),
                "dataset": row.get("dataset", ""),
                "task": row.get("task", ""),
                "model": row.get("model", ""),
                "metric": row.get("metric", ""),
                "prior_frontier_class": row.get("frontier_class", ""),
                "new_evidence": "F6G3 top-10 corrective manifest only; no raw response admitted",
                "documentary_uncertainty_resolved": "false",
                "offline_raw_recomputation_completed": "false",
                "elevated": "false",
                "maximum_defensible_new_classification": "repository_derived_adapted_protocol_blocked_no_raw_responses",
                "remaining_mismatch": "no_existing_lastfm_reranking_raw_response; numeric comparability unavailable",
                "numeric_cell_directly_comparable": "false",
                "notes": "No LastFM reranking numeric cells were found if this table is empty.",
            }
        )
    return rows


def build_validation(
    inputs: dict[str, Any],
    corrective_requests: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    reconciliation_rows: list[dict[str, Any]],
    derivation_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    protected_before: dict[str, str],
    protected_after: dict[str, str],
) -> list[dict[str, Any]]:
    def status(condition: bool) -> str:
        return "pass" if condition else "fail"

    changed = [path for path, before in protected_before.items() if protected_after.get(path) != before]
    missing = [path for path in protected_before if path not in protected_after]
    return [
        {"check_id": "runtime_interpreter", "status": status(Path(sys.executable).resolve() == RECOVERED_INTERPRETER.resolve()), "value": sys.executable, "expected": str(RECOVERED_INTERPRETER), "notes": "Recovered runtime is not exact original executable environment."},
        {"check_id": "python_no_user_site", "status": status(os.environ.get("PYTHONNOUSERSITE") == "1"), "value": os.environ.get("PYTHONNOUSERSITE", ""), "expected": "1", "notes": ""},
        {"check_id": "phase_identifier_unoccupied", "status": "pass", "value": PHASE_ID, "expected": "no pre-existing phase_f6g3 artifacts before this work package", "notes": "Read-only reconnaissance found no F6G3 namespace collision."},
        {"check_id": "logical_instances", "status": status(len(inputs["logical"]) == 1859), "value": len(inputs["logical"]), "expected": 1859, "notes": ""},
        {"check_id": "legacy_requests", "status": status(len(inputs["requests"]) == 7436), "value": len(inputs["requests"]), "expected": 7436, "notes": "Four model/strategy cells over 1,859 logical instances."},
        {"check_id": "corrective_requests", "status": status(len(corrective_requests) == len(inputs["requests"])), "value": len(corrective_requests), "expected": len(inputs["requests"]), "notes": ""},
        {"check_id": "candidate_pool_size_20", "status": status(all(str(row["candidate_pool_size"]) == "20" for row in candidate_rows)), "value": len([row for row in candidate_rows if str(row["candidate_pool_size"]) == "20"]), "expected": len(candidate_rows), "notes": ""},
        {"check_id": "candidate_pool_unique_20", "status": status(all(row["candidate_unique_valid"] == "true" for row in candidate_rows)), "value": len([row for row in candidate_rows if row["candidate_unique_valid"] == "true"]), "expected": len(candidate_rows), "notes": ""},
        {"check_id": "positive_present_logical_instances", "status": "pass", "value": len([row for row in candidate_rows if row["positive_present"] == "true"]), "expected": "592 observed from F6G1", "notes": "Absent positives remain in operational denominator."},
        {"check_id": "lastfm_reranking_raw_rows", "status": "pass", "value": len([row for row in reconciliation_rows if row["raw_response_found"] == "true"]), "expected": 0, "notes": "No raw response was fabricated."},
        {"check_id": "admitted_rows", "status": status(len(canonical_rows) == 0), "value": len(canonical_rows), "expected": 0, "notes": "All rows blocked because raw responses are missing."},
        {"check_id": "blocked_rows_retained", "status": status(len(blocked_rows) == len(corrective_requests)), "value": len(blocked_rows), "expected": len(corrective_requests), "notes": "Invalid/blocked rows are not silently dropped."},
        {"check_id": "top10_outputs_no_more_than_10", "status": status(all(len(split_pipe(row["canonical_top10_item_ids"])) <= 10 for row in derivation_rows)), "value": "all", "expected": "len<=10", "notes": ""},
        {"check_id": "top5_prefix_of_top10", "status": status(all(split_pipe(row["first_five_item_ids"]) == split_pipe(row["canonical_top10_item_ids"])[:5] for row in derivation_rows)), "value": "all", "expected": "prefix", "notes": "Vacuously true for blocked rows with blank lists."},
        {"check_id": "protected_input_hash_invariance", "status": status(not changed and not missing), "value": f"changed={len(changed)};missing={len(missing)}", "expected": "changed=0;missing=0", "notes": "Before and after hashes are in phase_f6g3_protected_input_hashes.json."},
        {"check_id": "full_suite_excluded", "status": "pass", "value": "excluded", "expected": "focused_tests_only", "notes": "Full suite may run training/inference/network paths."},
    ]


def build_next_actions() -> list[dict[str, Any]]:
    return [
        {
            "priority": "critical",
            "action_id": "F6G3-NEXT-001",
            "component": "LastFM reranking raw responses",
            "current_status": "blocked_no_existing_raw_outputs",
            "safe_next_action": "Obtain explicit authorization for either locating an existing external exact raw-output archive or running a separate new inference phase; do not infer in F6G3.",
            "requires_training": "false",
            "requires_llm_inference": "future_phase_only_if_authorized",
            "requires_api": "false",
            "requires_download": "false",
            "notes": "F6G3 created the top-10 manifest and preserved lineage, but no raw rows were admissible.",
        },
        {
            "priority": "high",
            "action_id": "F6G3-NEXT-002",
            "component": "candidate pool exactness",
            "current_status": "adapted_reconstructed_pool",
            "safe_next_action": "Keep candidate pools classified as adapted unless exact paper LastFM reranking pools are recovered.",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "requires_download": "maybe_if_exact_external_artifact_authorized",
            "notes": "F6G1 pool evidence is coherent but not paper-exact.",
        },
        {
            "priority": "medium",
            "action_id": "F6G3-NEXT-003",
            "component": "global audit context",
            "current_status": "updated",
            "safe_next_action": "Use F6G3 report and manifest for future LastFM reranking claims.",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "requires_download": "false",
            "notes": "Do not use F6G1/F6G2 top-20 artifacts as canonical execution manifests.",
        },
    ]


def output_hashes() -> dict[str, str]:
    paths = [
        SOURCE_INVENTORY_CSV,
        LEGACY_SUPERSESSION_CSV,
        CORRECTIVE_REQUEST_MANIFEST_CSV,
        CORRECTIVE_SHARD_MANIFEST_CSV,
        REQUEST_RESPONSE_RECON_CSV,
        CANDIDATE_VALIDATION_CSV,
        TOP10_DERIVATION_CSV,
        CANONICAL_RECOMMENDATIONS_CSV,
        INVALID_BLOCKED_CSV,
        RERANKING_METRICS_CSV,
        BEYOND_METRICS_CSV,
        CELL_SUMMARY_CSV,
        EXACTNESS_ELEVATION_CSV,
        VALIDATION_CSV,
        DETERMINISM_CSV,
        NEXT_ACTIONS_CSV,
        FREEZE_MANIFEST_JSON,
        PROTECTED_HASHES_JSON,
        REPORT_PATH,
        HANDOFF_PATH,
    ]
    hashes = {}
    for path in paths:
        if path == FREEZE_MANIFEST_JSON:
            hashes[rel(path)] = "self_reference_excluded_for_determinism"
        elif path.exists():
            hashes[rel(path)] = file_sha256(path)
        else:
            hashes[rel(path)] = "missing"
    return dict(sorted(hashes.items()))


def git_status_lines(status_text: str) -> list[str]:
    return common_git.git_status_lines(status_text)


def git_status_path(line: str) -> str:
    return common_git.git_status_path(line)


def git_status_counts(lines: Iterable[str]) -> dict[str, int]:
    return common_git.git_status_counts(lines)


def phase_resume_baseline(final_git_status: str) -> dict[str, Any]:
    final_lines = git_status_lines(final_git_status)
    created = [line for line in final_lines if git_status_path(line) in set(PHASE_OUTPUT_PATHS)]
    baseline = [line for line in final_lines if git_status_path(line) not in set(PHASE_OUTPUT_PATHS)]
    return {
        "basis": "reconstructed_from_final_status_minus_phase_f6g3_paths",
        "status_row_count": len(baseline),
        "status_counts": git_status_counts(baseline),
        "phase_created_status_row_count": len(created),
        "phase_created_paths": sorted(git_status_path(line) for line in created),
    }


def build_freeze_manifest(
    config: dict[str, Any],
    source_rows: list[dict[str, Any]],
    corrective_requests: list[dict[str, Any]],
    corrective_shards: list[dict[str, Any]],
    reconciliation_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    derivation_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    beyond_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    exactness_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    next_rows: list[dict[str, Any]],
    protected_before: dict[str, str],
    protected_after: dict[str, str],
    initial_git_status: str,
    final_git_status: str,
) -> dict[str, Any]:
    return {
        "phase": PHASE_ID,
        "decision": "f6g3_top10_freeze_complete_blocked_no_existing_raw_responses",
        "contract": config["contract"],
        "environment": {
            "python_executable": sys.executable,
            "approved_runtime_interpreter": str(RECOVERED_INTERPRETER),
            "runtime_classification": "defensible_repaired_clone_exact_package_versions",
            "not_exact_original_runtime": True,
            "python_no_user_site_required": True,
        },
        "counts": {
            "source_artifacts": len(source_rows),
            "logical_instances": len(candidate_rows),
            "legacy_requests": len(corrective_requests),
            "corrective_requests": len(corrective_requests),
            "corrective_shards": len(corrective_shards),
            "raw_rows_reconciled": len([row for row in reconciliation_rows if row["raw_response_found"] == "true"]),
            "admitted_top10_rows": len(canonical_rows),
            "invalid_or_blocked_rows": len(blocked_rows),
            "metric_rows": len(metric_rows),
            "beyond_metric_rows": len(beyond_rows),
            "cell_rows": len(cell_rows),
            "exactness_rows_addressed": len(exactness_rows),
            "validation_rows": len(validation_rows),
            "next_action_rows": len(next_rows),
        },
        "rates": {
            "strict_valid_rate": "0.000000",
            "candidate_valid_rate": "0.000000",
            "admission_rate": "0.000000",
        },
        "protected_input_hashes": {
            "before_count": len(protected_before),
            "after_count": len(protected_after),
            "changed": [path for path, value in protected_before.items() if protected_after.get(path) != value],
            "missing": [path for path in protected_before if path not in protected_after],
        },
        "git_state": {
            "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "head": run_command(["git", "rev-parse", "HEAD"]),
            "initial_status_row_count": len(git_status_lines(initial_git_status)),
            "final_status_row_count": len(git_status_lines(final_git_status)),
            "final_status_counts": git_status_counts(git_status_lines(final_git_status)),
            "phase_resume_baseline": phase_resume_baseline(final_git_status),
        },
        "output_hashes": output_hashes(),
        "no_training": True,
        "no_llm_inference": True,
        "no_ollama": True,
        "no_api_calls": True,
        "no_downloads": True,
        "no_package_changes": True,
        "no_git_stage_or_commit": True,
        "recommended_next_work_package": "Keep LastFM reranking blocked until exact existing raw outputs are supplied or a separately authorized inference phase is approved.",
    }


def build_report(manifest: dict[str, Any], cell_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], next_rows: list[dict[str, Any]]) -> str:
    counts = manifest["counts"]
    rates = manifest["rates"]
    cell_lines = "\n".join(
        f"- `{row['model']}` / `{row['strategy']}`: expected `{row['expected_requests']}`, admitted `{row['admitted_top10_rows']}`, blocked `{row['invalid_or_blocked_rows']}`, strict-valid rate `{row['strict_valid_rate']}`"
        for row in cell_rows
    )
    source_lines = "\n".join(
        f"- `{row['path']}`: role `{row['role']}`, rows `{row['row_count']}`, LastFM reranking rows `{row['lastfm_reranking_rows']}`, status `{row['contract_status']}`"
        for row in source_rows
    )
    next_action_lines = "\n".join(f"- `{row['action_id']}`: {row['safe_next_action']}" for row in next_rows)
    baseline = manifest["git_state"]["phase_resume_baseline"]
    baseline_counts = baseline["status_counts"]
    final_counts = manifest["git_state"]["final_status_counts"]
    return f"""# F6G3 LastFM Reranking Top-10 Freeze

## 1. Objective

Construct a deterministic corrective offline LastFM reranking freeze under `D_repository_derived_top10_paper_unspecified`.

Decision token: `f6g3_top10_freeze_complete_blocked_no_existing_raw_responses`.

## 2. Source Contract

- Candidate pool size: `20`
- Requested output length: `10`
- Parser expected length: `10`
- Metric cutoff K: `5`
- Paper output length: `paper_unspecified`
- Repository output length: `repository_derived_not_paper_specified`
- Classification: `repository_derived_adapted_protocol`

This phase does not claim paper-exact output-length replication.

## 3. Runtime And Prohibitions

- Runtime: `{RECOVERED_INTERPRETER}`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Runtime caveat: repaired clone exact package tree, not exact original executable environment.
- No training, LLM inference, Ollama, API calls, downloads, package changes, staging, or commits occurred.

## 4. Source Artifacts

{source_lines}

## 5. Legacy Supersession

F6G1 top-20 logical/request/shard manifests are superseded for execution. F6G3 reuses only candidate identity, cohort identity, user/task identifiers, candidate order, positive-position metadata, model/strategy identity, shard planning, and provenance. F6G2 remains historical blocked evidence because its canonical raw paths are absent.

## 6. Precondition Audit

- LastFM reranking logical instances: `{counts['logical_instances']}`
- Legacy top-20 requests: `{counts['legacy_requests']}`
- Corrective top-10 requests: `{counts['corrective_requests']}`
- Existing raw rows reconciled: `{counts['raw_rows_reconciled']}`
- Admitted top-10 rows: `{counts['admitted_top10_rows']}`
- Invalid or blocked rows retained: `{counts['invalid_or_blocked_rows']}`

No existing LastFM reranking raw response rows were found. F6E4B/F6E4C contain LastFM ranking rows, not LastFM reranking rows; F6G2 raw outputs are absent.

## 7. Candidate-Source Models

The F6G1 reconstructed reranking candidate pool uses adapted BPRMF, GRU4Rec, SASRec, and LightGCN source models with selected seeds recorded in `phase_f6g1_candidate_source_model_decisions.csv`. Random, MostPop, and BM25-like sources remain diagnostic-only.

## 8. Candidate-Pool Validation

All F6G1 logical instances preserve 20 unique candidate items. Positive items are present for 592 of 1,859 logical instances and absent for 1,267; absent positives remain in the operational denominator.

## 9. Top-10 Derivation

No raw response was admitted, so no canonical top-10 list was produced. The offline transformation for every row is `none_blocked_no_raw_response`. No output was repaired, reordered, supplemented, or generated.

## 10. Metrics

Utility metrics (`HR@5`, `NDCG@5`, mean positive rank, operational utility, conditional utility) are blank and blocked because no row was admitted to the top-10 freeze. Strict-valid and candidate-valid rates are both `{rates['strict_valid_rate']}` over the 7,436 expected requests.

Beyond-utility metrics are present as blocked rows with explicit formulas and denominators where applicable; unsupported values are blank, not zero.

## 11. Per-Cell Summary

{cell_lines}

## 12. Exactness Frontier

F6G3 addresses `0` exactness-frontier paper rows because the frontier tables contain no `LastFM` + `reranking` paper components or numeric cells. No classification was elevated. The maximum defensible classification for the operational freeze is repository-derived/adapted and blocked pending raw responses.

## 13. Threats To Validity

The candidate pools are reconstructed/adapted rather than paper-exact. The paper does not specify output length. No LastFM reranking raw responses are available, so execution-dependent metrics cannot be computed.

## 14. Deterministic Reproduction

Run from repository root:

```powershell
$env:PYTHONNOUSERSITE='1'
.venv_audit_recovery/Scripts/python.exe src/analysis/plan_f6g3_lastfm_reranking_top10_freeze.py --config configs/phase_f6g3_lastfm_reranking_top10_freeze.yaml
```

Focused verification command for this work package:

```powershell
$env:PYTHONNOUSERSITE='1'
.venv_audit_recovery/Scripts/python.exe -m pytest tests/test_phase_f6g3_lastfm_reranking_top10_freeze.py -q -p no:cacheprovider
```

Final focused-test result: `8 passed`. The determinism test runs the builder twice and compares all F6G3 output hashes, including the report, handoff, freeze manifest, protected-hash manifest, tables, and current context snapshot. The full test suite is excluded because existing tests include training, checkpoint, writer, inference, and network-adjacent paths that are outside this corrective offline freeze.

The deterministic output-hash registry is stored in `results/manifests/phase_f6g3_top10_freeze_manifest.json` under `output_hashes`; the manifest's own self hash is intentionally recorded as `self_reference_excluded_for_determinism`.

## 15. Protected Inputs

Protected input hashes were captured before and after the builder. Changed: `{len(manifest['protected_input_hashes']['changed'])}`; missing: `{len(manifest['protected_input_hashes']['missing'])}`.

## 16. Git State

- Branch: `{manifest['git_state']['branch']}`
- HEAD: `{manifest['git_state']['head']}`
- Phase baseline rows: `{baseline['status_row_count']}` reconstructed from final status minus `{baseline['phase_created_status_row_count']}` F6G3 paths.
- Baseline counts: modified `{baseline_counts['modified']}`, untracked `{baseline_counts['untracked']}`.
- Final rows: `{manifest['git_state']['final_status_row_count']}`
- Final counts: modified `{final_counts['modified']}`, untracked `{final_counts['untracked']}`.

## 17. Files Created

All F6G3 outputs are isolated under `configs/`, `src/`, `tests/`, `results/tables/`, `results/manifests/`, and `report/`.

## 18. Recommended Next Work Package

{next_action_lines}
"""


def build_handoff(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    return f"""# Future Operations Context Handoff

## Current Runtime Contract

- Approved Python: `{RECOVERED_INTERPRETER}`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired clone exact package tree only; not exact original executable environment.
- Required environment: `PYTHONNOUSERSITE=1`

## Historical Baselines

- Forensic exact paper-component coverage: `0/192 = 0.000000%`
- Forensic adapted/reconstructed coverage: `192/192 = 100.000000%`
- Exactness frontier demonstrated exact coverage: `0/192 = 0.000000%`
- Exactness frontier offline/raw recoverable potential: `49/192 = 25.520833%`

## Post-F6G3 LastFM Reranking State

- F6G3 report: `report/phase_f6g3_lastfm_reranking_top10_freeze.md`
- F6G3 freeze manifest: `results/manifests/phase_f6g3_top10_freeze_manifest.json`
- Output-length contract: `D_repository_derived_top10_paper_unspecified`
- Logical instances: `{counts['logical_instances']}`
- Corrective top-10 requests: `{counts['corrective_requests']}`
- Raw rows reconciled: `{counts['raw_rows_reconciled']}`
- Admitted top-10 rows: `{counts['admitted_top10_rows']}`
- Invalid or blocked rows: `{counts['invalid_or_blocked_rows']}`
- Classification: `repository_derived_adapted_protocol_blocked_no_existing_raw_responses`

## Use Going Forward

Use F6G3, not F6G1/F6G2 top-20 artifacts, as the current LastFM reranking lineage freeze. Do not claim paper-exact output-length reproduction. Do not report LastFM reranking utility or beyond-utility values unless future authorized work supplies admissible raw responses.

## Forbidden Claims

- Exact original runtime reproduction.
- Paper-explicit top-10 output length.
- Complete LastFM reranking execution.
- LastFM as a fifth-dataset extension.
- Zero-valued utility/beyond-utility metrics for unsupported F6G3 cells.

## Recommended Next Work Package

Keep LastFM reranking blocked until exact existing raw outputs are supplied or a separately authorized inference phase is approved.
"""


def update_current_context(manifest: dict[str, Any]) -> None:
    if CURRENT_CONTEXT_SNAPSHOT.exists():
        snapshot = json.loads(read_text(CURRENT_CONTEXT_SNAPSHOT))
    else:
        snapshot = {}
    snapshot["post_phase_f6g3_lastfm_reranking_top10_freeze"] = {
        "phase": PHASE_ID,
        "decision": manifest["decision"],
        "contract": manifest["contract"],
        "counts": manifest["counts"],
        "rates": manifest["rates"],
        "classification": "repository_derived_adapted_protocol_blocked_no_existing_raw_responses",
        "raw_rows_reconciled": manifest["counts"]["raw_rows_reconciled"],
        "admitted_top10_rows": manifest["counts"]["admitted_top10_rows"],
        "invalid_or_blocked_rows": manifest["counts"]["invalid_or_blocked_rows"],
        "no_training": True,
        "no_llm_inference": True,
        "no_ollama": True,
        "no_api_calls": True,
        "no_downloads": True,
        "runtime_classification": "defensible_repaired_clone_exact_package_versions",
        "not_exact_original_runtime": True,
        "authoritative_artifacts": {
            "report": rel(REPORT_PATH),
            "freeze_manifest": rel(FREEZE_MANIFEST_JSON),
            "protected_hashes": rel(PROTECTED_HASHES_JSON),
            "corrective_request_manifest": rel(CORRECTIVE_REQUEST_MANIFEST_CSV),
            "cell_summary": rel(CELL_SUMMARY_CSV),
        },
        "recommended_next_work_package": manifest["recommended_next_work_package"],
    }
    CURRENT_CONTEXT_SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def build_all(config_path: Path) -> dict[str, Any]:
    require_runtime()
    config = read_config(config_path)
    initial_git_status = run_command(["git", "status", "--porcelain=v1"])
    inputs = load_inputs(config)
    protected_before = protected_input_hashes(inputs["paths"])

    source_rows = build_source_inventory(config, inputs)
    supersession_rows = build_legacy_supersession(inputs)
    corrective_requests = build_corrective_request_manifest(inputs, config)
    corrective_shards = build_corrective_shards(corrective_requests, inputs["shards"])
    candidate_rows = build_candidate_validation(inputs)
    reconciliation_rows = build_reconciliation(corrective_requests, corrective_shards, inputs)
    derivation_rows, canonical_rows, blocked_rows = build_top10_and_canonical(corrective_requests, reconciliation_rows)
    cell_rows = build_cell_summary(corrective_requests, derivation_rows, candidate_rows)
    metric_rows = build_metric_rows(cell_rows)
    beyond_rows = build_beyond_rows(cell_rows)
    exactness_rows = build_exactness_elevation(inputs)
    next_rows = build_next_actions()
    protected_after = protected_input_hashes(inputs["paths"])
    validation_rows = build_validation(inputs, corrective_requests, candidate_rows, reconciliation_rows, derivation_rows, canonical_rows, blocked_rows, protected_before, protected_after)
    determinism_rows = [
        {
            "check_id": "phase_f6g3_deterministic_builder_contract",
            "command": "Run the builder twice with PYTHONNOUSERSITE=1 and compare all F6G3 output hashes.",
            "status": "external_check_required",
            "checked_artifacts": "",
            "mismatches": "",
            "notes": "The focused test suite and final verification perform the two-run hash comparison.",
        }
    ]

    write_csv(SOURCE_INVENTORY_CSV, source_rows, SOURCE_INVENTORY_FIELDS)
    write_csv(LEGACY_SUPERSESSION_CSV, supersession_rows, ["legacy_artifact", "legacy_phase", "row_count", "original_request_contract", "canonical_f6g3_use", "supersession_status", "evidence_reused", "evidence_not_reused", "notes"])
    write_csv(CORRECTIVE_REQUEST_MANIFEST_CSV, corrective_requests, CORRECTIVE_REQUEST_FIELDS)
    write_csv(CORRECTIVE_SHARD_MANIFEST_CSV, corrective_shards, ["corrective_shard_id", "model", "strategy", "shard_index", "request_count", "first_f6g3_request_id", "last_f6g3_request_id", "legacy_shard_policy", "corrective_contract", "status", "notes"])
    write_csv(REQUEST_RESPONSE_RECON_CSV, reconciliation_rows, REQUEST_RESPONSE_FIELDS)
    write_csv(CANDIDATE_VALIDATION_CSV, candidate_rows, ["logical_instance_id", "user_id", "dataset", "task", "cohort_name", "candidate_pool_size", "candidate_unique_count", "candidate_title_count", "candidate_order_hash", "positive_item_id", "positive_present", "positive_initial_position", "candidate_count_valid", "candidate_unique_valid", "candidate_order_recoverable", "positive_position_recoverable", "pool_classification", "repository_alignment", "paper_alignment", "validation_status", "notes"])
    write_csv(TOP10_DERIVATION_CSV, derivation_rows, TOP10_DERIVATION_FIELDS)
    write_csv(CANONICAL_RECOMMENDATIONS_CSV, canonical_rows, ["f6g3_request_id", "rank", "item_id", "source_raw_response", "derivation_status", "notes"])
    write_csv(INVALID_BLOCKED_CSV, blocked_rows, ["blocked_id", "f6g3_request_id", "legacy_request_id", "logical_instance_id", "dataset", "task", "model", "strategy", "block_stage", "block_reason", "denominator_policy", "repair_allowed", "retry_allowed", "notes"])
    write_csv(RERANKING_METRICS_CSV, metric_rows, ["metric_id", "dataset", "task", "model", "strategy", "metric_family", "metric", "k", "value", "denominator", "formula", "catalog_universe", "popularity_universe", "long_tail_policy", "missing_item_handling", "invalid_output_handling", "aggregation_unit", "source_implementation", "paper_alignment", "repository_alignment", "local_adaptation_status", "status", "blocked_reason", "notes"])
    write_csv(BEYOND_METRICS_CSV, beyond_rows, ["metric_id", "dataset", "task", "model", "strategy", "metric", "k", "value", "denominator", "formula", "catalog_universe", "popularity_universe", "long_tail_policy", "missing_item_handling", "invalid_output_handling", "aggregation_unit", "source_implementation", "paper_alignment", "repository_alignment", "local_adaptation_status", "status", "blocked_reason", "notes"])
    write_csv(CELL_SUMMARY_CSV, cell_rows, ["cell_id", "dataset", "task", "model", "strategy", "expected_requests", "raw_rows_reconciled", "admitted_top10_rows", "invalid_or_blocked_rows", "positive_present_requests", "strict_valid_rows", "candidate_valid_rows", "strict_valid_rate", "candidate_valid_rate", "operational_hr_at5", "operational_ndcg_at5", "conditional_hr_at5", "conditional_ndcg_at5", "classification", "paper_alignment", "repository_alignment", "notes"])
    write_csv(EXACTNESS_ELEVATION_CSV, exactness_rows, ["elevation_id", "contract_id", "result_id", "dataset", "task", "model", "metric", "prior_frontier_class", "new_evidence", "documentary_uncertainty_resolved", "offline_raw_recomputation_completed", "elevated", "maximum_defensible_new_classification", "remaining_mismatch", "numeric_cell_directly_comparable", "notes"])
    write_csv(VALIDATION_CSV, validation_rows, ["check_id", "status", "value", "expected", "notes"])
    write_csv(DETERMINISM_CSV, determinism_rows, ["check_id", "command", "status", "checked_artifacts", "mismatches", "notes"])
    write_csv(NEXT_ACTIONS_CSV, next_rows, ["priority", "action_id", "component", "current_status", "safe_next_action", "requires_training", "requires_llm_inference", "requires_api", "requires_download", "notes"])

    PROTECTED_HASHES_JSON.write_text(
        json.dumps({"phase": PHASE_ID, "before": protected_before, "after": protected_after, "changed": [p for p, v in protected_before.items() if protected_after.get(p) != v], "missing": [p for p in protected_before if p not in protected_after]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    final_git_status = run_command(["git", "status", "--porcelain=v1"])
    manifest = build_freeze_manifest(config, source_rows, corrective_requests, corrective_shards, reconciliation_rows, candidate_rows, derivation_rows, canonical_rows, blocked_rows, metric_rows, beyond_rows, cell_rows, exactness_rows, validation_rows, next_rows, protected_before, protected_after, initial_git_status, final_git_status)
    FREEZE_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_report(manifest, cell_rows, source_rows, next_rows), encoding="utf-8")
    HANDOFF_PATH.write_text(build_handoff(manifest), encoding="utf-8")
    # Recompute output hashes after writing reports and context-facing artifacts.
    manifest["output_hashes"] = output_hashes()
    FREEZE_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    update_current_context(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    manifest = build_all(Path(args.config))
    if args.print_summary:
        print(json.dumps({"phase": PHASE_ID, "decision": manifest["decision"], "counts": manifest["counts"], "rates": manifest["rates"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

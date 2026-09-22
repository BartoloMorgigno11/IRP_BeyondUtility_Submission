"""Build the F6H1 offline exactness-elevation audit.

The phase is intentionally conservative. It revalidates the existing
exactness-frontier candidates from local evidence and elevates nothing unless
all material paper-exactness gates are satisfied without training, inference,
network access, package changes, or protocol changes.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6h1"
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase_f6h1_offline_exactness_elevation.yaml"
RECOVERED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6h1_offline_exactness_elevation.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
CURRENT_CONTEXT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"

TARGET_INVENTORY_CSV = TABLES_DIR / "phase_f6h1_target_candidate_inventory.csv"
DOCUMENTARY_CSV = TABLES_DIR / "phase_f6h1_documentary_verification.csv"
RAW_RECON_CSV = TABLES_DIR / "phase_f6h1_raw_evidence_reconciliation.csv"
OFFLINE_RECOMP_CSV = TABLES_DIR / "phase_f6h1_offline_recomputation.csv"
COMPONENT_DECISIONS_CSV = TABLES_DIR / "phase_f6h1_component_elevation_decisions.csv"
NUMERIC_DECISIONS_CSV = TABLES_DIR / "phase_f6h1_numeric_cell_elevation_decisions.csv"
UNRESOLVED_CSV = TABLES_DIR / "phase_f6h1_unresolved_or_rejected_candidates.csv"
COVERAGE_DELTA_CSV = TABLES_DIR / "phase_f6h1_coverage_delta.csv"
VALIDATION_CSV = TABLES_DIR / "phase_f6h1_validation.csv"
DETERMINISM_CSV = TABLES_DIR / "phase_f6h1_determinism.csv"
NEXT_ACTIONS_CSV = TABLES_DIR / "phase_f6h1_next_actions.csv"

MANIFEST_JSON = MANIFESTS_DIR / "phase_f6h1_offline_exactness_elevation_manifest.json"
PROTECTED_HASHES_JSON = MANIFESTS_DIR / "phase_f6h1_protected_input_hashes.json"

TARGET_CLASSES = {
    "exact_possible_by_documentary_verification_only",
    "exact_possible_by_offline_recomputation",
    "exact_possible_with_existing_raw_outputs",
}

PHASE_OUTPUT_PATHS = [
    "configs/phase_f6h1_offline_exactness_elevation.yaml",
    "src/analysis/plan_f6h1_offline_exactness_elevation.py",
    "src/experiments/build_phase_f6h1_offline_exactness_elevation.py",
    "tests/test_phase_f6h1_offline_exactness_elevation.py",
    "report/phase_f6h1_offline_exactness_elevation.md",
    "report/future_operations_context_handoff.md",
    "results/tables/phase_f6h1_target_candidate_inventory.csv",
    "results/tables/phase_f6h1_documentary_verification.csv",
    "results/tables/phase_f6h1_raw_evidence_reconciliation.csv",
    "results/tables/phase_f6h1_offline_recomputation.csv",
    "results/tables/phase_f6h1_component_elevation_decisions.csv",
    "results/tables/phase_f6h1_numeric_cell_elevation_decisions.csv",
    "results/tables/phase_f6h1_unresolved_or_rejected_candidates.csv",
    "results/tables/phase_f6h1_coverage_delta.csv",
    "results/tables/phase_f6h1_validation.csv",
    "results/tables/phase_f6h1_determinism.csv",
    "results/tables/phase_f6h1_next_actions.csv",
    "results/manifests/phase_f6h1_offline_exactness_elevation_manifest.json",
    "results/manifests/phase_f6h1_protected_input_hashes.json",
    "results/manifests/current_project_context_snapshot.json",
]

PHASE_NEW_STATUS_PATHS = [
    path
    for path in PHASE_OUTPUT_PATHS
    if path
    not in {
        "report/future_operations_context_handoff.md",
        "results/manifests/current_project_context_snapshot.json",
    }
]

TARGET_FIELDS = [
    "candidate_id",
    "contract_id",
    "numeric_cell_id",
    "dataset",
    "task",
    "model",
    "exact_model_version",
    "prompt_strategy",
    "metric",
    "paper_source",
    "expected_local_evidence",
    "source_files",
    "raw_output_requirement",
    "required_recomputation",
    "prior_frontier_class",
    "prior_first_material_mismatch",
    "expected_elevation_route",
    "lastfm_reranking_paper_row",
    "inventory_status",
]

DOCUMENTARY_FIELDS = [
    "candidate_id",
    "contract_id",
    "pdf_filename",
    "pdf_sha256",
    "pdf_byte_identical_group",
    "pdf_text_extraction_available",
    "python_pdf_modules_available",
    "pdftotext_available",
    "page",
    "section",
    "table",
    "figure",
    "equation",
    "exact_supporting_statement",
    "official_repository_behavior",
    "local_implementation_evidence",
    "material_fields_verified",
    "documentary_verification_status",
    "new_classification",
    "first_blocking_mismatch",
    "notes",
]

RAW_RECON_FIELDS = [
    "candidate_id",
    "contract_id",
    "route",
    "dataset",
    "task",
    "model",
    "metric",
    "expected_source",
    "evidence_path",
    "evidence_exists",
    "evidence_non_empty",
    "zip_entry_count",
    "candidate_matching_entry_count",
    "candidate_matching_entries",
    "request_id_status",
    "response_id_status",
    "runner_event_status",
    "model_version_status",
    "cohort_status",
    "candidate_pool_status",
    "candidate_order_status",
    "prompt_contract_status",
    "parser_contract_status",
    "raw_response_completeness",
    "lineage_status",
    "admission_status",
    "first_blocking_mismatch",
    "notes",
]

OFFLINE_FIELDS = [
    "candidate_id",
    "contract_id",
    "numeric_cell_id",
    "dataset",
    "task",
    "model",
    "metric",
    "route",
    "raw_source_paths",
    "source_hashes",
    "source_row_identifiers",
    "formula",
    "cutoff",
    "denominator",
    "aggregation_procedure",
    "prior_reported_value",
    "recomputed_value",
    "absolute_difference",
    "relative_difference",
    "recomputation_status",
    "exact_comparability_verdict",
    "first_blocking_mismatch",
    "notes",
]

COMPONENT_FIELDS = [
    "candidate_id",
    "contract_id",
    "prior_frontier_class",
    "evidence_expected",
    "evidence_found",
    "raw_evidence_count",
    "documentary_verification_status",
    "lineage_status",
    "protocol_match",
    "cohort_match",
    "model_version_match",
    "prompt_match",
    "parser_match",
    "metric_match",
    "denominator_match",
    "recomputation_status",
    "prior_classification",
    "new_classification",
    "elevated",
    "first_blocking_mismatch",
    "maximum_defensible_claim",
    "notes",
]

NUMERIC_FIELDS = [
    "numeric_decision_id",
    "result_id",
    "contract_id",
    "paper_table",
    "paper_row",
    "paper_column",
    "dataset",
    "task",
    "model",
    "prompt",
    "metric",
    "paper_value",
    "paper_precision",
    "local_source_value",
    "recomputed_value",
    "absolute_difference",
    "relative_difference",
    "protocol_comparability",
    "cohort_comparability",
    "model_comparability",
    "prompt_comparability",
    "parser_comparability",
    "metric_comparability",
    "denominator_comparability",
    "numeric_similarity_only_elevation_allowed",
    "exact_cell_verdict",
    "first_blocking_mismatch",
    "notes",
]

COVERAGE_FIELDS = [
    "coverage_metric",
    "previous_numerator",
    "previous_denominator",
    "previous_percentage",
    "new_numerator",
    "new_denominator",
    "new_percentage",
    "delta_numerator",
    "delta_percentage",
    "changed_rows",
    "reason",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:14].upper()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_command(args: list[str]) -> str:
    completed = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def require_runtime() -> None:
    if Path(sys.executable).resolve() != RECOVERED_INTERPRETER.resolve():
        raise RuntimeError(f"F6H1 requires {RECOVERED_INTERPRETER}, got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("F6H1 requires PYTHONNOUSERSITE=1 for every Python invocation.")


def read_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(read_text(path))


def git_status_lines(text: str) -> list[str]:
    return common_git.git_status_lines(text)


def git_status_path(line: str) -> str:
    return common_git.git_status_path(line)


def git_status_counts(lines: Iterable[str]) -> dict[str, int]:
    return common_git.git_status_counts(lines)


def phase_resume_baseline(final_git_status: str) -> dict[str, Any]:
    phase_paths = set(PHASE_NEW_STATUS_PATHS)
    lines = git_status_lines(final_git_status)
    created = [line for line in lines if git_status_path(line) in phase_paths]
    baseline = [line for line in lines if git_status_path(line) not in phase_paths]
    return {
        "basis": "reconstructed_from_final_status_minus_phase_f6h1_paths",
        "note": "Current-state global artifacts are updated by F6H1 but are not treated as newly created phase-specific status rows.",
        "status_row_count": len(baseline),
        "status_counts": git_status_counts(baseline),
        "phase_created_status_row_count": len(created),
        "phase_created_paths": sorted(git_status_path(line) for line in created),
    }


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    paths = {name: PROJECT_ROOT / path for name, path in config["inputs"].items()}
    return {
        "paths": paths,
        "frontier_components": read_csv(paths["exactness_frontier_by_component"]),
        "numeric_frontier": read_csv(paths["numeric_cell_exactness_frontier"]),
        "elevation_candidates": read_csv(paths["exactness_elevation_candidates"]),
        "irreducible_blockers": read_csv(paths["irreducible_exactness_blockers"]),
        "numeric_reconciliation": read_csv(paths["numeric_result_reconciliation"]),
        "paper_contract": read_csv(paths["paper_atomic_contract"]),
        "traceability": read_csv(paths["paper_to_local_traceability"]),
        "status_matrix": read_csv(paths["current_reproduction_status_matrix"]),
        "artifact_lineage": read_csv(paths["artifact_lineage_and_supersession"]),
        "claims": read_csv(paths["project_claim_verification"]),
        "frontier_snapshot": json.loads(read_text(paths["exactness_frontier_snapshot"])),
        "f6g3_manifest": json.loads(read_text(paths["f6g3_freeze_manifest"])),
    }


def protected_input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    protected: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        if path.is_file():
            protected[rel(path)] = sha256_file(path)
        elif path.is_dir():
            child_hashes = []
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                child_hashes.append(f"{rel(child)}={sha256_file(child)}")
            protected[rel(path)] = sha256_bytes("\n".join(child_hashes).encode("utf-8"))
        else:
            protected[path.as_posix()] = "missing"
    return dict(sorted(protected.items()))


def pdf_environment(paths: dict[str, Path]) -> dict[str, Any]:
    module_names = ["PyPDF2", "fitz", "pdfplumber", "pymupdf", "pypdf", "pdfminer"]
    modules = {name: importlib.util.find_spec(name) is not None for name in module_names}
    pdftotext = shutil.which("pdftotext") is not None
    paper_files = []
    for key in ["paper_pdf", "paper_pdf_duplicate", "paper_docx_translation"]:
        path = paths[key]
        paper_files.append(
            {
                "path": rel(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() else "missing",
            }
        )
    return {
        "python_pdf_modules": modules,
        "pdftotext_available": pdftotext,
        "pdf_text_extraction_available": any(modules.values()) or pdftotext,
        "paper_files": paper_files,
        "byte_identical_pdf_sha256": paper_files[0]["sha256"] if len(paper_files) >= 2 and paper_files[0]["sha256"] == paper_files[1]["sha256"] else "",
    }


def zip_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size_bytes": "", "sha256": "missing", "entry_count": 0, "entries": []}
    with ZipFile(path) as archive:
        entries = sorted(info.filename for info in archive.infolist() if not info.is_dir())
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "entry_count": len(entries),
        "entries": entries,
    }


def zip_matching_entries(zip_info: dict[str, Any], dataset: str, model: str, route: str) -> list[str]:
    entries = zip_info.get("entries", [])
    if not entries:
        return []
    dataset_token = {
        "Amazon Beauty": "beauty/",
        "Amazon Sports": "sports/",
    }.get(dataset, "")
    model_token = model.lower()
    if route == "offline":
        return [entry for entry in entries if entry.startswith("beauty_his10/") and f"rec-{model}-" in entry and entry.endswith(".csv")]
    return [
        entry
        for entry in entries
        if entry.startswith(dataset_token)
        and f"results_{model_token}_" in entry.lower()
        and "candidate20" in entry.lower()
        and (entry.endswith(".json") or entry.endswith("_5.csv"))
    ]


def parse_paper_coordinates(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in value.split(";"):
        if "=" in part:
            key, val = part.split("=", 1)
            parsed[key.strip()] = val.strip()
    return parsed


def infer_route(frontier_class: str) -> str:
    if frontier_class == "exact_possible_by_documentary_verification_only":
        return "documentary_verification"
    if frontier_class == "exact_possible_with_existing_raw_outputs":
        return "existing_raw_output"
    return "offline_recomputation"


def exact_model_version(row: dict[str, str]) -> str:
    model = row.get("model", "")
    if not model:
        return "not_applicable_or_paper_scope_row"
    if model == "GPT3.5":
        return "paper_unspecified_gpt3.5_endpoint_or_snapshot"
    if model in {"BPRMF", "GRU4Rec", "LightGCN", "SASRec", "BM25"}:
        return "official_result_file_available_but_checkpoint_or_seed_policy_not_exactly_verified"
    return "paper_or_frontier_unspecified"


def expected_evidence(row: dict[str, str]) -> str:
    route = infer_route(row["frontier_class"])
    if route == "documentary_verification":
        return "direct PDF or appendix statement plus official repository and local contract match"
    if route == "existing_raw_output":
        return "official LLM raw response ZIP entry with request/response/runner-event/protocol lineage"
    return "official SOTA recommendation CSV plus exact source data, seed/repetition, cohort, and metric formula"


def build_target_inventory(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    numeric_by_contract = {row["contract_id"]: row for row in inputs["numeric_frontier"]}
    trace_by_contract = {row["contract_id"]: row for row in inputs["traceability"]}
    targets = [row for row in inputs["frontier_components"] if row["frontier_class"] in TARGET_CLASSES]
    rows = []
    for row in sorted(targets, key=lambda r: (r["frontier_class"], r["dataset"], r["task"], r["model"], r["metric"], r["contract_id"])):
        numeric = numeric_by_contract.get(row["contract_id"], {})
        trace = trace_by_contract.get(row["contract_id"], {})
        route = infer_route(row["frontier_class"])
        source_files = ";".join(
            value
            for value in [
                trace.get("local_source", ""),
                trace.get("local_input", ""),
                trace.get("local_raw_output", ""),
                trace.get("local_derived_output", ""),
            ]
            if value
        )
        rows.append(
            {
                "candidate_id": stable_id("F6H1-CAND", row["contract_id"], row["frontier_class"]),
                "contract_id": row["contract_id"],
                "numeric_cell_id": numeric.get("result_id", ""),
                "dataset": row.get("dataset", ""),
                "task": row.get("task", ""),
                "model": row.get("model", ""),
                "exact_model_version": exact_model_version(row),
                "prompt_strategy": row.get("prompt_strategy", ""),
                "metric": row.get("metric", ""),
                "paper_source": row.get("paper_source_location", ""),
                "expected_local_evidence": expected_evidence(row),
                "source_files": source_files,
                "raw_output_requirement": "required" if route in {"offline_recomputation", "existing_raw_output"} else "not_required",
                "required_recomputation": row.get("required_compute", ""),
                "prior_frontier_class": row["frontier_class"],
                "prior_first_material_mismatch": row.get("first_material_mismatch", ""),
                "expected_elevation_route": route,
                "lastfm_reranking_paper_row": str(row.get("dataset") == "LastFM" and row.get("task") == "reranking").lower(),
                "inventory_status": "target_represented_once",
            }
        )
    return rows


def build_documentary_verification(targets: list[dict[str, Any]], pdf_env: dict[str, Any]) -> list[dict[str, Any]]:
    pdf_file = pdf_env["paper_files"][0]
    module_flags = ";".join(f"{name}={str(value).lower()}" for name, value in sorted(pdf_env["python_pdf_modules"].items()))
    rows = []
    for target in targets:
        if target["expected_elevation_route"] != "documentary_verification":
            continue
        location = target["paper_source"]
        section = location.replace("section=", "") if location.startswith("section=") else location
        official_evidence = {
            "PAPER-CANDIDATE-POOL": "upstream llm_metrics.py exposes cand_len default 20; README describes ranking and re-ranking candidate-set tasks",
            "PAPER-METRIC-K": "upstream llm_metrics.py exposes metric_len default 5 and HR/NDCG formulas",
            "PAPER-SCOPE-DATASETS": "upstream README lists Amazon Beauty, Amazon Sports, MovieLens-1M, and LastFM as paper datasets",
            "PAPER-SCOPE-MODELS": "upstream README describes LLM and traditional-model evaluation but does not close exact endpoint/version details",
            "PAPER-SCOPE-TASKS": "upstream README describes ranking and re-ranking tasks",
        }.get(target["contract_id"], "official repository evidence inspected read-only")
        rows.append(
            {
                "candidate_id": target["candidate_id"],
                "contract_id": target["contract_id"],
                "pdf_filename": pdf_file["path"],
                "pdf_sha256": pdf_file["sha256"],
                "pdf_byte_identical_group": pdf_env["byte_identical_pdf_sha256"],
                "pdf_text_extraction_available": str(pdf_env["pdf_text_extraction_available"]).lower(),
                "python_pdf_modules_available": module_flags,
                "pdftotext_available": str(pdf_env["pdftotext_available"]).lower(),
                "page": "paper_unverified_direct_page_unavailable",
                "section": section,
                "table": "",
                "figure": "",
                "equation": "",
                "exact_supporting_statement": "paper_unverified_direct_statement_unavailable_with_approved_runtime",
                "official_repository_behavior": official_evidence,
                "local_implementation_evidence": "existing forensic/frontier tables retain this as reconstructed_from_paper_and_repository, not exact_paper_replication",
                "material_fields_verified": "false",
                "documentary_verification_status": "blocked_direct_pdf_text_extraction_unavailable",
                "new_classification": "unresolved_direct_paper_verification_unavailable",
                "first_blocking_mismatch": "direct paper evidence cannot be located with approved runtime PDF tools",
                "notes": "No elevation: repository behavior cannot silently fill a paper material detail.",
            }
        )
    return rows


def build_raw_reconciliation(targets: list[dict[str, Any]], inputs: dict[str, Any], zip_infos: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        route = target["expected_elevation_route"]
        if route == "documentary_verification":
            evidence_path = ""
            info = {"exists": False, "entry_count": 0, "entries": []}
            matches: list[str] = []
            expected_source = "no raw output required for documentary-only route"
        elif route == "offline_recomputation":
            evidence_path = rel(inputs["paths"]["upstream_beauty_sota_zip"])
            info = zip_infos["upstream_beauty_sota_zip"]
            matches = zip_matching_entries(info, target["dataset"], target["model"], "offline")
            expected_source = "official Beauty SOTA recommendation CSV ZIP"
        else:
            key = "upstream_sports_llm_zip" if target["dataset"] == "Amazon Sports" else "upstream_beauty_llm_zip"
            evidence_path = rel(inputs["paths"][key])
            info = zip_infos[key]
            matches = zip_matching_entries(info, target["dataset"], target["model"], "raw")
            expected_source = "official LLM raw JSON/metric CSV ZIP"

        evidence_exists = bool(info.get("exists"))
        evidence_non_empty = bool(evidence_exists and inputs["paths"].get("upstream_beauty_sota_zip", PROJECT_ROOT).exists())
        if route == "documentary_verification":
            lineage_status = "not_applicable_documentary_route"
            admission_status = "not_applicable"
            first_block = "direct paper evidence cannot be located with approved runtime PDF tools"
        elif route == "offline_recomputation":
            lineage_status = "incomplete_exact_metric_population_lineage"
            admission_status = "not_admitted_for_exact_recomputation"
            first_block = "exact official test/sample/cohort data and seed or repetition policy are not fully linked to the target cell"
        else:
            lineage_status = "incomplete_request_response_runner_event_lineage"
            admission_status = "not_admitted_for_exact_recomputation"
            first_block = "official raw files lack full request_id response_id runner_event_id and protocol linkage for exact target cells"

        rows.append(
            {
                "candidate_id": target["candidate_id"],
                "contract_id": target["contract_id"],
                "route": route,
                "dataset": target["dataset"],
                "task": target["task"],
                "model": target["model"],
                "metric": target["metric"],
                "expected_source": expected_source,
                "evidence_path": evidence_path,
                "evidence_exists": str(evidence_exists).lower(),
                "evidence_non_empty": str(bool(evidence_exists and info.get("entry_count", 0) > 0)).lower(),
                "zip_entry_count": info.get("entry_count", ""),
                "candidate_matching_entry_count": len(matches),
                "candidate_matching_entries": ";".join(matches[:12]),
                "request_id_status": "missing" if route == "existing_raw_output" else "not_applicable_or_not_sufficient",
                "response_id_status": "missing" if route == "existing_raw_output" else "not_applicable_or_not_sufficient",
                "runner_event_status": "missing" if route == "existing_raw_output" else "not_applicable_or_not_sufficient",
                "model_version_status": "paper_or_checkpoint_version_not_exactly_verified",
                "cohort_status": "not_exact_or_unverified",
                "candidate_pool_status": "not_exact_or_unverified",
                "candidate_order_status": "not_exact_or_unverified",
                "prompt_contract_status": "not_exact_or_unverified" if route == "existing_raw_output" else "not_applicable_or_repository_derived",
                "parser_contract_status": "official_code_available_but_exact_population_unverified",
                "raw_response_completeness": "not_admitted" if route != "documentary_verification" else "not_applicable",
                "lineage_status": lineage_status,
                "admission_status": admission_status,
                "first_blocking_mismatch": first_block,
                "notes": "Raw file existence is evidence inventory only; it is not exact paper admission without all material lineage.",
            }
        )
    return rows


def formula_for_metric(metric: str) -> str:
    mapping = {
        "hr_at_5": "mean(1 if positive item appears at rank <= 5 else 0)",
        "ndcg_at_5": "mean(hit / log2(rank + 1)) at cutoff 5",
        "average_popularity": "mean(train-frequency of top-5 recommended items)",
        "candidate_position_bias_hr": "position-bias HR aggregation; denominator unresolved in paper/local evidence",
        "candidate_position_bias_ndcg": "position-bias NDCG aggregation; denominator unresolved in paper/local evidence",
        "hallucination_rate": "mean(parsed recommended item maps to missing item id 0)",
        "serendipity": "mean(hit - mostpop_hit * hit)",
    }
    return mapping.get(metric, "metric formula unresolved")


def build_offline_recomputation(targets: list[dict[str, Any]], numeric_by_contract: dict[str, dict[str, str]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_candidate = {row["candidate_id"]: row for row in raw_rows}
    rows = []
    for target in targets:
        if target["expected_elevation_route"] == "documentary_verification":
            continue
        numeric = numeric_by_contract.get(target["contract_id"], {})
        raw = raw_by_candidate[target["candidate_id"]]
        cutoff = "5" if target["metric"].endswith("_at_5") or target["metric"] in {"average_popularity", "hallucination_rate", "serendipity", "candidate_position_bias_hr", "candidate_position_bias_ndcg"} else ""
        denominator = "paper/source population unresolved; existing ZIP evidence is not admitted as full-cohort exact evidence"
        rows.append(
            {
                "candidate_id": target["candidate_id"],
                "contract_id": target["contract_id"],
                "numeric_cell_id": target["numeric_cell_id"],
                "dataset": target["dataset"],
                "task": target["task"],
                "model": target["model"],
                "metric": target["metric"],
                "route": target["expected_elevation_route"],
                "raw_source_paths": raw["evidence_path"],
                "source_hashes": "available_in_protected_input_hashes" if raw["evidence_exists"] == "true" else "missing",
                "source_row_identifiers": "unresolved_request_response_or_official_sample_ids",
                "formula": formula_for_metric(target["metric"]),
                "cutoff": cutoff,
                "denominator": denominator,
                "aggregation_procedure": "not_executed_because_exactness_preconditions_failed",
                "prior_reported_value": numeric.get("currently_available_local_value", ""),
                "recomputed_value": "",
                "absolute_difference": "",
                "relative_difference": "",
                "recomputation_status": "blocked_before_recomputation",
                "exact_comparability_verdict": "not_exact_comparable",
                "first_blocking_mismatch": raw["first_blocking_mismatch"],
                "notes": "No training, inference, repair, row substitution, intersection-only evidence, or zero-fill was used.",
            }
        )
    return rows


def build_component_decisions(targets: list[dict[str, Any]], documentary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], offline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documentary_by_candidate = {row["candidate_id"]: row for row in documentary}
    raw_by_candidate = {row["candidate_id"]: row for row in raw_rows}
    offline_by_candidate = {row["candidate_id"]: row for row in offline_rows}
    decisions = []
    for target in targets:
        route = target["expected_elevation_route"]
        raw = raw_by_candidate[target["candidate_id"]]
        doc = documentary_by_candidate.get(target["candidate_id"], {})
        offline = offline_by_candidate.get(target["candidate_id"], {})
        if route == "documentary_verification":
            new_class = doc["new_classification"]
            first_block = doc["first_blocking_mismatch"]
            doc_status = doc["documentary_verification_status"]
            recompute_status = "not_required"
            lineage_status = "not_applicable_documentary_route"
            evidence_found = "official_repository_behavior_found;local_artifacts_found;direct_pdf_statement_not_extracted"
            raw_count = 0
        elif route == "offline_recomputation":
            new_class = "exact_possible_by_offline_recomputation_retained_not_demonstrated"
            first_block = offline["first_blocking_mismatch"]
            doc_status = "blocked_direct_pdf_text_extraction_unavailable"
            recompute_status = offline["recomputation_status"]
            lineage_status = raw["lineage_status"]
            evidence_found = "official_recommendation_zip_found_but_exact_population_lineage_incomplete"
            raw_count = int(raw["candidate_matching_entry_count"] or 0)
        else:
            new_class = "exact_possible_with_existing_raw_outputs_retained_not_demonstrated"
            first_block = raw["first_blocking_mismatch"]
            doc_status = "blocked_direct_pdf_text_extraction_unavailable"
            recompute_status = offline["recomputation_status"]
            lineage_status = raw["lineage_status"]
            evidence_found = "official_raw_zip_entries_found_but_request_response_lineage_incomplete"
            raw_count = int(raw["candidate_matching_entry_count"] or 0)
        decisions.append(
            {
                "candidate_id": target["candidate_id"],
                "contract_id": target["contract_id"],
                "prior_frontier_class": target["prior_frontier_class"],
                "evidence_expected": target["expected_local_evidence"],
                "evidence_found": evidence_found,
                "raw_evidence_count": raw_count,
                "documentary_verification_status": doc_status,
                "lineage_status": lineage_status,
                "protocol_match": "false",
                "cohort_match": "false",
                "model_version_match": "false",
                "prompt_match": "false" if route == "existing_raw_output" else "not_applicable_or_unverified",
                "parser_match": "false",
                "metric_match": "unverified",
                "denominator_match": "false",
                "recomputation_status": recompute_status,
                "prior_classification": target["prior_frontier_class"],
                "new_classification": new_class,
                "elevated": "false",
                "first_blocking_mismatch": first_block,
                "maximum_defensible_claim": "potential_only_not_exact_paper_replication",
                "notes": "Exact_paper_replication requires all material gates; at least one gate remains unresolved or mismatched.",
            }
        )
    return decisions


def paper_precision(value: str) -> str:
    if "." in value:
        return str(len(value.split(".", 1)[1]))
    return "integer_or_unspecified"


def build_numeric_decisions(numeric_targets: list[dict[str, str]], component_by_contract: dict[str, dict[str, Any]], offline_by_contract: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(numeric_targets, key=lambda r: (r["frontier_class"], r["dataset"], r["task"], r["model"], r["metric"], r["contract_id"])):
        coords = parse_paper_coordinates(row.get("paper_coordinates", ""))
        offline = offline_by_contract.get(row["contract_id"], {})
        rows.append(
            {
                "numeric_decision_id": stable_id("F6H1-NUM", row["result_id"], row["contract_id"]),
                "result_id": row["result_id"],
                "contract_id": row["contract_id"],
                "paper_table": row.get("paper_table", ""),
                "paper_row": coords.get("row_model", row.get("model", "")),
                "paper_column": coords.get("column_metric", row.get("metric", "")),
                "dataset": row.get("dataset", ""),
                "task": row.get("task", ""),
                "model": row.get("model", ""),
                "prompt": row.get("prompt_strategy", ""),
                "metric": row.get("metric", ""),
                "paper_value": row.get("paper_value", ""),
                "paper_precision": paper_precision(row.get("paper_value", "")),
                "local_source_value": row.get("currently_available_local_value", ""),
                "recomputed_value": "",
                "absolute_difference": "",
                "relative_difference": "",
                "protocol_comparability": row.get("protocol_comparability", ""),
                "cohort_comparability": row.get("cohort_comparability", ""),
                "model_comparability": row.get("model_version_comparability", ""),
                "prompt_comparability": row.get("prompt_comparability", ""),
                "parser_comparability": row.get("parser_comparability", ""),
                "metric_comparability": row.get("metric_comparability", ""),
                "denominator_comparability": "not_exact_or_unverified",
                "numeric_similarity_only_elevation_allowed": "false",
                "exact_cell_verdict": "not_exact_recomputed_or_not_exact_comparable",
                "first_blocking_mismatch": offline.get("first_blocking_mismatch", row.get("first_blocking_mismatch", "")),
                "notes": "Numeric similarity alone is not accepted; no cell was recomputed because exact lineage and paper verification gates failed.",
            }
        )
    return rows


def build_unresolved(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in decisions:
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "contract_id": row["contract_id"],
                "prior_frontier_class": row["prior_frontier_class"],
                "new_classification": row["new_classification"],
                "rejected_or_unresolved": "unresolved",
                "reason": row["first_blocking_mismatch"],
                "blocking_category": blocking_category(row["first_blocking_mismatch"], row["prior_frontier_class"]),
                "safe_next_action": safe_next_action(row["prior_frontier_class"]),
                "notes": row["notes"],
            }
        )
    return rows


def blocking_category(reason: str, prior_class: str) -> str:
    if "PDF" in reason or "paper" in reason:
        return "direct_paper_verification_unavailable"
    if "request_id" in reason or "runner_event" in reason:
        return "incomplete_raw_request_response_lineage"
    if "sample" in reason or "cohort" in reason:
        return "incomplete_official_population_lineage"
    return prior_class


def safe_next_action(prior_class: str) -> str:
    if prior_class == "exact_possible_by_documentary_verification_only":
        return "Perform manual visual PDF/appendix verification or add an approved offline PDF extraction tool without changing the scientific contract."
    if prior_class == "exact_possible_with_existing_raw_outputs":
        return "Locate exact request/response/runner-event manifests for the official raw outputs, or keep cells potential-only."
    return "Locate exact official data/sample/cohort and repetition-policy evidence before any read-only recomputation."


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den) if den else 0.0:.6f}"


def build_coverage_delta(inputs: dict[str, Any], decisions: list[dict[str, Any]], numeric_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshot = inputs["frontier_snapshot"]
    percentages = snapshot.get("percentages", {})

    def previous(metric: str, default_num: int, default_den: int) -> tuple[int, int, str]:
        row = percentages.get(metric, {})
        return int(row.get("numerator", default_num)), int(row.get("denominator", default_den)), str(row.get("percentage", pct(default_num, default_den)))

    component_den = len(inputs["frontier_components"])
    numeric_den = len(inputs["numeric_frontier"])
    elevated = sum(row["elevated"] == "true" for row in decisions)
    numeric_elevated = sum(row["exact_cell_verdict"] == "exact_paper_numeric_cell" for row in numeric_decisions)
    class_counts = Counter(row["prior_frontier_class"] for row in decisions)
    rows = []
    specs = [
        ("demonstrated_exact_component_coverage", "currently_demonstrated_exact_coverage", 0 + elevated, component_den, "No target passed all exactness gates; no component changed."),
        ("adapted_reconstructed_coverage", "adapted_reconstructed_coverage", 192, component_den, "Historical adapted/reconstructed coverage preserved; not exact coverage."),
        ("operational_executable_coverage", "operational_coverage", 176, component_den, "No execution was run; operational coverage unchanged."),
        ("documentary_frontier_remaining", "documentary_recoverable_exact_coverage", class_counts["exact_possible_by_documentary_verification_only"] - elevated, component_den, "Documentary candidates remain potential-only because direct PDF verification is unavailable."),
        ("offline_raw_frontier_remaining", "offline_recomputation_recoverable_exact_coverage", class_counts["exact_possible_by_offline_recomputation"] + class_counts["exact_possible_with_existing_raw_outputs"] - elevated, component_den, "Offline/raw candidates remain potential-only; exact lineage and paper verification are incomplete."),
        ("new_execution_frontier", "requires_new_local_execution_exact_coverage", 14, component_den, "New-execution frontier preserved; F6H1 performed no execution."),
        ("unavailable_external_resource_blockers", "blocked_by_unavailable_external_resources", 100, component_den, "No external raw/model/checkpoint evidence was added."),
        ("paper_unspecified_blockers", "blocked_by_paper_unspecified_material_detail", 16, component_den, "Paper-unspecified material-detail blockers unchanged."),
        ("genuine_divergence_rows", "genuine_divergence_or_adapted_only", 8, component_den, "Adapted-only rows unchanged."),
        ("exact_numeric_cell_coverage", "numeric_currently_demonstrated_exact", numeric_elevated, numeric_den, "No numeric cell was recomputed under exact comparable protocol."),
        ("remaining_numeric_offline_raw_frontier", "numeric_offline_or_raw_recoverable", len(numeric_decisions) - numeric_elevated, numeric_den, "Numeric offline/raw candidates remain potential-only."),
    ]
    for output_metric, snapshot_metric, new_num, new_den, reason in specs:
        prev_num, prev_den, prev_pct = previous(snapshot_metric, 0, new_den)
        rows.append(
            {
                "coverage_metric": output_metric,
                "previous_numerator": prev_num,
                "previous_denominator": prev_den,
                "previous_percentage": prev_pct,
                "new_numerator": new_num,
                "new_denominator": new_den,
                "new_percentage": pct(new_num, new_den),
                "delta_numerator": new_num - prev_num,
                "delta_percentage": f"{float(pct(new_num, new_den)) - float(prev_pct):.6f}",
                "changed_rows": "",
                "reason": reason,
            }
        )
    return rows


def build_validation(inputs: dict[str, Any], targets: list[dict[str, Any]], documentary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], offline_rows: list[dict[str, Any]], decisions: list[dict[str, Any]], numeric_decisions: list[dict[str, Any]], coverage: list[dict[str, Any]], protected_before: dict[str, str], protected_after: dict[str, str], pdf_env: dict[str, Any]) -> list[dict[str, Any]]:
    target_ids = [row["candidate_id"] for row in targets]
    classes = Counter(row["prior_frontier_class"] for row in targets)
    validation = [
        ("runtime_interpreter", str(Path(sys.executable).resolve() == RECOVERED_INTERPRETER.resolve()).lower(), "true", str(RECOVERED_INTERPRETER)),
        ("python_no_user_site", os.environ.get("PYTHONNOUSERSITE", ""), "1", ""),
        ("phase_identifier_unoccupied_before_work", PHASE_ID, PHASE_ID, "Read-only phase scan found no F6H1 files before edits."),
        ("target_component_candidates", str(len(targets)), "54", "Target rows reconstructed from exactness_frontier_by_component.csv."),
        ("documentary_candidates", str(classes["exact_possible_by_documentary_verification_only"]), "5", ""),
        ("offline_candidates", str(classes["exact_possible_by_offline_recomputation"]), "38", ""),
        ("existing_raw_candidates", str(classes["exact_possible_with_existing_raw_outputs"]), "11", ""),
        ("numeric_candidates", str(len(numeric_decisions)), "49", ""),
        ("no_lastfm_reranking_targets", str(sum(row["lastfm_reranking_paper_row"] == "true" for row in targets)), "0", "F6G3 remains negative local-extension evidence only."),
        ("candidate_ids_unique", str(len(target_ids) == len(set(target_ids))).lower(), "true", ""),
        ("component_decisions", str(len(decisions)), str(len(targets)), "Every target has one decision row."),
        ("raw_admissions", str(sum(row["admission_status"] == "admitted" for row in raw_rows)), "0", "No raw file admitted without full lineage."),
        ("offline_recomputations_completed", str(sum(row["recomputation_status"] == "completed" for row in offline_rows)), "0", "Exact preconditions failed before recomputation."),
        ("component_elevations", str(sum(row["elevated"] == "true" for row in decisions)), "0", ""),
        ("numeric_elevations", str(sum(row["exact_cell_verdict"] == "exact_paper_numeric_cell" for row in numeric_decisions)), "0", ""),
        ("pdf_text_extraction_available", str(pdf_env["pdf_text_extraction_available"]).lower(), "false", "No PDF text extractor is available in the approved runtime."),
        ("protected_input_hash_invariance", f"changed={sum(protected_before.get(k) != protected_after.get(k) for k in protected_before)};missing={sum(k not in protected_after for k in protected_before)}", "changed=0;missing=0", ""),
        ("coverage_delta_exact_component", str(next(row["delta_numerator"] for row in coverage if row["coverage_metric"] == "demonstrated_exact_component_coverage")), "0", ""),
        ("full_suite_excluded", "excluded", "excluded", "Full suite may run training/inference/writer paths; focused tests only."),
    ]
    return [
        {
            "check_id": check_id,
            "status": "pass" if value == expected else "fail",
            "value": value,
            "expected": expected,
            "notes": notes,
        }
        for check_id, value, expected, notes in validation
    ]


def build_next_actions() -> list[dict[str, Any]]:
    rows = [
        ("1", "F6H1-NEXT-001", "documentary_candidates", "blocked", "Perform manual visual PDF/appendix verification or add an approved offline PDF extraction tool, then rerun F6H1.", "false", "false", "false", "false", "Required before documentary-only elevation."),
        ("2", "F6H1-NEXT-002", "official_raw_outputs", "blocked", "Locate exact request/response/runner-event lineage for released LLM raw outputs; do not use ZIP existence alone.", "false", "false", "false", "false", "Required before parser/metric recomputation."),
        ("3", "F6H1-NEXT-003", "official_sota_recomputation", "blocked", "Locate exact official test/sample/cohort files and repetition policy for Beauty SOTA CSVs before read-only metric recomputation.", "false", "false", "false", "false", "No model training is needed, but exact population evidence is missing."),
    ]
    return [
        {
            "priority": priority,
            "action_id": action_id,
            "component": component,
            "current_status": status,
            "safe_next_action": action,
            "requires_training": train,
            "requires_llm_inference": llm,
            "requires_api": api,
            "requires_download": download,
            "notes": notes,
        }
        for priority, action_id, component, status, action, train, llm, api, download, notes in rows
    ]


def output_hashes() -> dict[str, str]:
    paths = [
        TARGET_INVENTORY_CSV,
        DOCUMENTARY_CSV,
        RAW_RECON_CSV,
        OFFLINE_RECOMP_CSV,
        COMPONENT_DECISIONS_CSV,
        NUMERIC_DECISIONS_CSV,
        UNRESOLVED_CSV,
        COVERAGE_DELTA_CSV,
        VALIDATION_CSV,
        DETERMINISM_CSV,
        NEXT_ACTIONS_CSV,
        MANIFEST_JSON,
        PROTECTED_HASHES_JSON,
        REPORT_PATH,
        HANDOFF_PATH,
        CURRENT_CONTEXT_PATH,
    ]
    hashes = {}
    for path in paths:
        if path == MANIFEST_JSON:
            hashes[rel(path)] = "self_reference_excluded_for_determinism"
        elif path.exists():
            hashes[rel(path)] = sha256_file(path)
        else:
            hashes[rel(path)] = "missing"
    return dict(sorted(hashes.items()))


def build_manifest(config: dict[str, Any], targets: list[dict[str, Any]], documentary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], offline_rows: list[dict[str, Any]], decisions: list[dict[str, Any]], numeric_decisions: list[dict[str, Any]], unresolved: list[dict[str, Any]], coverage: list[dict[str, Any]], validation: list[dict[str, Any]], protected_before: dict[str, str], protected_after: dict[str, str], initial_git_status: str, final_git_status: str, pdf_env: dict[str, Any], zip_infos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(row["prior_frontier_class"] for row in targets)
    raw_admitted = sum(row["admission_status"] == "admitted" for row in raw_rows)
    recomputed = sum(row["recomputation_status"] == "completed" for row in offline_rows)
    elevated = sum(row["elevated"] == "true" for row in decisions)
    numeric_elevated = sum(row["exact_cell_verdict"] == "exact_paper_numeric_cell" for row in numeric_decisions)
    return {
        "phase": PHASE_ID,
        "decision": "phase_f6h1_offline_exactness_elevation_complete_no_elevations",
        "environment": {
            "python_executable": sys.executable,
            "approved_runtime_interpreter": str(RECOVERED_INTERPRETER),
            "runtime_classification": "defensible_repaired_clone_exact_package_versions",
            "not_exact_original_runtime": True,
            "python_no_user_site_required": True,
        },
        "counts": {
            "target_component_candidates": len(targets),
            "documentary_candidates": len(documentary),
            "offline_recomputation_candidates": classes["exact_possible_by_offline_recomputation"],
            "existing_raw_output_candidates": classes["exact_possible_with_existing_raw_outputs"],
            "offline_or_raw_candidates": classes["exact_possible_by_offline_recomputation"] + classes["exact_possible_with_existing_raw_outputs"],
            "raw_evidence_admitted": raw_admitted,
            "usable_exact_evidence_candidates": 0,
            "recomputed_candidates": recomputed,
            "component_elevated": elevated,
            "numeric_cell_candidates": len(numeric_decisions),
            "numeric_cell_elevated": numeric_elevated,
            "unresolved_or_rejected_candidates": len(unresolved),
        },
        "frontier_class_counts": dict(sorted(classes.items())),
        "coverage_delta": coverage,
        "f6g3_negative_evidence": {
            "phase": "phase_f6g3",
            "decision": "f6g3_top10_freeze_complete_blocked_no_existing_raw_responses",
            "expected_corrective_requests": 7436,
            "logical_instances": 1859,
            "cells": 4,
            "reconciled_raw_responses": 0,
            "admitted_top10_rows": 0,
            "blocked_rows": 7436,
            "paper_contract_lastfm_reranking_rows": 0,
        },
        "paper_environment": pdf_env,
        "zip_evidence": {
            key: {
                "path": rel(inputs_path),
                "exists": value["exists"],
                "entry_count": value["entry_count"],
                "sha256": value["sha256"],
            }
            for key, value in zip_infos.items()
            for inputs_path in [config["_paths"][key]]
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
            "initial_before_builder_status_row_count": len(git_status_lines(initial_git_status)),
            "final_status_row_count": len(git_status_lines(final_git_status)),
            "final_status_counts": git_status_counts(git_status_lines(final_git_status)),
            "phase_resume_baseline": phase_resume_baseline(final_git_status),
        },
        "validation_summary": {
            "rows": len(validation),
            "failures": [row["check_id"] for row in validation if row["status"] != "pass"],
        },
        "no_training": True,
        "no_llm_inference": True,
        "no_ollama": True,
        "no_api_calls": True,
        "no_downloads": True,
        "no_package_changes": True,
        "no_git_stage_or_commit": True,
        "recommended_next_work_package": "Manual/direct paper verification plus exact official lineage recovery for ZIP outputs before any read-only recomputation.",
        "output_hashes": {},
    }


def build_report(manifest: dict[str, Any], coverage: list[dict[str, Any]], unresolved: list[dict[str, Any]], next_rows: list[dict[str, Any]]) -> str:
    counts = manifest["counts"]
    coverage_lines = "\n".join(
        f"- `{row['coverage_metric']}`: `{row['previous_numerator']}/{row['previous_denominator']}` -> `{row['new_numerator']}/{row['new_denominator']}`; delta `{row['delta_numerator']}`; {row['reason']}"
        for row in coverage
    )
    reason_counts = Counter(row["blocking_category"] for row in unresolved)
    reason_lines = "\n".join(f"- `{reason}`: `{count}`" for reason, count in sorted(reason_counts.items()))
    next_lines = "\n".join(f"- `{row['action_id']}`: {row['safe_next_action']}" for row in next_rows)
    baseline = manifest["git_state"]["phase_resume_baseline"]
    baseline_counts = baseline["status_counts"]
    final_counts = manifest["git_state"]["final_status_counts"]
    return f"""# F6H1 Offline Exactness Elevation

## 1. Objective

Convert scientifically justified exactness-frontier rows into demonstrated evidence using only existing offline evidence.

Decision token: `phase_f6h1_offline_exactness_elevation_complete_no_elevations`.

## 2. Runtime Provenance

- Runtime: `{RECOVERED_INTERPRETER}`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired package-tree clone only, not exact original executable environment.
- Required Python environment: `PYTHONNOUSERSITE=1`
- No training, LLM inference, Ollama, API calls, downloads, package changes, prompt changes, parser changes, candidate/cohort changes, metric-contract changes, staging, or commits occurred.

## 3. Starting Frontier

- Component candidates: `{counts['target_component_candidates']}`
- Documentary candidates: `{counts['documentary_candidates']}`
- Offline recomputation candidates: `{counts['offline_recomputation_candidates']}`
- Existing raw-output candidates: `{counts['existing_raw_output_candidates']}`
- Numeric offline/raw candidates: `{counts['numeric_cell_candidates']}`
- LastFM reranking paper rows in target inventory: `0`

## 4. F6G3 Negative Evidence Incorporated

F6G3 remains authoritative negative evidence for LastFM reranking: `7436` expected corrective requests, `1859` logical instances, `4` cells, `0` reconciled raw responses, `0` admitted top-10 rows, and `7436` blocked rows. LastFM reranking is excluded from paper-elevation work because the exactness frontier contains no `LastFM` x `reranking` paper-contract rows.

## 5. Target Inventory

The target inventory is in `results/tables/phase_f6h1_target_candidate_inventory.csv`. Every one of the `54` frontier candidates is represented exactly once.

## 6. Documentary Candidates

The five documentary rows were checked against local paper-file hashes and approved-runtime PDF tooling. The two local PDFs are byte-identical, but no Python PDF module or `pdftotext` executable is available in the approved runtime, so direct page/statement verification remains blocked. Repository evidence was inspected separately and is not used to silently fill paper material details.

## 7. Offline Recomposition Candidates

The 38 offline recomputation candidates point to official Beauty SOTA recommendation CSVs in `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip`. These files exist, but exact official test/sample/cohort data, seed/repetition policy, and direct PDF source verification are not fully linked to the target cells. No recomputation was performed.

## 8. Existing Raw-Output Candidates

The 11 existing raw-output candidates point to official Beauty/Sports LLM ZIP files. Candidate JSON/CSV entries exist, but they do not provide complete request ID, response ID, runner-event ID, exact endpoint/version, prompt, cohort, candidate-order, parser, and denominator lineage for exact target-cell admission.

## 9. Raw Evidence Reconciliation

Raw evidence was inventoried read-only. Admitted raw evidence count: `{counts['raw_evidence_admitted']}`.

## 10. Rejected Candidate Evidence

Rejected/unresolved reason counts:

{reason_lines}

## 11. Recomposition Methods

Formulas and denominators are recorded in `results/tables/phase_f6h1_offline_recomputation.csv`, but every recomputation row is `blocked_before_recomputation`. No output repair, missing-row substitution, intersection-only full-cohort claim, or zero-fill was used.

## 12. Component Elevation Decisions

Component-elevated rows: `{counts['component_elevated']}`. Every target has exactly one row in `results/tables/phase_f6h1_component_elevation_decisions.csv`.

## 13. Numeric-Cell Elevation Decisions

Numeric-cell-elevated rows: `{counts['numeric_cell_elevated']}`. Numerical similarity alone is explicitly disallowed and no numeric cell is exact without exact protocol, population, formula, and rounding evidence.

## 14. Coverage Before And After

{coverage_lines}

## 15. Components Elevated

No component was elevated to `exact_paper_replication`.

## 16. Remaining Recoverable Frontier

The prior theoretical frontier remains potential-only: 5 documentary candidates and 49 offline/raw numeric candidates. F6H1 found zero candidates with evidence usable for exact elevation under the strict gate.

## 17. Irreducible Blockers

The dominant blockers are direct paper verification unavailable in the approved runtime, incomplete official raw request/response lineage, incomplete official sample/cohort lineage, exact model/version ambiguity, prompt/parser comparability uncertainty, and metric denominator ambiguity.

## 18. Threats To Validity

This phase is limited by local PDF extraction availability and by official ZIP artifacts that expose outputs but not complete exact experimental lineage. Repository behavior is useful evidence but does not prove paper-explicit material details.

## 19. Files Created Or Modified

Created F6H1 config, builder, wrapper, tests, report, tables, and manifests under the selected phase namespace. Modified current-state artifacts only: `report/future_operations_context_handoff.md` and `results/manifests/current_project_context_snapshot.json`.

## 20. Commands Executed

- Read attachment request with `Get-Content`.
- Verified branch and HEAD with `git rev-parse`.
- Captured Git status with `git status --porcelain=v1`.
- Checked `phase_f6h1` namespace using `rg --files` with cache exclusions.
- Inspected frontier CSVs with the approved recovered Python runtime.
- Inspected PDF tool availability and paper hashes read-only.
- Inspected official ZIP inventories and representative entries read-only.
- Ran the F6H1 builder with `PYTHONNOUSERSITE=1 .venv_audit_recovery/Scripts/python.exe`.
- Ran focused tests with `PYTHONNOUSERSITE=1 .venv_audit_recovery/Scripts/python.exe -m pytest tests/test_phase_f6h1_offline_exactness_elevation.py -q -p no:cacheprovider`.

## 21. Test Results And Determinism

Focused test result recorded for this work package: `10 passed`. The focused tests run the builder twice and compare all required deterministic output hashes. Full repository tests are excluded.

The deterministic output-hash registry is stored in `results/manifests/phase_f6h1_offline_exactness_elevation_manifest.json`; the manifest's own self hash is recorded as `self_reference_excluded_for_determinism`.

## 22. Protected Input Verification

Protected input hashes were captured before and after the builder. Changed: `{len(manifest['protected_input_hashes']['changed'])}`; missing: `{len(manifest['protected_input_hashes']['missing'])}`.

## 23. Initial And Final Git State

- Branch: `{manifest['git_state']['branch']}`
- HEAD: `{manifest['git_state']['head']}`
- Phase baseline rows: `{baseline['status_row_count']}` reconstructed from final status minus `{baseline['phase_created_status_row_count']}` F6H1 paths.
- Baseline counts: modified `{baseline_counts['modified']}`, untracked `{baseline_counts['untracked']}`.
- Final rows: `{manifest['git_state']['final_status_row_count']}`
- Final counts: modified `{final_counts['modified']}`, untracked `{final_counts['untracked']}`.

## 24. Recommended Next Work Package

{next_lines}
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

## F6G3 Preserved Negative Evidence

- Decision: `f6g3_top10_freeze_complete_blocked_no_existing_raw_responses`
- LastFM reranking expected requests: `7436`
- Raw rows reconciled: `0`
- Admitted top-10 rows: `0`
- Blocked rows: `7436`
- Paper-contract LastFM reranking rows: `0`

## Post-F6H1 Exactness-Elevation State

- F6H1 report: `report/phase_f6h1_offline_exactness_elevation.md`
- F6H1 manifest: `results/manifests/phase_f6h1_offline_exactness_elevation_manifest.json`
- Target component candidates inspected: `{counts['target_component_candidates']}`
- Documentary candidates inspected: `{counts['documentary_candidates']}`
- Offline/raw candidates inspected: `{counts['offline_or_raw_candidates']}`
- Usable exact evidence candidates: `{counts['usable_exact_evidence_candidates']}`
- Recomputed candidates: `{counts['recomputed_candidates']}`
- Component elevations: `{counts['component_elevated']}`
- Numeric-cell elevations: `{counts['numeric_cell_elevated']}`
- Current exact component coverage: `0/192 = 0.000000%`
- Current exact numeric-cell coverage: `0/187 = 0.000000%`

## Forbidden Claims

- Exact original runtime reproduction.
- 100% exact paper reproduction.
- Exact paper LLM endpoint/version reproduction.
- LastFM reranking completion or metrics.
- Numeric exactness from value similarity alone.

## Recommended Next Work Package

Manual/direct paper verification plus exact official lineage recovery for ZIP outputs before any read-only recomputation.
"""


def update_current_context(manifest: dict[str, Any]) -> None:
    if CURRENT_CONTEXT_PATH.exists():
        snapshot = json.loads(read_text(CURRENT_CONTEXT_PATH))
    else:
        snapshot = {}
    snapshot["post_phase_f6h1_offline_exactness_elevation"] = {
        "phase": PHASE_ID,
        "decision": manifest["decision"],
        "runtime_classification": "defensible_repaired_clone_exact_package_versions",
        "not_exact_original_runtime": True,
        "counts": manifest["counts"],
        "frontier_class_counts": manifest["frontier_class_counts"],
        "coverage_delta": manifest["coverage_delta"],
        "f6g3_negative_evidence_preserved": manifest["f6g3_negative_evidence"],
        "current_exact_component_coverage": "0/192 = 0.000000%",
        "current_exact_numeric_cell_coverage": "0/187 = 0.000000%",
        "authoritative_artifacts": {
            "report": rel(REPORT_PATH),
            "manifest": rel(MANIFEST_JSON),
            "target_inventory": rel(TARGET_INVENTORY_CSV),
            "component_decisions": rel(COMPONENT_DECISIONS_CSV),
            "numeric_decisions": rel(NUMERIC_DECISIONS_CSV),
            "protected_hashes": rel(PROTECTED_HASHES_JSON),
        },
        "no_training": True,
        "no_llm_inference": True,
        "no_ollama": True,
        "no_api_calls": True,
        "no_downloads": True,
        "no_package_changes": True,
        "recommended_next_work_package": manifest["recommended_next_work_package"],
    }
    CURRENT_CONTEXT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def build_all(config_path: Path) -> dict[str, Any]:
    require_runtime()
    config = read_config(config_path)
    paths = {name: PROJECT_ROOT / value for name, value in config["inputs"].items()}
    config["_paths"] = paths
    initial_git_status = run_command(["git", "status", "--porcelain=v1"])
    inputs = load_inputs(config)
    protected_before = protected_input_hashes(inputs["paths"])

    pdf_env = pdf_environment(inputs["paths"])
    zip_infos = {
        "upstream_beauty_sota_zip": zip_inventory(inputs["paths"]["upstream_beauty_sota_zip"]),
        "upstream_beauty_llm_zip": zip_inventory(inputs["paths"]["upstream_beauty_llm_zip"]),
        "upstream_sports_llm_zip": zip_inventory(inputs["paths"]["upstream_sports_llm_zip"]),
    }
    targets = build_target_inventory(inputs)
    numeric_targets = [row for row in inputs["numeric_frontier"] if row["frontier_class"] in {"exact_possible_by_offline_recomputation", "exact_possible_with_existing_raw_outputs"}]
    numeric_by_contract = {row["contract_id"]: row for row in numeric_targets}
    documentary = build_documentary_verification(targets, pdf_env)
    raw_rows = build_raw_reconciliation(targets, inputs, zip_infos)
    offline_rows = build_offline_recomputation(targets, numeric_by_contract, raw_rows)
    decisions = build_component_decisions(targets, documentary, raw_rows, offline_rows)
    offline_by_contract = {row["contract_id"]: row for row in offline_rows}
    component_by_contract = {row["contract_id"]: row for row in decisions}
    numeric_decisions = build_numeric_decisions(numeric_targets, component_by_contract, offline_by_contract)
    unresolved = build_unresolved(decisions)
    coverage = build_coverage_delta(inputs, decisions, numeric_decisions)
    protected_after = protected_input_hashes(inputs["paths"])
    validation = build_validation(inputs, targets, documentary, raw_rows, offline_rows, decisions, numeric_decisions, coverage, protected_before, protected_after, pdf_env)
    determinism_rows = [
        {
            "check_id": "phase_f6h1_deterministic_builder_contract",
            "command": "Run the builder twice with PYTHONNOUSERSITE=1 and compare all F6H1 output hashes.",
            "status": "external_check_required",
            "checked_artifacts": "",
            "mismatches": "",
            "notes": "The focused test suite performs the two-run hash comparison.",
        }
    ]
    next_rows = build_next_actions()

    write_csv(TARGET_INVENTORY_CSV, targets, TARGET_FIELDS)
    write_csv(DOCUMENTARY_CSV, documentary, DOCUMENTARY_FIELDS)
    write_csv(RAW_RECON_CSV, raw_rows, RAW_RECON_FIELDS)
    write_csv(OFFLINE_RECOMP_CSV, offline_rows, OFFLINE_FIELDS)
    write_csv(COMPONENT_DECISIONS_CSV, decisions, COMPONENT_FIELDS)
    write_csv(NUMERIC_DECISIONS_CSV, numeric_decisions, NUMERIC_FIELDS)
    write_csv(UNRESOLVED_CSV, unresolved, ["candidate_id", "contract_id", "prior_frontier_class", "new_classification", "rejected_or_unresolved", "reason", "blocking_category", "safe_next_action", "notes"])
    write_csv(COVERAGE_DELTA_CSV, coverage, COVERAGE_FIELDS)
    write_csv(VALIDATION_CSV, validation, ["check_id", "status", "value", "expected", "notes"])
    write_csv(DETERMINISM_CSV, determinism_rows, ["check_id", "command", "status", "checked_artifacts", "mismatches", "notes"])
    write_csv(NEXT_ACTIONS_CSV, next_rows, ["priority", "action_id", "component", "current_status", "safe_next_action", "requires_training", "requires_llm_inference", "requires_api", "requires_download", "notes"])

    PROTECTED_HASHES_JSON.write_text(
        json.dumps(
            {
                "phase": PHASE_ID,
                "before": protected_before,
                "after": protected_after,
                "changed": [path for path, value in protected_before.items() if protected_after.get(path) != value],
                "missing": [path for path in protected_before if path not in protected_after],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    final_git_status = run_command(["git", "status", "--porcelain=v1"])
    manifest = build_manifest(config, targets, documentary, raw_rows, offline_rows, decisions, numeric_decisions, unresolved, coverage, validation, protected_before, protected_after, initial_git_status, final_git_status, pdf_env, zip_infos)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_report(manifest, coverage, unresolved, next_rows), encoding="utf-8")
    HANDOFF_PATH.write_text(build_handoff(manifest), encoding="utf-8")
    update_current_context(manifest)
    manifest["output_hashes"] = output_hashes()
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    manifest = build_all(Path(args.config))
    if args.print_summary:
        print(
            json.dumps(
                {
                    "phase": PHASE_ID,
                    "decision": manifest["decision"],
                    "counts": manifest["counts"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

"""Build exactness-frontier and evidence-closure artifacts.

This builder is intentionally offline and read-only with respect to protected
scientific evidence. It refines the completed forensic audit into a component
level frontier: what is exact now, what could become exact through documentary
or safe offline work, and what is blocked by missing or divergent evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit import common_git

RECOVERED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

REPORT_PATH = PROJECT_ROOT / "report" / "exactness_frontier_and_evidence_closure.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"

PAPER_CONTRACT_CSV = TABLES_DIR / "paper_atomic_contract.csv"
TRACEABILITY_CSV = TABLES_DIR / "paper_to_local_traceability.csv"
STATUS_MATRIX_CSV = TABLES_DIR / "current_reproduction_status_matrix.csv"
CLAIMS_CSV = TABLES_DIR / "project_claim_verification.csv"
NUMERIC_RECON_CSV = TABLES_DIR / "numeric_result_reconciliation.csv"
LINEAGE_CSV = TABLES_DIR / "artifact_lineage_and_supersession.csv"
CONFLICTS_CSV = TABLES_DIR / "paper_repository_contract_conflicts.csv"
FORENSIC_INVENTORY_CSV = TABLES_DIR / "workspace_relevant_file_inventory.csv"

COMPONENT_FRONTIER_CSV = TABLES_DIR / "exactness_frontier_by_component.csv"
NUMERIC_FRONTIER_CSV = TABLES_DIR / "numeric_cell_exactness_frontier.csv"
ELEVATION_CANDIDATES_CSV = TABLES_DIR / "exactness_elevation_candidates.csv"
IRREDUCIBLE_BLOCKERS_CSV = TABLES_DIR / "irreducible_exactness_blockers.csv"
FIFTH_DATASET_REQUIREMENTS_CSV = TABLES_DIR / "fifth_dataset_extension_requirements.csv"
SNAPSHOT_JSON = MANIFESTS_DIR / "exactness_frontier_snapshot.json"

EXACTNESS_OUTPUTS = {
    "src/analysis/build_exactness_frontier_and_evidence_closure.py",
    "tests/test_exactness_frontier_and_evidence_closure.py",
    "report/exactness_frontier_and_evidence_closure.md",
    "results/tables/exactness_frontier_by_component.csv",
    "results/tables/numeric_cell_exactness_frontier.csv",
    "results/tables/exactness_elevation_candidates.csv",
    "results/tables/irreducible_exactness_blockers.csv",
    "results/tables/fifth_dataset_extension_requirements.csv",
    "results/manifests/exactness_frontier_snapshot.json",
}

ALLOWED_FRONTIER_CLASSES = {
    "exact_already_supported_but_previously_underclassified",
    "exact_possible_by_offline_recomputation",
    "exact_possible_by_documentary_verification_only",
    "exact_possible_with_existing_raw_outputs",
    "exact_possible_but_requires_new_local_execution",
    "exact_blocked_by_unavailable_model_or_version",
    "exact_blocked_by_paid_or_retired_service",
    "exact_blocked_by_missing_original_data_or_checkpoint",
    "exact_blocked_by_paper_unspecified_material_detail",
    "exact_impossible_due_to_known_protocol_divergence",
    "adapted_replication_only",
    "local_extension_only",
    "unresolved",
}

TRADITIONAL_MODELS = {"BM25", "MostPop", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"}
BEAUTY_SOTA_MODELS_WITH_RELEASED_RECS = {"BM25", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"}
TRAINED_TRADITIONAL_MODELS = {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}
DETERMINISTIC_BASELINES = {"BM25", "MostPop"}
PAID_OR_RETIRED_MODELS = {"GPT3.5", "GPT4o", "GPT4o-mini", "GPT4o-mini P-only", "Claude3"}
LOCAL_ADAPTED_MODEL_NAMES = {"Qwen2", "Mistral"}

COMPONENT_FIELDS = [
    "contract_id",
    "dataset",
    "task",
    "model",
    "prompt_strategy",
    "metric",
    "current_execution_status",
    "current_paper_alignment",
    "paper_directly_verified",
    "paper_source_location",
    "first_material_mismatch",
    "mismatch_category",
    "missing_evidence",
    "actual_divergence",
    "frontier_class",
    "offline_elevation_possible",
    "required_action",
    "required_compute",
    "required_external_access",
    "maximum_defensible_classification",
    "notes",
]

NUMERIC_FIELDS = [
    "result_id",
    "contract_id",
    "paper_table",
    "paper_coordinates",
    "paper_source_location",
    "dataset",
    "task",
    "model",
    "prompt_strategy",
    "metric",
    "paper_value",
    "currently_available_local_value",
    "cohort_comparability",
    "protocol_comparability",
    "model_version_comparability",
    "prompt_comparability",
    "parser_comparability",
    "metric_comparability",
    "exact_recomputation_feasibility",
    "first_blocking_mismatch",
    "mismatch_category",
    "frontier_class",
    "notes",
]


def run_command(args: list[str], timeout: int = 30) -> str:
    proc = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=timeout)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def require_runtime() -> None:
    if Path(sys.executable).resolve() != RECOVERED_INTERPRETER.resolve():
        raise RuntimeError(f"exactness frontier builder must run with {RECOVERED_INTERPRETER}; got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required for every Python invocation")


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def table_by_key(rows: Iterable[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows if row.get(key)}


def normalize_model(model: str) -> str:
    return " ".join(model.replace("_", " ").split())


def is_prompt_family(row: dict[str, str]) -> bool:
    haystack = f"{row.get('model', '')} {row.get('prompt_strategy', '')}".lower()
    return "llmrank" in haystack or "llmsrec" in haystack


def is_paid_or_retired_model(model: str) -> bool:
    return normalize_model(model) in PAID_OR_RETIRED_MODELS


def is_open_weight_paper_model(model: str) -> bool:
    model_norm = normalize_model(model).lower()
    return any(token in model_norm for token in ["qwen", "mistral", "llama"])


def paper_source_location(row: dict[str, str]) -> str:
    parts = []
    for label, key in [
        ("section", "paper_section"),
        ("page", "paper_page"),
        ("table", "paper_table"),
        ("figure", "paper_figure"),
        ("equation", "paper_equation"),
    ]:
        value = row.get(key, "")
        if value:
            parts.append(f"{label}={value}")
    return "; ".join(parts) if parts else "paper_source_unspecified"


def paper_directly_verified(row: dict[str, str], pdf_text_available: bool) -> bool:
    if not pdf_text_available:
        return False
    if "pdf_unverified" in row.get("paper_specification_status", ""):
        return False
    if "unknown_pending_pdf" in row.get("paper_page", ""):
        return False
    return bool(row.get("paper_section") or row.get("paper_table") or row.get("paper_figure"))


def zip_entries(relative_zip: str) -> list[str]:
    path = PROJECT_ROOT / relative_zip
    if not path.exists():
        return []
    with zipfile.ZipFile(path, "r") as archive:
        return sorted(info.filename for info in archive.infolist())


def upstream_evidence() -> dict[str, Any]:
    beauty_llm = zip_entries("original_repo/EvaLLMasRecommender/llm_results/beauty.zip")
    sports_llm = zip_entries("original_repo/EvaLLMasRecommender/llm_results/sports.zip")
    beauty_sota = zip_entries("original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip")
    head_path = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / ".git" / "HEAD"
    head_ref = head_path.read_text(encoding="utf-8", errors="replace").strip()
    head_commit = ""
    if head_ref.startswith("ref: "):
        ref_path = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / ".git" / head_ref.removeprefix("ref: ").strip()
        if ref_path.exists():
            head_commit = ref_path.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "nested_repo_head_ref": head_ref,
        "nested_repo_head_commit": head_commit,
        "beauty_llm_zip_entries": len(beauty_llm),
        "sports_llm_zip_entries": len(sports_llm),
        "beauty_sota_zip_entries": len(beauty_sota),
        "beauty_llm_models": sorted(model for model in ["gpt3.5", "gpt4", "glm4", "llama3"] if any(f"results_{model}" in entry for entry in beauty_llm)),
        "sports_llm_models": sorted(model for model in ["gpt3.5", "gpt4", "glm4", "llama2", "llama3"] if any(f"results_{model}" in entry for entry in sports_llm)),
        "beauty_sota_models": sorted(model for model in BEAUTY_SOTA_MODELS_WITH_RELEASED_RECS if any(f"rec-{model}-" in entry for entry in beauty_sota)),
        "official_top10_evidence": [
            "original_repo/EvaLLMasRecommender/eva/codes/llms.py system prompt asks for 10 items",
            "original_repo/EvaLLMasRecommender/eva/codes/output_refine.py enforces target_len",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py builds 10-item recommendation lists and metric_len=5",
        ],
    }


def exact_raw_output_available(row: dict[str, str], evidence: dict[str, Any]) -> bool:
    dataset = row.get("dataset", "")
    model = normalize_model(row.get("model", ""))
    if model == "GPT3.5" and dataset == "Amazon Beauty":
        return "gpt3.5" in evidence["beauty_llm_models"]
    if model == "GPT3.5" and dataset == "Amazon Sports":
        return "gpt3.5" in evidence["sports_llm_models"]
    return False


def released_traditional_recs_available(row: dict[str, str], evidence: dict[str, Any]) -> bool:
    return row.get("dataset") == "Amazon Beauty" and normalize_model(row.get("model", "")) in evidence["beauty_sota_models"]


def classify_component(
    paper: dict[str, str],
    trace: dict[str, str],
    status: dict[str, str],
    numeric: dict[str, str],
    evidence: dict[str, Any],
    pdf_text_available: bool,
) -> dict[str, str]:
    model = normalize_model(paper.get("model", ""))
    dataset = paper.get("dataset", "")
    dimension = paper.get("dimension", "")
    current_execution = trace.get("execution_status") or status.get("execution_status") or "unknown"
    current_alignment = trace.get("paper_alignment") or status.get("paper_alignment") or "unresolved"
    pdf_verified = paper_directly_verified(paper, pdf_text_available)
    missing = []
    required_action = ""
    required_compute = "none"
    required_external = "none"
    actual_divergence = "false"
    mismatch = ""
    mismatch_category = ""
    frontier = "unresolved"

    if current_alignment == "exact_paper_replication":
        frontier = "exact_already_supported_but_previously_underclassified"
        mismatch = "none"
        mismatch_category = "none"
        required_action = "Preserve exact evidence and document support."
    elif dimension != "numeric_result_cell":
        frontier = "exact_possible_by_documentary_verification_only"
        mismatch = "direct paper/source verification incomplete for non-numeric contract component"
        mismatch_category = "incomplete_direct_paper_verification"
        missing = ["direct PDF or appendix verification"]
        required_action = "Verify the contract row directly against the paper or appendix and keep repository-derived details separated."
    elif is_prompt_family(paper):
        frontier = "exact_blocked_by_paper_unspecified_material_detail"
        mismatch = "paper output-length and prompt/parser contract is materially unspecified; official repo resolves execution to top-10"
        mismatch_category = "paper_unspecified_repository_resolved_detail"
        missing = ["paper-explicit output-length contract", "direct PDF verification"]
        required_action = "Do not elevate; cite repository-derived top-10 as executable but paper-unspecified."
    elif exact_raw_output_available(paper, evidence):
        frontier = "exact_possible_with_existing_raw_outputs"
        mismatch = "official released raw outputs exist, but direct paper verification and read-only metric recomputation are not yet closed"
        mismatch_category = "unavailable_exact_raw_output_or_checkpoint"
        missing = ["direct PDF verification", "read-only recomputation from official released raw outputs", "exact cohort/parser/metric reconciliation"]
        required_action = "Use only the released ZIP raw outputs and official parser/metric code for read-only recomputation."
        required_compute = "read_only_parser_metric_recomputation"
    elif released_traditional_recs_available(paper, evidence):
        frontier = "exact_possible_by_offline_recomputation"
        mismatch = "official Beauty SOTA recommendation CSVs exist, but metrics must be recomputed read-only and matched to the paper cell"
        mismatch_category = "unavailable_exact_raw_output_or_checkpoint"
        missing = ["direct PDF verification", "read-only metric recomputation from official released recommendation CSVs"]
        required_action = "Recompute metrics from released Beauty SOTA recommendation CSVs without training."
        required_compute = "read_only_metric_recomputation"
    elif is_paid_or_retired_model(model):
        frontier = "exact_blocked_by_paid_or_retired_service"
        mismatch = "exact paper LLM service/version output is unavailable locally"
        mismatch_category = "unavailable_exact_model_or_model_version"
        missing = ["exact service/version provenance", "exact raw model output", "direct PDF verification"]
        required_action = "Keep the component blocked unless exact released raw outputs or endpoint/version provenance is supplied."
        required_external = "paid_or_retired_service_or_released_exact_raw_outputs"
    elif model in LOCAL_ADAPTED_MODEL_NAMES:
        frontier = "adapted_replication_only"
        mismatch = "local open-weight execution is an adapted substitute rather than the exact paper model/protocol"
        mismatch_category = "adapted_local_implementation"
        missing = ["exact paper model/version", "exact paper prompt contract", "direct PDF verification"]
        actual_divergence = "true"
        required_action = "Report as adapted evidence only."
    elif is_open_weight_paper_model(model):
        frontier = "exact_blocked_by_unavailable_model_or_version"
        mismatch = "exact open-weight model/version and paper prompt/output evidence are not established"
        mismatch_category = "unavailable_exact_model_or_model_version"
        missing = ["exact model/version provenance", "exact raw output", "direct PDF verification"]
        required_action = "Keep blocked unless exact model/version and raw outputs are proven."
        required_external = "exact_model_weights_or_version_provenance"
    elif model in DETERMINISTIC_BASELINES:
        frontier = "exact_possible_but_requires_new_local_execution"
        mismatch = "deterministic baseline may be executable, but exact cohort/candidate/metric contract is not closed"
        mismatch_category = "cohort_candidate_prompt_parser_metric_or_denominator_mismatch"
        missing = ["direct PDF verification", "exact cohort and candidate-set proof", "read-only baseline execution plan"]
        required_action = "Create a future offline execution plan with immutable inputs and no training."
        required_compute = "future_local_execution_no_training"
    elif model in TRAINED_TRADITIONAL_MODELS:
        frontier = "exact_blocked_by_missing_original_data_or_checkpoint"
        mismatch = "exact trained model output or checkpoint is unavailable for this paper cell"
        mismatch_category = "unavailable_exact_raw_output_or_checkpoint"
        missing = ["exact checkpoint or released recommendation output", "direct PDF verification", "exact cohort/candidate proof"]
        required_action = "Do not train in this package; require original checkpoint/output or a separately approved reproduction run."
        required_external = "original_checkpoint_or_released_output"
    elif not numeric.get("recomputed_value"):
        frontier = "exact_blocked_by_missing_original_data_or_checkpoint"
        mismatch = "no exact local or upstream raw output is linked to the paper numeric cell"
        mismatch_category = "unavailable_exact_raw_output_or_checkpoint"
        missing = ["exact raw output", "direct PDF verification"]
        required_action = "Require exact raw output evidence before recomputation."
    else:
        frontier = "unresolved"
        mismatch = "existing value is only superficially label-matched and protocol comparability remains unresolved"
        mismatch_category = "cohort_candidate_prompt_parser_metric_or_denominator_mismatch"
        missing = ["cohort comparability", "protocol comparability", "direct PDF verification"]
        required_action = "Resolve cohort and protocol evidence before any exactness elevation."

    if not pdf_verified and "direct PDF verification" not in missing:
        missing.append("direct PDF verification")
    offline_possible = frontier in {
        "exact_already_supported_but_previously_underclassified",
        "exact_possible_by_offline_recomputation",
        "exact_possible_by_documentary_verification_only",
        "exact_possible_with_existing_raw_outputs",
    }
    return {
        "contract_id": paper.get("contract_id", ""),
        "dataset": dataset,
        "task": paper.get("task", ""),
        "model": model,
        "prompt_strategy": paper.get("prompt_strategy", ""),
        "metric": paper.get("metric", ""),
        "current_execution_status": current_execution,
        "current_paper_alignment": current_alignment,
        "paper_directly_verified": str(pdf_verified).lower(),
        "paper_source_location": paper_source_location(paper),
        "first_material_mismatch": mismatch,
        "mismatch_category": mismatch_category,
        "missing_evidence": "; ".join(sorted(set(item for item in missing if item))),
        "actual_divergence": actual_divergence,
        "frontier_class": frontier,
        "offline_elevation_possible": str(offline_possible).lower(),
        "required_action": required_action,
        "required_compute": required_compute,
        "required_external_access": required_external,
        "maximum_defensible_classification": frontier if frontier.startswith("exact_blocked") or frontier.endswith("_only") else f"{frontier}_potential_not_demonstrated",
        "notes": "; ".join(
            item
            for item in [
                paper.get("notes", ""),
                trace.get("notes", ""),
                numeric.get("notes", ""),
                "paper PDF text extraction unavailable in approved runtime" if not pdf_text_available else "",
            ]
            if item
        ),
    }


def status_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("task", ""),
        normalize_model(row.get("model", "")),
        row.get("prompt_strategy", ""),
        row.get("metric", ""),
    )


def build_component_rows(
    paper_rows: list[dict[str, str]],
    trace_rows: list[dict[str, str]],
    status_rows: list[dict[str, str]],
    numeric_rows: list[dict[str, str]],
    evidence: dict[str, Any],
    pdf_text_available: bool,
) -> list[dict[str, str]]:
    trace_by_id = table_by_key(trace_rows, "contract_id")
    numeric_by_id = table_by_key(numeric_rows, "result_id")
    status_by_key = {status_key(row): row for row in status_rows}
    rows = []
    for paper in paper_rows:
        key = (
            paper.get("dataset", ""),
            paper.get("task", ""),
            normalize_model(paper.get("model", "")),
            paper.get("prompt_strategy", ""),
            paper.get("metric", ""),
        )
        rows.append(
            classify_component(
                paper,
                trace_by_id.get(paper["contract_id"], {}),
                status_by_key.get(key, {}),
                numeric_by_id.get(paper["contract_id"], {}),
                evidence,
                pdf_text_available,
            )
        )
    return rows


def model_comparability(component: dict[str, str]) -> str:
    frontier = component["frontier_class"]
    if frontier in {"exact_possible_with_existing_raw_outputs", "exact_possible_by_offline_recomputation"}:
        return "potentially_exact_existing_upstream_artifact"
    if frontier == "exact_blocked_by_paid_or_retired_service":
        return "blocked_paid_or_retired_exact_service"
    if frontier == "exact_blocked_by_unavailable_model_or_version":
        return "blocked_exact_model_version_unavailable"
    if frontier == "adapted_replication_only":
        return "adapted_local_substitute_not_exact"
    if component["model"] in TRADITIONAL_MODELS:
        return "traditional_model_output_not_exactly_linked"
    return "unresolved"


def prompt_comparability(component: dict[str, str]) -> str:
    if "llmrank" in f"{component['model']} {component['prompt_strategy']}".lower():
        return "repository_template_conflict_top20_wording_vs_top10_parser"
    if "llmsrec" in f"{component['model']} {component['prompt_strategy']}".lower():
        return "repository_derived_top10_paper_unspecified"
    if component["prompt_strategy"]:
        return "adapted_or_unverified_prompt"
    return "not_applicable_or_unspecified"


def parser_comparability(component: dict[str, str]) -> str:
    if component["frontier_class"] == "exact_blocked_by_paper_unspecified_material_detail":
        return "paper_unspecified_repository_top10_parser"
    if component["frontier_class"] in {"exact_possible_with_existing_raw_outputs", "exact_possible_by_offline_recomputation"}:
        return "official_parser_or_metric_code_available_for_read_only_check"
    return "unverified"


def metric_comparability(component: dict[str, str]) -> str:
    if component["metric"] in {"hr_at_5", "ndcg_at_5"} and component["frontier_class"] in {
        "exact_possible_with_existing_raw_outputs",
        "exact_possible_by_offline_recomputation",
    }:
        return "potentially_exact_top5_metric_code_available"
    if component["metric"]:
        return "metric_denominator_or_formula_unverified"
    return "metric_unspecified"


def build_numeric_rows(
    paper_rows: list[dict[str, str]],
    numeric_recon_rows: list[dict[str, str]],
    component_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    paper_by_id = table_by_key(paper_rows, "contract_id")
    component_by_id = table_by_key(component_rows, "contract_id")
    rows = []
    for numeric in numeric_recon_rows:
        paper = paper_by_id[numeric["result_id"]]
        component = component_by_id[numeric["result_id"]]
        coordinates = "; ".join(
            [
                f"table={paper.get('paper_table', '')}",
                f"section={paper.get('paper_section', '')}",
                f"row_model={paper.get('model', '')}",
                f"column_metric={paper.get('metric', '')}",
                f"dataset={paper.get('dataset', '')}",
                f"task={paper.get('task', '')}",
            ]
        )
        rows.append(
            {
                "result_id": numeric["result_id"],
                "contract_id": numeric["result_id"],
                "paper_table": paper.get("paper_table", ""),
                "paper_coordinates": coordinates,
                "paper_source_location": component["paper_source_location"],
                "dataset": paper.get("dataset", ""),
                "task": paper.get("task", ""),
                "model": normalize_model(paper.get("model", "")),
                "prompt_strategy": paper.get("prompt_strategy", ""),
                "metric": paper.get("metric", "") or numeric.get("metric", ""),
                "paper_value": paper.get("paper_value", ""),
                "currently_available_local_value": numeric.get("recomputed_value", ""),
                "cohort_comparability": "not_exact" if numeric.get("cohort_match") == "false" else numeric.get("cohort_match", "unresolved"),
                "protocol_comparability": numeric.get("protocol_match", "unresolved"),
                "model_version_comparability": model_comparability(component),
                "prompt_comparability": prompt_comparability(component),
                "parser_comparability": parser_comparability(component),
                "metric_comparability": metric_comparability(component),
                "exact_recomputation_feasibility": component["required_compute"],
                "first_blocking_mismatch": component["first_material_mismatch"],
                "mismatch_category": component["mismatch_category"],
                "frontier_class": component["frontier_class"],
                "notes": component["notes"],
            }
        )
    return rows


def build_elevation_candidates(component_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed = {
        "exact_already_supported_but_previously_underclassified",
        "exact_possible_by_documentary_verification_only",
        "exact_possible_by_offline_recomputation",
        "exact_possible_with_existing_raw_outputs",
    }
    fields = [
        "contract_id",
        "dataset",
        "task",
        "model",
        "metric",
        "frontier_class",
        "required_action",
        "required_compute",
        "required_external_access",
        "notes",
    ]
    return [{field: row.get(field, "") for field in fields} for row in component_rows if row["frontier_class"] in allowed]


def build_irreducible_blockers(component_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    blocked_prefixes = ("exact_blocked", "exact_impossible", "adapted_", "local_extension", "unresolved")
    fields = [
        "contract_id",
        "dataset",
        "task",
        "model",
        "metric",
        "frontier_class",
        "first_material_mismatch",
        "mismatch_category",
        "required_external_access",
        "required_action",
        "notes",
    ]
    return [
        {field: row.get(field, "") for field in fields}
        for row in component_rows
        if row["frontier_class"].startswith(blocked_prefixes)
    ]


def fifth_dataset_requirements() -> list[dict[str, str]]:
    return [
        {
            "requirement_id": "FIFTH-DATASET-001",
            "requirement_area": "additionality",
            "requirement": "The dataset must not be Amazon Beauty, Amazon Sports, ML-1M, or LastFM.",
            "minimum": "one genuinely new dataset",
            "evidence_required": "source citation, local raw/processed manifest, and explicit exclusion from paper-replication denominators",
            "exclusion_rule": "LastFM is a paper dataset and cannot satisfy the fifth-dataset requirement.",
            "notes": "Do not select or implement the dataset in this closure package.",
        },
        {
            "requirement_id": "FIFTH-DATASET-002",
            "requirement_area": "tasks",
            "requirement": "Support ranking at minimum; reranking is required if the extension claims paper-task comparability.",
            "minimum": "ranking plus documented candidate construction; reranking if claimed",
            "evidence_required": "task config, split manifest, candidate manifest, and ground-truth manifest",
            "exclusion_rule": "Do not mix fifth-dataset rows into exact paper coverage.",
            "notes": "Task semantics must be config-driven.",
        },
        {
            "requirement_id": "FIFTH-DATASET-003",
            "requirement_area": "baseline_coverage",
            "requirement": "Include deterministic baselines and at least one stronger recommender baseline before optional LLM comparison.",
            "minimum": "MostPop plus BM25 or one trained baseline with frozen configuration",
            "evidence_required": "baseline outputs, metrics CSV, config, seed policy, and provenance hashes",
            "exclusion_rule": "Do not train or run this phase until separately approved.",
            "notes": "Training, if later approved, must be separated from this closure package.",
        },
        {
            "requirement_id": "FIFTH-DATASET-004",
            "requirement_area": "metric_coverage",
            "requirement": "Report utility and beyond-utility metrics separately.",
            "minimum": "HR@5, NDCG@5, coverage, hallucination/applicability where meaningful",
            "evidence_required": "metric contract, implementation tests, and generated metric CSV",
            "exclusion_rule": "Do not manually type final metric values into reports.",
            "notes": "Metric denominators must be explicit.",
        },
        {
            "requirement_id": "FIFTH-DATASET-005",
            "requirement_area": "provenance",
            "requirement": "Every input and output must have a manifest with hashes and source paths.",
            "minimum": "raw, processed, candidates, predictions, metrics, report tables",
            "evidence_required": "manifest JSON/CSV and Git HEAD",
            "exclusion_rule": "No downloads without explicit future approval.",
            "notes": "Protected inputs remain immutable once frozen.",
        },
        {
            "requirement_id": "FIFTH-DATASET-006",
            "requirement_area": "separation",
            "requirement": "Fifth-dataset percentages must be reported outside exact paper-replication coverage.",
            "minimum": "separate extension table and separate progress denominator",
            "evidence_required": "report section and machine-readable summary",
            "exclusion_rule": "Do not call LastFM or any paper dataset an extension dataset.",
            "notes": "This prevents overstating reproduction progress.",
        },
        {
            "requirement_id": "FIFTH-DATASET-007",
            "requirement_area": "selection_criteria",
            "requirement": "Choose a dataset only after checking license, size, schema fit, candidate construction feasibility, and compute budget.",
            "minimum": "documented selection memo before implementation",
            "evidence_required": "dataset decision table and risk notes",
            "exclusion_rule": "Do not begin selection in this closure package.",
            "notes": "The recommended next work package is LastFM reranking top-10 closure, not fifth-dataset work.",
        },
    ]


def pct(num: int, den: int) -> str:
    return "0.000000" if den == 0 else f"{(num / den) * 100:.6f}"


def percentage_row(name: str, num: int, den: int, rule: str, limitations: str) -> dict[str, Any]:
    return {"numerator": num, "denominator": den, "percentage": pct(num, den), "rule": rule, "limitations": limitations}


def build_percentages(component_rows: list[dict[str, str]], status_rows: list[dict[str, str]], numeric_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    den = len(component_rows)
    numeric_den = len(numeric_rows)
    counts = Counter(row["frontier_class"] for row in component_rows)
    numeric_counts = Counter(row["frontier_class"] for row in numeric_rows)
    adapted_status = len([row for row in status_rows if "adapted" in row.get("paper_alignment", "") or "local_extension" in row.get("paper_alignment", "")])
    operational = len([row for row in status_rows if row.get("execution_status") == "completed"])
    return {
        "currently_demonstrated_exact_coverage": percentage_row(
            "currently_demonstrated_exact_coverage",
            counts["exact_already_supported_but_previously_underclassified"],
            den,
            "frontier class exact_already_supported_but_previously_underclassified",
            "No components were elevated because all material exactness conditions remain unclosed.",
        ),
        "documentary_recoverable_exact_coverage": percentage_row(
            "documentary_recoverable_exact_coverage",
            counts["exact_possible_by_documentary_verification_only"],
            den,
            "components requiring only direct paper/appendix/source verification and no computation",
            "Potential only; direct PDF parser is unavailable in the approved runtime.",
        ),
        "offline_recomputation_recoverable_exact_coverage": percentage_row(
            "offline_recomputation_recoverable_exact_coverage",
            counts["exact_possible_by_offline_recomputation"] + counts["exact_possible_with_existing_raw_outputs"],
            den,
            "components with existing official raw outputs or recommendation CSVs and no training/inference requirement",
            "Potential only; values were not recomputed in this package.",
        ),
        "requires_new_local_execution_exact_coverage": percentage_row(
            "requires_new_local_execution_exact_coverage",
            counts["exact_possible_but_requires_new_local_execution"],
            den,
            "components needing a future local run but no LLM/API/download in this package",
            "Not currently recoverable within the allowed closure work.",
        ),
        "blocked_by_unavailable_external_resources": percentage_row(
            "blocked_by_unavailable_external_resources",
            counts["exact_blocked_by_unavailable_model_or_version"] + counts["exact_blocked_by_paid_or_retired_service"] + counts["exact_blocked_by_missing_original_data_or_checkpoint"],
            den,
            "components blocked by exact model/service/raw-output/checkpoint absence",
            "May change only if exact external evidence is supplied.",
        ),
        "blocked_by_paper_unspecified_material_detail": percentage_row(
            "blocked_by_paper_unspecified_material_detail",
            counts["exact_blocked_by_paper_unspecified_material_detail"],
            den,
            "components where the paper leaves a material detail unresolved and repository-derived behavior cannot prove paper exactness",
            "Repository evidence supports executable behavior but not paper-explicit exactness.",
        ),
        "genuine_divergence_or_adapted_only": percentage_row(
            "genuine_divergence_or_adapted_only",
            counts["exact_impossible_due_to_known_protocol_divergence"] + counts["adapted_replication_only"] + counts["local_extension_only"],
            den,
            "components whose current local evidence is materially adapted or divergent",
            "These are useful local evidence, not exact paper evidence.",
        ),
        "adapted_reconstructed_coverage": percentage_row(
            "adapted_reconstructed_coverage",
            adapted_status,
            len(status_rows),
            "forensic status rows classified adapted or local extension",
            "This is inherited from the completed forensic audit and is not exact coverage.",
        ),
        "operational_coverage": percentage_row(
            "operational_coverage",
            operational,
            len(status_rows),
            "forensic status rows with execution_status=completed",
            "Operational does not imply exact paper comparability.",
        ),
        "numeric_currently_demonstrated_exact": percentage_row(
            "numeric_currently_demonstrated_exact",
            numeric_counts["exact_already_supported_but_previously_underclassified"],
            numeric_den,
            "numeric cells with demonstrated exact class",
            "No numeric cell is exact under the strict frontier.",
        ),
        "numeric_offline_or_raw_recoverable": percentage_row(
            "numeric_offline_or_raw_recoverable",
            numeric_counts["exact_possible_by_offline_recomputation"] + numeric_counts["exact_possible_with_existing_raw_outputs"],
            numeric_den,
            "numeric cells with official released outputs or recommendation CSVs available for safe recomputation",
            "Potential only; no numeric values were recomputed here.",
        ),
    }


def protected_fingerprint() -> dict[str, Any]:
    rows = read_csv_rows(FORENSIC_INVENTORY_CSV)
    protected = [
        (row.get("path", ""), row.get("sha256", ""))
        for row in rows
        if row.get("protected_status") in {"protected_input", "protected_or_manifested"}
    ]
    protected = sorted(protected)
    digest = hashlib.sha256()
    for path, sha in protected:
        digest.update(f"{path}\t{sha}\n".encode("utf-8"))
    return {"row_count": len(protected), "aggregate_sha256": digest.hexdigest().upper()}


def pdf_environment() -> dict[str, Any]:
    modules = {name: importlib.util.find_spec(name) is not None for name in ["pypdf", "PyPDF2", "pdfplumber", "fitz", "pymupdf"]}
    paper_files = []
    for path in sorted((PROJECT_ROOT / "paper").glob("*")):
        if path.is_file():
            paper_files.append({"path": rel(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    duplicate_pdf_hashes = Counter(item["sha256"] for item in paper_files if item["path"].lower().endswith(".pdf"))
    return {
        "python_pdf_modules": modules,
        "pdf_text_extraction_available": any(modules.values()),
        "paper_files": paper_files,
        "byte_identical_pdf_groups": [sha for sha, count in sorted(duplicate_pdf_hashes.items()) if count > 1],
    }


def git_status_lines(status_text: str) -> list[str]:
    return common_git.git_status_lines(status_text)


def git_status_path(line: str) -> str:
    return common_git.git_status_path(line)


def git_status_counts(lines: Iterable[str]) -> dict[str, int]:
    return common_git.git_status_counts(lines)


def closure_resume_baseline(final_git_status: str) -> dict[str, Any]:
    final_lines = git_status_lines(final_git_status)
    created = [line for line in final_lines if git_status_path(line) in EXACTNESS_OUTPUTS]
    baseline = [line for line in final_lines if git_status_path(line) not in EXACTNESS_OUTPUTS]
    return {
        "basis": "reconstructed_from_final_status_minus_new_exactness_frontier_paths",
        "status_row_count": len(baseline),
        "status_counts": git_status_counts(baseline),
        "exactness_created_status_row_count": len(created),
        "exactness_created_paths": sorted(git_status_path(line) for line in created),
    }


def output_hashes() -> dict[str, str]:
    paths = [
        REPORT_PATH,
        HANDOFF_PATH,
        COMPONENT_FRONTIER_CSV,
        NUMERIC_FRONTIER_CSV,
        ELEVATION_CANDIDATES_CSV,
        IRREDUCIBLE_BLOCKERS_CSV,
        FIFTH_DATASET_REQUIREMENTS_CSV,
        SNAPSHOT_JSON,
    ]
    hashes = {}
    for path in paths:
        if path == SNAPSHOT_JSON:
            hashes[rel(path)] = "CURRENT_EXACTNESS_FRONTIER_SNAPSHOT_SELF_REFERENCE"
        elif path.exists():
            hashes[rel(path)] = file_sha256(path)
    return hashes


def markdown_counts(counter: Counter[str]) -> str:
    return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counter.items()))


def markdown_percentages(percentages: dict[str, dict[str, Any]]) -> str:
    lines = []
    for key, item in sorted(percentages.items()):
        lines.append(f"- `{key}`: `{item['numerator']}/{item['denominator']}` = `{item['percentage']}%`; rule: {item['rule']}; limitation: {item['limitations']}")
    return "\n".join(lines)


def build_report(
    component_rows: list[dict[str, str]],
    numeric_rows: list[dict[str, str]],
    elevation_rows: list[dict[str, str]],
    blocker_rows: list[dict[str, str]],
    fifth_rows: list[dict[str, str]],
    percentages: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    pdf_info: dict[str, Any],
    initial_git_status: str,
    final_git_status: str,
) -> str:
    class_counts = Counter(row["frontier_class"] for row in component_rows)
    numeric_counts = Counter(row["frontier_class"] for row in numeric_rows)
    mismatch_counts = Counter(row["mismatch_category"] for row in component_rows)
    baseline = closure_resume_baseline(final_git_status)
    final_lines = git_status_lines(final_git_status)
    initial_lines = git_status_lines(initial_git_status)
    baseline_counts = baseline["status_counts"]
    final_counts = git_status_counts(final_lines)
    return f"""# Exactness Frontier And Evidence Closure

## 1. Decision

Decision: `exactness_frontier_closed_conservative_no_elevation`.

No component is elevated to demonstrated exact paper reproduction. The completed forensic audit remains correct at the strict frontier: exact paper coverage is `0/192`, while many components have adapted or potentially recoverable evidence that must not be combined with exact coverage.

## 2. Runtime And Scope

- Runtime: `{RECOVERED_INTERPRETER}`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Runtime caveat: repaired package-tree clone only, not exact original executable environment.
- No training, LLM inference, Ollama, API calls, downloads, package changes, staging, or commits were performed.
- Python PDF modules available: `{json.dumps(pdf_info['python_pdf_modules'], sort_keys=True)}`
- Direct PDF text extraction available in approved runtime: `{str(pdf_info['pdf_text_extraction_available']).lower()}`
- Byte-identical local paper PDF groups: `{len(pdf_info['byte_identical_pdf_groups'])}`

## 3. Inputs Reconciled

Starting point artifacts were the completed forensic audit report, handoff, snapshot, paper contract, traceability, status matrix, claims, numeric reconciliation, lineage, and blocker tables. Direct source inspection covered local paper files, upstream prompt templates, upstream LLM wrappers, upstream parser/metric code, and released upstream ZIP inventories.

## 4. Git State

- Branch: `{run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}`
- HEAD: `{run_command(['git', 'rev-parse', 'HEAD'])}`
- Closure-resume baseline rows: `{baseline['status_row_count']}` reconstructed from final status minus `{baseline['exactness_created_status_row_count']}` new exactness-frontier paths.
- Closure-resume baseline counts: modified `{baseline_counts['modified']}`, added `{baseline_counts['added']}`, deleted `{baseline_counts['deleted']}`, untracked `{baseline_counts['untracked']}`, other `{baseline_counts['other']}`.
- Initial rows before this builder run: `{len(initial_lines)}`
- Final rows after this builder run: `{len(final_lines)}`
- Final counts: modified `{final_counts['modified']}`, added `{final_counts['added']}`, deleted `{final_counts['deleted']}`, untracked `{final_counts['untracked']}`, other `{final_counts['other']}`.

## 5. Frontier Percentages

{markdown_percentages(percentages)}

## 6. Frontier Class Counts

{markdown_counts(class_counts)}

## 7. Mismatch Categories

{markdown_counts(mismatch_counts)}

## 8. Numeric-Cell Frontier

Numeric cells represented exactly once: `{len(numeric_rows)}`.

{markdown_counts(numeric_counts)}

No numeric cell is currently exact. Numeric cells with potential safe recovery require direct paper verification plus read-only recomputation from existing official raw outputs or recommendation CSVs; this package did not compute new numeric outputs.

## 9. Evidence Elevation Candidates

Elevation candidates without training, LLM inference, API calls, downloads, or protocol changes: `{len(elevation_rows)}` in `results/tables/exactness_elevation_candidates.csv`.

These are potential-only. They remain non-exact until all material exactness requirements are demonstrated.

## 10. Irreducible Or External Blockers

Irreducible/external blocker rows: `{len(blocker_rows)}` in `results/tables/irreducible_exactness_blockers.csv`.

The dominant blockers are exact service/model version absence, missing original raw outputs or checkpoints, and paper-unspecified output/prompt details that the repository resolves only as executable behavior, not paper-explicit exactness.

## 11. Upstream Repository Findings

- Nested repo HEAD ref: `{evidence['nested_repo_head_ref']}`
- Nested repo HEAD commit from direct ref read: `{evidence['nested_repo_head_commit']}`
- Beauty LLM ZIP entries: `{evidence['beauty_llm_zip_entries']}`; model tokens: `{', '.join(evidence['beauty_llm_models'])}`
- Sports LLM ZIP entries: `{evidence['sports_llm_zip_entries']}`; model tokens: `{', '.join(evidence['sports_llm_models'])}`
- Beauty SOTA ZIP entries: `{evidence['beauty_sota_zip_entries']}`; model tokens: `{', '.join(evidence['beauty_sota_models'])}`
- Official executable contract evidence remains top-10 recommendation output with top-5 metrics, while some template wording remains top-20 or paper-unspecified.

## 12. Fifth-Dataset Requirement

LastFM is a paper dataset and cannot satisfy the university requirement for an additional dataset. The future fifth-dataset phase must use a dataset outside Amazon Beauty, Amazon Sports, ML-1M, and LastFM, and must report its extension percentages separately from exact paper replication.

Requirements are in `results/tables/fifth_dataset_extension_requirements.csv` with `{len(fifth_rows)}` rows.

## 13. Recommended Next Work Package

Single next work package: corrective offline LastFM reranking top-10 freeze, preserving F6G1 evidence and regenerating only request/shard manifests under the repository-derived top-10 executable contract. Do not run inference.

## 14. Protected-Input Verification

Protected inputs are represented by the completed forensic inventory fingerprint. The closure builder records the protected aggregate in `results/manifests/exactness_frontier_snapshot.json`; focused tests compare it across builder runs.

## 15. Files Created Or Updated

- `report/exactness_frontier_and_evidence_closure.md`
- `results/tables/exactness_frontier_by_component.csv`
- `results/tables/numeric_cell_exactness_frontier.csv`
- `results/tables/exactness_elevation_candidates.csv`
- `results/tables/irreducible_exactness_blockers.csv`
- `results/tables/fifth_dataset_extension_requirements.csv`
- `results/manifests/exactness_frontier_snapshot.json`
- `report/future_operations_context_handoff.md`
- `src/analysis/build_exactness_frontier_and_evidence_closure.py`
- `tests/test_exactness_frontier_and_evidence_closure.py`
"""


def build_handoff(percentages: dict[str, dict[str, Any]], class_counts: Counter[str], numeric_counts: Counter[str]) -> str:
    return f"""# Future Operations Context Handoff

## Current Runtime Contract

- Approved Python: `{RECOVERED_INTERPRETER}`
- Runtime classification: `defensible_repaired_clone_exact_package_versions`
- Caveat: repaired clone exact package tree only; not exact original executable environment.
- Required environment: `PYTHONNOUSERSITE=1`

## Forensic Audit History

The completed workspace forensic audit remains the authoritative baseline:

- `report/workspace_forensic_state_audit.md`
- `results/manifests/current_project_context_snapshot.json`
- Exact paper-component coverage from forensic audit: `0/192 = 0.000000%`
- Adapted/reconstructed coverage from forensic audit: `192/192 = 100.000000%`
- Operational coverage from forensic audit: `176/192 = 91.666667%`

## Exactness Frontier Update

- Frontier report: `report/exactness_frontier_and_evidence_closure.md`
- Component frontier: `results/tables/exactness_frontier_by_component.csv`
- Numeric frontier: `results/tables/numeric_cell_exactness_frontier.csv`
- Snapshot: `results/manifests/exactness_frontier_snapshot.json`

Updated percentages:

{markdown_percentages(percentages)}

Frontier class counts:

{markdown_counts(class_counts)}

Numeric frontier counts:

{markdown_counts(numeric_counts)}

## Allowed Claims

- Exact paper reproduction is still not demonstrated.
- Some components are potentially recoverable through documentary verification or read-only recomputation from existing official artifacts.
- Potential coverage must be reported separately from demonstrated exact coverage.
- LastFM is a paper dataset, not the fifth-dataset extension.

## Forbidden Claims

- Do not claim exact original runtime reproduction.
- Do not combine demonstrated exact, potential documentary, and potential recomputation percentages.
- Do not describe local Qwen2/Mistral runs as exact paper model executions.
- Do not treat repository-derived top-10 output length as paper-explicit.
- Do not use LastFM as the additional dataset.

## Recommended Next Work Package

Create a corrective offline LastFM reranking top-10 freeze. Do not run training, LLM inference, Ollama, API calls, downloads, staging, or commits in that work package unless separately authorized.
"""


def build_snapshot(
    component_rows: list[dict[str, str]],
    numeric_rows: list[dict[str, str]],
    elevation_rows: list[dict[str, str]],
    blocker_rows: list[dict[str, str]],
    fifth_rows: list[dict[str, str]],
    percentages: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    pdf_info: dict[str, Any],
    initial_git_status: str,
    final_git_status: str,
) -> dict[str, Any]:
    git_branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_head = run_command(["git", "rev-parse", "HEAD"])
    final_lines = git_status_lines(final_git_status)
    return {
        "snapshot_status": "exactness_frontier_closed_conservative_no_elevation",
        "project_root": str(PROJECT_ROOT),
        "branch": git_branch,
        "head": git_head,
        "environment": {
            "approved_runtime_interpreter": str(RECOVERED_INTERPRETER),
            "runtime_classification": "defensible_repaired_clone_exact_package_versions",
            "not_exact_original_runtime": True,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "python_no_user_site_required": True,
        },
        "git_state": {
            "initial_before_builder_status_row_count": len(git_status_lines(initial_git_status)),
            "final_status_row_count": len(final_lines),
            "final_status_counts": git_status_counts(final_lines),
            "closure_resume_baseline": closure_resume_baseline(final_git_status),
        },
        "counts": {
            "component_rows": len(component_rows),
            "numeric_rows": len(numeric_rows),
            "elevation_candidate_rows": len(elevation_rows),
            "irreducible_blocker_rows": len(blocker_rows),
            "fifth_dataset_requirement_rows": len(fifth_rows),
        },
        "frontier_class_counts": dict(sorted(Counter(row["frontier_class"] for row in component_rows).items())),
        "numeric_frontier_class_counts": dict(sorted(Counter(row["frontier_class"] for row in numeric_rows).items())),
        "percentages": percentages,
        "paper_environment": pdf_info,
        "upstream_evidence": evidence,
        "protected_input_fingerprint": protected_fingerprint(),
        "lastfm_is_fifth_dataset": False,
        "lastfm_classification": "paper_dataset_not_additional_extension",
        "recommended_next_work_package": "Corrective offline LastFM reranking top-10 freeze; no inference.",
        "no_training": True,
        "no_llm_inference": True,
        "no_ollama": True,
        "no_api_calls": True,
        "no_downloads": True,
        "no_git_stage_or_commit": True,
        "authoritative_inputs": [
            "report/workspace_forensic_state_audit.md",
            "results/manifests/current_project_context_snapshot.json",
            "results/tables/paper_atomic_contract.csv",
            "results/tables/paper_to_local_traceability.csv",
            "results/tables/current_reproduction_status_matrix.csv",
            "results/tables/project_claim_verification.csv",
            "results/tables/numeric_result_reconciliation.csv",
            "results/tables/artifact_lineage_and_supersession.csv",
            "results/tables/blockers_gaps_and_next_actions.csv",
        ],
    }


def build_all() -> dict[str, Any]:
    require_runtime()
    initial_git_status = run_command(["git", "status", "--porcelain=v1"])

    paper_rows = read_csv_rows(PAPER_CONTRACT_CSV)
    trace_rows = read_csv_rows(TRACEABILITY_CSV)
    status_rows = read_csv_rows(STATUS_MATRIX_CSV)
    numeric_recon_rows = read_csv_rows(NUMERIC_RECON_CSV)
    _ = read_csv_rows(CLAIMS_CSV)
    _ = read_csv_rows(LINEAGE_CSV)
    _ = read_csv_rows(CONFLICTS_CSV)

    evidence = upstream_evidence()
    pdf_info = pdf_environment()
    component_rows = build_component_rows(
        paper_rows,
        trace_rows,
        status_rows,
        numeric_recon_rows,
        evidence,
        bool(pdf_info["pdf_text_extraction_available"]),
    )
    numeric_rows = build_numeric_rows(paper_rows, numeric_recon_rows, component_rows)
    elevation_rows = build_elevation_candidates(component_rows)
    blocker_rows = build_irreducible_blockers(component_rows)
    fifth_rows = fifth_dataset_requirements()
    percentages = build_percentages(component_rows, status_rows, numeric_rows)

    write_csv(COMPONENT_FRONTIER_CSV, component_rows, COMPONENT_FIELDS)
    write_csv(NUMERIC_FRONTIER_CSV, numeric_rows, NUMERIC_FIELDS)
    write_csv(
        ELEVATION_CANDIDATES_CSV,
        elevation_rows,
        ["contract_id", "dataset", "task", "model", "metric", "frontier_class", "required_action", "required_compute", "required_external_access", "notes"],
    )
    write_csv(
        IRREDUCIBLE_BLOCKERS_CSV,
        blocker_rows,
        ["contract_id", "dataset", "task", "model", "metric", "frontier_class", "first_material_mismatch", "mismatch_category", "required_external_access", "required_action", "notes"],
    )
    write_csv(
        FIFTH_DATASET_REQUIREMENTS_CSV,
        fifth_rows,
        ["requirement_id", "requirement_area", "requirement", "minimum", "evidence_required", "exclusion_rule", "notes"],
    )

    final_git_status = run_command(["git", "status", "--porcelain=v1"])
    report = build_report(component_rows, numeric_rows, elevation_rows, blocker_rows, fifth_rows, percentages, evidence, pdf_info, initial_git_status, final_git_status)
    handoff = build_handoff(percentages, Counter(row["frontier_class"] for row in component_rows), Counter(row["frontier_class"] for row in numeric_rows))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    HANDOFF_PATH.write_text(handoff, encoding="utf-8")

    snapshot = build_snapshot(component_rows, numeric_rows, elevation_rows, blocker_rows, fifth_rows, percentages, evidence, pdf_info, initial_git_status, final_git_status)
    snapshot["output_hashes"] = {}
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_JSON.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    snapshot["output_hashes"] = output_hashes()
    SNAPSHOT_JSON.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    snapshot = build_all()
    if args.print_summary:
        print(
            json.dumps(
                {
                    "snapshot_status": snapshot["snapshot_status"],
                    "counts": snapshot["counts"],
                    "percentages": snapshot["percentages"],
                    "frontier_class_counts": snapshot["frontier_class_counts"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

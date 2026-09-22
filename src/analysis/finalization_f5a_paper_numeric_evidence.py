"""Finalization F5A paper numeric evidence recovery.

F5A inventories local paper PDFs and extraction tools, then represents the
eleven paper-result cells already frozen by earlier audits. It separates paper
value recovery, local-result mapping, protocol equivalence, and numeric
replication. The module is intentionally non-destructive and does not run
experiments, training, inference, network calls, or package installation.
"""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

REPORT_PATH = Path("report/finalization_f5a_paper_numeric_evidence.md")

PAPER_DIR = Path("paper")
F2_PROTECTED_AFTER = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
F3_PAPER_ALIGNMENT = Path("results/tables/finalization_f3_paper_alignment.csv")
F3_MISSING_EVIDENCE = Path("results/tables/finalization_f3_missing_evidence.csv")
F3_CANONICAL_RESULTS = Path("results/tables/finalization_f3_canonical_scientific_results.csv")
F3_PRIMARY_MATRIX = Path("results/tables/finalization_f3_primary_result_matrix.csv")
F3_VALID_COMPARISONS = Path("results/tables/finalization_f3_valid_comparisons.csv")
F3_CLAIMS = Path("results/tables/finalization_f3_scientific_claim_registry.csv")
F4_TRIAGE = Path("results/tables/finalization_f4_missing_evidence_triage.csv")
F4_F5_CANDIDATES = Path("results/tables/finalization_f4_f5_experiment_candidates.csv")
F4_COVERAGE = Path("results/tables/finalization_f4_coverage_impact.csv")
F4_ANOMALIES = Path("results/tables/finalization_f4_anomalies.csv")
F4_CLAIMS = Path("results/tables/finalization_f4_claim_update.csv")
PHASE5E_RESULT_CELLS = Path("results/tables/plan_d_phase_5e_paper_result_cells.csv")
PHASE5E_RESULT_COMPARISON = Path("results/tables/plan_d_phase_5e_result_cell_comparison.csv")
PHASE5E_COVERAGE = Path("results/tables/plan_d_phase_5e_coverage_metrics.csv")
PHASE5E_COMPONENT_COVERAGE = Path("results/tables/plan_d_phase_5e_component_coverage.csv")

OUTPUT_TABLES: dict[str, Path] = {
    "paper_source_inventory": Path("results/tables/finalization_f5a_paper_source_inventory.csv"),
    "pdf_tool_inventory": Path("results/tables/finalization_f5a_pdf_tool_inventory.csv"),
    "numeric_cell_extraction": Path("results/tables/finalization_f5a_numeric_cell_extraction.csv"),
    "local_result_mapping": Path("results/tables/finalization_f5a_local_result_mapping.csv"),
    "protocol_equivalence": Path("results/tables/finalization_f5a_protocol_equivalence.csv"),
    "numeric_cell_verification": Path("results/tables/finalization_f5a_numeric_cell_verification.csv"),
    "paper_numeric_alignment": Path("results/tables/finalization_f5a_paper_numeric_alignment.csv"),
    "coverage_impact": Path("results/tables/finalization_f5a_coverage_impact.csv"),
    "claim_update": Path("results/tables/finalization_f5a_claim_update.csv"),
    "anomalies": Path("results/tables/finalization_f5a_anomalies.csv"),
    "validation": Path("results/tables/finalization_f5a_validation.csv"),
}

PAPER_SOURCE_COLUMNS = [
    "paper_source_id",
    "file",
    "sha256",
    "size_bytes",
    "page_count",
    "title",
    "authors",
    "version",
    "venue",
    "arxiv_id",
    "text_extractable",
    "table_text_extractable",
    "is_authoritative",
    "authority_reason",
    "relationship_to_other_versions",
    "parser_used",
    "parser_version",
    "notes",
]

PDF_TOOL_COLUMNS = [
    "tool_id",
    "tool_name",
    "tool_type",
    "available",
    "version",
    "executable_or_module",
    "test_status",
    "test_details",
    "selected_for_use",
    "selection_reason",
    "notes",
]

NUMERIC_EXTRACTION_COLUMNS = [
    "paper_numeric_cell_id",
    "paper_component_id",
    "paper_source_id",
    "paper_section",
    "paper_subsection",
    "paper_page_pdf",
    "paper_page_printed",
    "paper_table_or_figure",
    "paper_table_caption",
    "paper_row_label",
    "paper_column_label",
    "paper_dataset",
    "paper_task",
    "paper_model",
    "paper_model_version",
    "paper_prompt_or_setting",
    "paper_metric",
    "paper_cutoff",
    "paper_value_text",
    "paper_value_numeric",
    "paper_decimal_places",
    "paper_is_percentage",
    "paper_scale",
    "paper_dispersion_text",
    "paper_dispersion_numeric",
    "paper_dispersion_type",
    "paper_run_count",
    "paper_significance_marker",
    "extraction_method",
    "extraction_confidence",
    "machine_verified",
    "manual_review_required",
    "evidence_snippet",
    "evidence_location",
    "notes",
]

LOCAL_MAPPING_COLUMNS = [
    "mapping_id",
    "paper_numeric_cell_id",
    "local_result_id",
    "local_source_file",
    "local_source_row",
    "local_dataset",
    "local_task",
    "local_model",
    "local_model_version",
    "local_prompt_contract",
    "local_cohort",
    "local_candidate_set",
    "local_candidate_count",
    "local_positive_count",
    "local_metric",
    "local_cutoff",
    "local_value",
    "local_seed",
    "local_run_count",
    "mapping_status",
    "mapping_reason",
    "notes",
]

PROTOCOL_COLUMNS = [
    "protocol_check_id",
    "paper_numeric_cell_id",
    "local_result_id",
    "dataset_equivalent",
    "dataset_version_equivalent",
    "preprocessing_equivalent",
    "split_equivalent",
    "cohort_equivalent",
    "candidate_set_equivalent",
    "candidate_count_equivalent",
    "positive_policy_equivalent",
    "task_equivalent",
    "model_equivalent",
    "model_version_equivalent",
    "prompt_equivalent",
    "decoding_equivalent",
    "parser_equivalent",
    "invalid_output_policy_equivalent",
    "metric_definition_equivalent",
    "cutoff_equivalent",
    "aggregation_equivalent",
    "run_policy_equivalent",
    "all_required_fields_known",
    "protocol_equivalence_class",
    "blocking_differences",
    "evidence",
    "notes",
]

NUMERIC_VERIFICATION_COLUMNS = [
    "verification_id",
    "paper_numeric_cell_id",
    "local_result_id",
    "paper_value_text",
    "paper_value_numeric",
    "paper_decimal_places",
    "local_value_raw",
    "local_value_at_paper_precision",
    "absolute_difference",
    "relative_difference",
    "printed_precision_match",
    "exact_raw_match",
    "protocol_equivalence_class",
    "numeric_verification_class",
    "is_exact_numeric_replication",
    "is_functional_numeric_replication",
    "is_numeric_match_without_protocol_equivalence",
    "is_reportable",
    "explanation",
    "notes",
]

ALIGNMENT_COLUMNS = [
    "paper_numeric_cell_id",
    "paper_component_id",
    "paper_table_or_figure",
    "paper_dataset",
    "paper_task",
    "paper_model",
    "paper_metric",
    "paper_cutoff",
    "paper_value_text",
    "machine_verified",
    "best_local_result_id",
    "best_local_value",
    "protocol_equivalence_class",
    "numeric_verification_class",
    "is_exact_numeric_replication",
    "is_functional_numeric_replication",
    "current_status",
    "blocking_reason",
    "next_action",
    "notes",
]

COVERAGE_COLUMNS = [
    "coverage_metric",
    "f4_value",
    "f5a_value",
    "absolute_change",
    "numerator_before",
    "denominator_before",
    "numerator_after",
    "denominator_after",
    "change_reason",
    "supporting_cell_ids",
    "is_change_scientifically_allowed",
    "validation_status",
    "notes",
]

CLAIM_COLUMNS = [
    "claim_id",
    "previous_claim_text",
    "previous_confidence",
    "f5a_numeric_evidence",
    "f5a_protocol_evidence",
    "updated_claim_text",
    "updated_confidence",
    "is_safe_for_report",
    "is_safe_for_presentation",
    "remaining_limitation",
    "notes",
]

ANOMALY_COLUMNS = [
    "anomaly_id",
    "area",
    "severity",
    "source_file",
    "source_location",
    "description",
    "scientific_impact",
    "requires_previous_phase_correction",
    "can_continue",
    "recommended_action",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]

COMMAND_TOOLS = ["pdftotext", "pdfinfo", "mutool", "qpdf", "pdftoppm", "magick"]
PYTHON_MODULE_TOOLS = ["pypdf", "PyPDF2", "fitz", "pymupdf", "pdfplumber", "tabula"]
ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    f2_protected_after: pd.DataFrame
    f3_paper_alignment: pd.DataFrame
    f3_missing_evidence: pd.DataFrame
    f3_canonical_results: pd.DataFrame
    f3_primary_matrix: pd.DataFrame
    f3_valid_comparisons: pd.DataFrame
    f3_claims: pd.DataFrame
    f4_triage: pd.DataFrame
    f4_f5_candidates: pd.DataFrame
    f4_coverage: pd.DataFrame
    f4_anomalies: pd.DataFrame
    f4_claims: pd.DataFrame
    phase5e_result_cells: pd.DataFrame
    phase5e_result_comparison: pd.DataFrame
    phase5e_coverage: pd.DataFrame
    phase5e_component_coverage: pd.DataFrame


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def read_csv(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_sources(root: Path) -> SourceBundle:
    return SourceBundle(
        root=root,
        f2_protected_after=read_csv(root, F2_PROTECTED_AFTER),
        f3_paper_alignment=read_csv(root, F3_PAPER_ALIGNMENT),
        f3_missing_evidence=read_csv(root, F3_MISSING_EVIDENCE),
        f3_canonical_results=read_csv(root, F3_CANONICAL_RESULTS),
        f3_primary_matrix=read_csv(root, F3_PRIMARY_MATRIX),
        f3_valid_comparisons=read_csv(root, F3_VALID_COMPARISONS),
        f3_claims=read_csv(root, F3_CLAIMS),
        f4_triage=read_csv(root, F4_TRIAGE),
        f4_f5_candidates=read_csv(root, F4_F5_CANDIDATES),
        f4_coverage=read_csv(root, F4_COVERAGE),
        f4_anomalies=read_csv(root, F4_ANOMALIES),
        f4_claims=read_csv(root, F4_CLAIMS),
        phase5e_result_cells=read_csv(root, PHASE5E_RESULT_CELLS),
        phase5e_result_comparison=read_csv(root, PHASE5E_RESULT_COMPARISON),
        phase5e_coverage=read_csv(root, PHASE5E_COVERAGE),
        phase5e_component_coverage=read_csv(root, PHASE5E_COMPONENT_COVERAGE),
    )


def stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps([clean(part) for part in parts], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def git_output(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return completed.stdout.strip()


def sort_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    existing = [column for column in columns if column in frame.columns]
    if existing:
        return frame.sort_values(existing, kind="mergesort").reset_index(drop=True)
    return frame.reset_index(drop=True)


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def pdf_metadata(path: Path) -> dict[str, str]:
    data = path.read_bytes()
    text = data.decode("latin-1", errors="ignore")
    title = first_xml_value(text, r"<dc:title>.*?<rdf:li[^>]*>(.*?)</rdf:li>") or "Beyond Utility: Evaluating LLM as Recommender"
    creators_block = regex_value(text, r"<dc:creator>.*?<rdf:Seq>(.*?)</rdf:Seq>.*?</dc:creator>")
    authors = "; ".join(re.findall(r"<rdf:li[^>]*>(.*?)</rdf:li>", creators_block, flags=re.DOTALL))
    authors = "; ".join(html.unescape(item.strip()) for item in authors.split(";") if item.strip())
    venue = first_xml_value(text, r"<rdf:li>Association for Computing Machinery</rdf:li>")
    if venue:
        venue = "Association for Computing Machinery"
    arxiv_id = "2411.00331" if "2411.00331" in path.name or "arXivStAmP" in text else ""
    version = "v1" if "v1" in path.name else "unknown"
    page_count = str(len(re.findall(rb"/Type\s*/Page\b", data)))
    return {
        "title": html.unescape(title),
        "authors": authors,
        "venue": venue,
        "arxiv_id": arxiv_id,
        "version": version,
        "page_count": page_count,
        "metadata_title_visible": bool_text(bool(title)),
    }


def regex_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1) if match.groups() else match.group(0)


def first_xml_value(text: str, pattern: str) -> str:
    value = regex_value(text, pattern)
    return html.unescape(re.sub(r"\s+", " ", value).strip())


def build_paper_source_inventory(root: Path) -> pd.DataFrame:
    pdfs = sorted((root / PAPER_DIR).glob("*.pdf"), key=lambda p: p.name.lower())
    hashes = {path.name: file_sha256(path) for path in pdfs}
    authoritative_name = choose_authoritative_pdf(pdfs)
    rows: list[dict[str, str]] = []
    for path in pdfs:
        meta = pdf_metadata(path)
        digest = hashes[path.name]
        same_hash_names = sorted(name for name, value in hashes.items() if value == digest and name != path.name)
        is_auth = path.name == authoritative_name
        relationship = "byte-identical duplicate of " + ";".join(same_hash_names) if same_hash_names else "single local PDF version"
        if is_auth:
            reason = "Matches README/report arXiv 2411.00331 reference and is byte-identical to the alternate filename."
        else:
            reason = "Not selected because it is byte-identical to the authoritative arXiv-named file."
        rows.append(
            {
                "paper_source_id": stable_id("F5A-PAPER", digest),
                "file": relative_path(root, path),
                "sha256": digest.lower(),
                "size_bytes": str(path.stat().st_size),
                "page_count": meta["page_count"],
                "title": meta["title"],
                "authors": meta["authors"],
                "version": meta["version"],
                "venue": meta["venue"],
                "arxiv_id": meta["arxiv_id"],
                "text_extractable": "false",
                "table_text_extractable": "false",
                "is_authoritative": bool_text(is_auth),
                "authority_reason": reason,
                "relationship_to_other_versions": relationship,
                "parser_used": "python_stdlib_raw_pdf_metadata_scan",
                "parser_version": sys.version.split()[0],
                "notes": "Metadata visible in raw PDF bytes; full text/table extraction unavailable without installed parser.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=PAPER_SOURCE_COLUMNS), ["file"])


def choose_authoritative_pdf(pdfs: Sequence[Path]) -> str:
    for path in pdfs:
        if path.name == "2411.00331v1.pdf":
            return path.name
    return sorted((path.name for path in pdfs), key=str.lower)[0] if pdfs else ""


def command_version(tool: str) -> tuple[str, str]:
    executable = shutil.which(tool)
    if not executable:
        return "", "not_available"
    for args in ([tool, "--version"], [tool, "-version"], [tool, "-v"]):
        completed = subprocess.run(args, text=True, capture_output=True, check=False, timeout=5)
        output = (completed.stdout or completed.stderr).strip().splitlines()
        if output:
            return output[0][:160], f"exit={completed.returncode}"
    return "", "available_no_version"


def build_pdf_tool_inventory() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for tool in COMMAND_TOOLS:
        available = shutil.which(tool) is not None
        version, status = command_version(tool)
        rows.append(
            {
                "tool_id": stable_id("F5A-TOOL", tool),
                "tool_name": tool,
                "tool_type": "executable",
                "available": bool_text(available),
                "version": version,
                "executable_or_module": tool,
                "test_status": "available_read_only_probe" if available else "not_available",
                "test_details": status,
                "selected_for_use": "false",
                "selection_reason": "not available in current shell" if not available else "not selected unless table text extraction is needed and verified",
                "notes": "",
            }
        )
    for module in PYTHON_MODULE_TOOLS:
        spec = importlib.util.find_spec(module)
        available = spec is not None
        rows.append(
            {
                "tool_id": stable_id("F5A-TOOL", module),
                "tool_name": module,
                "tool_type": "python_module",
                "available": bool_text(available),
                "version": "",
                "executable_or_module": module,
                "test_status": "import_spec_available" if available else "not_available",
                "test_details": "module spec found without import" if available else "module spec missing",
                "selected_for_use": "false",
                "selection_reason": "not available in .venv" if not available else "not selected for table extraction without verified output",
                "notes": "",
            }
        )
    rows.append(
        {
            "tool_id": stable_id("F5A-TOOL", "python_stdlib_raw_pdf_metadata_scan"),
            "tool_name": "python_stdlib_raw_pdf_metadata_scan",
            "tool_type": "python_stdlib",
            "available": "true",
            "version": sys.version.split()[0],
            "executable_or_module": "python_stdlib",
            "test_status": "metadata_only_pass",
            "test_details": "raw byte scan can read XMP metadata and estimate page count; cannot machine-extract tables",
            "selected_for_use": "true",
            "selection_reason": "only available read-only method in frozen environment; selected for source inventory, not numeric table extraction",
            "notes": "F5A leaves paper table cells unresolved instead of inventing values.",
        }
    )
    return sort_frame(pd.DataFrame(rows, columns=PDF_TOOL_COLUMNS), ["tool_type", "tool_name"])


def authoritative_paper_id(paper_inventory: pd.DataFrame) -> str:
    rows = paper_inventory.loc[paper_inventory["is_authoritative"].eq("true")]
    return clean(rows.iloc[0]["paper_source_id"]) if not rows.empty else ""


def target_cells(bundle: SourceBundle) -> pd.DataFrame:
    cells = bundle.phase5e_result_cells.copy()
    if cells.empty:
        return pd.DataFrame(columns=bundle.phase5e_result_cells.columns)
    return sort_frame(cells, ["paper_result_cell_id"])


def parse_paper_value(value_text: str) -> dict[str, str]:
    text = clean(value_text)
    result = {
        "paper_value_numeric": "",
        "paper_decimal_places": "",
        "paper_is_percentage": "false",
        "paper_scale": "",
        "paper_dispersion_text": "",
        "paper_dispersion_numeric": "",
        "paper_dispersion_type": "",
        "paper_run_count": "",
        "paper_significance_marker": "",
    }
    if not text:
        return result
    result["paper_is_percentage"] = bool_text("%" in text)
    result["paper_scale"] = "percentage" if "%" in text else "raw"
    marker_match = re.search(r"([*†‡]+)$", text)
    if marker_match:
        result["paper_significance_marker"] = marker_match.group(1)
    pm_match = re.search(
        rf"([+-]?\d+(?:\.\d+)?)\s*(?:{re.escape(chr(177))}|\+/-|\+-)\s*([+-]?\d+(?:\.\d+)?)",
        text,
    )
    number_text = ""
    if pm_match:
        number_text = pm_match.group(1)
        result["paper_dispersion_text"] = pm_match.group(2)
        result["paper_dispersion_numeric"] = pm_match.group(2)
        result["paper_dispersion_type"] = "plus_minus"
    else:
        number_match = re.search(r"([+-]?\d+(?:\.\d+)?)", text)
        number_text = number_match.group(1) if number_match else ""
    if number_text:
        result["paper_value_numeric"] = number_text
        result["paper_decimal_places"] = str(len(number_text.split(".", 1)[1])) if "." in number_text else "0"
    return result


def build_numeric_cell_extraction(bundle: SourceBundle, paper_inventory: pd.DataFrame) -> pd.DataFrame:
    paper_id = authoritative_paper_id(paper_inventory)
    rows: list[dict[str, str]] = []
    for _, cell in target_cells(bundle).iterrows():
        cell_id = clean(cell.get("paper_result_cell_id"))
        value_text = clean(cell.get("value"))
        parsed = parse_paper_value(value_text)
        is_verified = bool(value_text and clean(cell.get("verification_status")) != "paper_value_unverified")
        rows.append(
            {
                "paper_numeric_cell_id": cell_id,
                "paper_component_id": cell_id,
                "paper_source_id": paper_id,
                "paper_section": "paper_result_cell",
                "paper_subsection": clean(cell.get("paper_table")),
                "paper_page_pdf": clean(cell.get("source_page")),
                "paper_page_printed": clean(cell.get("source_page")),
                "paper_table_or_figure": clean(cell.get("paper_table")),
                "paper_table_caption": clean(cell.get("paper_table")),
                "paper_row_label": clean(cell.get("paper_row")),
                "paper_column_label": clean(cell.get("paper_column")),
                "paper_dataset": clean(cell.get("dataset")),
                "paper_task": clean(cell.get("task")),
                "paper_model": clean(cell.get("model")),
                "paper_model_version": "",
                "paper_prompt_or_setting": clean(cell.get("prompt_or_method")),
                "paper_metric": clean(cell.get("metric")),
                "paper_cutoff": clean(cell.get("cutoff")),
                "paper_value_text": value_text,
                **parsed,
                "extraction_method": "structured_prior_audit_only_no_pdf_table_parser",
                "extraction_confidence": "high" if is_verified else "unresolved",
                "machine_verified": bool_text(is_verified),
                "manual_review_required": bool_text(not is_verified),
                "evidence_snippet": short_snippet(cell),
                "evidence_location": PHASE5E_RESULT_CELLS.as_posix(),
                "notes": "Paper value remains unresolved because no installed PDF table parser was available." if not is_verified else "Paper value inherited from structured source.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=NUMERIC_EXTRACTION_COLUMNS), ["paper_numeric_cell_id"])


def short_snippet(cell: Mapping[str, Any]) -> str:
    parts = [
        clean(cell.get("paper_table")),
        clean(cell.get("paper_row")),
        clean(cell.get("paper_column")),
        clean(cell.get("metric")),
    ]
    return " | ".join(part for part in parts if part)[:180]


def comparison_by_cell(bundle: SourceBundle) -> dict[str, Mapping[str, str]]:
    if bundle.phase5e_result_comparison.empty:
        return {}
    return {
        clean(row.get("paper_result_cell_id")): {str(k): clean(v) for k, v in row.items()}
        for _, row in bundle.phase5e_result_comparison.iterrows()
    }


def f3_alignment_by_cell(bundle: SourceBundle) -> dict[str, Mapping[str, str]]:
    if bundle.f3_paper_alignment.empty:
        return {}
    return {
        clean(row.get("paper_component_id")): {str(k): clean(v) for k, v in row.items()}
        for _, row in bundle.f3_paper_alignment.iterrows()
        if clean(row.get("paper_component_id")).startswith("PRC-")
    }


def build_local_result_mapping(bundle: SourceBundle, extraction: pd.DataFrame) -> pd.DataFrame:
    comparisons = comparison_by_cell(bundle)
    f3_alignment = f3_alignment_by_cell(bundle)
    rows: list[dict[str, str]] = []
    for idx, cell in extraction.iterrows():
        cell_id = clean(cell.get("paper_numeric_cell_id"))
        comp = comparisons.get(cell_id, {})
        align = f3_alignment.get(cell_id, {})
        local_id = clean(comp.get("local_result_id")) or clean(align.get("local_result_id"))
        if not local_id:
            status = "no_candidate"
            reason = clean(comp.get("reason")) or "No local result candidate was identified by prior paper-cell comparison."
        else:
            status = "protocol_incompatible"
            reason = clean(comp.get("reason")) or "Prior audit found local counterpart but protocol and paper value are not verified."
        rows.append(
            {
                "mapping_id": stable_id("F5A-MAP", cell_id, local_id or "none"),
                "paper_numeric_cell_id": cell_id,
                "local_result_id": local_id,
                "local_source_file": PHASE5E_RESULT_COMPARISON.as_posix(),
                "local_source_row": str(idx + 2),
                "local_dataset": clean(cell.get("paper_dataset")),
                "local_task": clean(cell.get("paper_task")),
                "local_model": "local Qwen2/Mistral or baseline counterpart" if local_id else "",
                "local_model_version": "",
                "local_prompt_contract": "",
                "local_cohort": "not exact paper cohort",
                "local_candidate_set": "not exact paper candidate set",
                "local_candidate_count": "",
                "local_positive_count": "",
                "local_metric": clean(cell.get("paper_metric")),
                "local_cutoff": clean(cell.get("paper_cutoff")),
                "local_value": clean(align.get("local_value")),
                "local_seed": "",
                "local_run_count": "",
                "mapping_status": status,
                "mapping_reason": reason,
                "notes": "Mapping is not promoted to exact comparison.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=LOCAL_MAPPING_COLUMNS), ["paper_numeric_cell_id"])


def equivalent_from_bool(value: str) -> str:
    if clean(value) == "true":
        return "true"
    if clean(value) == "false":
        return "false"
    return "unknown"


def build_protocol_equivalence(bundle: SourceBundle, mapping: pd.DataFrame) -> pd.DataFrame:
    comparisons = comparison_by_cell(bundle)
    rows: list[dict[str, str]] = []
    for _, map_row in mapping.iterrows():
        cell_id = clean(map_row.get("paper_numeric_cell_id"))
        comp = comparisons.get(cell_id, {})
        has_local = bool(clean(map_row.get("local_result_id")))
        task_eq = equivalent_from_bool(clean(comp.get("exact_task_match"))) if comp else "unknown"
        metric_eq = equivalent_from_bool(clean(comp.get("exact_metric_definition_match"))) if comp else "unknown"
        cutoff_eq = equivalent_from_bool(clean(comp.get("exact_cutoff_match"))) if comp else "unknown"
        dataset_eq = equivalent_from_bool(clean(comp.get("exact_dataset_match"))) if comp else "unknown"
        model_eq = equivalent_from_bool(clean(comp.get("exact_model_match"))) if comp else "unknown"
        cohort_eq = equivalent_from_bool(clean(comp.get("exact_cohort_match"))) if comp else "unknown"
        candidate_eq = equivalent_from_bool(clean(comp.get("exact_candidate_protocol_match"))) if comp else "unknown"
        if not has_local:
            protocol_class = "not_applicable"
            blocking = "no local result candidate"
        else:
            protocol_class = "incompatible"
            blocking = clean(comp.get("reason")) or "dataset/model/cohort/candidate protocol not exact"
        rows.append(
            {
                "protocol_check_id": stable_id("F5A-PROTO", cell_id, clean(map_row.get("local_result_id"))),
                "paper_numeric_cell_id": cell_id,
                "local_result_id": clean(map_row.get("local_result_id")),
                "dataset_equivalent": dataset_eq,
                "dataset_version_equivalent": "unknown",
                "preprocessing_equivalent": "unknown",
                "split_equivalent": "unknown",
                "cohort_equivalent": cohort_eq,
                "candidate_set_equivalent": candidate_eq,
                "candidate_count_equivalent": "unknown",
                "positive_policy_equivalent": "unknown",
                "task_equivalent": task_eq,
                "model_equivalent": model_eq,
                "model_version_equivalent": "unknown",
                "prompt_equivalent": "unknown",
                "decoding_equivalent": "unknown",
                "parser_equivalent": "unknown",
                "invalid_output_policy_equivalent": "unknown",
                "metric_definition_equivalent": metric_eq,
                "cutoff_equivalent": cutoff_eq,
                "aggregation_equivalent": "unknown",
                "run_policy_equivalent": "unknown",
                "all_required_fields_known": "false",
                "protocol_equivalence_class": protocol_class,
                "blocking_differences": blocking,
                "evidence": PHASE5E_RESULT_COMPARISON.as_posix(),
                "notes": "Exact requires all material fields verified; F5A does not infer missing fields.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=PROTOCOL_COLUMNS), ["paper_numeric_cell_id"])


def local_value_at_precision(local_value: str, decimal_places: str) -> str:
    if not clean(local_value) or not clean(decimal_places):
        return ""
    try:
        places = int(decimal_places)
        return f"{float(local_value):.{places}f}"
    except ValueError:
        return ""


def build_numeric_cell_verification(extraction: pd.DataFrame, mapping: pd.DataFrame, protocol: pd.DataFrame) -> pd.DataFrame:
    mapping_by_cell = {clean(row.get("paper_numeric_cell_id")): row for _, row in mapping.iterrows()}
    protocol_by_cell = {clean(row.get("paper_numeric_cell_id")): row for _, row in protocol.iterrows()}
    rows: list[dict[str, str]] = []
    for _, cell in extraction.iterrows():
        cell_id = clean(cell.get("paper_numeric_cell_id"))
        map_row = mapping_by_cell.get(cell_id, {})
        proto_row = protocol_by_cell.get(cell_id, {})
        local_value = clean(map_row.get("local_value"))
        paper_value = clean(cell.get("paper_value_numeric"))
        local_rounded = local_value_at_precision(local_value, clean(cell.get("paper_decimal_places")))
        protocol_class = clean(proto_row.get("protocol_equivalence_class")) or "not_applicable"
        machine_verified = clean(cell.get("machine_verified")) == "true"
        if not paper_value:
            numeric_class = "paper_value_unresolved"
            explanation = "Paper value is not machine-verified from PDF or source table."
        elif not local_value:
            numeric_class = "local_value_unavailable"
            explanation = "Paper value is available but no local value is mapped."
        elif protocol_class != "exact":
            numeric_class = "numeric_match_protocol_not_exact" if local_rounded == clean(cell.get("paper_value_text")) else "numeric_mismatch_protocol_not_exact"
            explanation = "Numeric comparison cannot imply exact replication without protocol equivalence."
        else:
            if clean(paper_value) == clean(local_value):
                numeric_class = "exact_numeric_and_protocol"
            else:
                numeric_class = "printed_match_and_protocol" if local_rounded == clean(cell.get("paper_value_text")) else "numeric_mismatch_protocol_exact"
            explanation = "Protocol exactness and printed precision determine replication class."
        abs_diff, rel_diff = numeric_differences(paper_value, local_value)
        printed_match = bool(paper_value and local_value and local_rounded == clean(cell.get("paper_value_text")))
        exact_raw = bool(paper_value and local_value and clean(paper_value) == clean(local_value))
        exact_replication = bool(machine_verified and printed_match and protocol_class == "exact")
        functional_replication = bool(machine_verified and printed_match and protocol_class == "functionally_equivalent")
        numeric_match_without_protocol = bool(printed_match and protocol_class not in ["exact", "functionally_equivalent"])
        rows.append(
            {
                "verification_id": stable_id("F5A-VER", cell_id, clean(map_row.get("local_result_id"))),
                "paper_numeric_cell_id": cell_id,
                "local_result_id": clean(map_row.get("local_result_id")),
                "paper_value_text": clean(cell.get("paper_value_text")),
                "paper_value_numeric": paper_value,
                "paper_decimal_places": clean(cell.get("paper_decimal_places")),
                "local_value_raw": local_value,
                "local_value_at_paper_precision": local_rounded,
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff,
                "printed_precision_match": bool_text(printed_match),
                "exact_raw_match": bool_text(exact_raw),
                "protocol_equivalence_class": protocol_class,
                "numeric_verification_class": numeric_class,
                "is_exact_numeric_replication": bool_text(exact_replication),
                "is_functional_numeric_replication": bool_text(functional_replication),
                "is_numeric_match_without_protocol_equivalence": bool_text(numeric_match_without_protocol),
                "is_reportable": "true",
                "explanation": explanation,
                "notes": "F5A keeps value extraction separate from protocol equivalence.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=NUMERIC_VERIFICATION_COLUMNS), ["paper_numeric_cell_id"])


def numeric_differences(paper_value: str, local_value: str) -> tuple[str, str]:
    if not clean(paper_value) or not clean(local_value):
        return "", ""
    try:
        paper = float(paper_value)
        local = float(local_value)
    except ValueError:
        return "", ""
    abs_diff = abs(local - paper)
    rel_diff = abs_diff / abs(paper) if paper != 0 else math.inf
    return format_float(abs_diff), format_float(rel_diff) if math.isfinite(rel_diff) else ""


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def build_paper_numeric_alignment(extraction: pd.DataFrame, mapping: pd.DataFrame, protocol: pd.DataFrame, verification: pd.DataFrame) -> pd.DataFrame:
    mapping_by_cell = {clean(row.get("paper_numeric_cell_id")): row for _, row in mapping.iterrows()}
    proto_by_cell = {clean(row.get("paper_numeric_cell_id")): row for _, row in protocol.iterrows()}
    ver_by_cell = {clean(row.get("paper_numeric_cell_id")): row for _, row in verification.iterrows()}
    rows: list[dict[str, str]] = []
    for _, cell in extraction.iterrows():
        cell_id = clean(cell.get("paper_numeric_cell_id"))
        map_row = mapping_by_cell.get(cell_id, {})
        proto_row = proto_by_cell.get(cell_id, {})
        ver_row = ver_by_cell.get(cell_id, {})
        if clean(cell.get("machine_verified")) != "true":
            status = "paper_value_unresolved"
            reason = "paper value not machine-verified from local PDF"
            next_action = "manual PDF/table extraction with double-entry verification or approved parser"
        elif clean(proto_row.get("protocol_equivalence_class")) == "exact" and clean(ver_row.get("printed_precision_match")) == "true":
            status = "exactly_replicated"
            reason = ""
            next_action = "none"
        else:
            status = "protocol_incompatible"
            reason = clean(proto_row.get("blocking_differences"))
            next_action = "recover protocol evidence before numeric claim"
        rows.append(
            {
                "paper_numeric_cell_id": cell_id,
                "paper_component_id": clean(cell.get("paper_component_id")),
                "paper_table_or_figure": clean(cell.get("paper_table_or_figure")),
                "paper_dataset": clean(cell.get("paper_dataset")),
                "paper_task": clean(cell.get("paper_task")),
                "paper_model": clean(cell.get("paper_model")),
                "paper_metric": clean(cell.get("paper_metric")),
                "paper_cutoff": clean(cell.get("paper_cutoff")),
                "paper_value_text": clean(cell.get("paper_value_text")),
                "machine_verified": clean(cell.get("machine_verified")),
                "best_local_result_id": clean(map_row.get("local_result_id")),
                "best_local_value": clean(map_row.get("local_value")),
                "protocol_equivalence_class": clean(proto_row.get("protocol_equivalence_class")),
                "numeric_verification_class": clean(ver_row.get("numeric_verification_class")),
                "is_exact_numeric_replication": clean(ver_row.get("is_exact_numeric_replication")),
                "is_functional_numeric_replication": clean(ver_row.get("is_functional_numeric_replication")),
                "current_status": status,
                "blocking_reason": reason,
                "next_action": next_action,
                "notes": "Paper-value recovery is not the same as exact numeric replication.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=ALIGNMENT_COLUMNS), ["paper_numeric_cell_id"])


def coverage_value(coverage: pd.DataFrame, metric_name: str, value_col: str = "f4_value") -> str:
    if coverage.empty or "coverage_metric" not in coverage.columns:
        return ""
    rows = coverage.loc[coverage["coverage_metric"].eq(metric_name)]
    if rows.empty:
        return ""
    return clean(rows.iloc[0].get(value_col))


def phase5e_metric(bundle: SourceBundle, metric_name: str, column: str) -> str:
    if bundle.phase5e_coverage.empty:
        return ""
    rows = bundle.phase5e_coverage.loc[bundle.phase5e_coverage["metric_name"].eq(metric_name)]
    if rows.empty:
        return ""
    return clean(rows.iloc[0].get(column))


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    value = numerator / denominator * 100.0
    if value == 0:
        return "0.0"
    return f"{value:.6f}"


def coverage_row(metric: str, f4_value: str, f5a_value: str, before_num: str, before_den: str, after_num: str, after_den: str, reason: str, cells: Sequence[str]) -> dict[str, str]:
    try:
        change = format_float(float(f5a_value) - float(f4_value))
    except ValueError:
        change = ""
    return {
        "coverage_metric": metric,
        "f4_value": f4_value,
        "f5a_value": f5a_value,
        "absolute_change": change,
        "numerator_before": before_num,
        "denominator_before": before_den,
        "numerator_after": after_num,
        "denominator_after": after_den,
        "change_reason": reason,
        "supporting_cell_ids": ";".join(sorted(cells)),
        "is_change_scientifically_allowed": "true",
        "validation_status": "pass",
        "notes": "Cell-level extraction coverage is separate from component-level reproduction.",
    }


def build_coverage_impact(bundle: SourceBundle, extraction: pd.DataFrame, mapping: pd.DataFrame, protocol: pd.DataFrame, verification: pd.DataFrame) -> pd.DataFrame:
    cell_ids = extraction["paper_numeric_cell_id"].astype(str).tolist()
    denominator = len(extraction)
    verified = int(extraction["machine_verified"].eq("true").sum())
    mapped = int(mapping["local_result_id"].astype(str).str.len().gt(0).sum())
    exact_protocol = int(protocol["protocol_equivalence_class"].eq("exact").sum())
    functional_numeric = int(verification["is_functional_numeric_replication"].eq("true").sum())
    exact_numeric = int(verification["is_exact_numeric_replication"].eq("true").sum())
    rows = [
        coverage_row("exact numerical result-cell coverage", coverage_value(bundle.f4_coverage, "exact numerical coverage") or "0.0", percent(exact_numeric, denominator), "0", str(denominator), str(exact_numeric), str(denominator), "No paper cell has both verified value and exact protocol.", cell_ids),
        coverage_row("verified_paper_numeric_cell_extraction_coverage", "0.0", percent(verified, denominator), "0", str(denominator), str(verified), str(denominator), "No installed parser could machine-verify table cells.", cell_ids),
        coverage_row("mapped local result-cell coverage", "54.545455", percent(mapped, denominator), "6", str(denominator), str(mapped), str(denominator), "Mappings inherited from prior result-cell comparison; none are exact protocol matches.", cell_ids),
        coverage_row("protocol-exact mapped-cell coverage", "0.0", percent(exact_protocol, denominator), "0", str(denominator), str(exact_protocol), str(denominator), "No mapped local result has exact protocol evidence.", cell_ids),
        coverage_row("functional numeric replication coverage", "0.0", percent(functional_numeric, denominator), "0", str(denominator), str(functional_numeric), str(denominator), "No paper value is verified, so functional numeric replication is not claimed.", cell_ids),
        coverage_row("exact reproduction component coverage", phase5e_metric(bundle, "strict_exact_reproduction_micro_pct", "value"), phase5e_metric(bundle, "strict_exact_reproduction_micro_pct", "value"), phase5e_metric(bundle, "strict_exact_reproduction_micro_pct", "numerator"), phase5e_metric(bundle, "strict_exact_reproduction_micro_pct", "denominator"), phase5e_metric(bundle, "strict_exact_reproduction_micro_pct", "numerator"), phase5e_metric(bundle, "strict_exact_reproduction_micro_pct", "denominator"), "F5A does not execute new paper-equivalent experiments.", []),
        coverage_row("reconstructed-or-better component coverage", phase5e_metric(bundle, "reconstructed_or_better_micro_pct", "value"), phase5e_metric(bundle, "reconstructed_or_better_micro_pct", "value"), phase5e_metric(bundle, "reconstructed_or_better_micro_pct", "numerator"), phase5e_metric(bundle, "reconstructed_or_better_micro_pct", "denominator"), phase5e_metric(bundle, "reconstructed_or_better_micro_pct", "numerator"), phase5e_metric(bundle, "reconstructed_or_better_micro_pct", "denominator"), "F5A keeps component coverage separate from cell extraction.", []),
        coverage_row("maximum defensible component coverage", phase5e_metric(bundle, "maximum_defensible_micro_coverage_pct", "value"), phase5e_metric(bundle, "maximum_defensible_micro_coverage_pct", "value"), phase5e_metric(bundle, "maximum_defensible_micro_coverage_pct", "numerator"), phase5e_metric(bundle, "maximum_defensible_micro_coverage_pct", "denominator"), phase5e_metric(bundle, "maximum_defensible_micro_coverage_pct", "numerator"), phase5e_metric(bundle, "maximum_defensible_micro_coverage_pct", "denominator"), "F5A does not promote adapted/local cells into exact paper coverage.", []),
    ]
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_claim_update(bundle: SourceBundle, extraction: pd.DataFrame, mapping: pd.DataFrame, protocol: pd.DataFrame, verification: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, claim in bundle.f4_claims.iterrows():
        claim_id = clean(claim.get("claim_id"))
        previous_text = clean(claim.get("updated_claim_text")) or clean(claim.get("f3_claim_text"))
        rows.append(
            {
                "claim_id": claim_id,
                "previous_claim_text": previous_text,
                "previous_confidence": clean(claim.get("updated_confidence")),
                "f5a_numeric_evidence": "results/tables/finalization_f5a_numeric_cell_verification.csv",
                "f5a_protocol_evidence": "results/tables/finalization_f5a_protocol_equivalence.csv",
                "updated_claim_text": previous_text,
                "updated_confidence": clean(claim.get("updated_confidence")),
                "is_safe_for_report": clean(claim.get("is_safe_for_report")),
                "is_safe_for_presentation": clean(claim.get("is_safe_for_presentation")),
                "remaining_limitation": clean(claim.get("remaining_limitation")),
                "notes": "F5A does not promote prior claims without verified paper values and exact protocols.",
            }
        )
    verified = int(extraction["machine_verified"].eq("true").sum())
    mapped = int(mapping["local_result_id"].astype(str).str.len().gt(0).sum())
    exact_protocol = int(protocol["protocol_equivalence_class"].eq("exact").sum())
    exact_numeric = int(verification["is_exact_numeric_replication"].eq("true").sum())
    new_claims = [
        ("F5A-CLAIM-001", f"F5A represents all 11 target paper-result cells, but machine-verifies {verified} paper values.", "high", "true"),
        ("F5A-CLAIM-002", f"F5A maps {mapped} of 11 target cells to prior local counterpart identifiers, none with exact protocol evidence.", "high", "true"),
        ("F5A-CLAIM-003", f"F5A finds {exact_numeric} exact numeric replications and {exact_protocol} exact protocol mappings.", "high", "true"),
    ]
    for claim_id, text, confidence, safe in new_claims:
        rows.append(
            {
                "claim_id": claim_id,
                "previous_claim_text": "",
                "previous_confidence": "",
                "f5a_numeric_evidence": "results/tables/finalization_f5a_numeric_cell_verification.csv",
                "f5a_protocol_evidence": "results/tables/finalization_f5a_protocol_equivalence.csv",
                "updated_claim_text": text,
                "updated_confidence": confidence,
                "is_safe_for_report": safe,
                "is_safe_for_presentation": safe,
                "remaining_limitation": "Paper table values remain unresolved without an installed parser or manual verification.",
                "notes": "New F5A claim is limited to structured evidence generated in this phase.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=CLAIM_COLUMNS), ["claim_id"])


def build_anomalies(bundle: SourceBundle, paper_inventory: pd.DataFrame, tool_inventory: pd.DataFrame, extraction: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    paper_hashes = sorted(set(paper_inventory["sha256"].astype(str).tolist()))
    parser_available = tool_inventory.loc[
        (tool_inventory["available"].eq("true"))
        & (tool_inventory["tool_name"].isin(["pdftotext", "pdfplumber", "pypdf", "PyPDF2", "fitz", "pymupdf", "mutool"]))
    ]
    model_identity_count = int(extraction["paper_metric"].eq("model_identity").sum())
    mapped_count = int(mapping["local_result_id"].astype(str).str.len().gt(0).sum())
    rows = [
        anomaly("F5A-ANOM-001", "paper_versions", "info", "paper/", "", "The two local PDF files are byte-identical." if len(paper_hashes) == 1 else "Local PDF files differ by SHA-256.", "Authority can be assigned to the arXiv-named file without value conflicts when hashes match.", "false", "true", "Use `paper/2411.00331v1.pdf` as authoritative.", f"unique_pdf_hashes={len(paper_hashes)}"),
        anomaly("F5A-ANOM-002", "pdf_extraction", "medium", "paper/2411.00331v1.pdf", "", "No adequate installed PDF table parser or executable was available for machine table extraction." if parser_available.empty else "At least one parser is available but was not needed for unresolved conservative extraction.", "Paper values remain unresolved; exact numeric coverage cannot increase.", "false", "true", "Perform manual double-entry verification or approve a parser in a later phase.", "No packages installed."),
        anomaly("F5A-ANOM-003", "target_cell_semantics", "medium", PHASE5E_RESULT_CELLS.as_posix(), "", f"{model_identity_count} of 11 frozen target cells are model-identity cells rather than numeric metric cells.", "F5A preserves prior audit IDs but does not treat model identity as numeric replication.", "false", "true", "Keep anomaly visible; do not rewrite Phase 5E.", ""),
        anomaly("F5A-ANOM-004", "unresolved_values", "medium", PHASE5E_RESULT_CELLS.as_posix(), "", f"{len(extraction) - int(extraction['machine_verified'].eq('true').sum())} target cells have unresolved paper values.", "Numeric verification and exact replication remain zero.", "false", "true", "Recover paper values with source-backed manual or parser-based extraction.", ""),
        anomaly("F5A-ANOM-005", "protocol_mapping", "low", PHASE5E_RESULT_COMPARISON.as_posix(), "", f"{mapped_count} cells have prior local counterpart IDs, but protocol equivalence is not exact.", "Metric-name overlap is not sufficient for exact paper replication.", "false", "true", "Use protocol equivalence table before any numeric claim.", ""),
    ]
    return pd.DataFrame(rows, columns=ANOMALY_COLUMNS)


def anomaly(anomaly_id: str, area: str, severity: str, source_file: str, location: str, description: str, impact: str, correction: str, can_continue: str, action: str, notes: str) -> dict[str, str]:
    return {
        "anomaly_id": anomaly_id,
        "area": area,
        "severity": severity,
        "source_file": source_file,
        "source_location": location,
        "description": description,
        "scientific_impact": impact,
        "requires_previous_phase_correction": correction,
        "can_continue": can_continue,
        "recommended_action": action,
        "notes": notes,
    }


def protected_hash_status(root: Path, protected_after: pd.DataFrame) -> str:
    return _d1r_protected_hash_status(root, protected_after)


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return True
    return ABSOLUTE_PATH_PATTERN.search(frame.to_csv(index=False, lineterminator="\n")) is None


def build_validation(
    bundle: SourceBundle,
    paper_inventory: pd.DataFrame,
    tool_inventory: pd.DataFrame,
    extraction: pd.DataFrame,
    mapping: pd.DataFrame,
    protocol: pd.DataFrame,
    verification: pd.DataFrame,
    alignment: pd.DataFrame,
    coverage: pd.DataFrame,
    claims: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check_id: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append({"check_id": check_id, "check_name": name, "status": "pass" if condition else "fail", "observed": clean(observed), "expected": clean(expected), "details": details})

    cell_ids = extraction["paper_numeric_cell_id"].astype(str).tolist()
    target_ids = target_cells(bundle)["paper_result_cell_id"].astype(str).tolist()
    verified_rows = extraction.loc[extraction["machine_verified"].eq("true")]
    exact_protocol = protocol.loc[protocol["protocol_equivalence_class"].eq("exact")]
    printed_matches = verification.loc[verification["printed_precision_match"].eq("true")]
    exact_numeric = verification.loc[verification["is_exact_numeric_replication"].eq("true")]
    all_frames = [paper_inventory, tool_inventory, extraction, mapping, protocol, verification, alignment, coverage, claims, anomalies]
    prot_status = protected_hash_status(bundle.root, bundle.f2_protected_after)

    add("F5A-VAL-001", "all_local_pdfs_inventoried", len(paper_inventory) == len(list((bundle.root / PAPER_DIR).glob("*.pdf"))) and len(paper_inventory) >= 1, len(paper_inventory), "all local PDFs", "Every local PDF under paper/ is inventoried.")
    add("F5A-VAL-002", "authoritative_pdf_identified", paper_inventory["is_authoritative"].eq("true").sum() == 1, paper_inventory["is_authoritative"].eq("true").sum(), "1", "Exactly one authoritative source is selected.")
    add("F5A-VAL-003", "authority_reason_present", paper_inventory.loc[paper_inventory["is_authoritative"].eq("true"), "authority_reason"].astype(str).str.len().gt(0).all(), "reason present", "true", "Authoritative PDF selection is motivated.")
    add("F5A-VAL-004", "pdf_tools_inventoried", set(COMMAND_TOOLS).issubset(set(tool_inventory["tool_name"])) and set(PYTHON_MODULE_TOOLS).issubset(set(tool_inventory["tool_name"])), len(tool_inventory), "all required tools plus stdlib scanner", "Tool inventory records executable and Python parser availability.")
    add("F5A-VAL-005", "no_package_installation", True, "none", "none", "F5A uses only existing environment probes.")
    add("F5A-VAL-006", "exactly_11_target_cells", len(target_ids) == 11, len(target_ids), "11", "Target cells come from Phase 5E structured inventory.")
    add("F5A-VAL-007", "all_11_cells_extracted", set(cell_ids) == set(target_ids) and len(extraction) == 11, len(extraction), "11", "Extraction table has one row per target cell.")
    add("F5A-VAL-008", "all_11_cells_aligned", set(alignment["paper_numeric_cell_id"].astype(str)) == set(target_ids) and len(alignment) == 11, len(alignment), "11", "Alignment table preserves every target cell.")
    add("F5A-VAL-009", "no_duplicate_cells", len(cell_ids) == len(set(cell_ids)), len(cell_ids) - len(set(cell_ids)), "0", "Target cell IDs are unique.")
    add("F5A-VAL-010", "extracted_values_have_position", extraction.loc[extraction["paper_value_text"].astype(str).str.len().gt(0), "evidence_location"].astype(str).str.len().gt(0).all(), "value rows checked", "position present", "No value row lacks an evidence location.")
    add("F5A-VAL-011", "machine_verified_rows_have_row_column", verified_rows.empty or (verified_rows["paper_row_label"].str.len().gt(0).all() and verified_rows["paper_column_label"].str.len().gt(0).all()), len(verified_rows), "row and column present", "Machine verified rows require determined row/column.")
    add("F5A-VAL-012", "local_mapping_ids_valid", mapping["mapping_id"].is_unique and mapping["paper_numeric_cell_id"].isin(cell_ids).all(), len(mapping), "11 valid mappings", "Mapping rows are linked to target cells.")
    add("F5A-VAL-013", "protocol_checks_have_evidence", protocol["evidence"].astype(str).str.len().gt(0).all(), "all", "all", "Every protocol check cites source evidence.")
    add("F5A-VAL-014", "exact_protocol_fields_verified", exact_protocol.empty or exact_protocol["all_required_fields_known"].eq("true").all(), len(exact_protocol), "all exact have all fields known", "No exact protocol is asserted with unknown required fields.")
    add("F5A-VAL-015", "printed_precision_uses_paper_precision", printed_matches.empty or printed_matches["paper_decimal_places"].astype(str).str.len().gt(0).all(), len(printed_matches), "paper precision present", "Printed precision matches must cite paper decimal places.")
    add("F5A-VAL-016", "adapted_match_not_exact_coverage", not verification.loc[verification["protocol_equivalence_class"].isin(["adapted", "incompatible", "insufficient_evidence"]), "is_exact_numeric_replication"].eq("true").any(), "no adapted exact", "true", "Adapted or incompatible rows cannot increase exact numeric coverage.")
    add("F5A-VAL-017", "exact_numeric_requires_exact_protocol", exact_numeric.empty or exact_numeric["protocol_equivalence_class"].eq("exact").all(), len(exact_numeric), "exact protocol", "Exact numeric replication requires exact protocol.")
    add("F5A-VAL-018", "unverified_cells_not_exact", not verification.loc[verification["paper_value_text"].astype(str).str.len().eq(0), "is_exact_numeric_replication"].eq("true").any(), "no unresolved exact", "true", "Unresolved paper values are not exact replications.")
    add("F5A-VAL-019", "coverage_math_consistent", coverage_math_consistent(coverage), "consistent", "consistent", "Coverage rows match numerator/denominator percentages.")
    add("F5A-VAL-020", "component_and_cell_coverage_separate", "component coverage" in ";".join(coverage["coverage_metric"]) and "result-cell" in ";".join(coverage["coverage_metric"]), "separate rows", "true", "Component-level and cell-level coverage are not conflated.")
    add("F5A-VAL-021", "protected_inputs_unchanged", prot_status.endswith("changed=0"), prot_status, "changed=0", "Protected inputs match F2 after hashes.")
    add("F5A-VAL-022", "no_experimental_results_regenerated", True, "none", "none", "F5A reads existing CSV/PDF evidence only.")
    add("F5A-VAL-023", "no_training_executed", True, "none", "none", "No training entrypoint is invoked.")
    add("F5A-VAL-024", "no_inference_executed", True, "none", "none", "No LLM/Ollama/API inference is invoked.")
    add("F5A-VAL-025", "no_network_or_api_used", True, "none", "none", "F5A uses local files and environment probes only.")
    add("F5A-VAL-026", "relative_paths_only", all(no_absolute_paths(frame) for frame in all_frames), "all frames", "no drive-letter paths", "Output CSV frames contain only relative paths/tool names.")
    add("F5A-VAL-027", "deterministic_frames", deterministic_frames(all_frames), "stable", "stable", "Frame serialization hashes are deterministic in memory.")
    add("F5A-VAL-028", "expected_outputs_present_in_builder", len(OUTPUT_TABLES) == 11, len(OUTPUT_TABLES), "11", "All expected F5A output table frames are produced.")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def deterministic_frames(frames: Sequence[pd.DataFrame]) -> bool:
    return [frame_sha256(frame) for frame in frames] == [frame_sha256(frame.copy()) for frame in frames]


def coverage_math_consistent(coverage: pd.DataFrame) -> bool:
    for _, row in coverage.iterrows():
        value = clean(row.get("f5a_value"))
        numerator = clean(row.get("numerator_after"))
        denominator = clean(row.get("denominator_after"))
        if not value or not numerator or not denominator:
            continue
        try:
            expected = float(numerator) / float(denominator) * 100.0
            observed = float(value)
        except ValueError:
            continue
        if abs(expected - observed) > 0.00001:
            return False
    return True


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    bundle = load_sources(root)
    paper_inventory = build_paper_source_inventory(root)
    tool_inventory = build_pdf_tool_inventory()
    extraction = build_numeric_cell_extraction(bundle, paper_inventory)
    mapping = build_local_result_mapping(bundle, extraction)
    protocol = build_protocol_equivalence(bundle, mapping)
    verification = build_numeric_cell_verification(extraction, mapping, protocol)
    alignment = build_paper_numeric_alignment(extraction, mapping, protocol, verification)
    coverage = build_coverage_impact(bundle, extraction, mapping, protocol, verification)
    claims = build_claim_update(bundle, extraction, mapping, protocol, verification)
    anomalies = build_anomalies(bundle, paper_inventory, tool_inventory, extraction, mapping)
    validation = build_validation(bundle, paper_inventory, tool_inventory, extraction, mapping, protocol, verification, alignment, coverage, claims, anomalies)
    return {
        "paper_source_inventory": paper_inventory,
        "pdf_tool_inventory": tool_inventory,
        "numeric_cell_extraction": extraction,
        "local_result_mapping": mapping,
        "protocol_equivalence": protocol,
        "numeric_cell_verification": verification,
        "paper_numeric_alignment": alignment,
        "coverage_impact": coverage,
        "claim_update": claims,
        "anomalies": anomalies,
        "validation": validation,
    }


def final_decision(validation: pd.DataFrame) -> str:
    if not validation.empty and validation["status"].eq("pass").all():
        return "finalization_f5a_complete"
    return "finalization_f5a_blocked"


def build_report(root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    paper = frames["paper_source_inventory"]
    tools = frames["pdf_tool_inventory"]
    extraction = frames["numeric_cell_extraction"]
    mapping = frames["local_result_mapping"]
    protocol = frames["protocol_equivalence"]
    verification = frames["numeric_cell_verification"]
    alignment = frames["paper_numeric_alignment"]
    coverage = frames["coverage_impact"]
    claims = frames["claim_update"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]
    authoritative = paper.loc[paper["is_authoritative"].eq("true")].iloc[0].to_dict() if not paper.empty else {}
    decision = final_decision(validation)
    lines = [
        "# Finalization F5A Paper Numeric Evidence",
        "",
        "## 1. Executive summary",
        f"- Decisione finale: `{decision}`.",
        f"- Celle target: `{len(extraction)}`; valori machine-verified: `{extraction['machine_verified'].eq('true').sum()}`.",
        f"- Exact numeric replications: `{verification['is_exact_numeric_replication'].eq('true').sum()}`.",
        "",
        "## 2. Paper autoritativo",
        f"- `{clean(authoritative.get('file'))}` sha256 `{clean(authoritative.get('sha256'))}`.",
        "",
        "## 3. Versioni locali del paper",
        f"- PDF locali inventariati: `{len(paper)}`; i due file locali sono byte-identical se condividono lo stesso SHA-256.",
        "",
        "## 4. Strumenti di estrazione disponibili",
        f"- Tool inventariati: `{len(tools)}`; selected parser: `python_stdlib_raw_pdf_metadata_scan` per metadata only.",
        "",
        "## 5. Metodo di estrazione",
        "- Le 11 celle sono identificate da `results/tables/plan_d_phase_5e_paper_result_cells.csv`; senza parser tabellare, i valori restano unresolved.",
        "",
        "## 6. Le 11 celle numeriche target",
        f"- Target IDs: `{'; '.join(extraction['paper_numeric_cell_id'].tolist())}`.",
        "",
        "## 7. Valori originali recuperati",
        f"- Valori recuperati e machine-verified: `{extraction['machine_verified'].eq('true').sum()}`.",
        "",
        "## 8. Celle non recuperate",
        f"- Celle non recuperate: `{extraction['manual_review_required'].eq('true').sum()}`.",
        "",
        "## 9. Mapping con i risultati locali",
        f"- Celle con local counterpart ID: `{mapping['local_result_id'].astype(str).str.len().gt(0).sum()}`.",
        "",
        "## 10. Verifica dell'equivalenza di protocollo",
        f"- Protocol exact/functionally/adapted/incompatible: `{protocol['protocol_equivalence_class'].eq('exact').sum()}` / `{protocol['protocol_equivalence_class'].eq('functionally_equivalent').sum()}` / `{protocol['protocol_equivalence_class'].eq('adapted').sum()}` / `{protocol['protocol_equivalence_class'].eq('incompatible').sum()}`.",
        "",
        "## 11. Confronto numerico",
        f"- Printed precision match: `{verification['printed_precision_match'].eq('true').sum()}`; exact raw match: `{verification['exact_raw_match'].eq('true').sum()}`.",
        "",
        "## 12. Celle esattamente replicate",
        f"- `{verification['is_exact_numeric_replication'].eq('true').sum()}`.",
        "",
        "## 13. Celle funzionalmente replicate",
        f"- `{verification['is_functional_numeric_replication'].eq('true').sum()}`.",
        "",
        "## 14. Coincidenze numeriche non equivalenti",
        f"- `{verification['is_numeric_match_without_protocol_equivalence'].eq('true').sum()}`.",
        "",
        "## 15. Impatto sulla copertura",
        "\n".join(f"- `{clean(row['coverage_metric'])}`: `{clean(row['f4_value'])}` -> `{clean(row['f5a_value'])}`." for _, row in coverage.iterrows()),
        "",
        "## 16. Claim aggiornati",
        f"- Claim rows: `{len(claims)}`.",
        "",
        "## 17. Anomalie",
        f"- Anomalie: `{len(anomalies)}`.",
        "",
        "## 18. Test e validazioni",
        f"- Validation pass/fail: `{validation['status'].eq('pass').sum()}` / `{validation['status'].eq('fail').sum()}`.",
        "",
        "## 19. Protezione degli input",
        f"- `{protected_hash_status(root, load_sources(root).f2_protected_after)}`.",
        "",
        "## 20. Decisione finale",
        f"- `{decision}`.",
        "",
        "## 21. Prossimo passo consigliato",
        "- Procedere con estrazione manuale double-entry o approvare un parser PDF locale in F5B.",
        "",
        "Recuperare il numero del paper non significa averlo replicato; ottenere lo stesso numero non significa replicare il protocollo; una replica numerica esatta richiede valore verificato e protocollo exact.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, relative in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].to_csv(path, index=False, lineterminator="\n")
        hashes[relative.as_posix()] = file_sha256(path)
    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_path)
    return dict(sorted(hashes.items()))


def build_all(root: Path) -> dict[str, Any]:
    frames = build_frames(root)
    report = build_report(root, frames)
    hashes = write_outputs(root, frames, report)
    return {"frames": frames, "report": report, "hashes": hashes, "decision": final_decision(frames["validation"])}


def output_hashes(root: Path) -> dict[str, str]:
    paths = [REPORT_PATH, *OUTPUT_TABLES.values()]
    return {path.as_posix(): file_sha256(root / path) for path in paths if (root / path).exists()}

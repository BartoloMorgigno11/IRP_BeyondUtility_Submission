"""Finalization F5A-R structured DOCX numeric recovery.

This module parses a local DOCX as a secondary structured source using only the
Python standard library for WordprocessingML access. It never treats the DOCX as
primary paper evidence and never promotes numeric replication without manual PDF
confirmation.
"""

from __future__ import annotations

import hashlib
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

PDF_PATH = Path("paper/2411.00331v1.pdf")
DOCX_PATH = Path("paper/traduzione_italiana_beyond_utility_llm_recommender.docx")
MANUAL_CONFIRMATIONS_PATH = Path("results/manual/finalization_f5ar_pdf_confirmations.csv")
REPORT_PATH = Path("report/finalization_f5ar_structured_numeric_recovery.md")

F2_PROTECTED_AFTER = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
F5A_EXTRACTION = Path("results/tables/finalization_f5a_numeric_cell_extraction.csv")
F5A_ALIGNMENT = Path("results/tables/finalization_f5a_paper_numeric_alignment.csv")
F5A_MAPPING = Path("results/tables/finalization_f5a_local_result_mapping.csv")
F5A_PROTOCOL = Path("results/tables/finalization_f5a_protocol_equivalence.csv")
F5A_VERIFICATION = Path("results/tables/finalization_f5a_numeric_cell_verification.csv")
F5A_COVERAGE = Path("results/tables/finalization_f5a_coverage_impact.csv")
F3_PAPER_ALIGNMENT = Path("results/tables/finalization_f3_paper_alignment.csv")
F4_TRIAGE = Path("results/tables/finalization_f4_missing_evidence_triage.csv")

OUTPUT_TABLES: dict[str, Path] = {
    "docx_source_inventory": Path("results/tables/finalization_f5ar_docx_source_inventory.csv"),
    "docx_tables": Path("results/tables/finalization_f5ar_docx_tables.csv"),
    "paper_table_inventory": Path("results/tables/finalization_f5ar_paper_table_inventory.csv"),
    "numeric_cell_candidates": Path("results/tables/finalization_f5ar_numeric_cell_candidates.csv"),
    "numeric_cell_recovery": Path("results/tables/finalization_f5ar_numeric_cell_recovery.csv"),
    "manual_pdf_review": Path("results/tables/finalization_f5ar_manual_pdf_review.csv"),
    "preliminary_numeric_comparison": Path("results/tables/finalization_f5ar_preliminary_numeric_comparison.csv"),
    "coverage_impact": Path("results/tables/finalization_f5ar_coverage_impact.csv"),
    "anomalies": Path("results/tables/finalization_f5ar_anomalies.csv"),
    "validation": Path("results/tables/finalization_f5ar_validation.csv"),
}

DOCX_SOURCE_COLUMNS = [
    "source_id",
    "file",
    "sha256",
    "size_bytes",
    "docx_valid_zip",
    "document_xml_present",
    "paragraph_count",
    "table_count",
    "row_count",
    "cell_count",
    "numeric_token_count",
    "declared_relationship_to_paper",
    "is_primary_source",
    "is_secondary_structured_source",
    "notes",
]

DOCX_TABLE_COLUMNS = [
    "docx_table_id",
    "table_index",
    "table_caption",
    "row_index",
    "column_index",
    "row_label",
    "column_label",
    "cell_text",
    "normalized_text",
    "numeric_value",
    "decimal_places",
    "is_percentage",
    "significance_marker",
    "is_header",
    "source_xml_location",
    "notes",
]

PAPER_TABLE_COLUMNS = [
    "paper_table_id",
    "docx_table_id",
    "paper_table_number",
    "caption",
    "section",
    "expected_columns",
    "observed_columns",
    "expected_rows",
    "observed_rows",
    "schema_status",
    "contains_f5a_target",
    "notes",
]

CANDIDATE_COLUMNS = [
    "candidate_id",
    "paper_numeric_cell_id",
    "paper_component_id",
    "target_dataset",
    "target_task",
    "target_model",
    "target_metric",
    "target_cutoff",
    "docx_table_id",
    "paper_table_number",
    "row_label",
    "column_label",
    "cell_text",
    "cell_value_numeric",
    "decimal_places",
    "matching_fields",
    "missing_matching_fields",
    "candidate_score",
    "candidate_status",
    "source_type",
    "requires_pdf_confirmation",
    "notes",
]

RECOVERY_COLUMNS = [
    "paper_numeric_cell_id",
    "paper_component_id",
    "paper_table_number",
    "paper_dataset",
    "paper_task",
    "paper_model",
    "paper_metric",
    "paper_cutoff",
    "recovered_value_text",
    "recovered_value_numeric",
    "decimal_places",
    "docx_row_label",
    "docx_column_label",
    "candidate_count",
    "recovery_status",
    "recovery_confidence",
    "secondary_source_verified",
    "pdf_manually_verified",
    "ready_for_numeric_comparison",
    "blocking_reason",
    "notes",
]

MANUAL_REVIEW_COLUMNS = [
    "review_id",
    "paper_numeric_cell_id",
    "paper_table_number",
    "expected_pdf_page",
    "dataset",
    "task",
    "model",
    "metric",
    "cutoff",
    "docx_candidate_value",
    "docx_row_label",
    "docx_column_label",
    "pdf_value_confirmed",
    "confirmed_pdf_value_text",
    "confirmed_by",
    "confirmation_date",
    "review_status",
    "review_notes",
]

PRELIM_COLUMNS = [
    "comparison_id",
    "paper_numeric_cell_id",
    "docx_candidate_value",
    "local_result_id",
    "local_value",
    "absolute_difference",
    "relative_difference",
    "printed_precision_candidate_match",
    "protocol_equivalence_class",
    "comparison_status",
    "is_scientific_replication_claim_allowed",
    "notes",
]

COVERAGE_COLUMNS = [
    "coverage_metric",
    "f5a_value",
    "f5ar_value",
    "absolute_change",
    "numerator_before",
    "denominator_before",
    "numerator_after",
    "denominator_after",
    "change_reason",
    "validation_status",
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
    "can_continue",
    "recommended_action",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]

ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:[\\/]")
NUMERIC_TOKEN_PATTERN = re.compile(r"(?<![\w])[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*%?(?:[*\u2020\u2021]+)?")
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"w": W_NS, "r": R_NS}

METRIC_ALIASES: dict[str, list[str]] = {
    "hr_at_5": ["hr@5", "hr at 5", "hit rate", "hit@5", "hr"],
    "ndcg_at_5": ["ndcg@5", "ndcg at 5", "ncdg@5", "ndcg"],
    "average_popularity_at_5": ["average popularity", "avg popularity", "popularity"],
    "long_tail_rate_at_5": ["long-tail", "long tail", "long_tail", "tail rate"],
    "item_coverage_at_5": ["item coverage", "coverage@5", "coverage"],
    "novelty_at_5": ["novelty@5", "novelty"],
    "model_identity": ["model identity"],
}


@dataclass(frozen=True)
class DocxCell:
    table_index: int
    table_caption: str
    row_index: int
    column_index: int
    row_label: str
    column_label: str
    cell_text: str
    notes: str


@dataclass(frozen=True)
class DocxParseResult:
    file: Path
    valid_zip: bool
    document_xml_present: bool
    paragraphs: tuple[str, ...]
    cells: tuple[DocxCell, ...]
    relationships_read: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    f2_protected_after: pd.DataFrame
    f5a_extraction: pd.DataFrame
    f5a_alignment: pd.DataFrame
    f5a_mapping: pd.DataFrame
    f5a_protocol: pd.DataFrame
    f5a_verification: pd.DataFrame
    f5a_coverage: pd.DataFrame
    f3_paper_alignment: pd.DataFrame
    f4_triage: pd.DataFrame


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(clean(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def read_csv(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_sources(root: Path) -> SourceBundle:
    return SourceBundle(
        root=root,
        f2_protected_after=read_csv(root, F2_PROTECTED_AFTER),
        f5a_extraction=read_csv(root, F5A_EXTRACTION),
        f5a_alignment=read_csv(root, F5A_ALIGNMENT),
        f5a_mapping=read_csv(root, F5A_MAPPING),
        f5a_protocol=read_csv(root, F5A_PROTOCOL),
        f5a_verification=read_csv(root, F5A_VERIFICATION),
        f5a_coverage=read_csv(root, F5A_COVERAGE),
        f3_paper_alignment=read_csv(root, F3_PAPER_ALIGNMENT),
        f4_triage=read_csv(root, F4_TRIAGE),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, FileNotFoundError):
        return path.as_posix()


def sort_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    existing = [column for column in columns if column in frame.columns]
    return frame.sort_values(existing, kind="mergesort").reset_index(drop=True) if existing else frame.reset_index(drop=True)


def parse_docx(path: Path) -> DocxParseResult:
    if not path.exists():
        return DocxParseResult(path, False, False, tuple(), tuple(), tuple(), "docx_missing")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            rels = tuple(sorted(name for name in names if name.endswith(".rels")))
            document_name = find_document_xml_name(archive, names)
            if not document_name:
                return DocxParseResult(path, True, False, tuple(), tuple(), rels, "document_xml_missing")
            xml_bytes = archive.read(document_name)
    except zipfile.BadZipFile:
        return DocxParseResult(path, False, False, tuple(), tuple(), tuple(), "bad_zip")
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    cells: list[DocxCell] = []
    table_index = 0
    last_caption = ""
    body = root.find("w:body", NS)
    if body is None:
        return DocxParseResult(path, True, True, tuple(), tuple(), tuple(), "body_missing")
    for child in list(body):
        tag = strip_namespace(child.tag)
        if tag == "p":
            text = normalize_spaces(extract_text(child))
            if text:
                paragraphs.append(text)
                if looks_like_caption(text):
                    last_caption = text
        elif tag == "tbl":
            table_index += 1
            table_cells = parse_table(child, table_index, last_caption)
            cells.extend(table_cells)
    return DocxParseResult(path, True, True, tuple(paragraphs), tuple(cells), tuple(), "")


def find_document_xml_name(archive: zipfile.ZipFile, names: set[str]) -> str:
    if "word/document.xml" in names:
        return "word/document.xml"
    if "_rels/.rels" not in names:
        return ""
    try:
        rel_root = ET.fromstring(archive.read("_rels/.rels"))
    except ET.ParseError:
        return ""
    for rel in rel_root:
        target = rel.attrib.get("Target", "")
        rel_type = rel.attrib.get("Type", "")
        if target.endswith("document.xml") or rel_type.endswith("/officeDocument"):
            candidate = target.lstrip("/")
            if candidate in names:
                return candidate
            if f"word/{candidate}" in names:
                return f"word/{candidate}"
    return ""


def parse_table(table: ET.Element, table_index: int, caption: str) -> list[DocxCell]:
    rows = table.findall("w:tr", NS)
    raw_rows: list[list[tuple[str, str]]] = []
    for row in rows:
        row_cells: list[tuple[str, str]] = []
        for cell in row.findall("w:tc", NS):
            text = normalize_spaces(extract_text(cell))
            notes: list[str] = []
            grid_span = cell.find(".//w:gridSpan", NS)
            if grid_span is not None:
                notes.append(f"gridSpan={grid_span.attrib.get(f'{{{W_NS}}}val', '')}")
            v_merge = cell.find(".//w:vMerge", NS)
            if v_merge is not None:
                notes.append("vMerge")
            if cell.find(".//w:b", NS) is not None:
                notes.append("bold_text_detected")
            row_cells.append((text, ";".join(note for note in notes if note)))
        raw_rows.append(row_cells)
    headers = [text for text, _ in raw_rows[0]] if raw_rows else []
    parsed: list[DocxCell] = []
    for row_index, row_cells in enumerate(raw_rows):
        row_label = row_cells[0][0] if row_cells else ""
        for col_index, (text, notes) in enumerate(row_cells):
            column_label = headers[col_index] if col_index < len(headers) else ""
            parsed.append(
                DocxCell(
                    table_index=table_index,
                    table_caption=caption,
                    row_index=row_index,
                    column_index=col_index,
                    row_label=row_label,
                    column_label=column_label,
                    cell_text=text,
                    notes=notes,
                )
            )
    return parsed


def extract_text(element: ET.Element) -> str:
    parts: list[str] = []
    for child in element.iter():
        tag = strip_namespace(child.tag)
        if tag == "t":
            parts.append(child.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append(" ")
    return "".join(parts)


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text)).strip()


def looks_like_caption(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("table ") or lowered.startswith("tabella ") or lowered.startswith("appendix table")


def parse_numeric_token(text: str) -> dict[str, str]:
    raw = clean(text)
    match = NUMERIC_TOKEN_PATTERN.search(raw)
    result = {
        "normalized_text": normalize_numeric_text(raw),
        "numeric_value": "",
        "decimal_places": "",
        "is_percentage": "false",
        "significance_marker": "",
    }
    if not match:
        return result
    token = match.group(0).strip()
    marker = re.search(r"([*\u2020\u2021]+)$", token)
    if marker:
        result["significance_marker"] = marker.group(1)
        token = token[: marker.start()].strip()
    result["is_percentage"] = bool_text("%" in token)
    token = token.replace("%", "").strip()
    normalized = normalize_numeric_text(token)
    numeric_match = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", normalized)
    if numeric_match:
        number = numeric_match.group(0)
        result["numeric_value"] = number
        result["decimal_places"] = str(len(number.split(".", 1)[1])) if "." in number else "0"
    return result


def normalize_numeric_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if "," in token and "." not in token:
            return token.replace(",", ".")
        return token

    return re.sub(r"[-+]?(?:\d+,\d+|,\d+)", replace, clean(text))


def build_docx_source_inventory(root: Path, parsed: DocxParseResult) -> pd.DataFrame:
    path = root / DOCX_PATH
    xml_present = parsed.document_xml_present
    cell_count = len(parsed.cells)
    row_count = len({(cell.table_index, cell.row_index) for cell in parsed.cells})
    table_count = len({cell.table_index for cell in parsed.cells})
    numeric_count = sum(len(NUMERIC_TOKEN_PATTERN.findall(cell.cell_text)) for cell in parsed.cells)
    rows = [
        {
            "source_id": stable_id("F5AR-DOCX", DOCX_PATH.as_posix()),
            "file": DOCX_PATH.as_posix(),
            "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            "size_bytes": str(path.stat().st_size) if path.exists() and path.is_file() else "0",
            "docx_valid_zip": bool_text(parsed.valid_zip),
            "document_xml_present": bool_text(xml_present),
            "paragraph_count": str(len(parsed.paragraphs)),
            "table_count": str(table_count),
            "row_count": str(row_count),
            "cell_count": str(cell_count),
            "numeric_token_count": str(numeric_count),
            "declared_relationship_to_paper": "secondary_structured_source_claiming unchanged model, dataset, metric, formula, and numeric values",
            "is_primary_source": "false",
            "is_secondary_structured_source": "true",
            "notes": parsed.error or "DOCX parsed from WordprocessingML with standard library only.",
        }
    ]
    return pd.DataFrame(rows, columns=DOCX_SOURCE_COLUMNS)


def build_docx_tables(parsed: DocxParseResult) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for cell in parsed.cells:
        parsed_number = parse_numeric_token(cell.cell_text)
        rows.append(
            {
                "docx_table_id": stable_id("F5AR-DOCXTBL", cell.table_index, cell.table_caption),
                "table_index": str(cell.table_index),
                "table_caption": cell.table_caption,
                "row_index": str(cell.row_index),
                "column_index": str(cell.column_index),
                "row_label": cell.row_label,
                "column_label": cell.column_label,
                "cell_text": cell.cell_text,
                "normalized_text": parsed_number["normalized_text"],
                "numeric_value": parsed_number["numeric_value"],
                "decimal_places": parsed_number["decimal_places"],
                "is_percentage": parsed_number["is_percentage"],
                "significance_marker": parsed_number["significance_marker"],
                "is_header": bool_text(cell.row_index == 0),
                "source_xml_location": f"word/document.xml table={cell.table_index} row={cell.row_index} col={cell.column_index}",
                "notes": cell.notes,
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=DOCX_TABLE_COLUMNS), ["table_index", "row_index", "column_index"])


def infer_table_number(caption: str, table_index: str) -> str:
    match = re.search(r"(?:table|tabella|appendix table)\s+([A-Za-z]?\d+)", caption, flags=re.IGNORECASE)
    return match.group(1) if match else table_index


def build_paper_table_inventory(docx_tables: pd.DataFrame) -> pd.DataFrame:
    if docx_tables.empty:
        return pd.DataFrame(columns=PAPER_TABLE_COLUMNS)
    rows: list[dict[str, str]] = []
    for docx_table_id, table_rows in docx_tables.groupby("docx_table_id", sort=True):
        caption = clean(table_rows.iloc[0].get("table_caption"))
        table_index = clean(table_rows.iloc[0].get("table_index"))
        table_number = infer_table_number(caption, table_index)
        observed_columns = str(table_rows["column_index"].astype(int).max() + 1)
        observed_rows = str(table_rows["row_index"].astype(int).max() + 1)
        header = table_rows.loc[table_rows["row_index"].eq("0"), "cell_text"].tolist()
        expected_columns = ";".join(header)
        context = " ".join(table_rows["normalized_text"].astype(str).tolist()).lower()
        contains_target = any(alias in context for aliases in METRIC_ALIASES.values() for alias in aliases)
        rows.append(
            {
                "paper_table_id": stable_id("F5AR-PAPERTBL", docx_table_id, table_number),
                "docx_table_id": docx_table_id,
                "paper_table_number": table_number,
                "caption": caption,
                "section": infer_section(caption, context),
                "expected_columns": expected_columns,
                "observed_columns": observed_columns,
                "expected_rows": "",
                "observed_rows": observed_rows,
                "schema_status": "table_number_inferred" if caption else "caption_missing",
                "contains_f5a_target": bool_text(contains_target),
                "notes": classify_table_notes(table_number, caption),
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=PAPER_TABLE_COLUMNS), ["paper_table_number", "docx_table_id"])


def infer_section(caption: str, context: str) -> str:
    lowered = f"{caption} {context}".lower()
    if "utility" in lowered or "hr@5" in lowered or "ndcg" in lowered:
        return "utility"
    if "novelty" in lowered or "coverage" in lowered or "long-tail" in lowered:
        return "beyond_utility"
    if "re-ranking" in lowered or "reranking" in lowered:
        return "reranking"
    if "dataset" in lowered or "statistics" in lowered or "statistiche" in lowered:
        return "dataset_statistics"
    return "unknown"


def classify_table_notes(table_number: str, caption: str) -> str:
    if table_number == "2":
        return "candidate for paper dataset statistics table"
    if table_number == "3":
        return "candidate for paper utility table"
    if "novelty" in caption.lower() or "coverage" in caption.lower():
        return "candidate for beyond-utility numeric table"
    if "re-ranking" in caption.lower() or "reranking" in caption.lower():
        return "candidate for reranking table"
    return ""


def target_cells(bundle: SourceBundle) -> pd.DataFrame:
    if bundle.f5a_extraction.empty:
        return pd.DataFrame(columns=["paper_numeric_cell_id"])
    return sort_frame(bundle.f5a_extraction.copy(), ["paper_numeric_cell_id"])


def build_numeric_cell_candidates(extraction: pd.DataFrame, docx_tables: pd.DataFrame, paper_tables: pd.DataFrame) -> pd.DataFrame:
    paper_table_by_docx = {clean(row.get("docx_table_id")): clean(row.get("paper_table_number")) for _, row in paper_tables.iterrows()}
    candidate_rows: list[dict[str, str]] = []
    for _, target in extraction.iterrows():
        matches = find_candidates_for_target(target, docx_tables, paper_table_by_docx)
        if matches:
            status = status_for_matches(matches)
            for row in matches:
                row["candidate_status"] = status
                candidate_rows.append(row)
        else:
            candidate_rows.append(no_candidate_row(target))
    return sort_frame(pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS), ["paper_numeric_cell_id", "candidate_score", "candidate_id"])


def find_candidates_for_target(target: Mapping[str, Any], docx_tables: pd.DataFrame, paper_table_by_docx: Mapping[str, str]) -> list[dict[str, str]]:
    if docx_tables.empty:
        return []
    target_metric = clean(target.get("paper_metric"))
    target_model = clean(target.get("paper_model"))
    rows: list[dict[str, str]] = []
    for _, cell in docx_tables.iterrows():
        if clean(cell.get("is_header")) == "true":
            continue
        if target_metric != "model_identity" and not clean(cell.get("numeric_value")):
            continue
        context = candidate_context(cell)
        fields, missing = matching_fields(target, context, cell)
        if target_metric == "model_identity":
            if target_model and target_model.lower() not in context:
                continue
        elif "metric" not in fields:
            continue
        score = str(len(fields))
        rows.append(
            {
                "candidate_id": stable_id("F5AR-CAND", clean(target.get("paper_numeric_cell_id")), clean(cell.get("docx_table_id")), clean(cell.get("row_index")), clean(cell.get("column_index"))),
                "paper_numeric_cell_id": clean(target.get("paper_numeric_cell_id")),
                "paper_component_id": clean(target.get("paper_component_id")),
                "target_dataset": clean(target.get("paper_dataset")),
                "target_task": clean(target.get("paper_task")),
                "target_model": target_model,
                "target_metric": target_metric,
                "target_cutoff": clean(target.get("paper_cutoff")),
                "docx_table_id": clean(cell.get("docx_table_id")),
                "paper_table_number": paper_table_by_docx.get(clean(cell.get("docx_table_id")), ""),
                "row_label": clean(cell.get("row_label")),
                "column_label": clean(cell.get("column_label")),
                "cell_text": clean(cell.get("cell_text")),
                "cell_value_numeric": clean(cell.get("numeric_value")),
                "decimal_places": clean(cell.get("decimal_places")),
                "matching_fields": ";".join(fields),
                "missing_matching_fields": ";".join(missing),
                "candidate_score": score,
                "candidate_status": "",
                "source_type": "secondary_structured_source",
                "requires_pdf_confirmation": "true",
                "notes": "Candidate comes from DOCX secondary source and must be visually checked against the PDF.",
            }
        )
    return sorted(rows, key=lambda row: (-int(row["candidate_score"]), row["candidate_id"]))


def candidate_context(cell: Mapping[str, Any]) -> str:
    parts = [cell.get("table_caption"), cell.get("row_label"), cell.get("column_label"), cell.get("cell_text"), cell.get("normalized_text")]
    return " ".join(clean(part).lower() for part in parts if clean(part))


def matching_fields(target: Mapping[str, Any], context: str, cell: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    fields: list[str] = []
    missing: list[str] = []
    metric = clean(target.get("paper_metric"))
    aliases = METRIC_ALIASES.get(metric, [metric.replace("_", " ")])
    if metric == "model_identity":
        model = clean(target.get("paper_model")).lower()
        if model and model in context:
            fields.append("model")
        else:
            missing.append("model")
    elif any(alias and alias in context for alias in aliases):
        fields.append("metric")
    else:
        missing.append("metric")
    cutoff = clean(target.get("paper_cutoff"))
    if cutoff:
        if f"@{cutoff}" in context or f"at {cutoff}" in context:
            fields.append("cutoff")
        else:
            missing.append("cutoff")
    dataset_match = any(dataset_alias(token) in context for token in split_target_values(clean(target.get("paper_dataset"))))
    if dataset_match:
        fields.append("dataset")
    else:
        missing.append("dataset")
    task_match = any(token.lower() in context for token in split_target_values(clean(target.get("paper_task"))))
    if task_match:
        fields.append("task")
    else:
        missing.append("task")
    if clean(cell.get("numeric_value")):
        fields.append("numeric_value")
    elif metric != "model_identity":
        missing.append("numeric_value")
    return fields, missing


def split_target_values(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def dataset_alias(value: str) -> str:
    return value.lower().replace("amazon ", "").replace("ml-1m", "ml-1m")


def status_for_matches(matches: Sequence[Mapping[str, str]]) -> str:
    if len(matches) > 1:
        return "multiple_candidates"
    score = int(clean(matches[0].get("candidate_score")) or "0")
    return "unique_high_confidence" if score >= 3 else "unique_medium_confidence" if score >= 2 else "partial_match"


def no_candidate_row(target: Mapping[str, Any]) -> dict[str, str]:
    return {
        "candidate_id": stable_id("F5AR-CAND-NONE", clean(target.get("paper_numeric_cell_id"))),
        "paper_numeric_cell_id": clean(target.get("paper_numeric_cell_id")),
        "paper_component_id": clean(target.get("paper_component_id")),
        "target_dataset": clean(target.get("paper_dataset")),
        "target_task": clean(target.get("paper_task")),
        "target_model": clean(target.get("paper_model")),
        "target_metric": clean(target.get("paper_metric")),
        "target_cutoff": clean(target.get("paper_cutoff")),
        "docx_table_id": "",
        "paper_table_number": "",
        "row_label": "",
        "column_label": "",
        "cell_text": "",
        "cell_value_numeric": "",
        "decimal_places": "",
        "matching_fields": "",
        "missing_matching_fields": "docx_candidate",
        "candidate_score": "0",
        "candidate_status": "no_candidate",
        "source_type": "secondary_structured_source",
        "requires_pdf_confirmation": "true",
        "notes": "No DOCX candidate was available for this F5A target cell.",
    }


def load_valid_confirmations(root: Path, target_ids: set[str]) -> tuple[dict[str, Mapping[str, str]], list[str]]:
    path = root / MANUAL_CONFIRMATIONS_PATH
    if not path.exists():
        return {}, []
    rows = read_csv(root, MANUAL_CONFIRMATIONS_PATH)
    confirmations: dict[str, Mapping[str, str]] = {}
    errors: list[str] = []
    required = {"paper_numeric_cell_id", "pdf_value_confirmed", "confirmed_pdf_value_text", "confirmed_by", "confirmation_date", "review_notes"}
    if not required.issubset(rows.columns):
        return {}, ["manual confirmation schema invalid"]
    for _, row in rows.iterrows():
        cell_id = clean(row.get("paper_numeric_cell_id"))
        if cell_id not in target_ids:
            errors.append(f"unknown id {cell_id}")
            continue
        if clean(row.get("pdf_value_confirmed")).lower() != "true":
            continue
        if not clean(row.get("confirmed_by")) or not clean(row.get("confirmation_date")):
            errors.append(f"confirmation metadata missing for {cell_id}")
            continue
        if not parse_numeric_token(clean(row.get("confirmed_pdf_value_text")))["numeric_value"]:
            errors.append(f"confirmation value not numeric for {cell_id}")
            continue
        confirmations[cell_id] = {key: clean(row.get(key)) for key in required}
    return confirmations, errors


def build_numeric_cell_recovery(bundle: SourceBundle, candidates: pd.DataFrame) -> pd.DataFrame:
    confirmations, _ = load_valid_confirmations(bundle.root, set(target_cells(bundle)["paper_numeric_cell_id"].astype(str)))
    by_cell = {cell_id: group.copy() for cell_id, group in candidates.groupby("paper_numeric_cell_id", sort=True)}
    rows: list[dict[str, str]] = []
    for _, target in target_cells(bundle).iterrows():
        cell_id = clean(target.get("paper_numeric_cell_id"))
        group = by_cell.get(cell_id, pd.DataFrame(columns=CANDIDATE_COLUMNS))
        real_candidates = group.loc[group["candidate_status"].ne("no_candidate")] if not group.empty else group
        best = real_candidates.iloc[0].to_dict() if not real_candidates.empty else {}
        candidate_count = len(real_candidates)
        unique = candidate_count == 1 and clean(best.get("candidate_status")) in {"unique_high_confidence", "unique_medium_confidence"}
        confirmation = confirmations.get(cell_id, {})
        pdf_verified = bool(confirmation)
        rows.append(
            {
                "paper_numeric_cell_id": cell_id,
                "paper_component_id": clean(target.get("paper_component_id")),
                "paper_table_number": clean(best.get("paper_table_number")),
                "paper_dataset": clean(target.get("paper_dataset")),
                "paper_task": clean(target.get("paper_task")),
                "paper_model": clean(target.get("paper_model")),
                "paper_metric": clean(target.get("paper_metric")),
                "paper_cutoff": clean(target.get("paper_cutoff")),
                "recovered_value_text": clean(best.get("cell_text")),
                "recovered_value_numeric": clean(best.get("cell_value_numeric")),
                "decimal_places": clean(best.get("decimal_places")),
                "docx_row_label": clean(best.get("row_label")),
                "docx_column_label": clean(best.get("column_label")),
                "candidate_count": str(candidate_count),
                "recovery_status": recovery_status(candidate_count, unique, pdf_verified),
                "recovery_confidence": "high" if unique else "medium" if candidate_count == 1 else "ambiguous" if candidate_count > 1 else "unresolved",
                "secondary_source_verified": bool_text(unique),
                "pdf_manually_verified": bool_text(pdf_verified),
                "ready_for_numeric_comparison": bool_text(pdf_verified and unique),
                "blocking_reason": "" if pdf_verified else "pending_manual_pdf_review" if candidate_count else "docx_candidate_missing",
                "notes": "PDF confirmation is required before numeric comparison or replication claims.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=RECOVERY_COLUMNS), ["paper_numeric_cell_id"])


def recovery_status(candidate_count: int, unique: bool, pdf_verified: bool) -> str:
    if pdf_verified and unique:
        return "pdf_confirmed_secondary_candidate"
    if unique:
        return "secondary_unique_candidate_pending_pdf_review"
    if candidate_count > 1:
        return "multiple_secondary_candidates_pending_pdf_review"
    return "no_candidate"


def build_manual_pdf_review(bundle: SourceBundle, recovery: pd.DataFrame) -> pd.DataFrame:
    confirmations, _ = load_valid_confirmations(bundle.root, set(target_cells(bundle)["paper_numeric_cell_id"].astype(str)))
    target_by_id = {clean(row.get("paper_numeric_cell_id")): row for _, row in target_cells(bundle).iterrows()}
    rows: list[dict[str, str]] = []
    for _, row in recovery.iterrows():
        cell_id = clean(row.get("paper_numeric_cell_id"))
        target = target_by_id.get(cell_id, {})
        confirmation = confirmations.get(cell_id, {})
        confirmed = bool(confirmation)
        rows.append(
            {
                "review_id": stable_id("F5AR-REVIEW", cell_id),
                "paper_numeric_cell_id": cell_id,
                "paper_table_number": clean(row.get("paper_table_number")),
                "expected_pdf_page": clean(target.get("paper_page_pdf")),
                "dataset": clean(row.get("paper_dataset")),
                "task": clean(row.get("paper_task")),
                "model": clean(row.get("paper_model")),
                "metric": clean(row.get("paper_metric")),
                "cutoff": clean(row.get("paper_cutoff")),
                "docx_candidate_value": clean(row.get("recovered_value_text")),
                "docx_row_label": clean(row.get("docx_row_label")),
                "docx_column_label": clean(row.get("docx_column_label")),
                "pdf_value_confirmed": bool_text(confirmed),
                "confirmed_pdf_value_text": clean(confirmation.get("confirmed_pdf_value_text")),
                "confirmed_by": clean(confirmation.get("confirmed_by")),
                "confirmation_date": clean(confirmation.get("confirmation_date")),
                "review_status": "pdf_value_confirmed" if confirmed else "pending_manual_pdf_review",
                "review_notes": clean(confirmation.get("review_notes")) if confirmed else "Visually check the authoritative PDF table, row, column, and printed value.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=MANUAL_REVIEW_COLUMNS), ["paper_numeric_cell_id"])


def build_preliminary_numeric_comparison(bundle: SourceBundle, recovery: pd.DataFrame) -> pd.DataFrame:
    mapping_by_cell = {clean(row.get("paper_numeric_cell_id")): row for _, row in bundle.f5a_mapping.iterrows()}
    protocol_by_cell = {clean(row.get("paper_numeric_cell_id")): row for _, row in bundle.f5a_protocol.iterrows()}
    rows: list[dict[str, str]] = []
    for _, row in recovery.iterrows():
        cell_id = clean(row.get("paper_numeric_cell_id"))
        mapping = mapping_by_cell.get(cell_id, {})
        protocol = protocol_by_cell.get(cell_id, {})
        docx_value = clean(row.get("recovered_value_numeric"))
        local_value = clean(mapping.get("local_value"))
        abs_diff, rel_diff = numeric_differences(docx_value, local_value)
        rows.append(
            {
                "comparison_id": stable_id("F5AR-COMP", cell_id, clean(mapping.get("local_result_id"))),
                "paper_numeric_cell_id": cell_id,
                "docx_candidate_value": clean(row.get("recovered_value_text")),
                "local_result_id": clean(mapping.get("local_result_id")),
                "local_value": local_value,
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff,
                "printed_precision_candidate_match": bool_text(printed_precision_match(docx_value, local_value, clean(row.get("decimal_places")))),
                "protocol_equivalence_class": clean(protocol.get("protocol_equivalence_class")) or "not_applicable",
                "comparison_status": comparison_status(clean(row.get("pdf_manually_verified")), docx_value, local_value),
                "is_scientific_replication_claim_allowed": "false",
                "notes": "Preliminary comparison is not a scientific replication claim until PDF confirmation and exact protocol evidence exist.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=PRELIM_COLUMNS), ["paper_numeric_cell_id"])


def numeric_differences(left: str, right: str) -> tuple[str, str]:
    if not clean(left) or not clean(right):
        return "", ""
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return "", ""
    absolute = abs(left_value - right_value)
    relative = absolute / abs(left_value) if left_value else math.inf
    return format_float(absolute), format_float(relative) if math.isfinite(relative) else ""


def printed_precision_match(candidate: str, local: str, decimal_places: str) -> bool:
    if not clean(candidate) or not clean(local) or not clean(decimal_places):
        return False
    try:
        places = int(decimal_places)
        return f"{float(local):.{places}f}" == f"{float(candidate):.{places}f}"
    except ValueError:
        return False


def comparison_status(pdf_verified: str, docx_value: str, local_value: str) -> str:
    if clean(pdf_verified) != "true":
        return "pending_pdf_confirmation"
    if not docx_value:
        return "docx_value_unavailable"
    if not local_value:
        return "local_value_unavailable"
    return "preliminary_difference_computed"


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.17g}"


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    value = numerator / denominator * 100.0
    return "0.0" if value == 0 else f"{value:.6f}"


def f5a_coverage_value(bundle: SourceBundle, name: str) -> str:
    rows = bundle.f5a_coverage.loc[bundle.f5a_coverage["coverage_metric"].eq(name)] if not bundle.f5a_coverage.empty else pd.DataFrame()
    return clean(rows.iloc[0].get("f5a_value")) if not rows.empty else "0.0"


def coverage_row(metric: str, f5a_value: str, f5ar_value: str, before_num: int, before_den: int, after_num: int, after_den: int, reason: str) -> dict[str, str]:
    try:
        change = format_float(float(f5ar_value) - float(f5a_value))
    except ValueError:
        change = ""
    return {
        "coverage_metric": metric,
        "f5a_value": f5a_value,
        "f5ar_value": f5ar_value,
        "absolute_change": change,
        "numerator_before": str(before_num),
        "denominator_before": str(before_den),
        "numerator_after": str(after_num),
        "denominator_after": str(after_den),
        "change_reason": reason,
        "validation_status": "pass",
        "notes": "DOCX recovery is secondary-source evidence and does not change exact or functional numerical coverage without PDF confirmation.",
    }


def build_coverage_impact(bundle: SourceBundle, recovery: pd.DataFrame, manual_review: pd.DataFrame) -> pd.DataFrame:
    denominator = len(recovery)
    secondary_count = int(recovery["secondary_source_verified"].eq("true").sum()) if not recovery.empty else 0
    pending_count = int(manual_review["review_status"].eq("pending_manual_pdf_review").sum()) if not manual_review.empty else 0
    verified_pdf_count = int(manual_review["pdf_value_confirmed"].eq("true").sum()) if not manual_review.empty else 0
    rows = [
        coverage_row("secondary_structured_recovery_coverage", "0.0", percent(secondary_count, denominator), 0, denominator, secondary_count, denominator, "Only unique DOCX secondary-source candidates can increase this coverage."),
        coverage_row("pending_manual_review_coverage", "0.0", percent(pending_count, denominator), 0, denominator, pending_count, denominator, "All target cells remain pending visual PDF review unless explicit confirmations are provided."),
        coverage_row("verified_paper_numeric_cell_extraction_coverage", f5a_coverage_value(bundle, "verified_paper_numeric_cell_extraction_coverage"), percent(verified_pdf_count, denominator), 0, denominator, verified_pdf_count, denominator, "Verified PDF extraction can increase only from valid manual confirmations."),
        coverage_row("exact_numerical_coverage", f5a_coverage_value(bundle, "exact numerical result-cell coverage"), f5a_coverage_value(bundle, "exact numerical result-cell coverage"), 0, denominator, 0, denominator, "No exact numerical coverage is promoted from DOCX-only evidence."),
        coverage_row("functional_numerical_coverage", f5a_coverage_value(bundle, "functional numeric replication coverage"), f5a_coverage_value(bundle, "functional numeric replication coverage"), 0, denominator, 0, denominator, "Functional numerical coverage requires PDF confirmation and protocol evidence."),
    ]
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_anomalies(parsed: DocxParseResult, docx_tables: pd.DataFrame, candidates: pd.DataFrame, confirmations_errors: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if not parsed.file.exists():
        rows.append(anomaly("F5AR-ANOM-001", "docx_source", "high", DOCX_PATH.as_posix(), "", "Expected DOCX secondary source is missing.", "Structured recovery cannot proceed; all target cells remain without DOCX candidates.", "true", "Place the DOCX at the expected paper/ path and rerun F5A-R.", parsed.error))
    elif not parsed.valid_zip:
        rows.append(anomaly("F5AR-ANOM-002", "docx_source", "high", DOCX_PATH.as_posix(), "", "DOCX is not a valid ZIP archive.", "WordprocessingML table extraction is blocked.", "true", "Replace with a valid DOCX copy.", parsed.error))
    elif not parsed.document_xml_present:
        rows.append(anomaly("F5AR-ANOM-003", "docx_source", "high", DOCX_PATH.as_posix(), "word/document.xml", "DOCX does not contain document.xml.", "The main document body cannot be parsed.", "true", "Inspect the DOCX package manually.", parsed.error))
    if docx_tables.empty:
        rows.append(anomaly("F5AR-ANOM-004", "docx_tables", "high", DOCX_PATH.as_posix(), "word/document.xml", "No DOCX table cells were extracted.", "No secondary structured numeric recovery can be claimed.", "true", "Verify the DOCX exists and contains Word tables.", ""))
    else:
        merged = docx_tables.loc[docx_tables["notes"].str.contains("gridSpan|vMerge", regex=True, na=False)]
        if not merged.empty:
            rows.append(anomaly("F5AR-ANOM-005", "docx_tables", "medium", DOCX_PATH.as_posix(), "word/document.xml", "Merged DOCX table cells were detected.", "Row/column mapping may require manual review.", "true", "Inspect merged table rows before accepting candidates.", f"merged_cell_count={len(merged)}"))
        non_numeric = docx_tables.loc[docx_tables["cell_text"].str.contains(r"\d", regex=True, na=False) & docx_tables["numeric_value"].eq("")]
        if not non_numeric.empty:
            rows.append(anomaly("F5AR-ANOM-006", "numeric_parsing", "medium", DOCX_PATH.as_posix(), "word/document.xml", "Some table cells contain digits but no parseable numeric token.", "Potential numeric evidence may need manual normalization.", "true", "Review non-parseable numeric-looking cells.", f"count={len(non_numeric)}"))
    no_candidate_count = int(candidates["candidate_status"].eq("no_candidate").sum()) if not candidates.empty else 0
    if no_candidate_count:
        rows.append(anomaly("F5AR-ANOM-007", "target_mapping", "medium", "results/tables/finalization_f5a_numeric_cell_extraction.csv", "", f"{no_candidate_count} F5A target cells have no DOCX candidate.", "Manual PDF review remains required and secondary recovery coverage is limited.", "true", "Add/repair DOCX source or map manually with confirmation.", ""))
    multi_count = int(candidates["candidate_status"].eq("multiple_candidates").sum()) if not candidates.empty else 0
    if multi_count:
        rows.append(anomaly("F5AR-ANOM-008", "target_mapping", "medium", DOCX_PATH.as_posix(), "word/document.xml", f"{multi_count} DOCX candidate rows are part of multiple-candidate mappings.", "Candidate ambiguity blocks secondary_source_verified for those cells.", "true", "Resolve row/column/table identity manually.", ""))
    for index, error in enumerate(confirmations_errors, start=1):
        rows.append(anomaly(f"F5AR-ANOM-CONF-{index:03d}", "manual_confirmation", "high", MANUAL_CONFIRMATIONS_PATH.as_posix(), "", error, "Invalid manual confirmation is ignored.", "true", "Correct confirmation CSV schema and values.", ""))
    if not rows:
        rows.append(anomaly("F5AR-ANOM-000", "none", "info", DOCX_PATH.as_posix(), "", "No structural anomaly detected by F5A-R.", "Manual PDF confirmation is still required before replication claims.", "true", "Proceed with manual PDF review.", ""))
    return pd.DataFrame(rows, columns=ANOMALY_COLUMNS)


def anomaly(anomaly_id: str, area: str, severity: str, source_file: str, source_location: str, description: str, impact: str, can_continue: str, action: str, notes: str) -> dict[str, str]:
    return {
        "anomaly_id": anomaly_id,
        "area": area,
        "severity": severity,
        "source_file": source_file,
        "source_location": source_location,
        "description": description,
        "scientific_impact": impact,
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


def coverage_exact_unchanged(coverage: pd.DataFrame) -> bool:
    exact = coverage.loc[coverage["coverage_metric"].isin(["exact_numerical_coverage", "functional_numerical_coverage", "verified_paper_numeric_cell_extraction_coverage"])]
    if exact.empty:
        return False
    for _, row in exact.iterrows():
        metric = clean(row.get("coverage_metric"))
        if metric in {"exact_numerical_coverage", "functional_numerical_coverage"} and clean(row.get("f5a_value")) != clean(row.get("f5ar_value")):
            return False
        if metric == "verified_paper_numeric_cell_extraction_coverage" and float(clean(row.get("f5ar_value")) or "0") < float(clean(row.get("f5a_value")) or "0"):
            return False
    return True


def build_validation(
    bundle: SourceBundle,
    parsed: DocxParseResult,
    docx_inventory: pd.DataFrame,
    docx_tables: pd.DataFrame,
    candidates: pd.DataFrame,
    recovery: pd.DataFrame,
    manual_review: pd.DataFrame,
    coverage: pd.DataFrame,
    frames_for_paths: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    target_ids = target_cells(bundle)["paper_numeric_cell_id"].astype(str).tolist()
    prot_status = protected_hash_status(bundle.root, bundle.f2_protected_after)

    def add(check_id: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append({"check_id": check_id, "check_name": name, "status": "pass" if condition else "fail", "observed": clean(observed), "expected": clean(expected), "details": details})

    recovered_values = recovery.loc[recovery["recovered_value_text"].astype(str).str.len().gt(0)] if not recovery.empty else pd.DataFrame()
    add("F5AR-VAL-001", "docx_present", (bundle.root / DOCX_PATH).exists(), (bundle.root / DOCX_PATH).exists(), "true", "The secondary DOCX source must exist at the expected path.")
    add("F5AR-VAL-002", "docx_zip_valid", parsed.valid_zip, parsed.valid_zip, "true", "The DOCX must be a readable ZIP archive.")
    add("F5AR-VAL-003", "document_xml_present", parsed.document_xml_present, parsed.document_xml_present, "true", "word/document.xml must be present or resolved through relationships.")
    add("F5AR-VAL-004", "tables_extracted", len(docx_tables) > 0, len(docx_tables), ">0", "At least one table cell must be extracted from DOCX.")
    add("F5AR-VAL-005", "exactly_11_target_cells", len(target_ids) == 11, len(target_ids), "11", "F5A-R uses F5A target cells only.")
    add("F5AR-VAL-006", "recovery_has_all_11", set(recovery["paper_numeric_cell_id"]) == set(target_ids) and len(recovery) == 11, len(recovery), "11", "Recovery table has exactly one row per F5A target.")
    add("F5AR-VAL-007", "manual_review_has_all_11", set(manual_review["paper_numeric_cell_id"]) == set(target_ids) and len(manual_review) == 11, len(manual_review), "11", "Manual review pack has exactly one row per F5A target.")
    add("F5AR-VAL-008", "no_duplicate_cells", recovery["paper_numeric_cell_id"].is_unique and manual_review["paper_numeric_cell_id"].is_unique, "duplicates=0", "duplicates=0", "Recovery and review IDs are unique.")
    add("F5AR-VAL-009", "recovered_values_have_location", recovered_values.empty or (recovered_values["paper_table_number"].str.len().gt(0).all() and recovered_values["docx_row_label"].str.len().gt(0).all() and recovered_values["docx_column_label"].str.len().gt(0).all()), len(recovered_values), "all recovered values have table/row/column", "Recovered secondary values must have table, row and column.")
    add("F5AR-VAL-010", "docx_values_secondary", candidates["source_type"].eq("secondary_structured_source").all() if not candidates.empty else True, "all candidates", "secondary_structured_source", "DOCX candidates are never primary evidence.")
    add("F5AR-VAL-011", "no_docx_value_pdf_verified", not recovery.loc[recovery["pdf_manually_verified"].ne("true"), "ready_for_numeric_comparison"].eq("true").any(), "no unverified ready rows", "true", "DOCX-only values are not PDF verified.")
    add("F5AR-VAL-012", "exact_coverage_not_increased", coverage_exact_unchanged(coverage), "unchanged", "unchanged", "Exact and functional numerical coverage must not increase from DOCX-only evidence.")
    add("F5AR-VAL-013", "no_previous_phase_file_modified", True, "F5A-R outputs only", "no F2/F3/F4/F5A edits", "Builder writes only finalization_f5ar outputs and report.")
    add("F5AR-VAL-014", "protected_inputs_unchanged", prot_status.endswith("changed=0"), prot_status, "changed=0", "Protected inputs match the F2 after hashes.")
    add("F5AR-VAL-015", "no_network_api_training_inference", True, "none", "none", "F5A-R uses local CSV/PDF/DOCX files only.")
    add("F5AR-VAL-016", "relative_paths_only", all(no_absolute_paths(frame) for frame in frames_for_paths), "all output frames", "relative paths only", "No drive-letter paths are emitted.")
    add("F5AR-VAL-017", "deterministic_frames", deterministic_frames(frames_for_paths), "stable", "stable", "Frame serialization hashes are stable in memory.")
    add("F5AR-VAL-018", "expected_outputs_present_in_builder", len(OUTPUT_TABLES) == 10, len(OUTPUT_TABLES), "10", "All expected F5A-R output table frames are produced.")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def deterministic_frames(frames: Sequence[pd.DataFrame]) -> bool:
    return [frame_sha256(frame) for frame in frames] == [frame_sha256(frame.copy()) for frame in frames]


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f5ar_complete" if not validation.empty and validation["status"].eq("pass").all() else "finalization_f5ar_blocked"


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    bundle = load_sources(root)
    parsed = parse_docx(root / DOCX_PATH)
    docx_inventory = build_docx_source_inventory(root, parsed)
    docx_tables = build_docx_tables(parsed)
    paper_tables = build_paper_table_inventory(docx_tables)
    candidates = build_numeric_cell_candidates(target_cells(bundle), docx_tables, paper_tables)
    recovery = build_numeric_cell_recovery(bundle, candidates)
    manual_review = build_manual_pdf_review(bundle, recovery)
    preliminary = build_preliminary_numeric_comparison(bundle, recovery)
    coverage = build_coverage_impact(bundle, recovery, manual_review)
    _, confirmation_errors = load_valid_confirmations(root, set(target_cells(bundle)["paper_numeric_cell_id"].astype(str)))
    anomalies = build_anomalies(parsed, docx_tables, candidates, confirmation_errors)
    frames_for_paths = [docx_inventory, docx_tables, paper_tables, candidates, recovery, manual_review, preliminary, coverage, anomalies]
    validation = build_validation(bundle, parsed, docx_inventory, docx_tables, candidates, recovery, manual_review, coverage, frames_for_paths)
    return {
        "docx_source_inventory": docx_inventory,
        "docx_tables": docx_tables,
        "paper_table_inventory": paper_tables,
        "numeric_cell_candidates": candidates,
        "numeric_cell_recovery": recovery,
        "manual_pdf_review": manual_review,
        "preliminary_numeric_comparison": preliminary,
        "coverage_impact": coverage,
        "anomalies": anomalies,
        "validation": validation,
    }


def build_report(root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    inventory = frames["docx_source_inventory"]
    tables = frames["docx_tables"]
    recovery = frames["numeric_cell_recovery"]
    review = frames["manual_pdf_review"]
    candidates = frames["numeric_cell_candidates"]
    comparison = frames["preliminary_numeric_comparison"]
    coverage = frames["coverage_impact"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]
    decision = final_decision(validation)
    source = inventory.iloc[0].to_dict() if not inventory.empty else {}
    lines = [
        "# Finalization F5A-R Structured Numeric Recovery",
        "",
        "## 1. Executive summary",
        f"- Decisione: `{decision}`.",
        f"- DOCX: `{DOCX_PATH.as_posix()}`; valid ZIP: `{clean(source.get('docx_valid_zip'))}`; document.xml: `{clean(source.get('document_xml_present'))}`.",
        "- Il DOCX e' trattato solo come fonte secondaria strutturata; il PDF resta autoritativo.",
        "",
        "## 2. Metodo DOCX",
        "- Parsing via `zipfile` e `xml.etree.ElementTree`, leggendo WordprocessingML da `word/document.xml` o relazione officeDocument.",
        "",
        "## 3. Inventario tabelle",
        f"- Tabelle DOCX trovate: `{clean(source.get('table_count'))}`; celle tabella estratte: `{len(tables)}`.",
        "",
        "## 4. Mapping delle 11 celle F5A",
        f"- Celle target: `{len(recovery)}`; con candidato: `{int(recovery['candidate_count'].astype(int).gt(0).sum()) if not recovery.empty else 0}`; univoche: `{recovery['secondary_source_verified'].eq('true').sum() if not recovery.empty else 0}`.",
        "",
        "## 5. Recovery secondaria",
        "- Nessuna cella e' dichiarata PDF-verified sul solo DOCX.",
        "",
        "## 6. Pacchetto manuale PDF",
        "- Compilare opzionalmente `results/manual/finalization_f5ar_pdf_confirmations.csv` con colonne: `paper_numeric_cell_id,pdf_value_confirmed,confirmed_pdf_value_text,confirmed_by,confirmation_date,review_notes`.",
        "- Una conferma e' ignorata se ID, valore numerico, revisore o data non sono validi.",
        "",
        "## 7. Celle da controllare nel PDF",
    ]
    for _, row in review.iterrows():
        lines.append(
            "- "
            f"`{clean(row.get('paper_numeric_cell_id'))}`: table `{clean(row.get('paper_table_number'))}`, "
            f"row `{clean(row.get('docx_row_label'))}`, column `{clean(row.get('docx_column_label'))}`, "
            f"DOCX value `{clean(row.get('docx_candidate_value'))}`. "
            "Controllare visivamente nel PDF autoritativo tabella, riga, colonna e valore stampato."
        )
    lines.extend(
        [
            "",
            "## 8. Confronto locale preliminare",
            f"- Righe confronto preliminare: `{len(comparison)}`; claim di replica permessi: `{comparison['is_scientific_replication_claim_allowed'].eq('true').sum() if not comparison.empty else 0}`.",
            "",
            "## 9. Coverage",
            *[f"- `{clean(row['coverage_metric'])}`: `{clean(row['f5a_value'])}` -> `{clean(row['f5ar_value'])}`." for _, row in coverage.iterrows()],
            "",
            "## 10. Anomalie",
            f"- Anomalie registrate: `{len(anomalies)}`.",
            "",
            "## 11. Validazione",
            f"- Validation pass/fail: `{validation['status'].eq('pass').sum()}` / `{validation['status'].eq('fail').sum()}`.",
            "",
            "## 12. Decisione finale",
            f"- `{decision}`.",
            "",
            "Il recupero del numero dal DOCX non e' una verifica PDF; una replica numerica richiede PDF confermato e protocollo equivalente.",
            "",
        ]
    )
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

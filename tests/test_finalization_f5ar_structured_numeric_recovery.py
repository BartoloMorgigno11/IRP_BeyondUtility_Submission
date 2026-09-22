from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src.analysis.finalization_f5ar_structured_numeric_recovery import (
    ANOMALY_COLUMNS,
    CANDIDATE_COLUMNS,
    COVERAGE_COLUMNS,
    DOCX_SOURCE_COLUMNS,
    DOCX_TABLE_COLUMNS,
    MANUAL_REVIEW_COLUMNS,
    OUTPUT_TABLES,
    PAPER_TABLE_COLUMNS,
    PRELIM_COLUMNS,
    RECOVERY_COLUMNS,
    REPORT_PATH,
    VALIDATION_COLUMNS,
    build_docx_source_inventory,
    build_docx_tables,
    build_frames,
    build_numeric_cell_candidates,
    build_paper_table_inventory,
    build_report,
    final_decision,
    frame_sha256,
    load_sources,
    no_absolute_paths,
    output_hashes,
    parse_docx,
    parse_numeric_token,
    protected_hash_status,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def write_synthetic_docx(path: Path, body_xml: str) -> None:
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}"><w:body>{body_xml}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", "<Relationships/>")


def p(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def tc(text: str, *, bold: bool = False, grid_span: int | None = None) -> str:
    props = f'<w:tcPr><w:gridSpan w:val="{grid_span}"/></w:tcPr>' if grid_span else ""
    run_props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f"<w:tc>{props}<w:p><w:r>{run_props}<w:t>{text}</w:t></w:r></w:p></w:tc>"


def tr(*cells: str) -> str:
    return "<w:tr>" + "".join(cells) + "</w:tr>"


def tbl(*rows: str) -> str:
    return "<w:tbl>" + "".join(rows) + "</w:tbl>"


def synthetic_utility_docx(tmp_path: Path, duplicate_hr: bool = False) -> Path:
    rows = [
        tr(tc("Model", grid_span=1), tc("HR@5"), tc("NDCG@5")),
        tr(tc("MostPop"), tc("0,1734*", bold=True), tc("0.111\u2020")),
    ]
    if duplicate_hr:
        rows.append(tr(tc("Random"), tc("0.0500"), tc("0.0200")))
    path = tmp_path / "synthetic.docx"
    write_synthetic_docx(path, p("Tabella 3: Utility") + tbl(*rows))
    return path


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5ar_output_schemas() -> None:
    built = frames()
    assert list(built["docx_source_inventory"].columns) == DOCX_SOURCE_COLUMNS
    assert list(built["docx_tables"].columns) == DOCX_TABLE_COLUMNS
    assert list(built["paper_table_inventory"].columns) == PAPER_TABLE_COLUMNS
    assert list(built["numeric_cell_candidates"].columns) == CANDIDATE_COLUMNS
    assert list(built["numeric_cell_recovery"].columns) == RECOVERY_COLUMNS
    assert list(built["manual_pdf_review"].columns) == MANUAL_REVIEW_COLUMNS
    assert list(built["preliminary_numeric_comparison"].columns) == PRELIM_COLUMNS
    assert list(built["coverage_impact"].columns) == COVERAGE_COLUMNS
    assert list(built["anomalies"].columns) == ANOMALY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


def test_docx_zip_xml_paragraph_and_table_parsing(tmp_path: Path) -> None:
    path = synthetic_utility_docx(tmp_path)
    parsed = parse_docx(path)
    assert parsed.valid_zip is True
    assert parsed.document_xml_present is True
    assert parsed.paragraphs == ("Tabella 3: Utility",)
    assert len(parsed.cells) == 6
    inventory = build_docx_source_inventory(tmp_path, parsed)
    assert inventory.iloc[0]["docx_valid_zip"] == "true"
    assert inventory.iloc[0]["table_count"] == "1"
    assert inventory.iloc[0]["cell_count"] == "6"


def test_docx_table_extraction_preserves_numbers_markers_headers_and_fusion(tmp_path: Path) -> None:
    parsed = parse_docx(synthetic_utility_docx(tmp_path))
    tables = build_docx_tables(parsed)
    assert len(tables) == 6
    assert tables.loc[tables["cell_text"].eq("Model"), "is_header"].iloc[0] == "true"
    comma_value = tables.loc[tables["cell_text"].eq("0,1734*")].iloc[0]
    dot_value = tables.loc[tables["cell_text"].eq("0.111\u2020")].iloc[0]
    assert comma_value["normalized_text"] == "0.1734*"
    assert comma_value["numeric_value"] == "0.1734"
    assert comma_value["decimal_places"] == "4"
    assert comma_value["significance_marker"] == "*"
    assert "bold_text_detected" in comma_value["notes"]
    assert dot_value["numeric_value"] == "0.111"
    assert dot_value["significance_marker"] == "\u2020"
    assert tables["notes"].str.contains("gridSpan", regex=False).any()


def test_paper_table_inventory_detects_utility_table(tmp_path: Path) -> None:
    tables = build_docx_tables(parse_docx(synthetic_utility_docx(tmp_path)))
    inventory = build_paper_table_inventory(tables)
    assert len(inventory) == 1
    row = inventory.iloc[0]
    assert row["paper_table_number"] == "3"
    assert row["section"] == "utility"
    assert row["contains_f5a_target"] == "true"


def test_numeric_token_parser_handles_comma_dot_percentage_and_marker() -> None:
    comma = parse_numeric_token("0,1734*")
    dot_percent = parse_numeric_token("42.10%")
    assert comma["numeric_value"] == "0.1734"
    assert comma["decimal_places"] == "4"
    assert comma["significance_marker"] == "*"
    assert dot_percent["numeric_value"] == "42.10"
    assert dot_percent["decimal_places"] == "2"
    assert dot_percent["is_percentage"] == "true"


def test_mapping_target_unique_and_multiple_candidates(tmp_path: Path) -> None:
    target = pd.DataFrame(
        [
            {
                "paper_numeric_cell_id": "PRC-HR",
                "paper_component_id": "PRC-HR",
                "paper_dataset": "ML-1M",
                "paper_task": "ranking",
                "paper_model": "MostPop",
                "paper_metric": "hr_at_5",
                "paper_cutoff": "5",
            }
        ]
    )
    tables = build_docx_tables(parse_docx(synthetic_utility_docx(tmp_path)))
    inventory = build_paper_table_inventory(tables)
    candidates = build_numeric_cell_candidates(target, tables, inventory)
    assert len(candidates) == 1
    assert candidates.iloc[0]["candidate_status"] == "unique_high_confidence"
    assert candidates.iloc[0]["source_type"] == "secondary_structured_source"
    assert candidates.iloc[0]["requires_pdf_confirmation"] == "true"

    duplicate_tables = build_docx_tables(parse_docx(synthetic_utility_docx(tmp_path, duplicate_hr=True)))
    duplicate_inventory = build_paper_table_inventory(duplicate_tables)
    duplicate_candidates = build_numeric_cell_candidates(target, duplicate_tables, duplicate_inventory)
    assert duplicate_candidates["candidate_status"].eq("multiple_candidates").all()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_actual_recovery_manual_review_and_preliminary_comparison_have_11_rows() -> None:
    built = frames()
    recovery = built["numeric_cell_recovery"]
    manual = built["manual_pdf_review"]
    preliminary = built["preliminary_numeric_comparison"]
    assert len(recovery) == 11
    assert len(manual) == 11
    assert len(preliminary) == 11
    assert recovery["paper_numeric_cell_id"].is_unique
    assert manual["paper_numeric_cell_id"].is_unique
    assert manual["pdf_value_confirmed"].eq("false").all()
    assert manual["review_status"].eq("pending_manual_pdf_review").all()
    assert preliminary["is_scientific_replication_claim_allowed"].eq("false").all()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_coverage_policy_keeps_exact_and_verified_pdf_unchanged() -> None:
    coverage = {row["coverage_metric"]: row for _, row in frames()["coverage_impact"].iterrows()}
    assert "secondary_structured_recovery_coverage" in coverage
    assert coverage["verified_paper_numeric_cell_extraction_coverage"]["f5ar_value"] == "0.0"
    assert coverage["exact_numerical_coverage"]["f5a_value"] == coverage["exact_numerical_coverage"]["f5ar_value"]
    assert coverage["functional_numerical_coverage"]["f5a_value"] == coverage["functional_numerical_coverage"]["f5ar_value"]


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_validation_decision_reflects_missing_or_available_docx() -> None:
    built = frames()
    source = built["docx_source_inventory"].iloc[0]
    if source["docx_valid_zip"] == "true" and source["document_xml_present"] == "true" and len(built["docx_tables"]) > 0:
        assert final_decision(built["validation"]) == "finalization_f5ar_complete"
    else:
        assert final_decision(built["validation"]) == "finalization_f5ar_blocked"
        failed = set(built["validation"].loc[built["validation"]["status"].eq("fail"), "check_name"])
        assert {"docx_present", "docx_zip_valid", "document_xml_present", "tables_extracted"}.intersection(failed)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5ar_determinism_and_idempotent_tmp_outputs(tmp_path: Path) -> None:
    built = frames()
    first = {key: frame_sha256(value) for key, value in built.items()}
    second = {key: frame_sha256(value) for key, value in build_frames(PROJECT_ROOT).items()}
    assert first == second
    report = build_report(PROJECT_ROOT, built)
    write_outputs(tmp_path, built, report)
    first_hashes = output_hashes(tmp_path)
    write_outputs(tmp_path, built, report)
    second_hashes = output_hashes(tmp_path)
    assert first_hashes == second_hashes
    assert set(first_hashes) == {REPORT_PATH.as_posix(), *(path.as_posix() for path in OUTPUT_TABLES.values())}


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_no_absolute_paths_no_nan_and_protected_inputs_unchanged() -> None:
    built = frames()
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())
    protected_status = protected_hash_status(PROJECT_ROOT, load_sources(PROJECT_ROOT).f2_protected_after)
    assert protected_status == "checked=1144;changed=0"

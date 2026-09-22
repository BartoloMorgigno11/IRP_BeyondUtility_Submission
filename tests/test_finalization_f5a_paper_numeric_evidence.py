from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f5a_paper_numeric_evidence import (
    ALIGNMENT_COLUMNS,
    ANOMALY_COLUMNS,
    CLAIM_COLUMNS,
    COMMAND_TOOLS,
    COVERAGE_COLUMNS,
    NUMERIC_EXTRACTION_COLUMNS,
    NUMERIC_VERIFICATION_COLUMNS,
    OUTPUT_TABLES,
    PAPER_SOURCE_COLUMNS,
    PDF_TOOL_COLUMNS,
    PROTOCOL_COLUMNS,
    PYTHON_MODULE_TOOLS,
    REPORT_PATH,
    LOCAL_MAPPING_COLUMNS,
    VALIDATION_COLUMNS,
    build_frames,
    build_numeric_cell_verification,
    build_report,
    final_decision,
    frame_sha256,
    no_absolute_paths,
    output_hashes,
    parse_paper_value,
    protected_hash_status,
    write_outputs,
    load_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PDF_HASH = "a143265ae42f027c079766b1c006c438d698897b732634d78a010eb0a62f0f23"


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5a_output_schemas() -> None:
    built = frames()
    assert list(built["paper_source_inventory"].columns) == PAPER_SOURCE_COLUMNS
    assert list(built["pdf_tool_inventory"].columns) == PDF_TOOL_COLUMNS
    assert list(built["numeric_cell_extraction"].columns) == NUMERIC_EXTRACTION_COLUMNS
    assert list(built["local_result_mapping"].columns) == LOCAL_MAPPING_COLUMNS
    assert list(built["protocol_equivalence"].columns) == PROTOCOL_COLUMNS
    assert list(built["numeric_cell_verification"].columns) == NUMERIC_VERIFICATION_COLUMNS
    assert list(built["paper_numeric_alignment"].columns) == ALIGNMENT_COLUMNS
    assert list(built["coverage_impact"].columns) == COVERAGE_COLUMNS
    assert list(built["claim_update"].columns) == CLAIM_COLUMNS
    assert list(built["anomalies"].columns) == ANOMALY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5a_is_deterministic_in_memory() -> None:
    first = {key: frame_sha256(value) for key, value in frames().items()}
    second = {key: frame_sha256(value) for key, value in build_frames(PROJECT_ROOT).items()}
    assert first == second


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5a_output_hashes_are_idempotent_in_tmpdir(tmp_path: Path) -> None:
    built = frames()
    report = build_report(PROJECT_ROOT, built)
    write_outputs(tmp_path, built, report)
    first = output_hashes(tmp_path)
    write_outputs(tmp_path, built, report)
    second = output_hashes(tmp_path)
    assert first == second
    assert set(first) == {REPORT_PATH.as_posix(), *(path.as_posix() for path in OUTPUT_TABLES.values())}


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_pdf_inventory_and_authority_selection() -> None:
    paper = frames()["paper_source_inventory"]
    assert len(paper) == 2
    assert paper["sha256"].nunique() == 1
    assert paper["sha256"].iloc[0] == EXPECTED_PDF_HASH
    assert paper["is_authoritative"].eq("true").sum() == 1
    authoritative = paper.loc[paper["is_authoritative"].eq("true")].iloc[0]
    assert authoritative["file"] == "paper/2411.00331v1.pdf"
    assert "Beyond Utility" in authoritative["title"]
    assert authoritative["page_count"] == "12"
    assert authoritative["text_extractable"] == "false"
    assert authoritative["table_text_extractable"] == "false"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_pdf_tool_inventory_records_required_probes() -> None:
    tools = frames()["pdf_tool_inventory"]
    assert set(COMMAND_TOOLS).issubset(set(tools["tool_name"]))
    assert set(PYTHON_MODULE_TOOLS).issubset(set(tools["tool_name"]))
    selected = tools.loc[tools["selected_for_use"].eq("true")]
    assert selected["tool_name"].tolist() == ["python_stdlib_raw_pdf_metadata_scan"]
    assert selected["test_status"].tolist() == ["metadata_only_pass"]


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_all_target_cells_are_preserved_once() -> None:
    built = frames()
    extraction = built["numeric_cell_extraction"]
    alignment = built["paper_numeric_alignment"]
    assert len(extraction) == 11
    assert len(alignment) == 11
    assert extraction["paper_numeric_cell_id"].is_unique
    assert alignment["paper_numeric_cell_id"].is_unique
    assert set(extraction["paper_numeric_cell_id"]) == set(alignment["paper_numeric_cell_id"])
    assert extraction["paper_numeric_cell_id"].tolist() == sorted(extraction["paper_numeric_cell_id"].tolist())


def test_paper_value_parsing_preserves_semantics() -> None:
    decimal = parse_paper_value("42.1")
    percentage = parse_paper_value("42.1%")
    dispersion = parse_paper_value("42.1 ± 0.8")
    marked = parse_paper_value("42.100†")
    assert decimal["paper_value_numeric"] == "42.1"
    assert decimal["paper_decimal_places"] == "1"
    assert percentage["paper_is_percentage"] == "true"
    assert percentage["paper_scale"] == "percentage"
    assert dispersion["paper_value_numeric"] == "42.1"
    assert dispersion["paper_dispersion_numeric"] == "0.8"
    assert dispersion["paper_dispersion_type"] == "plus_minus"
    assert marked["paper_decimal_places"] == "3"
    assert marked["paper_significance_marker"] == "†"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_local_mapping_and_protocol_policy() -> None:
    built = frames()
    mapping = built["local_result_mapping"]
    protocol = built["protocol_equivalence"]
    assert mapping["local_result_id"].astype(str).str.len().gt(0).sum() == 6
    assert mapping["mapping_status"].eq("unique_candidate").sum() == 0
    assert mapping["mapping_status"].eq("no_candidate").sum() == 5
    assert mapping["mapping_status"].eq("protocol_incompatible").sum() == 6
    assert protocol["protocol_equivalence_class"].eq("exact").sum() == 0
    assert protocol["protocol_equivalence_class"].eq("functionally_equivalent").sum() == 0
    assert protocol["protocol_equivalence_class"].eq("adapted").sum() == 0
    assert protocol["protocol_equivalence_class"].eq("incompatible").sum() == 6
    assert protocol["protocol_equivalence_class"].eq("not_applicable").sum() == 5


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_numeric_verification_policy_on_actual_and_synthetic_cells() -> None:
    verification = frames()["numeric_cell_verification"]
    assert verification["printed_precision_match"].eq("true").sum() == 0
    assert verification["exact_raw_match"].eq("true").sum() == 0
    assert verification["is_exact_numeric_replication"].eq("true").sum() == 0
    assert verification["is_functional_numeric_replication"].eq("true").sum() == 0
    assert verification["is_numeric_match_without_protocol_equivalence"].eq("true").sum() == 0
    assert verification["numeric_verification_class"].eq("paper_value_unresolved").all()

    extraction = pd.DataFrame(
        [
            {
                "paper_numeric_cell_id": "synthetic_exact",
                "paper_value_text": "0.321",
                "paper_value_numeric": "0.321",
                "paper_decimal_places": "3",
                "machine_verified": "true",
            },
            {
                "paper_numeric_cell_id": "synthetic_incompatible",
                "paper_value_text": "0.321",
                "paper_value_numeric": "0.321",
                "paper_decimal_places": "3",
                "machine_verified": "true",
            },
        ]
    )
    mapping = pd.DataFrame(
        [
            {"paper_numeric_cell_id": "synthetic_exact", "local_result_id": "local1", "local_value": "0.321"},
            {"paper_numeric_cell_id": "synthetic_incompatible", "local_result_id": "local2", "local_value": "0.3214"},
        ]
    )
    protocol = pd.DataFrame(
        [
            {"paper_numeric_cell_id": "synthetic_exact", "protocol_equivalence_class": "exact"},
            {"paper_numeric_cell_id": "synthetic_incompatible", "protocol_equivalence_class": "incompatible"},
        ]
    )
    synthetic = build_numeric_cell_verification(extraction, mapping, protocol)
    exact = synthetic.loc[synthetic["paper_numeric_cell_id"].eq("synthetic_exact")].iloc[0]
    incompatible = synthetic.loc[synthetic["paper_numeric_cell_id"].eq("synthetic_incompatible")].iloc[0]
    assert exact["printed_precision_match"] == "true"
    assert exact["exact_raw_match"] == "true"
    assert exact["is_exact_numeric_replication"] == "true"
    assert incompatible["printed_precision_match"] == "true"
    assert incompatible["is_exact_numeric_replication"] == "false"
    assert incompatible["is_numeric_match_without_protocol_equivalence"] == "true"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_coverage_claims_anomalies_and_validation() -> None:
    built = frames()
    coverage = built["coverage_impact"]
    coverage_by_name = {row["coverage_metric"]: row for _, row in coverage.iterrows()}
    assert coverage_by_name["exact numerical result-cell coverage"]["f4_value"] == "0.0"
    assert coverage_by_name["exact numerical result-cell coverage"]["f5a_value"] == "0.0"
    assert coverage_by_name["verified_paper_numeric_cell_extraction_coverage"]["f5a_value"] == "0.0"
    assert coverage_by_name["mapped local result-cell coverage"]["f5a_value"] == "54.545455"
    assert len(built["claim_update"]) >= 12
    assert len(built["anomalies"]) >= 5
    assert built["anomalies"]["can_continue"].eq("true").all()
    assert built["validation"]["status"].eq("pass").all()
    assert final_decision(built["validation"]) == "finalization_f5a_complete"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_no_absolute_paths_no_nan_and_protected_inputs_unchanged() -> None:
    built = frames()
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())
    protected_status = protected_hash_status(PROJECT_ROOT, load_sources(PROJECT_ROOT).f2_protected_after)
    assert protected_status == "checked=1144;changed=0"

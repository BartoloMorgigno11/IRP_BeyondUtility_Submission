from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f6a_paper_scope_rebaseline import (
    ADDITIONAL_DATASET_COLUMNS,
    AFFECTED_ARTIFACT_COLUMNS,
    AUTHORITATIVE_SCOPE_COLUMNS,
    CLAIM_COLUMNS,
    COMPONENT_COLUMNS,
    CORRECTED_COVERAGE_COLUMNS,
    COVERAGE_FAMILY_COLUMNS,
    DATASET_COLUMNS,
    DIMENSION_COLUMNS,
    EXTENSION_COLUMNS,
    LASTFM_STATUS_COLUMNS,
    MISSING_REBASELINE_COLUMNS,
    MODEL_COLUMNS,
    OUTPUT_TABLES,
    PROMPT_COLUMNS,
    QWEN_MISTRAL_COLUMNS,
    REPORT_PATH,
    VALIDATION_COLUMNS,
    build_frames,
    build_report,
    final_decision,
    frame_sha256,
    no_absolute_paths,
    output_hashes,
    protected_hash_status,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def test_f6a_output_schemas() -> None:
    built = frames()
    assert list(built["authoritative_paper_scope"].columns) == AUTHORITATIVE_SCOPE_COLUMNS
    assert list(built["paper_component_inventory"].columns) == COMPONENT_COLUMNS
    assert list(built["dataset_taxonomy"].columns) == DATASET_COLUMNS
    assert list(built["model_taxonomy"].columns) == MODEL_COLUMNS
    assert list(built["prompt_taxonomy"].columns) == PROMPT_COLUMNS
    assert list(built["metric_dimension_taxonomy"].columns) == DIMENSION_COLUMNS
    assert list(built["lastfm_reproduction_status"].columns) == LASTFM_STATUS_COLUMNS
    assert list(built["qwen2_mistral_alignment"].columns) == QWEN_MISTRAL_COLUMNS
    assert list(built["extension_inventory"].columns) == EXTENSION_COLUMNS
    assert list(built["additional_dataset_requirement"].columns) == ADDITIONAL_DATASET_COLUMNS
    assert list(built["coverage_family_definitions"].columns) == COVERAGE_FAMILY_COLUMNS
    assert list(built["corrected_component_coverage"].columns) == CORRECTED_COVERAGE_COLUMNS
    assert list(built["missing_evidence_rebaseline"].columns) == MISSING_REBASELINE_COLUMNS
    assert list(built["affected_artifact_registry"].columns) == AFFECTED_ARTIFACT_COLUMNS
    assert list(built["claim_corrections"].columns) == CLAIM_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


def test_lastfm_is_original_paper_dataset_not_extension() -> None:
    dataset = frames()["dataset_taxonomy"].set_index("dataset").loc["LastFM"]
    assert dataset["authoritative_paper_dataset"] == "true"
    assert dataset["is_additional_dataset"] == "false"
    assert dataset["can_satisfy_extra_dataset_requirement"] == "false"
    assert dataset["local_status"] == "original_paper_component_blocked"
    assert dataset["local_interaction_rows"] == "92834"
    assert dataset["local_catalog_rows"] == "17632"
    assert dataset["local_candidate_set_exists"] == "false"
    assert dataset["local_ground_truth_exists"] == "false"


def test_qwen_mistral_are_paper_families_with_adapted_local_runs() -> None:
    models = frames()["model_taxonomy"].set_index("paper_model_family")
    assert len(models) == 7
    for family in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]:
        row = models.loc[family]
        assert row["paper_presence_verified"] == "true"
        assert row["model_family_equivalence"] == "true"
        assert row["exact_version_equivalence"] == "false"
        assert row["prompt_equivalence"] == "false"
        assert row["candidate_set_equivalence"] == "false"
        assert row["classification"] == "original_paper_component_locally_adapted"
        assert row["local_model_identifier"]


def test_qwen_mistral_alignment_counts_and_invalid_rows() -> None:
    alignment = frames()["qwen2_mistral_alignment"].set_index("paper_model_family")
    assert alignment.loc["Qwen2 7B Instruct", "local_model_identifier"] == "qwen2:7b-instruct"
    assert alignment.loc["Mistral 7B Instruct", "local_model_identifier"] == "mistral:7b-instruct"
    assert alignment.loc["Qwen2 7B Instruct", "invalid_output_rows"] == "26"
    assert alignment.loc["Mistral 7B Instruct", "invalid_output_rows"] == "0"
    assert alignment["candidate_count"].eq("20").all()
    assert alignment["datasets"].str.contains("Amazon Beauty;Amazon Sports;ML-1M", regex=False).all()


def test_calibration_extension_and_no_fifth_dataset() -> None:
    dims = frames()["metric_dimension_taxonomy"].set_index("dimension")
    assert dims.loc["calibration", "classification"] == "local_extension"
    additional = frames()["additional_dataset_requirement"].set_index("requirement_id")
    assert additional.loc["F6A-ADR-001", "answer"] == "no"
    assert additional.loc["F6A-ADR-002", "answer"] == "no"
    assert additional.loc["F6A-ADR-003", "answer"] == "yes"


def test_corrected_coverage_family_counts() -> None:
    coverage = frames()["corrected_component_coverage"].set_index("coverage_family")
    assert coverage.loc["paper dataset identification", "numerator"] == "4"
    assert coverage.loc["local evaluation-ready dataset coverage", "numerator"] == "3"
    assert coverage.loc["paper LLM model-family inventory", "denominator"] == "7"
    assert coverage.loc["exact LLM model-version coverage", "numerator"] == "0"
    assert coverage.loc["atomic paper numeric-cell extraction", "denominator"] == "187"
    assert coverage.loc["local result mapping coverage", "numerator"] == "44"
    assert coverage.loc["protocol-exact numeric replication", "numerator"] == "0"
    assert coverage.loc["corrected component-level denominator", "coverage_percent"] == "not_safely_derivable"


def test_missing_evidence_rebaseline_flags_taxonomy_corrections() -> None:
    missing = frames()["missing_evidence_rebaseline"]
    assert len(missing) >= 42
    lastfm = missing.loc[missing["f6a_classification"].eq("LastFM paper component")]
    assert not lastfm.empty
    assert lastfm["f6a_action"].str.contains("relabel_as_original_paper_dataset_blocked").any()
    split = missing.loc[missing["requires_split"].eq("true")]
    assert not split.empty
    assert split["f6a_action"].str.contains("split_model_family_presence").any()


def test_affected_artifacts_and_claim_corrections() -> None:
    affected = frames()["affected_artifact_registry"]
    assert len(affected) >= 17
    assert affected["artifact"].eq("results/tables/finalization_f5af_paper_model_inventory.csv").any()
    assert affected["must_be_superseded"].eq("true").sum() >= 17
    claims = frames()["claim_corrections"]
    assert claims["corrected_claim"].str.contains("Qwen2 and Mistral are original paper model families", regex=False).any()
    assert claims["corrected_claim"].str.contains("LastFM is an original paper dataset", regex=False).any()


def test_validation_decision_and_protected_inputs() -> None:
    built = frames()
    validation = built["validation"]
    assert validation["status"].eq("pass").all()
    assert final_decision(validation) == "finalization_f6a_complete"
    assert protected_hash_status(PROJECT_ROOT) == "checked=1144;changed=0"


def test_determinism_tmp_output_idempotence_and_f6a_only(tmp_path: Path) -> None:
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
    assert all("finalization_f6a_" in Path(path).name for path in first_hashes)


def test_no_absolute_paths_no_nan_and_report_sections() -> None:
    built = frames()
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())
    report = build_report(PROJECT_ROOT, built)
    for index in range(1, 23):
        assert f"# {index}. " in report

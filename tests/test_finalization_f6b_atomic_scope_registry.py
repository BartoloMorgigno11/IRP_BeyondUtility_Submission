from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f6b_atomic_scope_registry import (
    ALIGNMENT_COLUMNS,
    CLAIM_COLUMNS,
    COVERAGE_COLUMNS,
    DATASET_TASK_COLUMNS,
    DIMENSION_COLUMNS,
    EXTENSION_COLUMNS,
    MISSING_COLUMNS,
    MODEL_PROMPT_COLUMNS,
    OBSERVATION_COLUMNS,
    OUTPUT_TABLES,
    PAPER_COMPONENT_COLUMNS,
    PRIORITY_COLUMNS,
    REGRESSION_COLUMNS,
    REPORT_PATH,
    SUPERSESSION_COLUMNS,
    VALIDATION_COLUMNS,
    active_missing_count,
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


def test_f6b_output_schemas() -> None:
    built = frames()
    assert list(built["paper_component_inventory"].columns) == PAPER_COMPONENT_COLUMNS
    assert list(built["local_component_alignment"].columns) == ALIGNMENT_COLUMNS
    assert list(built["dataset_task_matrix"].columns) == DATASET_TASK_COLUMNS
    assert list(built["model_prompt_matrix"].columns) == MODEL_PROMPT_COLUMNS
    assert list(built["analysis_dimension_matrix"].columns) == DIMENSION_COLUMNS
    assert list(built["paper_observation_registry"].columns) == OBSERVATION_COLUMNS
    assert list(built["extension_inventory"].columns) == EXTENSION_COLUMNS
    assert list(built["corrected_coverage_dashboard"].columns) == COVERAGE_COLUMNS
    assert list(built["corrected_missing_evidence"].columns) == MISSING_COLUMNS
    assert list(built["experiment_priority"].columns) == PRIORITY_COLUMNS
    assert list(built["artifact_supersession_registry"].columns) == SUPERSESSION_COLUMNS
    assert list(built["regression_resolution"].columns) == REGRESSION_COLUMNS
    assert list(built["claim_corrections"].columns) == CLAIM_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


def test_authoritative_inventory_core_scope_and_atomicity() -> None:
    inv = frames()["paper_component_inventory"]
    assert inv["paper_component_id"].is_unique
    assert inv["is_atomic"].eq("true").all()
    assert inv["paper_evidence"].astype(str).str.len().gt(0).all()
    assert set(inv.loc[inv["component_type"].eq("dataset"), "dataset"]) == {"Amazon Beauty", "Amazon Sports", "ML-1M", "LastFM"}
    assert set(inv.loc[inv["component_type"].eq("task"), "task"]) == {"ranking", "reranking"}
    assert inv.loc[(inv["component_type"].eq("model_family")) & (inv["component_family"].eq("llm_model")), "model_family"].nunique() == 7
    assert inv.loc[(inv["component_type"].eq("model_family")) & (inv["component_family"].eq("traditional_model")), "model_family"].nunique() == 6
    assert inv.loc[inv["component_type"].eq("prompt_strategy"), "prompt_strategy"].nunique() == 3


def test_lastfm_qwen_mistral_and_calibration_classification() -> None:
    inv = frames()["paper_component_inventory"]
    align = frames()["local_component_alignment"]
    lastfm_id = inv.loc[(inv["component_type"].eq("dataset")) & (inv["dataset"].eq("LastFM")), "paper_component_id"].iloc[0]
    assert align.set_index("paper_component_id").loc[lastfm_id, "reproduction_class"] == "blocked_dataset_pipeline"
    assert "LastFM" not in " ".join(frames()["extension_inventory"]["extension_name"])
    assert {"Qwen2 7B Instruct", "Mistral 7B Instruct"}.issubset(set(inv["model_family"]))
    assert not inv["metric"].eq("calibration").any()
    assert frames()["extension_inventory"]["extension_name"].str.contains("Calibration", case=False).any()


def test_numeric_cells_are_separate_and_mapping_policy() -> None:
    inv = frames()["paper_component_inventory"]
    numeric = inv.loc[inv["component_type"].eq("numeric_result_cell")]
    assert len(numeric) == 187
    assert numeric["is_numeric"].eq("true").all()
    non_numeric = inv.loc[inv["component_type"].ne("numeric_result_cell")]
    assert non_numeric["is_numeric"].eq("false").all()
    align = frames()["local_component_alignment"]
    mapped = align.loc[(align["paper_component_id"].isin(numeric["paper_component_id"])) & (align["local_result_id"].astype(str).str.len().gt(0))]
    assert len(mapped) == 44
    assert align.loc[align["reproduction_class"].eq("exact"), "protocol_equivalence"].eq("exact").all()
    assert align["reproduction_class"].eq("exact").sum() == 0


def test_dataset_task_matrix_and_model_prompt_matrix() -> None:
    dt = frames()["dataset_task_matrix"]
    assert len(dt) == 8
    assert dt.loc[dt["dataset"].eq("LastFM"), "evaluation_ready"].eq("false").all()
    assert dt.loc[~dt["dataset"].eq("LastFM"), "evaluation_ready"].eq("true").all()
    mp = frames()["model_prompt_matrix"]
    assert mp.loc[mp["model_role"].eq("llm_as_recommender"), "model_family"].nunique() == 7
    assert mp.loc[mp["model_role"].eq("traditional_baseline"), "model_family"].nunique() == 6
    qwen = mp.loc[(mp["model_family"].eq("Qwen2 7B Instruct")) & (mp["prompt_strategy"].eq("OpenP5-derived base prompt"))].iloc[0]
    assert qwen["has_local_runs"] == "true"
    assert qwen["exact_version"] == "false"
    assert qwen["reproduction_class"] == "adapted"


def test_dimensions_observations_extensions_and_fifth_dataset() -> None:
    dims = frames()["analysis_dimension_matrix"]
    assert len(dims) == 8
    assert {"appendix fairness", "appendix diversity"}.issubset(set(dims["analysis_dimension"]))
    obs = frames()["paper_observation_registry"]
    assert len(obs) == 7
    assert set(obs["verification_status"]).issubset({"verified_exact", "verified_functional", "partially_supported", "not_supported", "not_tested", "blocked", "unknown"})
    ext = frames()["extension_inventory"].set_index("extension_id")
    assert ext.loc["EXT-FIFTH-DATASET-STATUS", "scientific_contribution"].endswith("no_if_mandatory")
    assert ext.loc["EXT-FIFTH-DATASET-STATUS", "implementation_status"] == "not_present"


def test_coverage_dashboard_denominator_policy() -> None:
    cov = frames()["corrected_coverage_dashboard"].set_index("coverage_family")
    assert cov.loc["dataset identification coverage", "numerator"] == "4"
    assert cov.loc["dataset evaluation-ready coverage", "numerator"] == "3"
    assert cov.loc["LLM-family local-run coverage", "numerator"] == "2"
    assert cov.loc["exact LLM-version coverage", "numerator"] == "0"
    assert cov.loc["traditional-model coverage", "denominator"] == "6"
    assert cov.loc["atomic numeric extraction coverage", "denominator"] == "187"
    assert cov.loc["local numeric mapping coverage", "numerator"] == "44"
    assert cov.loc["protocol-exact numeric replication coverage", "numerator"] == "0"
    assert cov.loc["flat overall component percentage", "percentage"] == "not_applicable"


def test_missing_evidence_accounting_split_and_supersession() -> None:
    missing = frames()["corrected_missing_evidence"]
    assert missing["source_missing_id"].astype(str).str.len().gt(0).sum() >= 42
    assert missing["is_newly_added"].eq("true").sum() == 2
    assert missing["corrected_classification"].str.contains("original_paper_dataset_blocked_pipeline").any()
    assert missing["corrected_classification"].str.contains("new_split_exact_protocol_gap").sum() == 2
    assert missing["is_superseded"].eq("true").sum() > 0
    assert active_missing_count(missing) > 0


def test_priority_artifact_claim_and_timeout_policy() -> None:
    priority = frames()["experiment_priority"]
    assert len(priority) >= 10
    assert priority.iloc[0]["experiment_title"] == "PDF/manual verification of paper numeric evidence"
    assert priority["estimated_monetary_cost"].eq("unknown").all()
    artifacts = frames()["artifact_supersession_registry"]
    assert len(artifacts) >= 17
    claims = frames()["claim_corrections"]
    assert len(claims) >= 10
    regression = frames()["regression_resolution"]
    assert regression.loc[regression["timed_out"].eq("true"), "is_conclusive"].eq("false").all()
    assert not regression.loc[regression["timed_out"].eq("true"), "execution_status"].str.contains("pass", case=False).any()


def test_validation_decision_protected_inputs_and_relative_paths() -> None:
    built = frames()
    validation = built["validation"]
    assert validation["status"].eq("pass").all()
    assert final_decision(validation) == "finalization_f6b_complete"
    assert protected_hash_status(PROJECT_ROOT) == "checked=1144;changed=0"
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())


def test_determinism_tmp_output_idempotence_and_report_sections(tmp_path: Path) -> None:
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
    for index in range(1, 28):
        assert f"# {index}. " in report

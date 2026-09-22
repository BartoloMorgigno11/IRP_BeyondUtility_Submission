"""Focused tests for the F6M reproduction closure."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

from src.analysis import phase_f6m_reproduction_closure as f6m


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def temp_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(f6m.load_config(f6m.ROOT / f6m.DEFAULT_CONFIG_PATH))
    config["outputs"] = {
        "report": str(tmp_path / "report.md"),
        "manifest": str(tmp_path / "manifest.json"),
        "tables": {
            "artifact_index": str(tmp_path / "artifact_index.csv"),
            "input_validation": str(tmp_path / "input_validation.csv"),
            "master_reproducibility_matrix": str(tmp_path / "master_reproducibility_matrix.csv"),
            "master_results_long": str(tmp_path / "master_results_long.csv"),
            "master_results_wide": str(tmp_path / "master_results_wide.csv"),
            "research_question_matrix": str(tmp_path / "research_question_matrix.csv"),
            "paper_claim_evidence_matrix": str(tmp_path / "paper_claim_evidence_matrix.csv"),
            "limitations_register": str(tmp_path / "limitations_register.csv"),
            "lastfm_final_status": str(tmp_path / "lastfm_final_status.csv"),
            "statistical_assessment": str(tmp_path / "statistical_assessment.csv"),
            "reproduction_extension_boundary": str(tmp_path / "reproduction_extension_boundary.csv"),
        },
    }
    return config


def write_yaml(path: Path, config: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_f6m_refuses_api_flag() -> None:
    with pytest.raises(RuntimeError, match="offline-only"):
        f6m.run_phase(f6m.ROOT / f6m.DEFAULT_CONFIG_PATH, allow_api_calls=True)


def test_input_validation_enforces_ranking_freeze_hash() -> None:
    config = copy.deepcopy(f6m.load_config(f6m.ROOT / f6m.DEFAULT_CONFIG_PATH))
    config["validation"]["expected_ranking_freeze_hash"] = "WRONG"
    with pytest.raises(RuntimeError, match="input validation failed"):
        f6m.validate_inputs(config)


def test_master_results_preserve_lastfm_primary_and_diagnostic_denominators() -> None:
    config = f6m.load_config(f6m.ROOT / f6m.DEFAULT_CONFIG_PATH)
    _, csv_inputs, _ = f6m.validate_inputs(config)
    rows = f6m.build_master_results(csv_inputs)

    ranking = [
        row
        for row in rows
        if row["dataset"] == "LastFM"
        and row["task"] == "ranking"
        and row["model_name"] == "qwen2:7b-instruct"
        and row["prompting_strategy"] == "base/OpenP5 title-list"
        and row["metric"] == "HR@5_fail_strict"
    ]
    assert len(ranking) == 1
    assert ranking[0]["value"] == "0.458"
    assert ranking[0]["positive_present_rows"] == ""

    reranking = [
        row
        for row in rows
        if row["task"] == "reranking"
        and row["model_name"] == "qwen2:7b-instruct"
        and row["prompting_strategy"] == "base/OpenP5-derived reranking"
        and row["metric"] == "HR@5_fail_strict"
    ]
    assert len(reranking) == 1
    assert reranking[0]["denominator"] == "1000"
    assert reranking[0]["positive_present_rows"] == "314"
    assert reranking[0]["metric_scope"] == "primary_fail_strict_all_1000_users"

    diagnostic = [
        row
        for row in rows
        if row["task"] == "reranking"
        and row["model_name"] == "qwen2:7b-instruct"
        and row["prompting_strategy"] == "base/OpenP5-derived reranking"
        and row["metric"] == "HR@5_positive_present_only"
    ]
    assert len(diagnostic) == 1
    assert diagnostic[0]["metric_scope"] == "diagnostic_positive_present_only"
    assert diagnostic[0]["denominator_policy"] == "positive_present_rows_only_diagnostic"


def test_run_phase_writes_closure_outputs_and_report_boundaries(tmp_path: Path) -> None:
    config = temp_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, config)

    manifest = f6m.run_phase(config_path)
    assert manifest["decision_token"] == f6m.DECISION_COMPLETE
    assert manifest["lastfm"]["reranking_aggregate_final_responses"] == 4000

    long_rows = read_csv(Path(config["outputs"]["tables"]["master_results_long"]))
    boundary_rows = read_csv(Path(config["outputs"]["tables"]["reproduction_extension_boundary"]))
    status_rows = read_csv(Path(config["outputs"]["tables"]["lastfm_final_status"]))
    report = Path(config["outputs"]["report"]).read_text(encoding="utf-8")

    assert any(row["task"] == "ranking" and row["dataset"] == "LastFM" for row in long_rows)
    assert any(row["task"] == "reranking" and row["positive_present_rows"] == "314" for row in long_rows)
    assert any(row["area"] == "future_extension" and row["status"] == "not_started" for row in boundary_rows)
    assert any(row["component"] == "local_llm_reranking_v4_1_execution" and row["evidence_status"] == "completed" for row in status_rows)
    assert "REPRODUCIBILITY CORE - CLOSED" in report
    assert "FUTURE EXTENSION - NOT STARTED" in report
    assert "LastFM remains incomplete" not in report


def test_statistical_assessment_does_not_invent_p_values() -> None:
    config = f6m.load_config(f6m.ROOT / f6m.DEFAULT_CONFIG_PATH)
    _, csv_inputs, _ = f6m.validate_inputs(config)
    rows = f6m.build_statistical_assessment(csv_inputs)
    assert rows
    assert all(row["p_value"] == "" for row in rows)
    assert any(row["status"] == "descriptive" for row in rows)

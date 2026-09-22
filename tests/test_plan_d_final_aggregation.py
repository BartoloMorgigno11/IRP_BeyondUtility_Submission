from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_final_aggregation import (
    BLOCKED_COLUMNS,
    EXPERIMENT_INDEX_COLUMNS,
    HYGIENE_COLUMNS,
    LLM_DIAGNOSTIC_COLUMNS,
    METRIC_SUMMARY_COLUMNS,
    REPRO_STATUS_COLUMNS,
    build_blocked_items,
    build_experiment_index,
    build_hygiene_audit,
    build_llm_diagnostic_summary,
    build_reproducibility_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "src/analysis/plan_d_final_aggregation.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/build_plan_d_final_reproducibility_package.py"

EXPERIMENT_INDEX_PATH = PROJECT_ROOT / "results/tables/plan_d_final_experiment_index.csv"
METRIC_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_final_metric_summary.csv"
LLM_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_final_llm_diagnostic_summary.csv"
REPRO_STATUS_PATH = PROJECT_ROOT / "results/tables/plan_d_final_reproducibility_status.csv"
BLOCKED_ITEMS_PATH = PROJECT_ROOT / "results/tables/plan_d_final_blocked_items.csv"
HYGIENE_AUDIT_PATH = PROJECT_ROOT / "results/tables/plan_d_repository_hygiene_audit.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_final_reproducibility_package.md"


EXPECTED_PHASES = {
    "D1",
    "D2",
    "D3",
    "D4",
    "D4.6",
    "D4.7",
    "D5B",
    "D5C0",
    "D5C1",
    "D5D",
    "D5E",
    "D5E.1",
    "D5F",
    "D5G",
    "D5H",
    "D5I",
    "D5J",
    "D5K",
}


def test_final_aggregation_source_files_exist() -> None:
    assert ANALYSIS_PATH.exists()
    assert RUNNER_PATH.exists()


def test_experiment_index_builder_contains_expected_d_phases() -> None:
    frame = build_experiment_index()

    assert set(EXPERIMENT_INDEX_COLUMNS).issubset(frame.columns)
    assert EXPECTED_PHASES.issubset(set(frame["phase_id"]))
    assert list(frame["phase_id"]).index("D1") < list(frame["phase_id"]).index("D5K")


def test_reproducibility_status_builder_includes_required_classifications() -> None:
    frame = build_reproducibility_status()
    text = " ".join(frame["classification"].tolist() + frame["allowed_claim"].tolist()).lower()

    assert set(REPRO_STATUS_COLUMNS).issubset(frame.columns)
    assert "exact reproduction" in text
    assert "reconstructed reproduction" in text
    assert "diagnostic_subset" in text
    assert "extended_subset" in text
    assert "extension" in text
    assert "blocked/conditional" in text
    assert "plan c is reconstructed reproduction" in text
    assert "llm full reproduction is not completed" in text


def test_blocked_items_include_required_blockers() -> None:
    frame = build_blocked_items()
    text = " ".join(frame["item"].tolist() + frame["reason"].tolist() + frame["current_status"].tolist()).lower()

    assert set(BLOCKED_COLUMNS).issubset(frame.columns)
    assert "claude 3 haiku" in text
    assert "llama3 70b" in text
    assert "closed-source llm" in text
    assert "lastfm" in text


def test_llm_diagnostic_summary_builder_includes_expected_phases() -> None:
    frame = build_llm_diagnostic_summary()

    assert set(LLM_DIAGNOSTIC_COLUMNS).issubset(frame.columns)
    assert {"D5E", "D5G", "D5I", "D5J", "D5K"}.issubset(set(frame["diagnostic_phase"]))
    assert "high hallucination" in " ".join(frame["main_finding"].tolist()).lower()
    assert "strict candidate-id-only" in " ".join(frame["main_finding"].tolist()).lower()


def test_hygiene_audit_builder_marks_raw_outputs_gitignored() -> None:
    frame = build_hygiene_audit()

    assert set(HYGIENE_COLUMNS).issubset(frame.columns)
    raw_row = frame.loc[frame["check_name"] == "raw_outputs_gitignored"].iloc[0]
    tracked_row = frame.loc[frame["check_name"] == "no_raw_jsonl_intended_for_commit"].iloc[0]
    assert raw_row["status"] == "passed"
    assert tracked_row["status"] == "passed"


def test_final_artifacts_if_present_have_expected_structure() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5L outputs are not generated before the final package build")
    index = pd.read_csv(EXPERIMENT_INDEX_PATH, dtype=str).fillna("")
    metrics = pd.read_csv(METRIC_SUMMARY_PATH, dtype=str).fillna("")
    llm = pd.read_csv(LLM_SUMMARY_PATH, dtype=str).fillna("")
    repro = pd.read_csv(REPRO_STATUS_PATH, dtype=str).fillna("")
    blocked = pd.read_csv(BLOCKED_ITEMS_PATH, dtype=str).fillna("")
    hygiene = pd.read_csv(HYGIENE_AUDIT_PATH, dtype=str).fillna("")

    assert set(EXPERIMENT_INDEX_COLUMNS).issubset(index.columns)
    assert set(METRIC_SUMMARY_COLUMNS).issubset(metrics.columns)
    assert set(LLM_DIAGNOSTIC_COLUMNS).issubset(llm.columns)
    assert set(REPRO_STATUS_COLUMNS).issubset(repro.columns)
    assert set(BLOCKED_COLUMNS).issubset(blocked.columns)
    assert set(HYGIENE_COLUMNS).issubset(hygiene.columns)
    assert EXPECTED_PHASES.issubset(set(index["phase_id"]))
    assert {"D5E", "D5G", "D5I", "D5J", "D5K"}.issubset(set(llm["diagnostic_phase"]))


def test_final_report_contains_required_boundaries_if_present() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5L report is not generated before the final package build")
    report = REPORT_PATH.read_text(encoding="utf-8").lower()

    assert "no new inference was run" in report
    assert "no ollama call was made" in report
    assert "no qwen2 call was made" in report
    assert "no mistral call was made" in report
    assert "no openai api call was made" in report
    assert "no anthropic api call was made" in report
    assert "no external api call was made" in report
    assert "no download was performed" in report
    assert "no model pull was performed" in report
    assert "no recommender training was performed" in report
    assert "no lastfm run was performed" in report
    assert "plan c was not modified" in report
    assert "no destructive cleanup was performed" in report
    assert "this is not exact reproduction" in report
    assert "this is not full llm reproduction" in report
    assert "plan c is reconstructed reproduction" in report


def test_generated_csv_ordering_is_deterministic_if_present() -> None:
    if not EXPERIMENT_INDEX_PATH.exists():
        pytest.skip("D5L outputs are not generated before the final package build")
    index = pd.read_csv(EXPERIMENT_INDEX_PATH, dtype=str).fillna("")
    blocked = pd.read_csv(BLOCKED_ITEMS_PATH, dtype=str).fillna("")

    assert index["phase_id"].tolist()[:4] == ["D1", "D2", "D3", "D4"]
    assert blocked["item"].tolist()[:4] == [
        "Claude 3 Haiku exact model",
        "Llama3 70B local execution",
        "Closed-source LLM original matrix",
        "LastFM utility evaluation",
    ]


def test_tests_do_not_call_inference_or_external_apis() -> None:
    test_text = Path(__file__).read_text(encoding="utf-8").lower()

    assert ("api" + "/generate") not in test_text
    assert ("ollama" + " run") not in test_text
    assert ("import " + "openai") not in test_text
    assert ("import " + "anthropic") not in test_text
    assert ("openai" + ".") not in test_text
    assert ("anthropic" + ".") not in test_text

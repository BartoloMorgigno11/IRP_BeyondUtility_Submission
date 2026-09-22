from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_exam_readiness_review import (
    CHECKLIST_COLUMNS,
    CLAIM_AUDIT_COLUMNS,
    ORAL_POINTS_COLUMNS,
    build_claim_safety_audit,
    build_exam_readiness_checklist,
    build_oral_defense_key_points,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "src/analysis/plan_d_exam_readiness_review.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/build_plan_d_exam_readiness_review.py"
CHECKLIST_PATH = PROJECT_ROOT / "results/tables/plan_d_exam_readiness_checklist.csv"
CLAIM_AUDIT_PATH = PROJECT_ROOT / "results/tables/plan_d_claim_safety_audit.csv"
ORAL_POINTS_PATH = PROJECT_ROOT / "results/tables/plan_d_oral_defense_key_points.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_exam_readiness_review.md"


REQUIRED_CHECK_AREAS = {
    "final report states no exact LLM reproduction",
    "final report states no full LLM reproduction",
    "Plan C classified as reconstructed reproduction",
    "D5E-D5K classified as diagnostic_subset",
    "blocked/conditional items are explicit",
    "strict prompt mitigation result is clearly explained",
    "hallucination/candidate validity distinction is clear",
    "position bias and history-length sensitivity are discussed",
    "Qwen vs Mistral comparison is interpreted correctly",
    "re-ranking diagnostic is interpreted without overclaiming",
    "raw outputs are said to be Git-ignored",
    "no API/no-download/no-training boundaries are stated",
}

REQUIRED_TOPICS = {
    "What exactly was reproduced?",
    "What was reconstructed?",
    "What is only diagnostic_subset?",
    "Why is the LLM part not exact reproduction?",
    "Why are hallucination and candidate validity important?",
    "What did strict prompting show?",
    "What did candidate position bias show?",
    "What did Qwen vs Mistral show?",
    "What did re-ranking show?",
    "What did history length sensitivity show?",
    "What is blocked and why?",
    "What is the main contribution of the project?",
    "What would be Plan E?",
}


def test_exam_readiness_source_files_exist() -> None:
    assert ANALYSIS_PATH.exists()
    assert RUNNER_PATH.exists()


def test_checklist_builder_contains_required_areas() -> None:
    checklist = build_exam_readiness_checklist()

    assert set(CHECKLIST_COLUMNS).issubset(checklist.columns)
    assert REQUIRED_CHECK_AREAS.issubset(set(checklist["area"]))
    d5h_row = checklist.loc[checklist["check_id"] == "C13"].iloc[0]
    assert d5h_row["status"] == "pass"
    assert "local Mistral smoke test passed" in d5h_row["finding"]


def test_claim_safety_audit_contains_risky_claims_and_safe_rewrites() -> None:
    claim_audit = build_claim_safety_audit()
    text = " ".join(claim_audit["risky_claim"].tolist() + claim_audit["safe_rewrite"].tolist()).lower()

    assert set(CLAIM_AUDIT_COLUMNS).issubset(claim_audit.columns)
    for risky in (
        "we reproduced the full llm part",
        "exact reproduction of beyond utility",
        "mistral/qwen results generalize to the full benchmark",
        "llms outperform neural recommenders",
        "hallucination is solved",
        "lastfm was reproduced",
        "closed-source llms were reproduced",
        "plan d is exact reproduction",
    ):
        assert risky in text
    assert "diagnostic_subset" in text
    assert "conditional/api-dependent" in text
    d5h = claim_audit.loc[claim_audit["claim_area"] == "D5H status"].iloc[0]
    assert d5h["status"] == "resolved_safe_rewrite"


def test_oral_defense_key_points_contain_required_topics() -> None:
    points = build_oral_defense_key_points()

    assert set(ORAL_POINTS_COLUMNS).issubset(points.columns)
    assert REQUIRED_TOPICS.issubset(set(points["topic"]))


def test_generated_artifacts_if_present_have_expected_structure() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5M outputs are not generated before the exam-readiness review build")
    checklist = pd.read_csv(CHECKLIST_PATH, dtype=str).fillna("")
    claim_audit = pd.read_csv(CLAIM_AUDIT_PATH, dtype=str).fillna("")
    oral_points = pd.read_csv(ORAL_POINTS_PATH, dtype=str).fillna("")

    assert set(CHECKLIST_COLUMNS).issubset(checklist.columns)
    assert set(CLAIM_AUDIT_COLUMNS).issubset(claim_audit.columns)
    assert set(ORAL_POINTS_COLUMNS).issubset(oral_points.columns)
    assert REQUIRED_CHECK_AREAS.issubset(set(checklist["area"]))
    assert REQUIRED_TOPICS.issubset(set(oral_points["topic"]))
    assert checklist.loc[checklist["check_id"] == "C13"].iloc[0]["status"] == "pass"


def test_report_contains_required_exam_readiness_boundaries_if_present() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5M report is not generated before the exam-readiness review build")
    report = REPORT_PATH.read_text(encoding="utf-8").lower()

    assert "not exact reproduction" in report
    assert "diagnostic_subset" in report
    assert "no new inference/api/download/training was run in d5m" in report
    assert "no openai api call was made" in report
    assert "no anthropic api call was made" in report
    assert "no external api call was made" in report
    assert "no download was performed" in report
    assert "no recommender training was performed" in report
    assert "no destructive cleanup was performed" in report


def test_generated_csv_ordering_is_deterministic_if_present() -> None:
    if not CHECKLIST_PATH.exists():
        pytest.skip("D5M outputs are not generated before the exam-readiness review build")
    checklist = pd.read_csv(CHECKLIST_PATH, dtype=str).fillna("")
    claim_audit = pd.read_csv(CLAIM_AUDIT_PATH, dtype=str).fillna("")
    oral_points = pd.read_csv(ORAL_POINTS_PATH, dtype=str).fillna("")

    assert checklist["check_id"].tolist()[:3] == ["C01", "C02", "C03"]
    assert claim_audit["risky_claim"].tolist()[0] == "we reproduced the full LLM part"
    assert oral_points["topic"].tolist()[0] == "What exactly was reproduced?"


def test_tests_do_not_call_inference_or_external_apis() -> None:
    test_text = Path(__file__).read_text(encoding="utf-8").lower()

    assert ("api" + "/generate") not in test_text
    assert ("ollama" + " run") not in test_text
    assert ("import " + "openai") not in test_text
    assert ("import " + "anthropic") not in test_text
    assert ("openai" + ".") not in test_text
    assert ("anthropic" + ".") not in test_text

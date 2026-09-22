"""Focused checks for F6K scientific gap readiness artifacts."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]

ROOT = Path(__file__).resolve().parents[1]
APPROVED_INTERPRETER = ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

CSV_PATH = ROOT / "results" / "tables" / "phase_f6k_scientific_gap_readiness.csv"
REPORT_PATH = ROOT / "report" / "phase_f6k_scientific_gap_readiness.md"
HANDOFF_PATH = ROOT / "report" / "future_operations_context_handoff.md"

EXPECTED_COLUMNS = [
    "gap_id",
    "related_rq_or_claim",
    "current_status",
    "authoritative_evidence",
    "required_inputs",
    "available_inputs",
    "missing_inputs",
    "requires_training",
    "requires_llm_inference",
    "requires_ollama",
    "requires_network",
    "requires_download",
    "estimated_requests",
    "estimated_runtime",
    "expected_scientific_value",
    "comparability_risk",
    "reproducibility_risk",
    "decision",
    "decision_reason",
    "recommended_execution_order",
    "stop_condition",
]

EXPECTED_ORDER = [
    "profile_generation_effectiveness",
    "runtime_cost_characterization",
    "lastfm_llm_reranking",
]

EXPECTED_DECISIONS = {
    "profile_generation_effectiveness": "blocked_missing_frozen_inputs",
    "runtime_cost_characterization": "execute_now_high_value",
    "lastfm_llm_reranking": "execute_now_limited_value",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_runtime_is_the_repaired_audit_interpreter() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"


def test_csv_schema_order_and_decisions_are_exact() -> None:
    rows = _rows(CSV_PATH)
    assert len(rows) == 3
    assert list(rows[0]) == EXPECTED_COLUMNS
    assert [row["gap_id"] for row in rows] == EXPECTED_ORDER
    assert {row["decision"] for row in rows} == set(EXPECTED_DECISIONS.values())
    for row in rows:
        assert row["decision"] == EXPECTED_DECISIONS[row["gap_id"]]
        for boolean_column in [
            "requires_training",
            "requires_llm_inference",
            "requires_ollama",
            "requires_network",
            "requires_download",
        ]:
            assert row[boolean_column] in {"true", "false"}


def test_gap_rows_preserve_source_evidence_and_execution_boundaries() -> None:
    rows = {row["gap_id"]: row for row in _rows(CSV_PATH)}

    profile = rows["profile_generation_effectiveness"]
    assert "RQ8" in profile["related_rq_or_claim"]
    assert "phase_f6e1_prompt_source_inventory.csv" in profile["authoritative_evidence"]
    assert "canonical_llm_execution_matrix.csv" in profile["authoritative_evidence"]
    assert profile["estimated_requests"].startswith("0 executable now")
    assert profile["requires_training"] == "false"
    assert profile["requires_llm_inference"] == "true"
    assert profile["requires_ollama"] == "true"
    assert "No frozen generated profiles" in profile["missing_inputs"]

    runtime = rows["runtime_cost_characterization"]
    assert "RQ14" in runtime["related_rq_or_claim"]
    assert "phase_f6e4b_runtime_summary.csv" in runtime["authoritative_evidence"]
    assert "phase_f6e2_runtime_estimates.csv" in runtime["authoritative_evidence"]
    assert "plan_e_api_cost_scenarios.csv" in runtime["authoritative_evidence"]
    assert runtime["estimated_requests"] == "0"
    assert runtime["requires_training"] == "false"
    assert runtime["requires_llm_inference"] == "false"
    assert runtime["requires_ollama"] == "false"
    assert runtime["requires_network"] == "false"
    assert runtime["requires_download"] == "false"

    lastfm = rows["lastfm_llm_reranking"]
    assert "RQ13" in lastfm["related_rq_or_claim"]
    assert "phase_f6g3_corrective_request_manifest.csv" in lastfm["authoritative_evidence"]
    assert "canonical_local_llm_reranking_top10_parser_contract.csv" in lastfm["authoritative_evidence"]
    assert "phase_f6g2r_ollama_preflight.csv" in lastfm["authoritative_evidence"]
    assert lastfm["estimated_requests"] == "7436"
    assert lastfm["requires_training"] == "false"
    assert lastfm["requires_llm_inference"] == "true"
    assert lastfm["requires_ollama"] == "true"
    assert "592/1859" in lastfm["decision_reason"]


def test_report_answers_required_f6k_questions_and_contains_token() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "f6k_scientific_gap_readiness_complete" in report
    assert "151 passed, 1698 deselected in 404.58s" in report
    assert "Execute now only the observational runtime/cost characterization" in report
    assert "LastFM reranking is useful only if the user authorizes" in report
    assert "Profile-generation effectiveness remains blocked" in report
    assert "Local monetary cost can be reported as `0` only for API/cash spend" in report
    assert "Estimated new executions for F6K: `0`" in report
    assert "Estimated future LastFM reranking requests: `7436`" in report
    assert "After F6K, finalization can proceed" in report
    assert "paper-exact reproduction ready" not in report.lower()
    assert "official current pricing" in report


def test_no_f6k_training_api_download_or_raw_output_generation_is_claimed() -> None:
    combined = CSV_PATH.read_text(encoding="utf-8") + "\n" + REPORT_PATH.read_text(encoding="utf-8")
    forbidden_claims = [
        "training completed",
        "api calls performed",
        "downloads performed",
        "raw responses generated",
        "ollama started",
    ]
    for claim in forbidden_claims:
        assert claim not in combined.lower()

    raw_roots = [
        ROOT / "results" / "llm",
        ROOT / "results" / "raw_outputs",
        ROOT / "results" / "raw",
    ]
    offenders: list[Path] = []
    for raw_root in raw_roots:
        if raw_root.exists():
            offenders.extend(raw_root.glob("**/*phase_f6k*"))
    assert offenders == []


def test_handoff_keeps_f6j_context_and_adds_minimal_f6k_state() -> None:
    handoff = HANDOFF_PATH.read_text(encoding="utf-8")
    assert "f6j_scientific_evidence_synthesis_complete" in handoff
    assert "151 passed, 1698 deselected" in handoff
    assert "f6k_scientific_gap_readiness_complete" in handoff
    for gap_id, decision in EXPECTED_DECISIONS.items():
        assert gap_id in handoff
        assert decision in handoff
    assert "runtime/cost characterization first" in handoff
    assert "LastFM reranking only as separate explicitly authorized adapted local inference" in handoff
    assert "profile generation remains blocked" in handoff


def test_f6k_created_only_the_authorized_artifact_family() -> None:
    expected_paths = {
        CSV_PATH.resolve(),
        REPORT_PATH.resolve(),
        Path(__file__).resolve(),
    }
    discovered: set[Path] = set()
    for pattern_root in [
        ROOT / "src",
        ROOT / "configs",
        ROOT / "results" / "tables",
        ROOT / "results" / "manifests",
        ROOT / "report",
        ROOT / "tests",
    ]:
        if pattern_root.exists():
            discovered.update(path.resolve() for path in pattern_root.glob("**/*phase_f6k*") if path.is_file())
    assert discovered == expected_paths


def test_no_paths_are_staged_by_f6k() -> None:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""

"""Focused checks for the F6H1 offline exactness-elevation audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "src" / "analysis" / "plan_f6h1_offline_exactness_elevation.py"
CONFIG = PROJECT_ROOT / "configs" / "phase_f6h1_offline_exactness_elevation.yaml"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

TABLES = PROJECT_ROOT / "results" / "tables"
MANIFESTS = PROJECT_ROOT / "results" / "manifests"
REPORT = PROJECT_ROOT / "report" / "phase_f6h1_offline_exactness_elevation.md"
HANDOFF = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
CURRENT_CONTEXT = MANIFESTS / "current_project_context_snapshot.json"

OUTPUTS = [
    TABLES / "phase_f6h1_target_candidate_inventory.csv",
    TABLES / "phase_f6h1_documentary_verification.csv",
    TABLES / "phase_f6h1_raw_evidence_reconciliation.csv",
    TABLES / "phase_f6h1_offline_recomputation.csv",
    TABLES / "phase_f6h1_component_elevation_decisions.csv",
    TABLES / "phase_f6h1_numeric_cell_elevation_decisions.csv",
    TABLES / "phase_f6h1_unresolved_or_rejected_candidates.csv",
    TABLES / "phase_f6h1_coverage_delta.csv",
    TABLES / "phase_f6h1_validation.csv",
    TABLES / "phase_f6h1_determinism.csv",
    TABLES / "phase_f6h1_next_actions.csv",
    MANIFESTS / "phase_f6h1_offline_exactness_elevation_manifest.json",
    MANIFESTS / "phase_f6h1_protected_input_hashes.json",
    REPORT,
    HANDOFF,
    CURRENT_CONTEXT,
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_builder() -> None:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(BUILDER), "--config", str(CONFIG)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_runtime_manifest_and_counts() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"

    manifest = json.loads((MANIFESTS / "phase_f6h1_offline_exactness_elevation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision"] == "phase_f6h1_offline_exactness_elevation_complete_no_elevations"
    assert manifest["environment"]["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert manifest["environment"]["not_exact_original_runtime"] is True
    assert manifest["counts"]["target_component_candidates"] == 54
    assert manifest["counts"]["documentary_candidates"] == 5
    assert manifest["counts"]["offline_recomputation_candidates"] == 38
    assert manifest["counts"]["existing_raw_output_candidates"] == 11
    assert manifest["counts"]["offline_or_raw_candidates"] == 49
    assert manifest["counts"]["usable_exact_evidence_candidates"] == 0
    assert manifest["counts"]["component_elevated"] == 0
    assert manifest["counts"]["numeric_cell_elevated"] == 0


def test_required_tables_have_stable_counts_and_no_lastfm_reranking() -> None:
    expected_counts = {
        "phase_f6h1_target_candidate_inventory.csv": 54,
        "phase_f6h1_documentary_verification.csv": 5,
        "phase_f6h1_raw_evidence_reconciliation.csv": 54,
        "phase_f6h1_offline_recomputation.csv": 49,
        "phase_f6h1_component_elevation_decisions.csv": 54,
        "phase_f6h1_numeric_cell_elevation_decisions.csv": 49,
        "phase_f6h1_unresolved_or_rejected_candidates.csv": 54,
        "phase_f6h1_coverage_delta.csv": 11,
        "phase_f6h1_validation.csv": 19,
        "phase_f6h1_next_actions.csv": 3,
    }
    for filename, count in expected_counts.items():
        assert len(read_rows(TABLES / filename)) == count, filename

    targets = read_rows(TABLES / "phase_f6h1_target_candidate_inventory.csv")
    assert Counter(row["prior_frontier_class"] for row in targets) == {
        "exact_possible_by_documentary_verification_only": 5,
        "exact_possible_by_offline_recomputation": 38,
        "exact_possible_with_existing_raw_outputs": 11,
    }
    assert all(row["lastfm_reranking_paper_row"] == "false" for row in targets)
    semantic_keys = [(row["prior_frontier_class"], row["dataset"], row["task"], row["model"], row["metric"], row["contract_id"]) for row in targets]
    assert semantic_keys == sorted(semantic_keys)


def test_every_candidate_has_exactly_one_decision_and_no_elevation() -> None:
    targets = read_rows(TABLES / "phase_f6h1_target_candidate_inventory.csv")
    decisions = read_rows(TABLES / "phase_f6h1_component_elevation_decisions.csv")
    target_ids = {row["candidate_id"] for row in targets}
    decision_ids = [row["candidate_id"] for row in decisions]
    assert set(decision_ids) == target_ids
    assert len(decision_ids) == len(set(decision_ids))
    assert all(row["elevated"] == "false" for row in decisions)
    assert all(row["new_classification"] != "exact_paper_replication" for row in decisions)
    assert all(row["maximum_defensible_claim"] == "potential_only_not_exact_paper_replication" for row in decisions)


def test_documentary_candidates_remain_blocked_without_direct_pdf_statement() -> None:
    rows = read_rows(TABLES / "phase_f6h1_documentary_verification.csv")
    assert len(rows) == 5
    assert all(row["pdf_text_extraction_available"] == "false" for row in rows)
    assert all(row["exact_supporting_statement"] == "paper_unverified_direct_statement_unavailable_with_approved_runtime" for row in rows)
    assert all(row["material_fields_verified"] == "false" for row in rows)
    assert all(row["new_classification"] == "unresolved_direct_paper_verification_unavailable" for row in rows)


def test_raw_evidence_is_not_admitted_without_full_lineage() -> None:
    rows = read_rows(TABLES / "phase_f6h1_raw_evidence_reconciliation.csv")
    route_counts = Counter((row["route"], row["admission_status"]) for row in rows)
    assert route_counts[("documentary_verification", "not_applicable")] == 5
    assert route_counts[("offline_recomputation", "not_admitted_for_exact_recomputation")] == 38
    assert route_counts[("existing_raw_output", "not_admitted_for_exact_recomputation")] == 11
    assert sum(row["admission_status"] == "admitted" for row in rows) == 0
    assert all(row["evidence_non_empty"] == "true" for row in rows if row["route"] != "documentary_verification")
    assert all(row["lineage_status"] != "complete_exact_lineage" for row in rows)


def test_recomputation_and_numeric_exactness_gate() -> None:
    recomputation = read_rows(TABLES / "phase_f6h1_offline_recomputation.csv")
    numeric = read_rows(TABLES / "phase_f6h1_numeric_cell_elevation_decisions.csv")
    assert len(recomputation) == 49
    assert all(row["recomputation_status"] == "blocked_before_recomputation" for row in recomputation)
    assert all(row["recomputed_value"] == "" for row in recomputation)
    assert all(row["denominator"] != "" for row in recomputation)
    assert all(row["numeric_similarity_only_elevation_allowed"] == "false" for row in numeric)
    assert all(row["exact_cell_verdict"] == "not_exact_recomputed_or_not_exact_comparable" for row in numeric)


def test_coverage_delta_reconciles_with_zero_elevations() -> None:
    coverage = {row["coverage_metric"]: row for row in read_rows(TABLES / "phase_f6h1_coverage_delta.csv")}
    assert coverage["demonstrated_exact_component_coverage"]["previous_numerator"] == "0"
    assert coverage["demonstrated_exact_component_coverage"]["new_numerator"] == "0"
    assert coverage["demonstrated_exact_component_coverage"]["delta_numerator"] == "0"
    assert coverage["exact_numeric_cell_coverage"]["previous_numerator"] == "0"
    assert coverage["exact_numeric_cell_coverage"]["new_numerator"] == "0"
    assert coverage["remaining_numeric_offline_raw_frontier"]["new_numerator"] == "49"
    assert coverage["offline_raw_frontier_remaining"]["new_numerator"] == "49"


def test_rejected_reason_counts_and_validation_passes() -> None:
    unresolved = read_rows(TABLES / "phase_f6h1_unresolved_or_rejected_candidates.csv")
    assert Counter(row["blocking_category"] for row in unresolved) == {
        "direct_paper_verification_unavailable": 5,
        "incomplete_official_population_lineage": 38,
        "incomplete_raw_request_response_lineage": 11,
    }
    validation = read_rows(TABLES / "phase_f6h1_validation.csv")
    assert all(row["status"] == "pass" for row in validation)


def test_protected_hash_invariance() -> None:
    payload = json.loads((MANIFESTS / "phase_f6h1_protected_input_hashes.json").read_text(encoding="utf-8"))
    assert payload["changed"] == []
    assert payload["missing"] == []
    assert payload["before"] == payload["after"]


def test_deterministic_two_run_output_hashes() -> None:
    run_builder()
    first = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in OUTPUTS}
    run_builder()
    second = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in OUTPUTS}
    assert first == second

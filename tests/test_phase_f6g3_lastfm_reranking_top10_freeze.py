"""Focused checks for the F6G3 LastFM reranking top-10 freeze."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "src" / "analysis" / "plan_f6g3_lastfm_reranking_top10_freeze.py"
CONFIG = PROJECT_ROOT / "configs" / "phase_f6g3_lastfm_reranking_top10_freeze.yaml"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

TABLES = PROJECT_ROOT / "results" / "tables"
MANIFESTS = PROJECT_ROOT / "results" / "manifests"
REPORT = PROJECT_ROOT / "report" / "phase_f6g3_lastfm_reranking_top10_freeze.md"
HANDOFF = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
CURRENT_CONTEXT = MANIFESTS / "current_project_context_snapshot.json"

OUTPUTS = [
    TABLES / "phase_f6g3_source_artifact_inventory.csv",
    TABLES / "phase_f6g3_legacy_manifest_supersession.csv",
    TABLES / "phase_f6g3_corrective_request_manifest.csv",
    TABLES / "phase_f6g3_corrective_shard_manifest.csv",
    TABLES / "phase_f6g3_request_response_reconciliation.csv",
    TABLES / "phase_f6g3_candidate_pool_validation.csv",
    TABLES / "phase_f6g3_top10_derivation.csv",
    TABLES / "phase_f6g3_canonical_recommendations.csv",
    TABLES / "phase_f6g3_invalid_or_blocked_rows.csv",
    TABLES / "phase_f6g3_reranking_metrics.csv",
    TABLES / "phase_f6g3_beyond_utility_metrics.csv",
    TABLES / "phase_f6g3_cell_summary.csv",
    TABLES / "phase_f6g3_exactness_elevation.csv",
    TABLES / "phase_f6g3_validation.csv",
    TABLES / "phase_f6g3_determinism.csv",
    TABLES / "phase_f6g3_next_actions.csv",
    MANIFESTS / "phase_f6g3_top10_freeze_manifest.json",
    MANIFESTS / "phase_f6g3_protected_input_hashes.json",
    REPORT,
    HANDOFF,
    CURRENT_CONTEXT,
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


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


def test_runtime_contract_and_manifest_counts() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"

    manifest = json.loads((MANIFESTS / "phase_f6g3_top10_freeze_manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision"] == "f6g3_top10_freeze_complete_blocked_no_existing_raw_responses"
    assert manifest["environment"]["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert manifest["environment"]["not_exact_original_runtime"] is True
    assert manifest["counts"]["logical_instances"] == 1859
    assert manifest["counts"]["corrective_requests"] == 7436
    assert manifest["counts"]["raw_rows_reconciled"] == 0
    assert manifest["counts"]["admitted_top10_rows"] == 0
    assert manifest["counts"]["invalid_or_blocked_rows"] == 7436
    assert manifest["rates"]["strict_valid_rate"] == "0.000000"
    assert manifest["rates"]["candidate_valid_rate"] == "0.000000"


def test_required_output_schemas_and_row_counts() -> None:
    expected_counts = {
        "phase_f6g3_source_artifact_inventory.csv": 13,
        "phase_f6g3_legacy_manifest_supersession.csv": 3,
        "phase_f6g3_corrective_request_manifest.csv": 7436,
        "phase_f6g3_corrective_shard_manifest.csv": 16,
        "phase_f6g3_request_response_reconciliation.csv": 7436,
        "phase_f6g3_candidate_pool_validation.csv": 1859,
        "phase_f6g3_top10_derivation.csv": 7436,
        "phase_f6g3_canonical_recommendations.csv": 0,
        "phase_f6g3_invalid_or_blocked_rows.csv": 7436,
        "phase_f6g3_reranking_metrics.csv": 40,
        "phase_f6g3_beyond_utility_metrics.csv": 28,
        "phase_f6g3_cell_summary.csv": 4,
        "phase_f6g3_exactness_elevation.csv": 0,
        "phase_f6g3_validation.csv": 16,
        "phase_f6g3_next_actions.csv": 3,
    }
    for filename, row_count in expected_counts.items():
        path = TABLES / filename
        assert path.exists(), filename
        assert len(read_rows(path)) == row_count, filename

    assert read_header(TABLES / "phase_f6g3_canonical_recommendations.csv") == [
        "f6g3_request_id",
        "rank",
        "item_id",
        "source_raw_response",
        "derivation_status",
        "notes",
    ]


def test_candidate_pool_and_top10_fail_closed_contract() -> None:
    candidate_rows = read_rows(TABLES / "phase_f6g3_candidate_pool_validation.csv")
    assert all(row["candidate_pool_size"] == "20" for row in candidate_rows)
    assert all(row["candidate_unique_count"] == "20" for row in candidate_rows)
    assert sum(row["positive_present"] == "true" for row in candidate_rows) == 592

    request_rows = read_rows(TABLES / "phase_f6g3_corrective_request_manifest.csv")
    assert all(row["corrective_requested_output_length"] == "10" for row in request_rows)
    assert all(row["corrective_parser_expected_length"] == "10" for row in request_rows)
    assert all(row["metric_cutoff_k"] == "5" for row in request_rows)
    assert all(row["output_length_resolution"] == "D_repository_derived_top10_paper_unspecified" for row in request_rows)

    derivation_rows = read_rows(TABLES / "phase_f6g3_top10_derivation.csv")
    assert all(row["admitted_to_freeze"] == "false" for row in derivation_rows)
    assert all(row["derived_output_length"] == "" for row in derivation_rows)
    assert all(row["canonical_top10_item_ids"] == "" for row in derivation_rows)
    assert all(row["first_five_item_ids"] == "" for row in derivation_rows)
    assert all(row["block_reason"] == "missing_existing_lastfm_reranking_raw_response" for row in derivation_rows)


def test_request_reconciliation_and_blocked_denominator_identity() -> None:
    request_ids = {row["f6g3_request_id"] for row in read_rows(TABLES / "phase_f6g3_corrective_request_manifest.csv")}
    reconciliation = read_rows(TABLES / "phase_f6g3_request_response_reconciliation.csv")
    derivation = read_rows(TABLES / "phase_f6g3_top10_derivation.csv")
    blocked = read_rows(TABLES / "phase_f6g3_invalid_or_blocked_rows.csv")

    assert {row["f6g3_request_id"] for row in reconciliation} == request_ids
    assert {row["f6g3_request_id"] for row in derivation} == request_ids
    assert {row["f6g3_request_id"] for row in blocked} == request_ids
    assert all(row["raw_response_found"] == "false" for row in reconciliation)
    assert all(row["raw_output_file"] == "" for row in reconciliation)
    assert all(row["admission_status"] == "blocked" for row in reconciliation)
    assert all(row["denominator_policy"] == "retained_in_operational_denominator" for row in blocked)
    assert all(row["repair_allowed"] == "false" for row in blocked)


def test_metric_rows_do_not_zero_fill_blocked_utility_or_beyond_utility() -> None:
    metric_rows = read_rows(TABLES / "phase_f6g3_reranking_metrics.csv")
    beyond_rows = read_rows(TABLES / "phase_f6g3_beyond_utility_metrics.csv")
    utility_rows = [row for row in metric_rows if row["metric_family"] == "utility"]
    quality_zero_rows = [row for row in metric_rows if row["metric"] in {"strict_parse_validity", "candidate_validity"}]

    assert utility_rows
    assert all(row["value"] == "" for row in utility_rows)
    assert all(row["status"] == "blocked_no_admitted_raw_rows" for row in utility_rows)
    assert {row["value"] for row in quality_zero_rows} == {"0.000000"}
    assert all(row["denominator"] == "1859" for row in quality_zero_rows)
    assert all(row["value"] == "" for row in beyond_rows)
    assert all(row["status"] == "blocked_no_admitted_top10_recommendations" for row in beyond_rows)


def test_legacy_supersession_and_exactness_no_elevation() -> None:
    supersession = read_rows(TABLES / "phase_f6g3_legacy_manifest_supersession.csv")
    assert any(row["legacy_phase"] == "phase_f6g1" and "superseded" in row["supersession_status"] for row in supersession)
    assert any(row["legacy_phase"] == "phase_f6g2" and "historical_blocked" in row["supersession_status"] for row in supersession)
    assert read_rows(TABLES / "phase_f6g3_exactness_elevation.csv") == []


def test_protected_hash_invariance() -> None:
    payload = json.loads((MANIFESTS / "phase_f6g3_protected_input_hashes.json").read_text(encoding="utf-8"))
    assert payload["changed"] == []
    assert payload["missing"] == []
    assert payload["before"] == payload["after"]


def test_deterministic_two_run_output_hashes() -> None:
    run_builder()
    first = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in OUTPUTS}
    run_builder()
    second = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in OUTPUTS}
    assert first == second

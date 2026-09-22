from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.audit.common_io import read_csv_rows as _read_csv_rows


pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
BUILDER = PROJECT_ROOT / "src" / "analysis" / "build_workspace_forensic_state_audit.py"


REQUIRED_TABLES = {
    "results/tables/workspace_relevant_file_inventory.csv": ["path", "sha256", "category", "currentness"],
    "results/tables/workspace_phase_and_supersession_registry.csv": ["phase_id", "current_status", "superseded_by"],
    "results/tables/paper_atomic_contract.csv": ["contract_id", "paper_specification_status", "requested_output_length"],
    "results/tables/paper_repository_contract_conflicts.csv": ["conflict_id", "current_resolution"],
    "results/tables/paper_to_local_traceability.csv": ["contract_id", "paper_alignment", "evidence_quality"],
    "results/tables/project_claim_verification.csv": ["claim_id", "verdict", "corrective_wording"],
    "results/tables/current_reproduction_status_matrix.csv": ["dataset", "execution_status", "paper_alignment"],
    "results/tables/artifact_lineage_and_supersession.csv": ["artifact_path", "status", "canonical_for"],
    "results/tables/numeric_result_reconciliation.csv": ["result_id", "verification_status"],
    "results/tables/blockers_gaps_and_next_actions.csv": ["priority", "component", "safe_next_action"],
}


def run_builder() -> None:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run([str(PYTHON), str(BUILDER)], cwd=PROJECT_ROOT, env=env, check=True)


def read_csv(path: str) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(path, root=PROJECT_ROOT, encoding="utf-8-sig")


def test_builder_requires_recovered_runtime_and_outputs_schemas():
    run_builder()
    for relative, columns in REQUIRED_TABLES.items():
        rows = read_csv(relative)
        assert rows, relative
        for column in columns:
            assert column in rows[0], f"{relative} missing {column}"


def test_runtime_snapshot_is_repaired_clone_not_exact_original():
    run_builder()
    snapshot = json.loads((PROJECT_ROOT / "results/manifests/current_project_context_snapshot.json").read_text(encoding="utf-8"))
    env = snapshot["environment"]
    assert env["approved_runtime_interpreter"].endswith(".venv_audit_recovery\\Scripts\\python.exe")
    assert env["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert env["not_exact_original_runtime"] is True


def test_conservative_exactness_and_blockers_are_present():
    run_builder()
    status = read_csv("results/tables/current_reproduction_status_matrix.csv")
    assert not [row for row in status if row["paper_alignment"] == "exact_paper_replication"]
    blockers = read_csv("results/tables/blockers_gaps_and_next_actions.csv")
    assert any(row["component"] == "LastFM LLM reranking" for row in blockers)
    conflicts = read_csv("results/tables/paper_repository_contract_conflicts.csv")
    assert any(row["current_resolution"] == "D_repository_derived_top10_paper_unspecified" for row in conflicts)


def test_claim_verification_flags_overstated_reproduction_claims():
    run_builder()
    claims = read_csv("results/tables/project_claim_verification.csv")
    assert claims
    assert any(row["verdict"] in {"unsupported", "verified_historical_only", "partially_verified"} for row in claims)

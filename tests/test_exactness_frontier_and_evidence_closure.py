from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.audit.common_io import read_csv_rows as _read_csv_rows


pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
BUILDER = PROJECT_ROOT / "src" / "analysis" / "build_exactness_frontier_and_evidence_closure.py"

ALLOWED_FRONTIER_CLASSES = {
    "exact_already_supported_but_previously_underclassified",
    "exact_possible_by_offline_recomputation",
    "exact_possible_by_documentary_verification_only",
    "exact_possible_with_existing_raw_outputs",
    "exact_possible_but_requires_new_local_execution",
    "exact_blocked_by_unavailable_model_or_version",
    "exact_blocked_by_paid_or_retired_service",
    "exact_blocked_by_missing_original_data_or_checkpoint",
    "exact_blocked_by_paper_unspecified_material_detail",
    "exact_impossible_due_to_known_protocol_divergence",
    "adapted_replication_only",
    "local_extension_only",
    "unresolved",
}

REQUIRED_OUTPUTS = [
    "report/exactness_frontier_and_evidence_closure.md",
    "report/future_operations_context_handoff.md",
    "results/tables/exactness_frontier_by_component.csv",
    "results/tables/numeric_cell_exactness_frontier.csv",
    "results/tables/exactness_elevation_candidates.csv",
    "results/tables/irreducible_exactness_blockers.csv",
    "results/tables/fifth_dataset_extension_requirements.csv",
    "results/manifests/exactness_frontier_snapshot.json",
]

COMPONENT_COLUMNS = {
    "contract_id",
    "dataset",
    "task",
    "model",
    "prompt_strategy",
    "metric",
    "current_execution_status",
    "current_paper_alignment",
    "paper_directly_verified",
    "paper_source_location",
    "first_material_mismatch",
    "mismatch_category",
    "missing_evidence",
    "actual_divergence",
    "frontier_class",
    "offline_elevation_possible",
    "required_action",
    "required_compute",
    "required_external_access",
    "maximum_defensible_classification",
    "notes",
}

NUMERIC_COLUMNS = {
    "result_id",
    "contract_id",
    "paper_table",
    "paper_coordinates",
    "paper_source_location",
    "dataset",
    "task",
    "model",
    "prompt_strategy",
    "metric",
    "paper_value",
    "currently_available_local_value",
    "cohort_comparability",
    "protocol_comparability",
    "model_version_comparability",
    "prompt_comparability",
    "parser_comparability",
    "metric_comparability",
    "exact_recomputation_feasibility",
    "first_blocking_mismatch",
    "mismatch_category",
    "frontier_class",
    "notes",
}


def run_builder() -> None:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run([str(PYTHON), str(BUILDER)], cwd=PROJECT_ROOT, env=env, check=True)


def read_csv(relative: str) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(relative, root=PROJECT_ROOT, encoding="utf-8-sig")


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (PROJECT_ROOT / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_required_schemas_and_exact_row_counts():
    run_builder()
    components = read_csv("results/tables/exactness_frontier_by_component.csv")
    numeric = read_csv("results/tables/numeric_cell_exactness_frontier.csv")

    assert len(components) == 192
    assert len({row["contract_id"] for row in components}) == 192
    assert len(numeric) == 187
    assert len({row["result_id"] for row in numeric}) == 187
    assert COMPONENT_COLUMNS.issubset(components[0].keys())
    assert NUMERIC_COLUMNS.issubset(numeric[0].keys())


def test_allowed_frontier_classes_and_consistency():
    run_builder()
    components = read_csv("results/tables/exactness_frontier_by_component.csv")
    numeric = read_csv("results/tables/numeric_cell_exactness_frontier.csv")

    assert {row["frontier_class"] for row in components}.issubset(ALLOWED_FRONTIER_CLASSES)
    assert {row["frontier_class"] for row in numeric}.issubset(ALLOWED_FRONTIER_CLASSES)
    assert all(row["first_material_mismatch"] for row in components if row["frontier_class"] != "exact_already_supported_but_previously_underclassified")
    assert all(row["offline_elevation_possible"] == "false" for row in components if row["frontier_class"].startswith("exact_blocked"))
    assert all(row["actual_divergence"] == "true" for row in components if row["frontier_class"] in {"adapted_replication_only", "exact_impossible_due_to_known_protocol_divergence"})


def test_elevation_candidates_are_safe_offline_only():
    run_builder()
    rows = read_csv("results/tables/exactness_elevation_candidates.csv")
    allowed = {
        "exact_already_supported_but_previously_underclassified",
        "exact_possible_by_documentary_verification_only",
        "exact_possible_by_offline_recomputation",
        "exact_possible_with_existing_raw_outputs",
    }
    assert rows
    assert {row["frontier_class"] for row in rows}.issubset(allowed)
    assert all(row["required_external_access"] == "none" for row in rows)
    assert all("training" not in row["required_compute"].lower() for row in rows)


def test_fifth_dataset_contract_excludes_lastfm():
    run_builder()
    rows = read_csv("results/tables/fifth_dataset_extension_requirements.csv")
    joined = "\n".join(" ".join(row.values()) for row in rows)
    assert "LastFM is a paper dataset" in joined
    assert "cannot satisfy the fifth-dataset requirement" in joined
    assert len(rows) >= 7


def test_determinism_and_protected_input_fingerprint():
    run_builder()
    first_hashes = {relative: sha256(relative) for relative in REQUIRED_OUTPUTS}
    first_snapshot = json.loads((PROJECT_ROOT / "results/manifests/exactness_frontier_snapshot.json").read_text(encoding="utf-8"))

    run_builder()
    second_hashes = {relative: sha256(relative) for relative in REQUIRED_OUTPUTS}
    second_snapshot = json.loads((PROJECT_ROOT / "results/manifests/exactness_frontier_snapshot.json").read_text(encoding="utf-8"))

    assert first_hashes == second_hashes
    assert first_snapshot["protected_input_fingerprint"] == second_snapshot["protected_input_fingerprint"]
    assert first_snapshot["protected_input_fingerprint"]["row_count"] == 136

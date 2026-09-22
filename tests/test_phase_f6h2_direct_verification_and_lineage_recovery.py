"""Focused checks for F6H2 direct verification and lineage recovery."""

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
BUILDER = PROJECT_ROOT / "src" / "analysis" / "plan_f6h2_direct_verification_and_lineage_recovery.py"
CONFIG = PROJECT_ROOT / "configs" / "phase_f6h2_direct_verification_and_lineage_recovery.yaml"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

TABLES = PROJECT_ROOT / "results" / "tables"
MANIFESTS = PROJECT_ROOT / "results" / "manifests"
REPORT = PROJECT_ROOT / "report" / "phase_f6h2_direct_verification_and_lineage_recovery.md"
HANDOFF = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
CURRENT_CONTEXT = MANIFESTS / "current_project_context_snapshot.json"

OUTPUTS = [
    TABLES / "phase_f6h2_target_inventory.csv",
    TABLES / "phase_f6h2_paper_evidence_register.csv",
    TABLES / "phase_f6h2_documentary_verification.csv",
    TABLES / "phase_f6h2_official_population_lineage.csv",
    TABLES / "phase_f6h2_population_fingerprints.csv",
    TABLES / "phase_f6h2_population_comparison.csv",
    TABLES / "phase_f6h2_request_response_lineage.csv",
    TABLES / "phase_f6h2_request_response_reconciliation.csv",
    TABLES / "phase_f6h2_orphan_and_duplicate_inventory.csv",
    TABLES / "phase_f6h2_source_code_lineage.csv",
    TABLES / "phase_f6h2_evidence_recovery_decisions.csv",
    TABLES / "phase_f6h2_recomputation_readiness.csv",
    TABLES / "phase_f6h2_validation.csv",
    TABLES / "phase_f6h2_determinism.csv",
    TABLES / "phase_f6h2_next_actions.csv",
    MANIFESTS / "phase_f6h2_evidence_recovery_manifest.json",
    MANIFESTS / "phase_f6h2_protected_input_hashes.json",
    REPORT,
    HANDOFF,
    CURRENT_CONTEXT,
]

PDF_HASH = "A143265AE42F027C079766B1C006C438D698897B732634D78A010EB0A62F0F23"


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


def test_runtime_manifest_and_core_counts() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"

    manifest = json.loads((MANIFESTS / "phase_f6h2_evidence_recovery_manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision"] == "phase_f6h2_evidence_recovery_complete_no_strict_readiness"
    assert manifest["environment"]["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert manifest["environment"]["not_exact_original_runtime"] is True
    assert manifest["environment"]["approved_runtime_interpreter"] == str(APPROVED_INTERPRETER)
    assert manifest["counts"] == {
        "target_candidates": 54,
        "documentary_candidates": 5,
        "documentary_directly_verified": 0,
        "documentary_paper_unspecified_confirmed": 0,
        "population_candidates": 38,
        "complete_official_population_lineages": 0,
        "repository_only_population_lineages": 0,
        "request_response_candidates": 11,
        "complete_request_response_lineages": 0,
        "ready_for_strict_offline_recomputation": 0,
        "ready_for_repository_derived_recomputation_only": 0,
        "candidates_still_blocked": 54,
        "permanently_unrecoverable": 0,
    }
    assert manifest["f6g3_negative_evidence"]["expected_requests"] == 7436
    assert manifest["f6g3_negative_evidence"]["raw_responses_reconciled"] == 0
    assert manifest["f6h1_zero_elevation"]["exact_component_coverage"] == "0/192"


def test_required_tables_have_stable_counts_and_sorting() -> None:
    expected_counts = {
        "phase_f6h2_target_inventory.csv": 54,
        "phase_f6h2_paper_evidence_register.csv": 5,
        "phase_f6h2_documentary_verification.csv": 5,
        "phase_f6h2_official_population_lineage.csv": 38,
        "phase_f6h2_population_fingerprints.csv": 76,
        "phase_f6h2_population_comparison.csv": 38,
        "phase_f6h2_request_response_lineage.csv": 11,
        "phase_f6h2_request_response_reconciliation.csv": 11,
        "phase_f6h2_orphan_and_duplicate_inventory.csv": 5,
        "phase_f6h2_source_code_lineage.csv": 5,
        "phase_f6h2_evidence_recovery_decisions.csv": 54,
        "phase_f6h2_recomputation_readiness.csv": 54,
        "phase_f6h2_validation.csv": 24,
        "phase_f6h2_next_actions.csv": 3,
    }
    for filename, row_count in expected_counts.items():
        assert len(read_rows(TABLES / filename)) == row_count, filename

    targets = read_rows(TABLES / "phase_f6h2_target_inventory.csv")
    semantic_keys = [(row["f6h2_track"], row["dataset"], row["task"], row["model"], row["metric"], row["contract_id"]) for row in targets]
    assert semantic_keys == sorted(semantic_keys)
    assert Counter(row["f6h2_track"] for row in targets) == {
        "track_a_direct_paper_verification": 5,
        "track_b_official_population_lineage": 38,
        "track_c_request_response_lineage": 11,
    }
    assert all(row["lastfm_reranking_paper_row"] == "false" for row in targets)


def test_paper_evidence_is_hash_only_and_not_directly_verified() -> None:
    manifest = json.loads((MANIFESTS / "phase_f6h2_evidence_recovery_manifest.json").read_text(encoding="utf-8"))
    paper_files = manifest["paper_environment"]["paper_files"]
    assert paper_files[0]["sha256"] == PDF_HASH
    assert paper_files[1]["sha256"] == PDF_HASH
    assert paper_files[0]["byte_identical_to_primary_pdf"] is True
    assert paper_files[1]["byte_identical_to_primary_pdf"] is True
    assert not any(manifest["paper_environment"]["python_pdf_modules"].values())
    assert not any(manifest["paper_environment"]["local_pdf_commands"].values())

    register = read_rows(TABLES / "phase_f6h2_paper_evidence_register.csv")
    documentary = read_rows(TABLES / "phase_f6h2_documentary_verification.csv")
    assert {row["pdf_hash"] for row in register} == {PDF_HASH}
    assert all(row["page"] == "verification_unavailable" for row in register)
    assert all(row["verification_method"] == "pdf_hash_and_tool_inventory;secondary_local_artifact_inventory" for row in register)
    assert all(row["specification_status"] == "verification_unavailable" for row in register)
    assert all(row["directly_verified"] == "false" for row in documentary)
    assert all(row["paper_unspecified_confirmed"] == "false" for row in documentary)
    assert all(row["documentary_recovery_status"] == "direct_paper_verification_not_recovered" for row in documentary)


def test_population_lineage_remains_partial_and_unadmitted() -> None:
    lineage = read_rows(TABLES / "phase_f6h2_official_population_lineage.csv")
    comparison = read_rows(TABLES / "phase_f6h2_population_comparison.csv")
    fingerprints = read_rows(TABLES / "phase_f6h2_population_fingerprints.csv")

    assert all(row["official_population_status"] == "official_population_lineage_partial" for row in lineage)
    assert all(row["execution_evidence"] == "official_zip_entry_exists_without_full_population_chain" for row in lineage)
    assert all(row["identifier_mapping"] == "not_bijective_verified" for row in lineage)
    assert all(row["equality_status"] == "unverifiable" for row in comparison)
    assert all(row["strict_population_equivalence_claimed"] == "false" for row in comparison)
    assert all(row["admitted_for_exact_recomputation"] == "false" for row in comparison)
    assert Counter(row["population_role"] for row in fingerprints) == {
        "local_reconstructed_population": 38,
        "official_repository_recommendation_outputs": 38,
    }


def test_request_response_lineage_remains_partial_and_unadmitted() -> None:
    lineage = read_rows(TABLES / "phase_f6h2_request_response_lineage.csv")
    reconciliation = read_rows(TABLES / "phase_f6h2_request_response_reconciliation.csv")

    assert all(row["request_id_status"] == "missing_from_official_zip" for row in lineage)
    assert all(row["runner_event_id_status"] == "missing_from_official_zip" for row in lineage)
    assert all(row["lineage_graph_status"] == "partial_lineage_no_request_response_graph" for row in lineage)
    assert all(row["request_response_status"] == "partial_lineage" for row in reconciliation)
    assert all(row["admitted_for_exact_evidence"] == "false" for row in reconciliation)
    assert all(row["transport_only"] == "false" for row in reconciliation)
    assert all(row["superseded_protocol"] == "unverified" for row in reconciliation)
    assert not any(row["request_response_status"].startswith("complete") for row in reconciliation)


def test_orphans_duplicates_and_empty_artifacts_are_not_admitted() -> None:
    rows = {row["finding_type"]: row for row in read_rows(TABLES / "phase_f6h2_orphan_and_duplicate_inventory.csv")}
    assert rows["responses_without_requests"]["artifact_count"] == "226"
    assert rows["parsed_or_metric_rows_without_request_graph"]["artifact_count"] == "105"
    assert rows["recommendation_outputs_without_source_population_chain"]["artifact_count"] == "26"
    assert rows["empty_artifacts"]["artifact_count"] == "0"
    assert rows["duplicate_response_or_output_hashes"]["artifact_count"] == "2"
    assert all(row["admitted"] == "false" for row in rows.values())


def test_recovery_decisions_and_readiness_are_unique_and_blocked() -> None:
    decisions = read_rows(TABLES / "phase_f6h2_evidence_recovery_decisions.csv")
    readiness = read_rows(TABLES / "phase_f6h2_recomputation_readiness.csv")
    target_ids = {row["f6h2_candidate_id"] for row in read_rows(TABLES / "phase_f6h2_target_inventory.csv")}

    assert {row["f6h2_candidate_id"] for row in decisions} == target_ids
    assert {row["f6h2_candidate_id"] for row in readiness} == target_ids
    assert len({row["f6h2_candidate_id"] for row in decisions}) == 54
    assert len({row["f6h2_candidate_id"] for row in readiness}) == 54
    assert Counter(row["decision_class"] for row in decisions) == {
        "lineage_partial_only": 49,
        "unresolved": 5,
    }
    assert Counter(row["readiness_class"] for row in readiness) == {
        "not_ready_missing_paper_specification": 5,
        "not_ready_missing_population_lineage": 38,
        "not_ready_missing_request_response_lineage": 11,
    }
    assert all(row["ready_for_strict_offline_recomputation"] == "false" for row in readiness)
    assert all(row["ready_for_repository_derived_recomputation_only"] == "false" for row in readiness)


def test_source_lineage_and_validation_fail_closed() -> None:
    source_rows = read_rows(TABLES / "phase_f6h2_source_code_lineage.csv")
    assert len(source_rows) == 5
    assert all(row["runtime_environment_classification"] == "defensible_repaired_clone_exact_package_versions" for row in source_rows)
    assert all(row["not_exact_original_runtime"] == "true" for row in source_rows)
    assert all(row["lineage_status"] != "historical_producer_chain_complete" for row in source_rows)

    validation = read_rows(TABLES / "phase_f6h2_validation.csv")
    assert all(row["status"] == "pass" for row in validation)
    assert {row["check_id"] for row in validation} >= {
        "complete_population_lineages",
        "repository_only_population_lineages",
        "claimed_equivalent_populations",
        "complete_request_response_lineages",
        "protected_input_hash_invariance",
        "full_suite_excluded",
    }


def test_protected_hash_invariance_and_current_context() -> None:
    payload = json.loads((MANIFESTS / "phase_f6h2_protected_input_hashes.json").read_text(encoding="utf-8"))
    assert payload["changed"] == []
    assert payload["missing"] == []
    assert payload["before"] == payload["after"]

    context = json.loads(CURRENT_CONTEXT.read_text(encoding="utf-8"))
    phase_context = context["post_phase_f6h2_direct_verification_and_lineage_recovery"]
    assert phase_context["phase"] == "phase_f6h2"
    assert phase_context["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert phase_context["not_exact_original_runtime"] is True
    assert phase_context["counts"]["target_candidates"] == 54


def test_deterministic_two_run_output_hashes() -> None:
    run_builder()
    first = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in OUTPUTS}
    run_builder()
    second = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in OUTPUTS}
    assert first == second

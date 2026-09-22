"""Focused checks for F6I2K manifest-generation consolidation."""

from __future__ import annotations

import csv
import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2k_manifest_generation_consolidation_execution import (
    APPROVED_INTERPRETER,
    CANONICAL_APIS,
    CONFIG_PATH,
    DECISION_COMPLETE,
    F6I2J_DECISION_TOKEN,
    MIGRATED_WRAPPERS,
    OUTPUT_TABLES,
    PHASE_ID,
    PROJECT_ROOT,
    build,
    semantic_digest,
)
from src.audit import common_manifests


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]


@lru_cache(maxsize=1)
def built_result():
    return build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)


def test_runtime_and_authoritative_contracts() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    result = built_result()
    manifest = result.manifests["consolidation_manifest"]
    assert PHASE_ID == "phase_f6i2k"
    assert manifest["previous_phase_decision_token"] == F6I2J_DECISION_TOKEN
    assert manifest["selected_group"] == "manifest_generation"
    assert manifest["runtime"]["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert manifest["not_exact_original_runtime"] is True
    assert result.decision_token == DECISION_COMPLETE


def test_common_manifest_json_contracts(tmp_path: Path) -> None:
    payload = {"b": 2, "a": {"items": [3, 1], "empty": [], "none": None}}
    sorted_text = common_manifests.canonicalize_manifest_payload(payload)
    assert sorted_text.startswith('{\n  "a"')
    assert sorted_text.endswith("\n")

    compact_text = common_manifests.canonicalize_manifest_payload(
        payload,
        indent=None,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert compact_text == '{"a":{"empty":[],"items":[3,1],"none":null},"b":2}\n'

    target = tmp_path / "nested" / "manifest.json"
    common_manifests.write_json_manifest(target, payload, atomic=True)
    assert common_manifests.read_json_manifest(target) == payload
    with pytest.raises(FileExistsError):
        common_manifests.write_json_manifest(target, payload, overwrite=False)


def test_common_manifest_csv_contracts(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("artifact_id,dataset,extra\na1, synthetic , value \n", encoding="utf-8")
    rows = common_manifests.read_csv_manifest_rows(
        path,
        required_columns=["artifact_id", "dataset"],
        strip=True,
        project_columns=["artifact_id", "dataset"],
    )
    assert rows == [{"artifact_id": "a1", "dataset": "synthetic"}]
    unstripped = common_manifests.read_csv_manifest_rows(path, strip=False)
    assert unstripped[0]["dataset"] == " synthetic "
    assert common_manifests.read_csv_manifest_rows(tmp_path / "missing.csv", missing="empty") == []
    with pytest.raises(FileNotFoundError):
        common_manifests.read_csv_manifest_rows(tmp_path / "missing.csv")
    with pytest.raises(ValueError):
        common_manifests.read_csv_manifest_rows(path, required_columns=["missing_column"])

    out = tmp_path / "written.csv"
    common_manifests.write_csv_manifest_rows(out, [{"artifact_id": "a2"}], ["artifact_id", "dataset"])
    assert list(csv.DictReader(out.open("r", encoding="utf-8", newline=""))) == [
        {"artifact_id": "a2", "dataset": ""}
    ]
    assert common_manifests.compare_csv_manifest_rows(out, out)
    with pytest.raises(ValueError):
        common_manifests.read_csv_manifest_rows(out, missing="unsupported")


def test_wrappers_retained_delegate_and_source_imports_are_not_duplicated() -> None:
    result = built_result()
    wrappers = result.tables["wrapper_compatibility"]
    migrated = [row for row in wrappers if row["delegates_to_canonical"] == "true"]
    assert len(migrated) == len(MIGRATED_WRAPPERS)
    assert all(row["wrapper_retained"] == "true" for row in wrappers)
    assert all(row["duplicate_manifest_body_removed"] == "true" for row in migrated)
    assert all(row["hashing_logic_migrated"] == "false" for row in wrappers)

    expected_sources = {
        "src/audit/check_plan_c_source_urls.py": "common_manifests.read_csv_manifest_rows",
        "src/audit/inspect_plan_c_downloads.py": "common_manifests.read_csv_manifest_rows",
        "src/audit/validate_plan_c_artifacts.py": "common_manifests.read_csv_manifest_rows",
        "src/analysis/plan_f6e4b_adapted_full_run.py": "common_manifests.read_csv_manifest_rows",
        "tests/test_check_plan_c_source_urls.py": "common_manifests.write_csv_manifest_rows",
        "tests/test_validate_plan_c_artifacts.py": "common_manifests.write_csv_manifest_rows",
    }
    for relative_path, needle in expected_sources.items():
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert text.count("from src.audit import common_manifests") == 1
        assert needle in text


def test_function_schema_output_and_builder_equivalence_pass() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["function_equivalence"])
    assert all(row["output_mismatch"] == "false" for row in result.tables["output_hash_comparison"])
    assert all(row["status"] == "pass" for row in result.tables["schema_equivalence"])
    assert all(row["status"] == "pass" for row in result.tables["builder_equivalence"])
    assert {row["manifest_family"] for row in result.tables["manifest_family_inventory"]} >= {
        "deterministic_scientific_manifest",
        "protected_input_manifest",
        "current_state_snapshot",
        "archive_rollback_manifest",
        "preflight_manifest",
        "runtime_log_manifest",
        "non_deterministic_execution_metadata",
    }


def test_hashing_boundary_protected_inputs_and_git_metadata_pass() -> None:
    result = built_result()
    common_text = (PROJECT_ROOT / "src" / "audit" / "common_manifests.py").read_text(encoding="utf-8")
    assert "hashlib" not in common_text
    assert "sha256" not in common_text.lower()
    assert all(row["status"] == "pass" for row in result.tables["hashing_boundary_validation"])
    assert result.protected_hashes["status"] == "pass"
    assert result.protected_hashes["changed"] == []
    assert result.protected_hashes["missing"] == []
    assert result.manifests["git_metadata_guard"]["status"] == "pass"


def test_dependency_future_readiness_and_wrapper_removal_are_conservative() -> None:
    result = built_result()
    graph = result.tables["dependency_graph_comparison"]
    assert graph[-1]["status"] == "pass"
    assert int(graph[-1]["after_common_manifest_importers"]) >= 6
    assert graph[-1]["circular_imports"] == ""
    assert all(row["status"] == "pass" for row in result.tables["future_dependency_validation"])
    readiness = {row["wrapper_removal_status"] for row in result.tables["wrapper_removal_readiness"]}
    assert "retain_active_external_consumers" in readiness
    assert "retain_phase_provenance" in readiness
    assert "retain_historical_schema" in readiness


def test_validation_outputs_and_full_suite_exclusion_are_recorded() -> None:
    result = built_result()
    assert set(result.tables) == set(OUTPUT_TABLES)
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    current_suite = {row["suite"]: row for row in result.tables["current_suite_validation"]}
    assert current_suite["full_suite"]["expected"] == "excluded"
    assert current_suite["f6i2k_focused_tests"]["expected"] == "pass"
    assert result.manifests["consolidation_manifest"]["next_helper_group"] == "validation_table_creation"
    assert result.manifests["consolidation_manifest"]["counts"]["files_deleted"] == 0


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

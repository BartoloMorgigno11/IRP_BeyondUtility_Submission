from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f2_runtime_verification import (
    ARTIFACT_COLUMNS,
    ENVIRONMENT_COLUMNS,
    HASH_VALIDATION_COLUMNS,
    OUTPUT_TABLES,
    REGISTRY_COLUMNS,
    TEST_INVENTORY_COLUMNS,
    F2OutputExistsError,
    artifact_classification,
    canonical_result_registry,
    ensure_no_existing_outputs,
    result_conflicts,
    safe_test_files,
    validate_hashes,
    validate_schema,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_schema_constants_are_exact() -> None:
    assert "check_name" in ENVIRONMENT_COLUMNS
    assert "test_node_id" in TEST_INVENTORY_COLUMNS
    assert "evaluation_policy_id" in REGISTRY_COLUMNS
    assert "denominator_policy" in REGISTRY_COLUMNS
    assert "status" in HASH_VALIDATION_COLUMNS
    assert {"source_file", "artifact_category", "status"}.issubset(ARTIFACT_COLUMNS)
    assert {"path", "artifact_category", "git_status"}.issubset(ARTIFACT_COLUMNS)


def test_output_guard_blocks_existing_f2_outputs(tmp_path: Path) -> None:
    ensure_no_existing_outputs(tmp_path)
    first = tmp_path / next(iter(OUTPUT_TABLES.values()))
    first.parent.mkdir(parents=True)
    first.write_text("already exists\n", encoding="utf-8")
    with pytest.raises(F2OutputExistsError):
        ensure_no_existing_outputs(tmp_path)


def test_safe_test_files_exclude_level4() -> None:
    files = safe_test_files()
    assert set(files) == {"LEVEL0", "LEVEL1", "LEVEL2"}
    all_files = " ".join(path for group in files.values() for path in group)
    assert "check_plan_c_source_urls" not in all_files
    assert "cuda_first_step" not in all_files
    assert "plan_e_bprmf_training.py" not in all_files


def test_hash_validation_detects_unchanged_and_changed_rows() -> None:
    before = pd.DataFrame(
        [
            {"relative_path": "a.txt", "exists": "true", "sha256": "1", "source_hash_table": "before"},
            {"relative_path": "b.txt", "exists": "true", "sha256": "2", "source_hash_table": "before"},
        ]
    )
    after = pd.DataFrame(
        [
            {"relative_path": "a.txt", "exists": "true", "sha256": "1", "source_hash_table": "after"},
            {"relative_path": "b.txt", "exists": "true", "sha256": "3", "source_hash_table": "after"},
        ]
    )
    validation = validate_hashes(before, after)
    assert validate_schema(validation, HASH_VALIDATION_COLUMNS)
    assert validation.loc[validation["relative_path"].eq("a.txt"), "status"].iloc[0] == "unchanged"
    assert validation.loc[validation["relative_path"].eq("b.txt"), "status"].iloc[0] == "changed"


def test_registry_distinguishes_mostpop_ml1m_plan_b_and_plan_e() -> None:
    registry = canonical_result_registry(PROJECT_ROOT)
    conflicts = result_conflicts(registry)
    rows = conflicts[
        conflicts["model"].eq("MostPop")
        & conflicts["dataset"].eq("ML-1M")
        & conflicts["metric"].eq("hr_at_5")
    ]
    assert not rows.empty
    row = rows.iloc[0]
    assert "0.320364238410596" in row["values"]
    assert "0.671" in row["values"]
    assert row["is_true_conflict"] == "false"
    assert "6040 users" in row["possible_explanation"]


def test_registry_schema_and_dedup_key_columns() -> None:
    registry = canonical_result_registry(PROJECT_ROOT)
    assert validate_schema(registry, REGISTRY_COLUMNS)
    assert registry["evidence_id"].is_unique
    required = [
        "phase",
        "dataset",
        "task",
        "model",
        "cohort_name",
        "candidate_set_name",
        "metric",
        "cutoff",
        "source_file",
        "source_row",
    ]
    assert not registry[required].isna().any().any()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_artifact_classification_has_required_columns() -> None:
    frame = artifact_classification(PROJECT_ROOT)
    assert validate_schema(frame, ARTIFACT_COLUMNS)
    assert {"source implementation", "tests", "config"}.intersection(set(frame["artifact_category"]))

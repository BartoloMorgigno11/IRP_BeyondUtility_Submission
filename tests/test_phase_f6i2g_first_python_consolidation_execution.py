"""Focused checks for F6I2G deterministic CSV I/O consolidation."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_f6i2g_first_python_consolidation_execution import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_COMPLETE,
    F6I2F_DECISION_TOKEN,
    PHASE_ID,
    PROJECT_ROOT,
    build,
    semantic_digest,
)
from src.audit.common_io import (
    read_csv_dataframe,
    read_csv_rows,
    write_csv_rows,
    write_dataframe_csv,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]


@lru_cache(maxsize=1)
def built_result():
    return build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)


def test_runtime_and_f6i2f_authority() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    result = built_result()
    assert PHASE_ID == "phase_f6i2g"
    assert result.manifests["consolidation_manifest"]["previous_phase_decision_token"] == F6I2F_DECISION_TOKEN
    assert result.decision_token == DECISION_COMPLETE


def test_common_io_dataframe_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="f6i2g_test_", dir=PROJECT_ROOT) as temp_dir:
        root = Path(temp_dir)
        path = root / "nested" / "frame.csv"
        frame = pd.DataFrame(
            [
                {"a": "alpha", "b": 1, "c": 1.25},
                {"a": "comma,value", "b": 2, "c": None},
            ],
            columns=["a", "b", "c"],
        )
        write_dataframe_csv(path, frame, columns=["b", "a"], lineterminator="\n")
        assert path.read_text(encoding="utf-8").splitlines()[0] == "b,a"
        read_back = read_csv_dataframe(path, dtype=str, keep_default_na=False)
        assert list(read_back.columns) == ["b", "a"]
        assert read_back.iloc[1]["a"] == "comma,value"

        with pytest.raises(FileExistsError, match="refusing"):
            write_dataframe_csv(
                path,
                frame,
                overwrite=False,
                existing_error_message=f"refusing to overwrite existing output: {path}",
            )


def test_common_io_row_contract_and_atomic_guard() -> None:
    with tempfile.TemporaryDirectory(prefix="f6i2g_test_", dir=PROJECT_ROOT) as temp_dir:
        root = Path(temp_dir)
        path = root / "rows" / "plan_e_lightgcn_fixture.csv"
        rows = [
            {"a": "alpha", "b": 1, "c": ""},
            {"a": 'quote"value', "b": 2, "c": "comma,value"},
        ]
        write_csv_rows(
            path,
            ["a", "b", "c"],
            rows,
            required_name_prefix="plan_e_lightgcn_",
            lineterminator="\n",
            value_transform=str,
            atomic=True,
        )
        assert not path.with_name(path.name + ".tmp").exists()
        assert read_csv_rows(path) == [
            {"a": "alpha", "b": "1", "c": ""},
            {"a": 'quote"value', "b": "2", "c": "comma,value"},
        ]

        bad_path = root / "bad.csv"
        with pytest.raises(RuntimeError, match="non-LightGCN"):
            write_csv_rows(
                bad_path,
                ["a"],
                [{"a": "x"}],
                required_name_prefix="plan_e_lightgcn_",
                prefix_error_message=f"refusing to write non-LightGCN output: {bad_path}",
            )


def test_wrappers_retained_and_delegate_without_duplicate_bodies() -> None:
    result = built_result()
    wrappers = result.tables["wrapper_compatibility"]
    assert len(wrappers) == 33
    assert all(row["signature_preserved"] == "true" for row in wrappers)
    assert all(row["wrapper_retained"] == "true" for row in wrappers)
    assert all(row["delegates_to_canonical"] == "true" for row in wrappers)
    assert all(row["duplicated_csv_body_removed"] == "true" for row in wrappers)
    assert all(row["status"] == "pass" for row in wrappers)


def test_function_and_output_equivalence() -> None:
    result = built_result()
    assert len(result.tables["function_equivalence"]) == 33
    assert all(row["status"] == "pass" for row in result.tables["function_equivalence"])
    assert all(row["output_mismatch"] == "false" for row in result.tables["output_hash_comparison"])
    guarded = [
        row
        for row in result.tables["function_equivalence"]
        if row["source_path"] == "src/experiments/build_plan_e_lightgcn_preflight.py"
        and row["function_name"] == "write_csv"
    ][0]
    assert guarded["pre_status"] == "fail"
    assert guarded["post_status"] == "fail"
    assert guarded["same_exception"] == "true"


def test_dependency_future_and_protected_validation() -> None:
    result = built_result()
    graph_after = result.tables["dependency_graph_comparison"][-1]
    assert graph_after["status"] == "pass"
    assert graph_after["circular_imports"] == ""
    assert int(graph_after["common_io_importer_count"]) >= 22
    assert all(row["status"] == "pass" for row in result.tables["future_dependency_validation"])
    assert result.protected_hashes["status"] == "pass"
    assert all(row["status"] == "pass" for row in result.tables["validation"])


def test_no_file_reduction_and_wrapper_removal_deferred() -> None:
    result = built_result()
    manifest = result.manifests["consolidation_manifest"]
    assert manifest["counts"]["files_deleted"] == 0
    assert manifest["counts"]["file_reduction_claimed"] == 0
    assert {row["wrapper_removal_status"] for row in result.tables["wrapper_removal_readiness"]} == {
        "not_ready_retain_wrapper"
    }


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

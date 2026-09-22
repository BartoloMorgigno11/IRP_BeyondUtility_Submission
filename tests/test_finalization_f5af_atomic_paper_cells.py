from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f5af_atomic_paper_cells import (
    ANOMALY_COLUMNS,
    ATOMIC_COLUMNS,
    COVERAGE_COLUMNS,
    MAPPING_COLUMNS,
    MODEL_COLUMNS,
    OUTPUT_TABLES,
    REPORT_PATH,
    SUPERSEDED_COLUMNS,
    VALIDATION_COLUMNS,
    build_frames,
    build_report,
    final_decision,
    frame_sha256,
    load_sources,
    no_absolute_paths,
    output_hashes,
    protected_hash_status,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def test_f5af_output_schemas() -> None:
    built = frames()
    assert list(built["superseded_target_audit"].columns) == SUPERSEDED_COLUMNS
    assert list(built["paper_model_inventory"].columns) == MODEL_COLUMNS
    assert list(built["atomic_paper_result_cells"].columns) == ATOMIC_COLUMNS
    assert list(built["local_atomic_mapping"].columns) == MAPPING_COLUMNS
    assert list(built["corrected_coverage"].columns) == COVERAGE_COLUMNS
    assert list(built["anomalies"].columns) == ANOMALY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


def test_superseded_classification_is_5_plus_6_and_no_old_numeric_cells() -> None:
    superseded = frames()["superseded_target_audit"]
    assert len(superseded) == 11
    assert superseded["superseded_status"].eq("superseded_non_atomic_numeric_inventory").all()
    assert superseded["corrected_target_type"].eq("paper_model_inventory_component").sum() == 5
    assert superseded["corrected_target_type"].eq("non_atomic_metric_family").sum() == 6
    assert superseded["is_numeric"].eq("true").sum() == 0
    assert superseded["is_atomic"].eq("true").sum() == 0


def test_model_identities_are_excluded_from_numeric_denominator() -> None:
    built = frames()
    models = built["paper_model_inventory"]
    coverage = {row["coverage_metric"]: row for _, row in built["corrected_coverage"].iterrows()}
    assert len(models) == 5
    assert models["is_numeric_result"].eq("false").all()
    assert coverage["atomic paper value extraction coverage"]["denominator"] == "187"
    assert coverage["atomic paper value extraction coverage"]["previous_denominator"] == "11"
    assert coverage["paper model inventory coverage"]["denominator"] == "5"


def test_atomic_tables_3_4_5_counts_and_unique_ids() -> None:
    atomic = frames()["atomic_paper_result_cells"]
    assert len(atomic) == 187
    assert atomic["atomic_cell_id"].is_unique
    assert len(atomic.loc[atomic["paper_table_number"].eq("3")]) == 128
    assert len(atomic.loc[atomic["paper_table_number"].eq("4")]) == 24
    assert len(atomic.loc[atomic["paper_table_number"].eq("5")]) == 35
    required = ["paper_table_number", "row_label", "column_label", "dataset", "task", "model", "metric", "source_type"]
    assert atomic[required].astype(str).apply(lambda col: col.str.len().gt(0).all()).all()


def test_table3_utility_cells_and_significance_marker() -> None:
    atomic = frames()["atomic_paper_result_cells"]
    sasrec = atomic.loc[
        atomic["paper_table_number"].eq("3")
        & atomic["dataset"].eq("Amazon Beauty")
        & atomic["model"].eq("SASRec")
        & atomic["metric"].eq("hr_at_5")
    ].iloc[0]
    lightgcn = atomic.loc[
        atomic["paper_table_number"].eq("3")
        & atomic["dataset"].eq("Amazon Beauty")
        & atomic["model"].eq("LightGCN")
        & atomic["metric"].eq("hr_at_5")
    ].iloc[0]
    assert sasrec["paper_value_text"] == "0,647"
    assert sasrec["paper_value_numeric"] == "0.647"
    assert lightgcn["significance_marker"] == "**"
    assert lightgcn["paper_value_numeric"] == "0.673"


def test_table4_hallucination_cells() -> None:
    atomic = frames()["atomic_paper_result_cells"]
    row = atomic.loc[
        atomic["paper_table_number"].eq("4")
        & atomic["dataset"].eq("ML-1M")
        & atomic["model"].eq("Claude3")
    ].iloc[0]
    assert row["metric"] == "hallucination_rate"
    assert row["task"] == "llm_output_validation"
    assert row["paper_value_numeric"] == "0.0658"


def test_table5_reranking_cells_marker_and_missing_marker() -> None:
    atomic = frames()["atomic_paper_result_cells"]
    gpt = atomic.loc[
        atomic["paper_table_number"].eq("5")
        & atomic["model"].eq("GPT3.5")
        & atomic["metric"].eq("hr_at_5")
    ].iloc[0]
    missing = atomic.loc[
        atomic["paper_table_number"].eq("5")
        & atomic["model"].eq("BPRMF")
        & atomic["metric"].eq("hallucination_rate")
    ].iloc[0]
    assert gpt["paper_value_text"] == "0,0320**"
    assert gpt["paper_value_numeric"] == "0.0320"
    assert gpt["significance_marker"] == "**"
    assert missing["paper_value_text"] == "-"
    assert missing["paper_value_numeric"] == ""


def test_mapping_policy_blocks_substitute_llms_and_keeps_protocol_non_exact() -> None:
    built = frames()
    atomic = built["atomic_paper_result_cells"]
    mapping = built["local_atomic_mapping"]
    by_id = mapping.set_index("atomic_cell_id")
    qwen_id = atomic.loc[atomic["model"].eq("Qwen2-7b")].iloc[0]["atomic_cell_id"]
    assert by_id.loc[qwen_id, "mapping_status"] == "blocked_model_prompt_protocol_mismatch"
    assert mapping["protocol_equivalence_class"].eq("exact").sum() == 0
    assert mapping["exact_numeric_replication"].eq("true").sum() == 0
    assert mapping["local_result_id"].astype(str).str.len().gt(0).sum() > 0


def test_denominator_policy_and_no_arbitrary_f5ar_value_promotion() -> None:
    built = frames()
    coverage = {row["coverage_metric"]: row for _, row in built["corrected_coverage"].iterrows()}
    mapping = built["local_atomic_mapping"]
    atomic = built["atomic_paper_result_cells"]
    flagged_ids = set(atomic.loc[atomic["paper_value_numeric"].isin(["0.3954", "0.0259", "0.647"]), "atomic_cell_id"])
    assert coverage["exact numeric replication coverage"]["numerator"] == "0"
    assert coverage["exact numeric replication coverage"]["denominator"] == "187"
    assert not mapping.loc[mapping["atomic_cell_id"].isin(flagged_ids), "exact_numeric_replication"].eq("true").any()


def test_anomalies_and_validation_complete() -> None:
    built = frames()
    anomalies = built["anomalies"]
    validation = built["validation"]
    assert len(anomalies) >= 5
    assert {"F5AF-ANOM-001", "F5AF-ANOM-002", "F5AF-ANOM-003", "F5AF-ANOM-004"}.issubset(set(anomalies["anomaly_id"]))
    assert validation["status"].eq("pass").all()
    assert final_decision(validation) == "finalization_f5af_complete"


def test_determinism_and_tmp_output_idempotence(tmp_path: Path) -> None:
    built = frames()
    first = {key: frame_sha256(value) for key, value in built.items()}
    second = {key: frame_sha256(value) for key, value in build_frames(PROJECT_ROOT).items()}
    assert first == second
    report = build_report(PROJECT_ROOT, built)
    write_outputs(tmp_path, built, report)
    first_hashes = output_hashes(tmp_path)
    write_outputs(tmp_path, built, report)
    second_hashes = output_hashes(tmp_path)
    assert first_hashes == second_hashes
    assert set(first_hashes) == {REPORT_PATH.as_posix(), *(path.as_posix() for path in OUTPUT_TABLES.values())}


def test_no_absolute_paths_no_nan_and_protected_inputs_unchanged() -> None:
    built = frames()
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())
    protected_status = protected_hash_status(PROJECT_ROOT, load_sources(PROJECT_ROOT).protected_inputs)
    assert protected_status == "checked=1144;changed=0"

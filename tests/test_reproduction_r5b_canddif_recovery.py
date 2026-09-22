from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.reproduction_r5b_canddif_recovery import (
    DECISION_COMPLETE,
    EXPECTED_R5A_MANIFEST_SHA256,
    FAIL_STRICT_DENOMINATOR_POLICY,
    P0_EXPECTED,
    build_r5b,
    canddif_value,
    load_config,
    load_r5a_manifest,
    parse_output_record,
    protected_hash_check,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "reproduction_r5b_canddif_recovery.yaml"


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def manifest_frames(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    return load_r5a_manifest(PROJECT_ROOT, config)


@pytest.fixture(scope="session")
def r5b_outputs() -> dict[str, pd.DataFrame]:
    result = build_r5b(PROJECT_ROOT, config_path=CONFIG_PATH, write=True)
    assert result.decision == DECISION_COMPLETE
    return result.frames


def test_manifest_hash_and_scope_are_frozen(manifest_frames: tuple[pd.DataFrame, pd.DataFrame, str]) -> None:
    manifest, canddif, manifest_sha = manifest_frames
    assert manifest_sha == EXPECTED_R5A_MANIFEST_SHA256
    assert len(manifest) == 4800
    assert len(canddif) == 1200
    assert set(canddif["recovery_family"]) == {"CANDDIF_RECOVERY"}
    assert set(canddif["task"]) == {"ranking"}
    assert set(canddif["dataset"]) == {"Beauty", "Sports", "LastFM"}
    assert not canddif["dataset"].eq("ML-1M").any()
    assert canddif["request_id"].is_unique
    assert canddif["output_destination"].is_unique


def test_first_random_pairs_preserve_only_the_frozen_candidate_set(
    r5b_outputs: dict[str, pd.DataFrame],
) -> None:
    pairs = r5b_outputs["pair_validation"]
    assert len(pairs) == 600
    assert pairs["pair_valid"].eq("true").all()
    assert pairs["same_candidate_items"].eq("true").all()
    assert pairs["different_order_as_expected"].eq("true").all()
    assert pairs["same_history"].eq("true").all()
    assert pairs["same_prompt_family"].eq("true").all()
    assert pairs["random_order_matches_seed"].eq("true").all()


def test_request_execution_accounting_is_complete(r5b_outputs: dict[str, pd.DataFrame]) -> None:
    status = r5b_outputs["request_execution_status"]
    assert len(status) == 1200
    assert status["execution_status"].eq("SUCCEEDED").sum() == 1200
    assert status["execution_status"].eq("BACKEND_FAILED").sum() == 0
    assert status["execution_status"].eq("NOT_EXECUTED").sum() == 0
    assert status["result_hash"].astype(str).str.len().eq(64).all()
    assert status["result_file_sha256"].astype(str).str.len().eq(64).all()


def test_output_validity_uses_fail_strict_denominator(r5b_outputs: dict[str, pd.DataFrame]) -> None:
    validity = r5b_outputs["output_validity"]
    assert len(validity) == 1200
    assert validity["metric_denominator_policy"].eq(FAIL_STRICT_DENOMINATOR_POLICY).all()
    assert validity["hr_at_5"].astype(float).between(0.0, 1.0).all()
    assert validity["ndcg_at_5"].astype(float).between(0.0, 1.0).all()
    invalid = validity.loc[validity["parse_valid"].eq("false")]
    assert invalid["hr_at_5"].astype(float).eq(0.0).all()
    assert invalid["ndcg_at_5"].astype(float).eq(0.0).all()


def test_metrics_and_canddif_formula_are_offline(r5b_outputs: dict[str, pd.DataFrame]) -> None:
    metrics = r5b_outputs["metrics_by_condition"]
    canddif = r5b_outputs["canddif_results"]
    assert len(metrics) == 12
    assert len(canddif) == 6
    assert set(metrics["experimental_condition"]) == {"CANDDIF_FIRST", "CANDDIF_RANDOM"}
    assert metrics["denominator_policy"].eq(FAIL_STRICT_DENOMINATOR_POLICY).all()
    for row in canddif.to_dict("records"):
        expected_hr, _ = canddif_value(float(row["HR_first"]), float(row["HR_random"]))
        expected_ndcg, _ = canddif_value(float(row["NDCG_first"]), float(row["NDCG_random"]))
        assert float(row["CandDif_HR"]) == pytest.approx(expected_hr)
        assert float(row["CandDif_NDCG"]) == pytest.approx(expected_ndcg)


def test_canddif_formula_unit_cases() -> None:
    value, safety = canddif_value(0.25, 0.50)
    assert value == pytest.approx(-0.4054651081)
    assert safety == "NONE"
    value, safety = canddif_value(1.0, 0.50)
    assert value > 20
    assert safety == "DENOMINATOR_CLAMPED_EPSILON_1E-12"


def test_strict_parser_preserves_invalid_outputs_as_zero(manifest_frames: tuple[pd.DataFrame, pd.DataFrame, str]) -> None:
    _, canddif, _ = manifest_frames
    row = canddif.iloc[0].to_dict()
    parsed = parse_output_record(row, '{"ranked_candidate_indices":[1,1,2,3,4]}')
    assert parsed["parse_valid"] is False
    assert parsed["parse_status"] == "duplicate_indices"
    assert parsed["hr_at_5"] == 0.0
    assert parsed["ndcg_at_5"] == 0.0
    parsed = parse_output_record(row, '{"ranked_candidate_indices":[1,2,3,4,5]}')
    assert parsed["parse_valid"] is True
    assert parsed["hr_at_5"] == 1.0


def test_p0_gap_resolution_set_is_exact(r5b_outputs: dict[str, pd.DataFrame]) -> None:
    gaps = r5b_outputs["gap_resolution"]
    observed = set(gaps[["dataset", "metric"]].apply(tuple, axis=1))
    assert observed == P0_EXPECTED
    assert gaps["resolution_status"].eq("RECOVERED_PAPER_ALIGNED_ADAPTED").all()
    assert gaps["not_executed"].astype(int).sum() == 0
    assert gaps["backend_failed"].astype(int).sum() == 0


def test_validation_passes_and_protected_artifacts_unchanged(r5b_outputs: dict[str, pd.DataFrame]) -> None:
    validation = r5b_outputs["validation"]
    assert len(validation) == 20
    assert validation["status"].eq("pass").all()
    protected = protected_hash_check(PROJECT_ROOT)
    assert len(protected) == 12
    assert protected["protection_status"].eq("PROTECTED_UNCHANGED").all()
    assert protected["tracked"].eq("false").all()

from pathlib import Path

import pandas as pd

from src.analysis import reproduction_r5c_selective_rerun_preflight as preflight


def test_hard_preflight_rejects_same_hash_selective_rerun() -> None:
    result = preflight.build_preflight(Path("."), write=False)

    assert result.summary["final_decision"] == "R5C_SELECTIVE_RERUN_NOT_ALLOWED_UNDER_FROZEN_PROTOCOL"
    assert result.summary["selective_rerun_scientifically_allowed"] is False
    assert result.summary["preflight"] == "FAIL"
    assert result.summary["llm_calls"] == 0
    assert result.summary["selective_rows_executed"] == 0
    assert result.summary["r5c_final_valid"] == 2385
    assert result.summary["r5c_final_invalid"] == 15


def test_no_retry_evidence_is_explicit() -> None:
    result = preflight.build_preflight(Path("."), write=False)
    checks = result.checks.set_index("check_id")

    assert checks.loc["PREFLIGHT-003", "status"] == "pass"
    assert checks.loc["PREFLIGHT-003", "observed"] == "0"
    assert checks.loc["PREFLIGHT-004", "status"] == "pass"
    assert checks.loc["PREFLIGHT-005", "status"] == "pass"
    assert checks.loc["PREFLIGHT-006", "observed"] == "0"


def test_bias_and_protocol_mixing_are_blocking_failures() -> None:
    result = preflight.build_preflight(Path("."), write=False)
    checks = result.checks.set_index("check_id")

    assert checks.loc["PREFLIGHT-014", "status"] == "fail"
    assert checks.loc["PREFLIGHT-014", "blocks_selective_rerun"] == "true"
    assert checks.loc["PREFLIGHT-015", "status"] == "fail"
    assert checks.loc["PREFLIGHT-015", "blocks_selective_rerun"] == "true"
    assert "PREFLIGHT-014" in result.summary["blocking_failures"]
    assert "PREFLIGHT-015" in result.summary["blocking_failures"]


def test_all_fifteen_original_invalid_raw_outputs_are_preserved() -> None:
    rerun_rows = pd.read_csv(preflight.RERUN_ROWS, dtype=str, keep_default_na=False)
    preserved, missing = preflight.count_original_raw_preserved(Path("."), rerun_rows)

    assert preserved == 15
    assert missing == []

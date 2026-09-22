from __future__ import annotations

from pathlib import Path

from src.analysis.reproduction_r5c_invalid_output_forensic_audit import (
    CLASS_EXACT,
    DECISION_SELECTIVE_RERUN,
    build_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_fifteen_invalid_r5c_rows_are_exact_item_id_mappable() -> None:
    result = build_audit(PROJECT_ROOT, write=False)
    assert len(result.audit) == 15
    assert result.audit["request_classification"].eq(CLASS_EXACT).all()
    assert result.summary["exact_offline_mappable"] == 15
    assert result.summary["partial_or_ambiguous"] == 0
    assert result.summary["genuine_invalid"] == 0
    assert result.summary["item_id_to_index_mapping"] == "DETERMINISTIC"
    assert len(result.values) == 75
    assert result.values["candidate_item_id_match"].eq("true").all()
    assert result.values["item_id_mapping_status"].eq("exact").all()


def test_protocol_decision_requires_selective_rerun_not_posthoc_repair() -> None:
    result = build_audit(PROJECT_ROOT, write=False)
    assert result.decision == DECISION_SELECTIVE_RERUN
    assert result.summary["post_hoc_normalization_scientifically_allowed"] is False
    assert result.summary["can_repair_from_existing_raw_outputs_scientifically"] is False
    assert result.summary["technical_mapping_possible"] is True
    assert result.summary["selective_rerun_required"] is True
    assert len(result.rerun_rows) == 15
    assert set(result.rerun_rows["model_name"]) == {"qwen2:7b-instruct"}
    assert set(result.rerun_rows["dataset"]) == {"ML-1M"}


def test_hypothetical_mapping_is_not_marked_freeze_ready() -> None:
    result = build_audit(PROJECT_ROOT, write=False)
    hypothetical = result.summary["if_offline_recovered_hypothetical"]
    assert hypothetical["valid"] == 2400
    assert hypothetical["target_gaps_all_valid"] == 4
    assert hypothetical["r5c_freeze_ready_if_normalization_were_allowed"] is False

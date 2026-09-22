from __future__ import annotations

from collections import Counter

import pytest

from src.analysis.plan_f6i2a_substantive_repository_consolidation_design import (
    PHASE_ID,
    build,
    build_tables,
    delete_gate_passes,
    load_phase_data,
    semantic_digest,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
]


def test_phase_identifier_and_f6i1_reconciliation() -> None:
    data = load_phase_data()
    assert PHASE_ID == "phase_f6i2a"
    assert len(data.inventory) == 1948
    assert len(data.edges) == 45826
    assert all(row["status"] == "pass" for row in data.validation)


def test_every_file_receives_one_retention_decision() -> None:
    data = load_phase_data()
    tables = build_tables(data)
    inventory_paths = sorted(row["path"] for row in data.inventory)
    decision_paths = sorted(row["path"] for row in tables["retention_decisions"])
    assert decision_paths == inventory_paths
    assert len(decision_paths) == len(set(decision_paths))


def test_delete_bundle_is_limited_to_gate_passing_zero_byte_logs() -> None:
    data = load_phase_data()
    tables = build_tables(data)
    for row in tables["low_risk_delete_bundle"]:
        assert row["action"] == "delete_in_f6i2b_obsolete_safe"
        assert delete_gate_passes(row["path"], data)


def test_semantic_merge_groups_and_config_groups_are_present() -> None:
    data = load_phase_data()
    tables = build_tables(data)
    group_names = {row["group_name"] for row in tables["python_semantic_groups"]}
    assert "deterministic_csv_io" in group_names
    assert "hashing_helpers" in group_names
    config_groups = {row["group_name"] for row in tables["config_merge_plan"]}
    assert "phase_specific_archive" in config_groups


def test_roadmap_protection_covers_all_future_task_groups() -> None:
    data = load_phase_data()
    tables = build_tables(data)
    counts = Counter(row["future_task_group"] for row in tables["future_dependency_protection"])
    required = {
        "F6J dataset/candidate/reranking closure",
        "F6K traditional models and prompt recovery",
        "F6L model availability and execution",
        "F6M complete ranking/reranking matrix",
        "F6N history/no-history and reduced-data D'",
        "F6O candidate-position bias",
        "F6P profile generation",
        "F6Q hallucination and reliability",
        "F6R metrics",
        "F6S repeated runs and statistical analysis",
        "F7 fifth dataset",
        "F8 scientific extensions",
        "F9 final deliverables",
    }
    assert not [task for task in required if counts[task] == 0]


def test_build_is_deterministic_and_validates_without_writing() -> None:
    first = build(update_current_state=False)
    second = build(update_current_state=False)
    assert semantic_digest(first.tables) == semantic_digest(second.tables)
    assert all(row["status"] == "pass" for row in first.tables["validation"])
    assert first.protected_hashes["all_unchanged"] is True

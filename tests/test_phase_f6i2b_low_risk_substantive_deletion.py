from __future__ import annotations

from collections import Counter

import pytest

from src.analysis.plan_f6i2b_low_risk_substantive_deletion import (
    DELETION_MANIFEST_PATH,
    EXPECTED_CANDIDATES,
    PHASE_ID,
    build,
    load_inputs,
    project_path,
    selected_candidate_paths,
    semantic_digest,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
]


def test_phase_identifier_and_authoritative_bundle_count() -> None:
    inputs = load_inputs()
    paths = selected_candidate_paths(inputs)
    assert PHASE_ID == "phase_f6i2b"
    assert len(paths) == EXPECTED_CANDIDATES
    assert len(set(paths)) == EXPECTED_CANDIDATES


def test_deletion_actions_are_exclusive_to_f6i2a_bundle() -> None:
    inputs = load_inputs()
    result = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False)
    bundle_paths = set(selected_candidate_paths(inputs))
    action_paths = {row["path"] for row in result.tables["deletion_actions"]}
    assert action_paths == bundle_paths
    assert all(path.startswith("results/logs/phase_f6e4b_") for path in action_paths)


def test_preflight_preserves_zero_byte_hash_and_metadata() -> None:
    result = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False)
    rows = result.tables["candidate_preflight"]
    assert len(rows) == EXPECTED_CANDIDATES
    assert {row["recorded_size_bytes"] for row in rows} == {"0"}
    assert {row["recorded_sha256"] for row in rows} == {
        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
    }
    assert {row["preflight_status"] for row in rows} == {"passed"}
    assert all(row["canonical_replacement"] for row in rows)
    assert all(row["rollback_source"] for row in rows)


def test_restart_safe_state_matches_manifest_presence() -> None:
    result = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False)
    action_status = Counter(row["final_status"] for row in result.tables["deletion_actions"])
    if DELETION_MANIFEST_PATH.exists():
        assert action_status == {"deleted": EXPECTED_CANDIDATES}
        assert not [row["path"] for row in result.tables["deletion_actions"] if project_path(row["path"]).exists()]
    else:
        assert action_status == {"pending_execute_deletion": EXPECTED_CANDIDATES}


def test_dependency_future_reference_and_producer_validations_pass() -> None:
    result = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False)
    for table_name in [
        "pre_post_dependency_comparison",
        "future_dependency_validation",
        "canonical_producer_validation",
        "reference_validation",
    ]:
        rows = result.tables[table_name]
        assert rows
        assert not [row for row in rows if row["status"] != "pass"]


@pytest.mark.superseded_invariant
def test_protected_inputs_and_builder_determinism() -> None:
    first = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False)
    second = build(execute_deletion=False, write_outputs_enabled=False, update_current_state=False)
    assert first.protected_hashes["comparison"]["all_unchanged"] is True
    assert semantic_digest(first) == semantic_digest(second)

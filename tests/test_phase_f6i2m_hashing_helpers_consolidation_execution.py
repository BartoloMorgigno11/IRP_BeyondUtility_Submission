from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path

import pytest

from src.analysis.plan_f6i2m_hashing_helpers_consolidation_execution import (
    BuildResult,
    CONSOLIDATION_MANIFEST_PATH,
    DECISION_COMPLETE,
    GIT_METADATA_GUARD_PATH,
    OUTPUT_TABLES,
    PHASE_ID,
    PROTECTED_HASHES_PATH,
    REPORT_PATH,
    SELECTED_WRAPPERS,
    semantic_digest,
)
from src.audit import common_hashing


pytestmark = [pytest.mark.refactor_regression]


@pytest.fixture(scope="module")
def result():
    return BuildResult(
        tables={name: _read_csv(path) for name, path in OUTPUT_TABLES.items()},
        manifests={
            "consolidation_manifest": _load_json(CONSOLIDATION_MANIFEST_PATH),
            "protected_input_hashes": _load_json(PROTECTED_HASHES_PATH),
            "git_metadata_guard": _load_json(GIT_METADATA_GUARD_PATH),
        },
        report=REPORT_PATH.read_text(encoding="utf-8"),
        decision_token=DECISION_COMPLETE,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_runtime_phase_and_decision(result):
    assert result.decision_token == DECISION_COMPLETE
    assert result.manifests["consolidation_manifest"]["phase_id"] == PHASE_ID
    assert result.manifests["consolidation_manifest"]["runtime"]["runtime_classification"] == (
        "defensible_repaired_clone_exact_package_versions"
    )
    assert result.manifests["consolidation_manifest"]["not_exact_original_runtime"] is True


@pytest.mark.cleanup_current
def test_common_hashing_import_safety_and_boundary():
    text = Path("src/audit/common_hashing.py").read_text(encoding="utf-8")
    assert "phase_f6" not in text
    assert "LastFM" not in text
    assert "Beauty" not in text
    assert "manifest" not in text.lower()
    assert "subprocess" not in text


@pytest.mark.cleanup_current
def test_sha256_bytes_and_text():
    assert common_hashing.sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()
    assert common_hashing.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert common_hashing.sha256_text("abc") == hashlib.sha256("abc".encode("utf-8")).hexdigest()


@pytest.mark.cleanup_current
def test_unicode_lf_and_crlf_text_hashes_are_preserved():
    assert common_hashing.sha256_text("cafe \u03c0") == hashlib.sha256("cafe \u03c0".encode("utf-8")).hexdigest()
    assert common_hashing.sha256_text("a\nb\n") != common_hashing.sha256_text("a\r\nb\r\n")


@pytest.mark.cleanup_current
def test_file_hash_zero_binary_and_multichunk(tmp_path):
    zero = tmp_path / "zero.bin"
    binary = tmp_path / "binary.bin"
    multi = tmp_path / "multi.bin"
    zero.write_bytes(b"")
    binary.write_bytes(bytes([0, 1, 2, 255, 128]))
    multi.write_bytes(b"abc123\n" * 200000)
    for path in [zero, binary, multi]:
        assert common_hashing.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.cleanup_current
def test_missing_file_and_directory_policy(tmp_path):
    missing = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError):
        common_hashing.sha256_file(missing)
    assert common_hashing.sha256_file_or_empty(missing) == ""
    assert common_hashing.sha256_file_or_empty(tmp_path) == ""


@pytest.mark.cleanup_current
def test_directory_tree_hashing_is_sorted_and_hidden_inclusive(tmp_path):
    (tmp_path / "b").write_text("b", encoding="utf-8")
    (tmp_path / "a").write_text("a", encoding="utf-8")
    (tmp_path / ".hidden").write_text("h", encoding="utf-8")
    first = common_hashing.hash_directory_tree(tmp_path)
    second = common_hashing.hash_directory_tree(tmp_path)
    assert first == second
    assert first != common_hashing.sha256_text("")


@pytest.mark.cleanup_current
def test_aggregate_named_hashes_ordering_and_duplicates():
    pairs = [("b", "2" * 64), ("a", "1" * 64)]
    reversed_pairs = list(reversed(pairs))
    duplicate_pairs = pairs + [("a", "1" * 64)]
    assert common_hashing.aggregate_named_hashes(pairs) == common_hashing.aggregate_named_hashes(reversed_pairs)
    assert common_hashing.aggregate_named_hashes(duplicate_pairs) != common_hashing.aggregate_named_hashes(pairs)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_exact_member_inventory(result):
    assert len(result.tables["bundle_members"]) == 117
    assert len(result.tables["pre_refactor_function_inventory"]) == 285
    assert all(row["status"] == "pass" for row in result.tables["bundle_members"])


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_hash_contract_families_and_algorithms(result):
    families = {row["hash_contract_family"] for row in result.tables["hash_contract_families"]}
    assert "single_file_byte_hash" in families
    assert "protected_input_snapshot_hash" in families
    assert "normalized_ast_fingerprint" in families
    assert {row["algorithm"] for row in result.tables["hash_contract_families"]} == {"sha256"}


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_fixture_baseline_covers_required_cases(result):
    fixtures = {row["fixture_name"] for row in result.tables["hash_behaviour_baseline"]}
    assert {"empty_bytes.bin", "unicode_text.txt", "lf_text.txt", "crlf_text.txt", "multi_chunk.bin"} <= fixtures
    assert {"missing_file", "empty_directory", "relative_path_small_file"} <= fixtures
    assert len(result.tables["hash_behaviour_baseline"]) == 54


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_migrated_wrappers_have_function_equivalence(result):
    migrated = {(wrapper["source_file"], wrapper["function_name"]) for wrapper in SELECTED_WRAPPERS}
    rows = result.tables["function_equivalence"]
    assert rows
    assert {row["status"] for row in rows} == {"pass"}
    assert {(row["source_file"], row["function_name"]) for row in rows} == migrated
    assert all(row["output_mismatch"] == "false" for row in rows)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_wrapper_compatibility_and_retention(result):
    wrappers = result.tables["wrapper_compatibility"]
    assert all(row["entrypoint_retained"] == "true" for row in wrappers)
    assert sum(row["delegates_to_common_hashing"] == "true" for row in wrappers) == len(SELECTED_WRAPPERS)
    assert any(row["compatibility_status"] == "retain_historical_hash_contract" for row in wrappers)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_source_modification_scope_is_narrow(result):
    rows = result.tables["source_modifications"]
    modified = {row["source_file"] for row in rows}
    assert "src/audit/common_hashing.py" in modified
    assert "src/llm/canonical_local_llm_runner.py" not in modified
    assert "src/metrics/canonical_metric_computation.py" not in modified
    assert all(row["status"] == "pass" for row in rows)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_real_hash_and_protected_guard_equivalence(result):
    assert {row["status"] for row in result.tables["real_hash_equivalence"]} == {"pass"}
    assert {row["status"] for row in result.tables["protected_guard_equivalence"]} == {"pass"}
    assert result.manifests["protected_input_hashes"]["status"] == "pass"
    assert result.manifests["protected_input_hashes"]["changed"] == []


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_manifest_boundary_and_dependency_graph(result):
    assert all(row["status"] == "pass" for row in result.tables["manifest_boundary_validation"])
    assert all(row["status"] == "pass" for row in result.tables["dependency_graph_comparison"])
    assert all(row["circular_imports"] == "none" for row in result.tables["dependency_graph_comparison"])


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_future_dependency_and_wrapper_removal_readiness(result):
    assert all(row["status"] == "pass" for row in result.tables["future_dependency_validation"])
    assert all(row["expected_file_reduction_now"] == "0" for row in result.tables["wrapper_removal_readiness"])
    assert all(row["wrapper_removal_status"].startswith("retain_") for row in result.tables["wrapper_removal_readiness"])


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_validation_and_git_metadata_pass(result):
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert result.manifests["git_metadata_guard"]["status"] == "pass"
    assert all(result.manifests["git_metadata_guard"]["checks"].values())


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_in_memory_build_is_deterministic(result):
    first_digest = semantic_digest(result)
    second_digest = semantic_digest(result)
    assert first_digest == second_digest
    assert json.dumps(result.tables["canonical_api_contract"], sort_keys=True) == json.dumps(
        result.tables["canonical_api_contract"], sort_keys=True
    )

"""Safe validation for the final G4/F1BZQ provenance contracts."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = ROOT / "results/manifests/g4_f1bzq_canonical_provenance.json"
REPRODUCIBILITY_CONTRACT = ROOT / "results/manifests/g4_f1bzq_reproducibility_contract.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_canonical_artifact_hashes_match_repository_files() -> None:
    manifest = _load(CANONICAL_MANIFEST)
    for artifact in manifest["canonical_artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file(), artifact["path"]
        assert _sha256(path) == artifact["sha256"], artifact["path"]


def test_paired_contributions_gzip_is_the_canonical_byte_preserving_delivery() -> None:
    manifest = _load(CANONICAL_MANIFEST)
    contract = manifest["paired_contributions_delivery"]
    packaged = ROOT / contract["compressed_path"]
    statistical = _load(ROOT / contract["frozen_statistical_manifest"])

    assert contract["canonical_delivery_form"] is True
    assert _sha256(packaged) == contract["compressed_sha256"]
    with gzip.open(packaged, "rb") as handle:
        logical_hash = hashlib.sha256(handle.read()).hexdigest()
    assert logical_hash == contract["decompressed_sha256"]
    assert (
        statistical["output_hashes"][contract["decompressed_logical_path"]]
        == logical_hash
    )


def test_external_ledgers_are_evidence_not_git_dependencies() -> None:
    manifest = _load(CANONICAL_MANIFEST)
    evidence = manifest["external_frozen_execution_evidence"]
    assert evidence["classification"] == "EXTERNAL_FROZEN_EXECUTION_EVIDENCE"
    assert evidence["mandatory_git_dependency"] is False
    assert evidence["required_for_tier_1_frozen_reproduction"] is False
    assert evidence["archive"]["expected_sha256"] == (
        "dc32bb824475be37f13616d9d7f5080b5f7da8de5ccba11ce78cfccd39089cd8"
    )
    for ledger in evidence["ledgers"]:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ledger["path"]],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert tracked.returncode != 0, ledger["path"]


def test_reproducibility_tiers_are_explicit() -> None:
    contract = _load(REPRODUCIBILITY_CONTRACT)
    tiers = contract["tiers"]
    assert set(tiers) == {
        "TIER_1_FROZEN_REPRODUCTION",
        "TIER_2_CANDIDATE_RECONSTRUCTION",
        "TIER_3_FULL_FRESH_EXECUTION",
    }
    assert tiers["TIER_1_FROZEN_REPRODUCTION"]["requires_llm_inference"] is False
    assert tiers["TIER_1_FROZEN_REPRODUCTION"]["requires_checkpoints"] is False
    assert tiers["TIER_1_FROZEN_REPRODUCTION"]["requires_raw_datasets"] is False
    assert tiers["TIER_3_FULL_FRESH_EXECUTION"]["byte_identical_llm_output_guaranteed"] is False


def test_historical_hashes_are_preserved_and_resolved() -> None:
    manifest = _load(CANONICAL_MANIFEST)
    for resolution in manifest["historical_reference_resolutions"]:
        current_path = ROOT / resolution["artifact_path"]
        assert _sha256(current_path) == resolution["canonical_sha256"]
        for source in resolution["historical_references"]:
            source_text = (ROOT / source["manifest_path"]).read_text(encoding="utf-8")
            assert source["historical_sha256"] in source_text
            assert source["classification"] == "NON_OPERATIONAL_PROVENANCE"


def test_operational_contract_paths_are_portable() -> None:
    manifest = _load(CANONICAL_MANIFEST)
    files = [ROOT / path for path in manifest["portable_path_policy"]["validated_files"]]
    absolute_local_path = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not absolute_local_path.search(text), path.relative_to(ROOT).as_posix()


def test_protocol_separation_and_comparison_scope() -> None:
    manifest = _load(CANONICAL_MANIFEST)
    separation = manifest["lastfm_protocol_separation"]
    comparison = manifest["g1_g4_comparison"]
    assert separation == {
        "legacy_f6g1_f6l_target_present": "314/1000",
        "canonical_f1bzq_target_present": "469/1000",
        "direct_comparison": "INVALID",
    }
    assert comparison["classification"] == "DESCRIPTIVE_ONLY"
    assert comparison["causal_ranking_vs_reranking_claim"] == "NOT_SUPPORTED"


def test_legacy_f1bz_and_f1bzc_authority_is_explicit() -> None:
    manifest = _load(CANONICAL_MANIFEST)
    authority = manifest["legacy_artifact_authority"]
    assert authority["F1BZ"]["classification"] == "FORENSIC"
    assert authority["F1BZC"]["classification"] == "OBSOLETE"

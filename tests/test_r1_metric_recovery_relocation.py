from __future__ import annotations

import hashlib
from pathlib import Path

from src.analysis import reproduction_r2_official_metric_calculability as r2
from src.analysis import reproduction_r3_complete_local_metric_audit as r3
from src.analysis import reproduction_r5a_controlled_recovery_manifest as r5a
from src.analysis import reproduction_r5b_canddif_recovery as r5b
from src.analysis.r1_metric_recovery_paths import (
    RELOCATION_DIR,
    relocation_map,
    resolve_r1_metric_recovery_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_relocated_artifacts_resolve_and_match_manifest() -> None:
    mapping = relocation_map(ROOT)
    assert len(mapping) == 13
    for historical, record in mapping.items():
        resolved = resolve_r1_metric_recovery_artifact(ROOT, historical)
        assert resolved == ROOT / record.current_path
        assert resolved.parent == ROOT / RELOCATION_DIR
        assert not (ROOT / historical).exists()
        assert resolved.is_file()
        assert resolved.stat().st_size == record.size_bytes
        assert hashlib.sha256(resolved.read_bytes()).hexdigest() == record.sha256


def test_unrelated_paths_are_not_remapped() -> None:
    relative = "results/tables/unrelated.csv"
    assert resolve_r1_metric_recovery_artifact(ROOT, relative) == ROOT / relative


def test_historical_identity_is_preserved() -> None:
    mapping = relocation_map(ROOT)
    assert set(mapping) == {record.historical_path for record in mapping.values()}


def test_missing_relocated_artifact_is_not_silently_accepted(tmp_path: Path, monkeypatch) -> None:
    manifest_dir = tmp_path / "results/manifests"
    manifest_dir.mkdir(parents=True)
    source = ROOT / "results/manifests/r1_root_artifact_relocation.csv"
    (manifest_dir / source.name).write_bytes(source.read_bytes())
    historical = next(iter(relocation_map(tmp_path)))
    assert not resolve_r1_metric_recovery_artifact(tmp_path, historical).exists()
    monkeypatch.setattr(r2, "PROTECTED_ROOT_FILES", (historical,))
    protected = r2.protected_hash_rows(tmp_path)
    assert protected[0]["protection_status"] == "PROTECTION_FAILURE"


def test_recovery_validators_accept_relocated_protected_artifacts() -> None:
    r3_config = r3.load_config(ROOT / r3.DEFAULT_CONFIG_PATH)
    r5a_config = r5a.load_config(ROOT / "configs/reproduction_r5a_controlled_recovery_manifest.yaml")
    protected_frames = [
        r2.protected_hash_rows(ROOT),
        r3.protected_hash_rows(ROOT, r3_config),
        r5a.build_protected_hash_check(ROOT, r5a_config).to_dict("records"),
        r5b.protected_hash_check(ROOT).to_dict("records"),
    ]
    expected = set(relocation_map(ROOT)) - {"final_metric_paper_comparison.csv"}
    for rows in protected_frames:
        assert {row["path"] for row in rows} == expected
        assert all(row["protection_status"] == "PROTECTED_UNCHANGED" for row in rows)

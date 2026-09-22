from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

from src.audit import common_manifests
from src.audit.validate_plan_c_artifacts import REQUIRED_COLUMNS, validate_manifest


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    common_manifests.write_csv_manifest_rows(path, rows, REQUIRED_COLUMNS, fill_missing=True)


def base_row(artifact_id: str, local_path: str) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "dataset": "synthetic",
        "task": "ranking",
        "artifact_name": artifact_id,
        "expected_filename": Path(local_path).name if local_path else "",
        "expected_relative_path": local_path,
        "artifact_category": "test_artifact",
        "source_url": "https://example.invalid/artifact",
        "source_type": "unknown",
        "local_path": local_path,
        "found_locally": "false",
        "validation_status": "not_validated",
        "reproducibility_status": "blocked_unclear_protocol",
        "can_reconstruct_from_raw": "false",
        "last_checked": "2026-06-29",
    }


def read_details(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "plan_c_artifact_validation_details.csv")


def read_summary(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "plan_c_artifact_validation_summary.csv")


def test_existing_and_missing_files_are_reported(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("abc", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "out"
    write_manifest(
        manifest,
        [
            base_row("existing", "existing.txt"),
            base_row("missing", "missing.txt"),
        ],
    )

    validate_manifest(manifest, project_root=tmp_path, output_dir=output_dir)

    details = read_details(output_dir).set_index("artifact_id")
    summary = read_summary(output_dir).iloc[0]
    assert details.loc["existing", "validation_status"] == "valid"
    assert bool(details.loc["existing", "actual_found_locally"]) is True
    assert details.loc["missing", "validation_status"] == "missing"
    assert summary["total_artifacts"] == 2
    assert summary["found_locally"] == 1
    assert summary["missing"] == 1


def test_size_validation_matches_and_mismatches(tmp_path: Path) -> None:
    file_path = tmp_path / "sized.txt"
    file_path.write_text("abcd", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "out"
    matching = base_row("matching_size", "sized.txt")
    matching["expected_size_bytes"] = "4"
    mismatching = base_row("mismatching_size", "sized.txt")
    mismatching["expected_size_bytes"] = "5"
    write_manifest(manifest, [matching, mismatching])

    validate_manifest(manifest, project_root=tmp_path, output_dir=output_dir)

    details = read_details(output_dir).set_index("artifact_id")
    assert details.loc["matching_size", "validation_status"] == "valid"
    assert details.loc["matching_size", "size_check_status"] == "matched"
    assert details.loc["mismatching_size", "validation_status"] == "size_mismatch"
    assert details.loc["mismatching_size", "size_check_status"] == "mismatch"


def test_hash_validation_only_when_enabled(tmp_path: Path) -> None:
    file_path = tmp_path / "hashed.txt"
    file_path.write_text("hash me", encoding="utf-8")
    expected_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    row = base_row("hashed", "hashed.txt")
    row["expected_hash_sha256"] = expected_hash
    write_manifest(manifest, [row])

    output_no_hash = tmp_path / "out_no_hash"
    validate_manifest(manifest, project_root=tmp_path, output_dir=output_no_hash)
    details = read_details(output_no_hash).set_index("artifact_id")
    assert details.loc["hashed", "validation_status"] == "found_hash_not_checked"
    assert details.loc["hashed", "hash_check_status"] == "not_checked"

    output_hash = tmp_path / "out_hash"
    validate_manifest(manifest, project_root=tmp_path, output_dir=output_hash, hash_files=True)
    details = read_details(output_hash).set_index("artifact_id")
    assert details.loc["hashed", "validation_status"] == "valid"
    assert details.loc["hashed", "hash_check_status"] == "matched"
    assert details.loc["hashed", "actual_hash_sha256"] == expected_hash


def test_wrong_hash_fails_when_hash_enabled(tmp_path: Path) -> None:
    file_path = tmp_path / "hashed.txt"
    file_path.write_text("hash me", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    row = base_row("wrong_hash", "hashed.txt")
    row["expected_hash_sha256"] = "0" * 64
    write_manifest(manifest, [row])

    output_dir = tmp_path / "out"
    validate_manifest(manifest, project_root=tmp_path, output_dir=output_dir, hash_files=True)

    details = read_details(output_dir).set_index("artifact_id")
    summary = read_summary(output_dir).iloc[0]
    assert details.loc["wrong_hash", "validation_status"] == "hash_mismatch"
    assert summary["hash_mismatch"] == 1


def test_empty_expected_size_and_hash_do_not_fail(tmp_path: Path) -> None:
    file_path = tmp_path / "plain.txt"
    file_path.write_text("plain", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [base_row("plain", "plain.txt")])

    output_dir = tmp_path / "out"
    validate_manifest(manifest, project_root=tmp_path, output_dir=output_dir, hash_files=True)

    details = read_details(output_dir).set_index("artifact_id")
    assert details.loc["plain", "validation_status"] == "valid"
    assert details.loc["plain", "size_check_status"] == "not_applicable"
    assert details.loc["plain", "hash_check_status"] == "not_applicable"


def test_missing_local_path_is_clear(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "out"
    write_manifest(manifest, [base_row("missing_local_path", "")])

    validate_manifest(manifest, project_root=tmp_path, output_dir=output_dir)

    details = read_details(output_dir).set_index("artifact_id")
    summary = read_summary(output_dir).iloc[0]
    assert details.loc["missing_local_path", "validation_status"] == "missing_local_path"
    assert summary["missing_local_path"] == 1


def test_summary_counts_blocked_unavailable_or_unknown(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("abc", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    row = base_row("blocked_unknown", "existing.txt")
    row["source_type"] = "unknown"
    row["reproducibility_status"] = "blocked_unclear_protocol"
    write_manifest(manifest, [row])

    output_dir = tmp_path / "out"
    validate_manifest(manifest, project_root=tmp_path, output_dir=output_dir)

    summary = read_summary(output_dir).iloc[0]
    assert summary["blocked_unclear_protocol"] == 1
    assert summary["blocked_unavailable_or_unknown"] == 1


def test_validator_does_not_download_from_source_url(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    row = base_row("remote_only", "missing_remote.txt")
    row["source_url"] = "https://example.invalid/should-not-be-downloaded.csv"
    write_manifest(manifest, [row])

    output_dir = tmp_path / "out"
    validate_manifest(manifest, project_root=tmp_path, output_dir=output_dir)

    assert not (tmp_path / "should-not-be-downloaded.csv").exists()
    details = read_details(output_dir).set_index("artifact_id")
    assert details.loc["remote_only", "validation_status"] == "missing"

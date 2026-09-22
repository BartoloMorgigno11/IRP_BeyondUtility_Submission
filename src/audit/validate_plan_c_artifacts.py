"""Validate local Plan C artifact manifest entries.

Run from the repository root:

    python src/audit/validate_plan_c_artifacts.py --manifest data/artifact_manifests/plan_c_artifact_manifest.csv

This validator checks only local filesystem state. It never downloads files,
calls APIs, trains models, modifies raw artifacts, or modifies ``original_repo/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit import common_hashing
from src.audit import common_manifests


REQUIRED_COLUMNS = [
    "artifact_id",
    "dataset",
    "task",
    "artifact_name",
    "expected_filename",
    "expected_relative_path",
    "artifact_category",
    "source_url",
    "source_type",
    "license_or_terms_notes",
    "expected_size_bytes",
    "expected_hash_sha256",
    "local_path",
    "found_locally",
    "validation_status",
    "reproducibility_status",
    "can_reconstruct_from_raw",
    "reconstruction_notes",
    "blocker_reason",
    "last_checked",
    "notes",
]

DETAIL_EXTRA_COLUMNS = [
    "resolved_local_path",
    "actual_found_locally",
    "actual_size_bytes",
    "actual_hash_sha256",
    "size_check_status",
    "hash_check_status",
    "validation_notes",
    "validated_at",
]

SUMMARY_COLUMNS = [
    "manifest_path",
    "total_artifacts",
    "found_locally",
    "missing",
    "missing_local_path",
    "valid",
    "found_hash_not_checked",
    "size_mismatch",
    "hash_mismatch",
    "hash_checked",
    "hash_enabled",
    "blocked_missing_artifact",
    "blocked_unavailable_url",
    "blocked_unclear_protocol",
    "requires_manual_download",
    "requires_author_contact",
    "requires_api",
    "requires_training",
    "extension_only",
    "unknown_reproducibility_status",
    "blocked_unavailable_or_unknown",
    "details_path",
    "notes",
    "timestamp",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Plan C artifact manifest entries.")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the Plan C artifact manifest CSV.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute SHA-256 and compare it when expected_hash_sha256 is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory for validation summary/detail CSVs.",
    )
    return parser.parse_args(argv)


def validate_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    output_dir: str | Path = "results/tables",
    hash_files: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Validate a Plan C artifact manifest and write summary/detail CSV files."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    manifest = _resolve_path(Path(manifest_path), root)
    rows = _read_manifest(manifest)

    timestamp = datetime.now(timezone.utc).isoformat()
    details = [
        _validate_row(row, root=root, hash_files=hash_files, timestamp=timestamp)
        for row in rows
    ]
    details_df = pd.DataFrame(details, columns=REQUIRED_COLUMNS + DETAIL_EXTRA_COLUMNS)

    out_dir = _resolve_path(Path(output_dir), root)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "plan_c_artifact_validation_details.csv"
    summary_path = out_dir / "plan_c_artifact_validation_summary.csv"
    details_df.to_csv(details_path, index=False)

    summary = _build_summary(
        details_df,
        manifest_path=_display_path(manifest, root),
        details_path=_display_path(details_path, root),
        hash_files=hash_files,
        timestamp=timestamp,
    )
    summary_df = pd.DataFrame([summary], columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_path, index=False)
    return details_df, summary_df, details_path, summary_path


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    return common_manifests.read_csv_manifest_rows(
        manifest_path,
        required_columns=REQUIRED_COLUMNS,
        missing="raise",
        strip=True,
        project_columns=REQUIRED_COLUMNS,
    )


def _validate_row(
    row: dict[str, str],
    *,
    root: Path,
    hash_files: bool,
    timestamp: str,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(row)
    local_path_value = row.get("local_path", "").strip()
    resolved = _resolve_path(Path(local_path_value), root) if local_path_value else None

    result["resolved_local_path"] = _display_path(resolved, root) if resolved else ""
    result["actual_found_locally"] = False
    result["actual_size_bytes"] = ""
    result["actual_hash_sha256"] = ""
    result["size_check_status"] = "not_applicable"
    result["hash_check_status"] = "not_applicable"
    result["validation_notes"] = ""
    result["validated_at"] = timestamp

    if resolved is None:
        result["validation_status"] = "missing_local_path"
        result["validation_notes"] = "No local_path was provided in the manifest."
        return result

    if not resolved.exists():
        result["validation_status"] = "missing"
        result["validation_notes"] = "Configured local_path does not exist."
        return result

    result["actual_found_locally"] = True
    is_file = resolved.is_file()
    if is_file:
        actual_size = resolved.stat().st_size
        result["actual_size_bytes"] = actual_size
    else:
        actual_size = None

    size_status, size_valid = _check_size(row.get("expected_size_bytes", ""), actual_size, is_file)
    result["size_check_status"] = size_status
    if size_status == "invalid_expected_size":
        result["validation_status"] = "invalid_expected_size"
        result["validation_notes"] = "Expected size is not an integer."
        return result
    if not size_valid:
        result["validation_status"] = "size_mismatch"
        result["validation_notes"] = "Actual file size differs from expected_size_bytes."
        return result

    hash_status, hash_valid, actual_hash = _check_hash(
        row.get("expected_hash_sha256", ""),
        resolved,
        hash_files=hash_files,
    )
    result["hash_check_status"] = hash_status
    result["actual_hash_sha256"] = actual_hash
    if hash_status == "not_checked":
        result["validation_status"] = "found_hash_not_checked"
        result["validation_notes"] = "Expected hash is present, but --hash was not enabled."
        return result
    if not hash_valid:
        result["validation_status"] = "hash_mismatch" if hash_status == "mismatch" else "hash_unsupported"
        result["validation_notes"] = "SHA-256 validation did not pass."
        return result

    result["validation_status"] = "valid"
    result["validation_notes"] = "Local artifact exists and provided size/hash expectations passed."
    return result


def _check_size(expected_size: str, actual_size: int | None, is_file: bool) -> tuple[str, bool]:
    expected_size = expected_size.strip()
    if not expected_size:
        return "not_applicable", True
    try:
        expected = int(expected_size)
    except ValueError:
        return "invalid_expected_size", False
    if not is_file or actual_size is None:
        return "unsupported_for_directory", False
    if actual_size == expected:
        return "matched", True
    return "mismatch", False


def _check_hash(
    expected_hash: str,
    path: Path,
    *,
    hash_files: bool,
) -> tuple[str, bool, str]:
    expected_hash = expected_hash.strip().lower()
    if not expected_hash:
        return "not_applicable", True, ""
    if not hash_files:
        return "not_checked", True, ""
    if not path.is_file():
        return "unsupported_for_directory", False, ""

    actual_hash = _sha256(path)
    if actual_hash.lower() == expected_hash:
        return "matched", True, actual_hash
    return "mismatch", False, actual_hash


def _sha256(path: Path) -> str:
    return common_hashing.sha256_file(path)


def _build_summary(
    details_df: pd.DataFrame,
    *,
    manifest_path: str,
    details_path: str,
    hash_files: bool,
    timestamp: str,
) -> dict[str, Any]:
    statuses = details_df["validation_status"].fillna("")
    repro_statuses = details_df["reproducibility_status"].fillna("")
    source_types = details_df["source_type"].fillna("")

    blocked_unavailable_or_unknown = (
        repro_statuses.str.startswith("blocked_")
        | repro_statuses.isin(["unknown"])
        | source_types.isin(["unavailable", "unknown"])
    ).sum()

    return {
        "manifest_path": manifest_path,
        "total_artifacts": int(len(details_df)),
        "found_locally": int(details_df["actual_found_locally"].astype(bool).sum()),
        "missing": int((statuses == "missing").sum()),
        "missing_local_path": int((statuses == "missing_local_path").sum()),
        "valid": int((statuses == "valid").sum()),
        "found_hash_not_checked": int((statuses == "found_hash_not_checked").sum()),
        "size_mismatch": int((statuses == "size_mismatch").sum()),
        "hash_mismatch": int((statuses == "hash_mismatch").sum()),
        "hash_checked": int((details_df["hash_check_status"] == "matched").sum() + (details_df["hash_check_status"] == "mismatch").sum()),
        "hash_enabled": bool(hash_files),
        "blocked_missing_artifact": int((repro_statuses == "blocked_missing_artifact").sum()),
        "blocked_unavailable_url": int((repro_statuses == "blocked_unavailable_url").sum()),
        "blocked_unclear_protocol": int((repro_statuses == "blocked_unclear_protocol").sum()),
        "requires_manual_download": int((repro_statuses == "requires_manual_download").sum()),
        "requires_author_contact": int((repro_statuses == "requires_author_contact").sum()),
        "requires_api": int((repro_statuses == "requires_api").sum()),
        "requires_training": int((repro_statuses == "requires_training").sum()),
        "extension_only": int((repro_statuses == "extension_only").sum()),
        "unknown_reproducibility_status": int((repro_statuses == "unknown").sum()),
        "blocked_unavailable_or_unknown": int(blocked_unavailable_or_unknown),
        "details_path": details_path,
        "notes": "Validation checks local paths only; no downloads, APIs, training, or metrics were run.",
        "timestamp": timestamp,
    }


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, summary_df, details_path, summary_path = validate_manifest(
        args.manifest,
        output_dir=args.output_dir,
        hash_files=args.hash,
    )
    summary = summary_df.iloc[0].to_dict()
    print(
        "Plan C artifact validation complete: "
        f"total={summary['total_artifacts']}, "
        f"found={summary['found_locally']}, "
        f"missing={summary['missing']}, "
        f"missing_local_path={summary['missing_local_path']}"
    )
    print(f"Details: {_display_path(details_path, PROJECT_ROOT)}")
    print(f"Summary: {_display_path(summary_path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

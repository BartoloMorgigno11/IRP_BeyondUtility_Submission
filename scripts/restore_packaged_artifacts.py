"""Restore or verify byte-preserving packaged CSV artifacts.

This utility performs packaging checks only. It does not compute metrics,
call models, access the network, or modify scientific outputs except when
explicitly restoring original CSV bytes from deterministic gzip packages.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


CHUNK_SIZE = 1024 * 1024
DEFAULT_MANIFEST = Path("results/manifests/github_packaged_artifacts.json")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_data_rows(path: Path) -> int:
    """Return data-row count for a headered CSV using byte lines only."""

    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def load_artifacts(manifest_path: Path) -> list[dict[str, Any]]:
    """Load artifact entries from the delivery manifest."""

    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"{manifest_path} does not contain an artifacts list")
    return artifacts


def verify_package(artifact: dict[str, Any], root: Path) -> None:
    """Verify that a packaged artifact exists and matches its manifest SHA."""

    package_path = root / artifact["packaged_path"]
    expected = artifact["packaged_sha256"]
    if not package_path.is_file():
        raise FileNotFoundError(f"missing package: {package_path}")
    actual = sha256_file(package_path)
    if actual != expected:
        raise ValueError(
            f"package SHA mismatch for {package_path}: {actual} != {expected}"
        )


def verify_restored(artifact: dict[str, Any], root: Path) -> None:
    """Verify that a restored original exists and matches SHA and row count."""

    original_path = root / artifact["original_path"]
    expected_sha = artifact["original_sha256"]
    expected_rows = int(artifact["row_count"])
    if not original_path.is_file():
        raise FileNotFoundError(f"missing restored original: {original_path}")
    actual_sha = sha256_file(original_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"restored SHA mismatch for {original_path}: {actual_sha} != {expected_sha}"
        )
    actual_rows = csv_data_rows(original_path)
    if actual_rows != expected_rows:
        raise ValueError(
            f"row-count mismatch for {original_path}: {actual_rows} != {expected_rows}"
        )


def restore_artifact(artifact: dict[str, Any], root: Path) -> None:
    """Restore one artifact after package verification and overwrite checks."""

    verify_package(artifact, root)
    original_path = root / artifact["original_path"]
    package_path = root / artifact["packaged_path"]

    if original_path.exists():
        current_sha = sha256_file(original_path)
        if current_sha != artifact["original_sha256"]:
            raise FileExistsError(
                "refusing to overwrite existing file with different SHA: "
                f"{original_path} ({current_sha})"
            )
        verify_restored(artifact, root)
        print(f"OK already restored: {original_path}")
        return

    original_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=original_path.parent, prefix=f".{original_path.name}.tmp."
    ) as temp_handle:
        temp_path = Path(temp_handle.name)
        with gzip.open(package_path, "rb") as gz_handle:
            shutil.copyfileobj(gz_handle, temp_handle, CHUNK_SIZE)

    try:
        actual_sha = sha256_file(temp_path)
        if actual_sha != artifact["original_sha256"]:
            raise ValueError(
                f"restored SHA mismatch for {package_path}: "
                f"{actual_sha} != {artifact['original_sha256']}"
            )
        actual_rows = csv_data_rows(temp_path)
        if actual_rows != int(artifact["row_count"]):
            raise ValueError(
                f"restored row-count mismatch for {package_path}: "
                f"{actual_rows} != {artifact['row_count']}"
            )
        temp_path.replace(original_path)
        print(f"OK restored: {original_path}")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> int:
    """Run the requested restore/verify mode."""

    root = Path.cwd()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    artifacts = load_artifacts(manifest_path)
    for artifact in artifacts:
        if args.verify_package:
            verify_package(artifact, root)
            print(f"OK package: {artifact['packaged_path']}")
        elif args.restore:
            restore_artifact(artifact, root)
        elif args.verify_restored:
            verify_restored(artifact, root)
            print(f"OK restored: {artifact['original_path']}")

    print(f"PASS {len(artifacts)} artifacts")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Artifact manifest path (default: {DEFAULT_MANIFEST})",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-package", action="store_true")
    mode.add_argument("--restore", action="store_true")
    mode.add_argument("--verify-restored", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)

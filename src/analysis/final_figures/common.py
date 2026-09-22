"""Shared deterministic infrastructure for final scientific figures."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


class FigureValidationError(ValueError):
    """Raised when a final-figure provenance contract is violated."""


REQUIRED_MANIFEST_FIELDS = {
    "figure_id",
    "rq",
    "scientific_purpose",
    "canonical_source_artifacts",
    "canonical_source_sha256",
    "figure_values_csv",
    "figure_values_sha256",
    "generator",
    "generator_sha256",
    "output_png",
    "output_png_sha256",
    "output_pdf",
    "output_pdf_sha256",
    "output_svg",
    "output_svg_sha256",
    "scientific_interpretation",
    "limitations",
    "created_from_commit",
}

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_PATH_FIELDS = {
    "canonical_source_artifacts",
    "figure_values_csv",
    "generator",
    "output_png",
    "output_pdf",
    "output_svg",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for repository artifacts."""

    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(root: Path, relative: str | Path) -> Path:
    """Resolve a safe repository-relative path below *root*."""

    root = root.resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or _WINDOWS_ABSOLUTE.match(str(relative)):
        raise FigureValidationError(f"absolute path is forbidden: {relative}")
    if ".." in candidate.parts:
        raise FigureValidationError(f"parent traversal is forbidden: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FigureValidationError(f"path leaves repository root: {relative}") from exc
    return resolved


def write_json(path: Path, value: Any) -> None:
    """Write deterministic UTF-8 JSON with LF newlines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8", newline="\n")


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministic CSV while preserving the supplied row order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def git_head(root: Path) -> str:
    """Read the current commit without modifying the repository."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _iter_path_values(manifest: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in _PATH_FIELDS:
        value = manifest.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return values


def validate_figure_manifest(
    source_root: Path,
    manifest_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Validate sources, values, outputs, generator, hashes, and path portability."""

    source_root = source_root.resolve()
    artifact_root = (artifact_root or source_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise FigureValidationError(f"manifest fields missing: {sorted(missing)}")

    for value in _iter_path_values(manifest):
        repository_path(source_root, value)

    sources = manifest["canonical_source_artifacts"]
    source_hashes = manifest["canonical_source_sha256"]
    if set(sources) != set(source_hashes):
        raise FigureValidationError("canonical source paths and hashes differ")
    for relative in sources:
        path = repository_path(source_root, relative)
        if not path.is_file():
            raise FigureValidationError(f"source does not exist: {relative}")
        if sha256_file(path) != source_hashes[relative]:
            raise FigureValidationError(f"source hash mismatch: {relative}")

    generator = repository_path(source_root, manifest["generator"])
    if not generator.is_file() or sha256_file(generator) != manifest["generator_sha256"]:
        raise FigureValidationError("generator missing or hash mismatch")

    checks = [
        ("figure_values_csv", "figure_values_sha256"),
        ("output_png", "output_png_sha256"),
        ("output_pdf", "output_pdf_sha256"),
        ("output_svg", "output_svg_sha256"),
    ]
    for path_key, hash_key in checks:
        relative = manifest[path_key]
        path = repository_path(artifact_root, relative)
        if not path.is_file():
            raise FigureValidationError(f"figure artifact does not exist: {relative}")
        if sha256_file(path) != manifest[hash_key]:
            raise FigureValidationError(f"figure artifact hash mismatch: {relative}")

    if not re.fullmatch(r"[0-9a-f]{40}", manifest["created_from_commit"]):
        raise FigureValidationError("created_from_commit must be a full Git commit")
    return manifest

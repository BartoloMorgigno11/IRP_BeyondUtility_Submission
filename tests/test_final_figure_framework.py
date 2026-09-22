from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.final_figures.common import (
    FigureValidationError,
    repository_path,
    sha256_file,
    validate_figure_manifest,
    write_dataframe,
    write_json,
)


def test_repository_path_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    with pytest.raises(FigureValidationError):
        repository_path(tmp_path, "C:/private/input.csv")
    with pytest.raises(FigureValidationError):
        repository_path(tmp_path, "../input.csv")
    assert repository_path(tmp_path, "results/value.csv") == tmp_path / "results/value.csv"


def test_writers_are_byte_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"dataset": "beauty", "value": "0.25"}])
    csv_a, csv_b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_dataframe(csv_a, frame)
    write_dataframe(csv_b, frame)
    assert csv_a.read_bytes() == csv_b.read_bytes()

    payload = {"z": 1, "a": [2, 3]}
    json_a, json_b = tmp_path / "a.json", tmp_path / "b.json"
    write_json(json_a, payload)
    write_json(json_b, payload)
    assert json_a.read_bytes() == json_b.read_bytes()
    assert list(json.loads(json_a.read_text(encoding="utf-8"))) == ["a", "z"]


def test_manifest_validator_checks_every_declared_hash(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifacts"
    for relative in ["source.csv", "generator.py"]:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    for relative in ["values.csv", "figure.png", "figure.pdf", "figure.svg"]:
        path = artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    manifest = {
        "figure_id": "figure",
        "rq": "RQX",
        "scientific_purpose": "test",
        "canonical_source_artifacts": ["source.csv"],
        "canonical_source_sha256": {"source.csv": sha256_file(source_root / "source.csv")},
        "figure_values_csv": "values.csv",
        "figure_values_sha256": sha256_file(artifact_root / "values.csv"),
        "generator": "generator.py",
        "generator_sha256": sha256_file(source_root / "generator.py"),
        "output_png": "figure.png",
        "output_png_sha256": sha256_file(artifact_root / "figure.png"),
        "output_pdf": "figure.pdf",
        "output_pdf_sha256": sha256_file(artifact_root / "figure.pdf"),
        "output_svg": "figure.svg",
        "output_svg_sha256": sha256_file(artifact_root / "figure.svg"),
        "scientific_interpretation": "test",
        "limitations": ["test"],
        "created_from_commit": "a" * 40,
    }
    manifest_path = artifact_root / "manifest.json"
    write_json(manifest_path, manifest)
    validate_figure_manifest(source_root, manifest_path, artifact_root=artifact_root)

    (artifact_root / "figure.png").write_text("changed", encoding="utf-8")
    with pytest.raises(FigureValidationError, match="hash mismatch"):
        validate_figure_manifest(source_root, manifest_path, artifact_root=artifact_root)

"""Checks for the final RQ6 construct and frozen reliability figure."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.final_figures.rq6 import (
    ROOT, FIGURE_ID, AUTHORITY, CONTRACT, SPEC, G1, G2, G3, G4,
    SOURCES, HASHES, MANIFEST, figure_values, generate, render,
    validate, validate_sources,
)


def _hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_authority_hashes_and_source_separation() -> None:
    authority, contract, spec = validate_sources(ROOT)
    assert {AUTHORITY, CONTRACT, SPEC, G1, G2, G3, G4} == set(SOURCES)
    assert all(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected for relative, expected in HASHES.items())
    assert authority["local_canonical_evidence"] == "STRUCTURAL_RELIABILITY"
    assert authority["direct_textual_hallucination_reproduced"] is False
    assert contract["direct_comparability"] is False
    assert spec["panels"][0]["quantitative_foreground"] is False
    assert not set(spec["forbidden_quantitative_sources"]).intersection(SOURCES)
    assert all("reproduction_r1" not in relative for relative in SOURCES)


def test_canonical_counts_rates_and_failure_partition() -> None:
    values = figure_values(ROOT)
    assert values.experiment.tolist() == ["G1", "G2", "G3", "G4"]
    expected = {
        "G1": (29292, 29024, 268, 238, 30, 0),
        "G2": (88050, 86768, 1282, 1090, 182, 10),
        "G3": (12000, 11834, 166, 114, 51, 1),
        "G4": (29292, 28986, 306, 255, 51, 0),
    }
    for row in values.itertuples(index=False):
        assert (row.total_requests, row.strict_valid, row.invalid_total, row.invalid_length,
                row.duplicate_index, row.unknown_or_out_of_candidate) == expected[row.experiment]
        assert row.strict_valid + row.invalid_total + row.backend_failure == row.total_requests
        assert row.invalid_length + row.duplicate_index + row.unknown_or_out_of_candidate + row.parser_or_schema_failure + row.backend_failure == row.invalid_total
        assert abs(row.strict_valid_rate - row.strict_valid / row.total_requests) < 1e-15
        assert row.evidence_type == "STRUCTURAL_RELIABILITY"
        assert row.comparability_to_paper == "NOT_DIRECTLY_COMPARABLE"
    assert values.backend_failure.eq(0).all()
    assert values.parser_or_schema_failure.eq(0).all()


def test_figure_labels_keep_constructs_and_scales_distinct() -> None:
    figure = render(figure_values(ROOT))
    try:
        text = " ".join(label.get_text() for label in figure.texts)
        assert "TEXTUAL HALLUCINATION  ≠  STRUCTURAL INVALIDITY" in text
        assert "NOT DIRECTLY COMPARABLE" in text
        assert "G2 unknown/out-of-candidate: 10 requests (16 index values)" in text
        assert len(figure.axes) == 4
        validity, taxonomy = figure.axes[2:]
        assert validity.get_xlim() == (0, 100)
        assert "counts; separate scale" in taxonomy.get_xlabel()
        assert taxonomy.get_xlim()[1] > 1282
        assert all("hallucination rate" not in label.get_text().lower() for ax in figure.axes for label in ax.texts)
    finally:
        plt.close(figure)


def test_deterministic_regeneration_and_portable_chain(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate(ROOT, artifact_root=first, created_from_commit="a" * 40)
    generate(ROOT, artifact_root=second, created_from_commit="a" * 40)
    assert _hashes(first) == _hashes(second)
    assert len(_hashes(first)) == 5
    manifest = validate(ROOT, artifact_root=first)
    assert manifest["figure_id"] == FIGURE_ID
    assert manifest["direct_comparability"] is False
    assert manifest["r3_r4_quantitative_dependency"] is False
    assert manifest["r1_quantitative_dependency"] is False
    assert manifest["rq5_views_summed"] is False
    assert manifest["created_from_commit"] == "a" * 40
    for key in ("canonical_source_artifacts", "figure_values_csv", "generator", "output_png", "output_pdf", "output_svg"):
        paths = manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
        assert all(not Path(path).is_absolute() and ":" not in path for path in paths)


def test_saved_final_figure_values_and_hashes() -> None:
    manifest = validate(ROOT)
    saved = pd.read_csv(ROOT / manifest["figure_values_csv"], dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(saved, figure_values(ROOT).astype(str), check_dtype=False, check_exact=True)
    assert (ROOT / MANIFEST).is_file()

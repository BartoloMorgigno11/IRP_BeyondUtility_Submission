"""Regression checks for presentation-only RQ5 visual authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.final_figures.rq5 import (
    ROOT, FIG1, FIG2, FIG5, IDS, MANIFESTS, METRICS, BOOTSTRAP, HISTORY,
    CONDITIONS, DECISIONS, METRIC_LABELS, SOURCE_HASHES,
    _plot_conditions, _plot_deltas, _plot_history,
    generate_all, validate_all, validate_sources, values_for,
)


def _hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_frozen_sources_and_untouched_supporting_figures() -> None:
    hashes = validate_sources(ROOT)
    for relative, expected in SOURCE_HASHES.items():
        assert hashes[relative] == expected


def test_conditions_are_categorical_and_exact() -> None:
    values = values_for(FIG1)
    source = pd.read_csv(ROOT / METRICS, dtype=str, keep_default_na=False)
    assert len(values) == 48
    assert set(values.condition) == set(CONDITIONS)
    assert values.requests.eq("200").all()
    assert values.denominator_policy.eq("FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO").all()
    for row in values.itertuples(index=False):
        match = source.loc[source.dataset.eq(row.dataset) &
                           source.recommender.eq(row.recommender) &
                           source.profile_generator.eq(row.source_profile_generator) &
                           source.condition.eq(row.condition)]
        assert len(match) == 1
        assert abs(float(row.value) - float(match.iloc[0][row.metric])) < 1e-15
        assert row.strict_valid == match.iloc[0].strict_valid
    figure = _plot_conditions(values)
    try:
        assert all(len(ax.lines) == 0 for ax in figure.axes)
        labels = [item.get_text() for item in figure.legends[0].texts]
        assert labels == ["History-only", "Profile-only", "Profile + History"]
        assert "inferential treatment effects are shown separately" in " ".join(text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


def test_delta_orientation_and_exact_frozen_ci_decisions() -> None:
    values = values_for(FIG2)
    source = pd.read_csv(ROOT / BOOTSTRAP, dtype=str, keep_default_na=False)
    assert len(values) == len(source) == 32
    assert values.paired_n.eq("200").all()
    assert values.bootstrap_resamples.eq("10000").all()
    assert values.p_value.eq("").all()
    assert values.multiplicity_correction.eq("not_computed").all()
    assert values.ci_marker.eq(values.decision.map(DECISIONS)).all()
    for column in ("mean_delta", "ci_lower", "ci_upper"):
        assert np.allclose(values[column].astype(float), source[column].astype(float), atol=1e-15, rtol=0)
    assert values.decision.eq(source.decision).all()
    figure = _plot_deltas(values)
    try:
        visible = " ".join(text.get_text() for text in figure.texts)
        assert "Profile-only minus History-only" in visible
        assert "Profile + History minus History-only" in visible
        assert "No marker: CI includes zero" in visible
        assert "unadjusted for multiplicity" in visible
    finally:
        plt.close(figure)


def test_history_v2_has_independent_raw_metric_panels() -> None:
    values = values_for(FIG5)
    source = pd.read_csv(ROOT / HISTORY, dtype=str, keep_default_na=False)
    assert values.equals(source)
    assert len(values) == 24
    figure = _plot_history(values)
    try:
        assert len(figure.axes) == len(METRIC_LABELS) == 4
        assert not any(axis.get_shared_y_axes().joined(figure.axes[0], axis) for axis in figure.axes[1:])
        for axis, metric in zip(figure.axes, METRIC_LABELS, strict=True):
            assert len(axis.lines) == 3
            for line, condition in zip(axis.lines, CONDITIONS, strict=True):
                expected = values.loc[values.condition.eq(condition)].sort_values("history_length", key=lambda col: col.astype(int))
                assert np.array_equal(line.get_xdata(), expected.history_length.astype(int).to_numpy())
                assert np.allclose(line.get_ydata(), expected[metric].astype(float).to_numpy(), atol=1e-15, rtol=0)
            assert axis.get_ylim()[0] == 0
        assert len({axis.get_ylim()[1] for axis in figure.axes}) == 4
        assert "Beauty / Qwen only" in " ".join(text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


def test_deterministic_generation_and_manifest_portability(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate_all(ROOT, artifact_root=first, created_from_commit="a" * 40)
    generate_all(ROOT, artifact_root=second, created_from_commit="a" * 40)
    assert _hashes(first) == _hashes(second)
    assert len(_hashes(first)) == 15
    manifests = validate_all(ROOT, artifact_root=first)
    assert {entry["figure_id"] for entry in manifests} == set(IDS)
    for figure_id in IDS:
        manifest = json.loads((first / MANIFESTS / f"{figure_id}_manifest.json").read_text(encoding="utf-8"))
        assert manifest["created_from_commit"] == "a" * 40
        assert manifest["presentation_only"] is True
        for key in ("canonical_source_artifacts", "figure_values_csv", "generator", "output_png", "output_pdf", "output_svg"):
            paths = manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
            assert all(not Path(path).is_absolute() and ":" not in path for path in paths)


def test_saved_final_renderings_validate() -> None:
    assert {entry["figure_id"] for entry in validate_all(ROOT)} == set(IDS)

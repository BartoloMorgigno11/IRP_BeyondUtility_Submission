"""Regression checks for the two frozen RQ2 comparative figures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.analysis.final_figures.rq2_traditional_llm import (
    B, F, FIGURES, G, SOURCES, build_values, generate_all, validate_all,
    validate_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_corrected_sources_and_calibration_authority() -> None:
    assert validate_sources(ROOT) == SOURCES
    assert SOURCES[G] == "2e5c774f552cb47f16622f2a989ce033ee05036ad814558926b40148e8857c5b"
    assert "15ca9025aada73feba7513e06756b6c3ede42542befd82c1a76dbc757b018ae1" not in SOURCES.values()


def test_values_match_all_selected_canonical_cells() -> None:
    for figure_id, (metrics, _) in FIGURES.items():
        values = build_values(ROOT, figure_id)
        assert len(values) == 195
        assert set(values.metric) == set(metrics)
        assert values.groupby(["dataset", "metric"]).size().eq(13).all()
        assert not values.metric.isin(["popreo_at_5", "calibration_at_5"]).any()
        if "item_coverage_at_5" in metrics:
            assert values.loc[values.metric.eq("item_coverage_at_5"), "comparability_class"].eq("DESCRIPTIVE_ONLY").all()
        else:
            assert values.comparability_class.eq("ADAPTED_NUMERICAL_COMPARISON").all()
        for source_path, column in ((B, "value"), (F, "mean_value"), (G, "value")):
            source = pd.read_csv(ROOT / source_path, dtype=str, keep_default_na=False)
            by_id = source.set_index("result_id")[column].to_dict()
            selected = values.loc[values.source_artifact.eq(source_path)]
            assert all(row.value == by_id[row.source_result_id] for row in selected.itertuples())


def test_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    commit = "a" * 40
    generate_all(ROOT, artifact_root=first, created_from_commit=commit)
    generate_all(ROOT, artifact_root=second, created_from_commit=commit)
    assert _hashes(first) == _hashes(second)
    assert len(_hashes(first)) == 10
    manifests = validate_all(ROOT, artifact_root=first)
    assert {m["figure_id"] for m in manifests} == set(FIGURES)
    assert all(m["calibration_design_decision"]["traditional_counterpart_exists"] for m in manifests)
    assert all(m["population_contract"]["llm"] == "strict_valid_plus_metric_defined_users" for m in manifests)


def test_committed_artifacts_validate() -> None:
    assert {m["figure_id"] for m in validate_all(ROOT)} == set(FIGURES)

"""Focused checks for the cross-RQ map and global editorial figure freeze."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.final_figures import cross_rq, global_freeze
from src.analysis.final_figures.common import repository_path, sha256_file


def test_cross_rq_summary_and_frozen_figure_chain() -> None:
    cross_rq.validate()
    values = cross_rq.build_values().set_index("rq")
    assert set(values.index) == {f"RQ{i}" for i in range(1, 8)}
    assert "57 pos / 1 neg / 32 inc" in values.loc["RQ3", "main_result"]
    assert "3 pos / 18 neg / 69 inc" in values.loc["RQ3", "main_result"]
    assert "different eligible cohorts" in values.loc["RQ3", "critical_qualification"]
    assert "85 pos / 1 neg / 4 inc" in values.loc["RQ4", "main_result"]
    assert "98.5" in values.loc["RQ6", "main_result"]
    assert "99.1" in values.loc["RQ6", "main_result"]
    assert "0.4" in values.loc["RQ7", "main_result"]
    assert "46.9" in values.loc["RQ7", "main_result"]
    assert "not local structural invalidity" in values.loc["RQ6", "critical_qualification"]
    assert "No same-pool superiority test" in values.loc["RQ7", "critical_qualification"]


def test_cross_rq_byte_identical_regeneration(tmp_path: Path) -> None:
    frozen = cross_rq.validate()
    regenerated = cross_rq.generate(artifact_root=tmp_path,
                                    created_from_commit=frozen["created_from_commit"])
    for key in ("figure_values_sha256", "output_png_sha256", "output_pdf_sha256", "output_svg_sha256"):
        assert regenerated[key] == frozen[key]
    assert sha256_file(repository_path(tmp_path, cross_rq.MANIFEST)) == sha256_file(
        repository_path(cross_rq.ROOT, cross_rq.MANIFEST))


def test_global_roles_and_all_active_hashes() -> None:
    freeze = global_freeze.validate()
    index = pd.read_csv(repository_path(global_freeze.ROOT, global_freeze.INDEX),
                        dtype=str, keep_default_na=False)
    assert freeze["role_counts"] == {"MAIN_TEXT": 12, "APPENDIX": 6,
                                     "REPOSITORY_ONLY": 7, "EXCLUDED": 1}
    assert index.figure_id.is_unique
    assert index.set_index("figure_id").loc["fig_rq1_utility_heatmap_appendix", "role"] == "MAIN_TEXT"
    assert index.set_index("figure_id").loc["fig_rq3_paired_bootstrap_decisions", "role"] == "APPENDIX"
    assert index.set_index("figure_id").loc["fig_rq4_targeted_r5b_canddif_recovery", "role"] == "REPOSITORY_ONLY"
    excluded = index.loc[index.figure_id.eq("fig_rq5_profile_example_qualitative")].iloc[0]
    assert excluded.role == "EXCLUDED" and not excluded.png_path
    assert all(freeze["global_assertions"].values())
    assert sha256_file(repository_path(global_freeze.ROOT, global_freeze.INDEX)) == freeze[
        "global_figure_index_sha256"]

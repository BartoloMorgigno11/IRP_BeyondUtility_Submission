"""Regression checks for the frozen RQ1 traditional-versus-LLM figure."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.analysis.final_figures.rq1_traditional_llm import (
    FIGURE_ID, SOURCES, build_values, generate, validate, validate_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_sources_and_protocol_alignment() -> None:
    assert validate_sources(ROOT) == SOURCES


def test_values_preserve_canonical_cells_and_exclude_mrr() -> None:
    values = build_values(ROOT)
    assert len(values) == 130
    assert set(values.metric) == {"hr_at_5", "ndcg_at_5"}
    assert set(values.system_family) == {"TRADITIONAL", "LLM"}
    assert values.groupby(["dataset", "metric"]).size().eq(13).all()
    assert values.loc[values.model.isin(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]), "seed_mean_flag"].eq("true").all()
    assert values.loc[values.system_family.eq("LLM"), "prompt"].isin(["BASE", "LLMRANK"]).all()
    source_paths = set(values.source_artifact)
    assert source_paths == {
        "results/tables/f1bx_baseline_utility_results.csv",
        "results/tables/f1bz_aggregate_utility_results.csv",
        "results/tables/g1_primary_ranking_utility.csv",
    }
    for source_path, value_column in (
        ("results/tables/f1bx_baseline_utility_results.csv", "value"),
        ("results/tables/f1bz_aggregate_utility_results.csv", "mean_value"),
        ("results/tables/g1_primary_ranking_utility.csv", "value"),
    ):
        source = pd.read_csv(ROOT / source_path, dtype=str, keep_default_na=False)
        expected = source.set_index("result_id")[value_column].to_dict()
        subset = values.loc[values.source_artifact.eq(source_path)]
        assert all(row.value == expected[row.source_result_id] for row in subset.itertuples())


def test_generation_is_deterministic_and_manifest_valid(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    commit = "a" * 40
    generate(ROOT, artifact_root=first, created_from_commit=commit)
    generate(ROOT, artifact_root=second, created_from_commit=commit)
    assert _hashes(first) == _hashes(second)
    assert len(_hashes(first)) == 5
    manifest = validate(ROOT, artifact_root=first)
    assert manifest["figure_id"] == FIGURE_ID
    assert manifest["comparison_contract"]["excluded_metric"] == "mrr"
    assert manifest["comparison_contract"]["f1bz_variability"].startswith("sample_standard_deviation")


def test_committed_artifacts_validate() -> None:
    assert validate(ROOT)["figure_id"] == FIGURE_ID

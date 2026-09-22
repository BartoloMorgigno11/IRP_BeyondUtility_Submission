from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.final_figures.rq1 import (
    FIGURE_IDS,
    MANIFEST_ROOT,
    SOURCE_HASHES,
    build_figure_values,
    generate_all,
    validate_all,
    validate_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_rq1_canonical_sources_and_mrr_contract() -> None:
    assert validate_sources(ROOT) == SOURCE_HASHES


def test_rq1_values_are_exact_complete_canonical_grid() -> None:
    values = build_figure_values(ROOT)
    assert len(values) == 30
    assert not values.duplicated(["dataset", "model_name", "prompt_variant"]).any()
    assert list(values.columns) == [
        "dataset",
        "model_name",
        "prompt_variant",
        "hr_at_5",
        "ndcg_at_5",
        "mrr",
        "mrr_display_label",
        "mrr_inspected_depth",
    ]
    assert values["mrr_display_label"].eq("MRR").all()
    assert values["mrr_inspected_depth"].eq("10").all()

    source = pd.read_csv(
        ROOT / "results/tables/g1_primary_ranking_utility.csv",
        dtype=str,
        keep_default_na=False,
    )
    for row in values.itertuples(index=False):
        for metric in ["hr_at_5", "ndcg_at_5", "mrr"]:
            expected = source.loc[
                source["dataset"].eq(row.dataset)
                & source["model_name"].eq(row.model_name)
                & source["prompt_variant"].eq(row.prompt_variant)
                & source["metric"].eq(metric),
                "value",
            ]
            assert len(expected) == 1
            assert getattr(row, metric) == expected.iloc[0]


def test_rq1_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    commit = "a" * 40
    generate_all(ROOT, artifact_root=first, created_from_commit=commit)
    generate_all(ROOT, artifact_root=second, created_from_commit=commit)
    assert _tree_hashes(first) == _tree_hashes(second)
    assert len(_tree_hashes(first)) == 10
    assert {item["figure_id"] for item in validate_all(ROOT, artifact_root=first)} == set(FIGURE_IDS)

    for figure_id in FIGURE_IDS:
        manifest = json.loads(
            (first / MANIFEST_ROOT / f"{figure_id}_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["created_from_commit"] == commit
        assert manifest["metric_contract"]["MRR"] == "mrr"
        assert manifest["metric_contract"]["MRR_INSPECTED_DEPTH"] == 10
        for key in [
            "canonical_source_artifacts",
            "figure_values_csv",
            "generator",
            "output_png",
            "output_pdf",
            "output_svg",
        ]:
            values = manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
            assert all(not Path(value).is_absolute() for value in values)
            assert all(":" not in value for value in values)


def test_committed_rq1_artifacts_match_manifests() -> None:
    validated = validate_all(ROOT)
    assert {item["figure_id"] for item in validated} == set(FIGURE_IDS)
    assert all(item["created_from_commit"] == "fd74c74186ffd84d2d35baffab488484dc6ff9bf" for item in validated)

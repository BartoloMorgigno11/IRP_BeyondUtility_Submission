from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.final_figures.rq2_heatmaps import (
    HEATMAP_IDS,
    HEATMAP_SPECS,
    MANIFEST_ROOT,
    PRE_FIX_HASH,
    SOURCE_BEYOND,
    SOURCE_HASHES,
    build_heatmap_values,
    generate_all,
    validate_all,
    validate_heatmap_sources,
)

ROOT = Path(__file__).resolve().parents[1]
STARTING_COMMIT = "c9d99e196aabbd9564e1756fef9d7d7c23e2d5ac"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_rq2_heatmap_source_identity_and_complete_configuration_grid() -> None:
    assert validate_heatmap_sources(ROOT) == SOURCE_HASHES
    for figure_id, spec in HEATMAP_SPECS.items():
        values = build_heatmap_values(figure_id, ROOT)
        assert len(values) == 30 * len(spec["metrics"])
        assert set(values["metric_id"]) == set(spec["metrics"])
        assert values.groupby("metric_id").size().eq(30).all()
        assert not values.duplicated(
            ["metric_id", "dataset", "model_name", "prompt_variant"]
        ).any()
        assert set(values["heatmap_row"].astype(int)) == set(range(5))
        assert set(values["heatmap_column"].astype(int)) == set(range(6))
        assert values["display_transform"].eq(
            "PER_METRIC_LINEAR_COLOR_NORMALIZATION_RAW_ANNOTATIONS"
        ).all()


def test_rq2_heatmap_values_equal_canonical_source_exactly() -> None:
    source = pd.read_csv(ROOT / SOURCE_BEYOND, dtype=str, keep_default_na=False)
    source = source.rename(columns={"metric": "metric_id"})
    keys = ["metric_id", "dataset", "model_name", "prompt_variant"]
    for figure_id in HEATMAP_IDS:
        values = build_heatmap_values(figure_id, ROOT)
        relevant_source = source.loc[
            source["metric_id"].isin(HEATMAP_SPECS[figure_id]["metrics"])
        ]
        merged = values.merge(
            relevant_source[keys + ["value", "status"]],
            on=keys,
            how="outer",
            suffixes=("_heatmap", "_source"),
            validate="one_to_one",
            indicator=True,
        )
        assert merged["_merge"].eq("both").all()
        assert merged["value_heatmap"].eq(merged["value_source"]).all()
        assert merged["status_heatmap"].eq(merged["status_source"]).all()


def test_rq2_heatmap_contracts_preserve_missingness_and_metric_identity() -> None:
    longtail = build_heatmap_values("fig_rq2_llm_longtail_novelty_heatmap", ROOT)
    assert "aplt_at_5" in set(longtail["metric_id"])
    assert "aclt_at_5" not in set(longtail["metric_id"])
    popreo = longtail.loc[longtail["metric_id"].eq("popreo_at_5")]
    noncomputed = popreo.loc[~popreo["status"].eq("COMPUTED")]
    assert len(noncomputed) == 6
    assert noncomputed["dataset"].eq("foodcom").all()
    assert noncomputed["status"].eq("DEGENERATE_PROTOCOL_INPUT").all()
    assert noncomputed["value"].eq("").all()
    assert noncomputed["cell_annotation"].eq("N/C").all()

    serendipity = build_heatmap_values(
        "fig_rq2_llm_serendipity_coverage_heatmap", ROOT
    )
    assert set(serendipity["metric_id"]) == {
        "serendipity_at_5",
        "item_coverage_at_5",
        "catalog_coverage_at_5",
    }
    assert serendipity.loc[
        serendipity["metric_id"].eq("serendipity_at_5"), "status"
    ].eq("COMPUTED").all()

    calibration = build_heatmap_values("fig_rq2_llm_calibration_heatmap", ROOT)
    assert calibration["status"].eq("COMPUTED").all()
    assert calibration["cell_annotation"].str.contains(r"\ne=\d+\nu=\d+", regex=True).all()
    assert (
        calibration["eligible_user_count"].astype(int)
        + calibration["undefined_user_count"].astype(int)
        == calibration["planned_users"].astype(int)
    ).all()
    assert calibration["undefined_user_count"].astype(int).max() == 740
    assert calibration.loc[
        calibration["dataset"].eq("foodcom"), "eligible_user_count"
    ].astype(int).between(260, 273).all()


def test_rq2_heatmap_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    commit = "a" * 40
    generate_all(ROOT, artifact_root=first, created_from_commit=commit)
    generate_all(ROOT, artifact_root=second, created_from_commit=commit)
    assert _tree_hashes(first) == _tree_hashes(second)
    assert len(_tree_hashes(first)) == 20
    assert {item["figure_id"] for item in validate_all(ROOT, artifact_root=first)} == set(
        HEATMAP_IDS
    )

    for figure_id in HEATMAP_IDS:
        manifest_path = first / MANIFEST_ROOT / f"{figure_id}_manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        assert PRE_FIX_HASH not in manifest_text
        assert "plan-d" not in manifest_text.lower()
        assert "plan_d" not in manifest_text.lower()
        assert manifest["created_from_commit"] == commit
        assert manifest["display_contract"]["direction_reversal"] is False
        assert manifest["display_contract"]["cross_metric_aggregate"] is False
        assert manifest["display_contract"]["imputation"] is False
        for key in [
            "canonical_source_artifacts",
            "figure_values_csv",
            "generator",
            "output_png",
            "output_pdf",
            "output_svg",
        ]:
            paths = manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
            assert all(not Path(path).is_absolute() for path in paths)
            assert all(":" not in path for path in paths)


def test_committed_rq2_heatmaps_match_manifests() -> None:
    validated = validate_all(ROOT)
    assert {item["figure_id"] for item in validated} == set(HEATMAP_IDS)
    assert all(item["created_from_commit"] == STARTING_COMMIT for item in validated)

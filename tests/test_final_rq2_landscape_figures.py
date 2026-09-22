from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.final_figures.rq2 import (
    FIGURE_IDS,
    FIGURE_SPECS,
    MANIFEST_ROOT,
    PRE_FIX_HASH,
    SOURCE_BEYOND,
    SOURCE_HASHES,
    build_figure_values,
    generate_all,
    load_and_validate_canonical,
    validate_all,
    validate_sources,
)

ROOT = Path(__file__).resolve().parents[1]
STARTING_COMMIT = "4f85c9abe18d9f916366146d8a5cdbd1f4800240"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_rq2_source_identity_configuration_and_contracts() -> None:
    assert validate_sources(ROOT) == SOURCE_HASHES
    source = load_and_validate_canonical(ROOT)
    assert len(source) == 420
    assert source[["dataset", "model_name", "prompt_variant"]].drop_duplicates().shape[0] == 30
    assert source.groupby("metric").size().eq(30).all()
    assert "aplt_at_5" in set(source["metric"])
    assert "aclt_at_5" not in set(source["metric"])

    contract = json.loads((ROOT / "results/manifests/g1_metric_contract_and_provenance.json").read_text(encoding="utf-8"))
    expected_long_tail = {
        "aclt_is_separate_metric": False,
        "canonical_metric": "APLT@5",
        "canonical_metric_id": "aplt_at_5",
        "implementation_alias": "calACLT",
    }
    assert {
        key: contract["long_tail_contract"][key] for key in expected_long_tail
    } == expected_long_tail
    authority = contract["canonical_artifact_authority"]
    assert authority["canonical_beyond_utility_path"] == SOURCE_BEYOND
    assert authority["canonical_beyond_utility_sha256"] == SOURCE_HASHES[SOURCE_BEYOND]
    assert authority["historical_pre_fix_sha256"] == PRE_FIX_HASH


def test_rq2_figure_values_preserve_exact_metrics_and_statuses() -> None:
    expected_lengths = {
        "fig_rq2_llm_longtail_novelty_landscape": 120,
        "fig_rq2_llm_serendipity_coverage_landscape": 90,
        "fig_rq2_llm_diversity_distribution_landscape": 180,
        "fig_rq2_llm_calibration_landscape": 30,
    }
    for figure_id, expected_length in expected_lengths.items():
        values = build_figure_values(figure_id, ROOT)
        assert len(values) == expected_length
        assert set(values["metric_id"]) == set(FIGURE_SPECS[figure_id]["metrics"])
        assert values["display_transform"].eq("NONE_RAW_CANONICAL_VALUE").all()

    longtail = build_figure_values("fig_rq2_llm_longtail_novelty_landscape", ROOT)
    popreo = longtail.loc[longtail["metric_id"].eq("popreo_at_5")]
    missing = popreo.loc[~popreo["status"].eq("COMPUTED")]
    assert len(missing) == 6
    assert missing["dataset"].eq("foodcom").all()
    assert missing["status"].eq("DEGENERATE_PROTOCOL_INPUT").all()
    assert missing["value"].eq("").all()

    coverage = build_figure_values("fig_rq2_llm_serendipity_coverage_landscape", ROOT)
    assert {"item_coverage_at_5", "catalog_coverage_at_5"}.issubset(set(coverage["metric_id"]))
    assert coverage.loc[coverage["metric_id"].eq("serendipity_at_5"), "status"].eq("COMPUTED").all()

    calibration = build_figure_values("fig_rq2_llm_calibration_landscape", ROOT)
    assert calibration["status"].eq("COMPUTED").all()
    assert calibration["value"].ne("").all()
    assert calibration["eligible_user_count"].ne("").all()
    food = calibration.loc[calibration["dataset"].eq("foodcom")]
    assert set(food["eligible_user_count"].astype(int)).issubset(set(range(260, 274)))
    assert food["eligibility_status"].eq("PARTIAL_USER_ELIGIBILITY").all()


def test_rq2_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    commit = "a" * 40
    generate_all(ROOT, artifact_root=first, created_from_commit=commit)
    generate_all(ROOT, artifact_root=second, created_from_commit=commit)
    assert _tree_hashes(first) == _tree_hashes(second)
    assert len(_tree_hashes(first)) == 20
    assert {item["figure_id"] for item in validate_all(ROOT, artifact_root=first)} == set(FIGURE_IDS)

    for figure_id in FIGURE_IDS:
        manifest_path = first / MANIFEST_ROOT / f"{figure_id}_manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        assert PRE_FIX_HASH not in manifest_text
        assert manifest["created_from_commit"] == commit
        assert manifest["display_contract"]["transform"] == "NONE_RAW_CANONICAL_VALUE"
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


def test_committed_rq2_artifacts_match_manifests() -> None:
    validated = validate_all(ROOT)
    assert {item["figure_id"] for item in validated} == set(FIGURE_IDS)
    assert all(item["created_from_commit"] == STARTING_COMMIT for item in validated)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.final_figures.rq3 import (
    BOOTSTRAP_ID, CONTRASTS, GE10, GE20, GRID, HASHES, HISTORY_ID,
    MANIFESTS, METRICS, ROOT, STATS, bootstrap_values, generate_all,
    history_values, validate_all, validate_sources,
)

STARTING_COMMIT = "7f9efd8b747928e0ed6e19f3ff655ae0febf73eb"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_rq3_frozen_sources_and_complete_grids() -> None:
    assert validate_sources(ROOT) == HASHES
    grid = pd.read_csv(ROOT / GRID, dtype=str, keep_default_na=False)
    rows = grid.loc[grid.row_type.eq("metric")]
    assert set(rows.loc[rows.cohort.eq("GE10"), "history_length"].astype(int)) == set(GE10)
    assert set(rows.loc[rows.cohort.eq("GE20"), "history_length"].astype(int)) == set(GE20)
    assert {3, 4, 15}.issubset(set(rows.history_length.astype(int)))
    assert len(rows.loc[rows.cohort.eq("GE20")]) == 810


def test_rq3_history_values_preserve_ge20_population_and_exact_dataset_means() -> None:
    values = history_values(ROOT)
    source = pd.read_csv(ROOT / GRID, dtype=str, keep_default_na=False)
    source = source.loc[source.row_type.eq("metric") & source.cohort.eq("GE20")]
    assert len(values) == 810
    assert values.cohort.eq("GE20").all()
    assert values.analysis_membership.eq("SECONDARY_FULL_CURVE_GE20").all()
    assert set(values.history_length) == set(GE20)
    assert set(values.metric_id) == set(METRICS)
    keys = ["history_length", "dataset", "model", "prompt_strategy", "metric_id"]
    source = source.copy()
    source["history_length"] = source.history_length.astype(int)
    compared = values.merge(source[keys + ["value"]], on=keys, validate="one_to_one")
    assert len(compared) == 810
    assert compared.raw_canonical_value.eq(compared.value).all()
    for _, group in values.groupby(["history_length", "model", "prompt_strategy", "metric_id"]):
        assert len(group) == 5
        assert set(group.dataset) == {"beauty", "sports", "ml1m", "lastfm", "foodcom"}
        assert group.displayed_value.nunique() == 1
        assert np.isclose(float(group.displayed_value.iloc[0]), group.raw_canonical_value.astype(float).mean(), rtol=0, atol=1e-15)


def test_rq3_paired_cis_decisions_and_orientation_equal_frozen_source() -> None:
    values = bootstrap_values(ROOT)
    source = pd.read_csv(ROOT / STATS, dtype=str, keep_default_na=False)
    assert len(values) == len(source) == 450
    assert set(values.contrast) == set(CONTRASTS)
    assert values.p_value_status.eq("NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD").all()
    compared = values.merge(source, on="result_id", validate="one_to_one")
    assert len(compared) == 450
    for left, right in [("mean_delta_x", "mean_delta_y"), ("ci_low_x", "ci_low_y"), ("ci_high_x", "ci_high_y"), ("decision_x", "decision_y"), ("n_pairs_x", "n_pairs_y")]:
        assert compared[left].eq(compared[right]).all()
    assert (values.orientation == values.history_length_a + "_TO_" + values.history_length_b + "_B_MINUS_A").all()
    assert values.loc[values.contrast.isin(CONTRASTS[:3]), "eligibility_population"].eq("PRIMARY_B1_GE10").all()
    assert values.loc[values.contrast.isin(CONTRASTS[3:]), "eligibility_population"].eq("SECONDARY_FULL_CURVE_GE20").all()
    low, high = values.ci_low.astype(float), values.ci_high.astype(float)
    expected = np.where(low > 0, "POSITIVE", np.where(high < 0, "NEGATIVE", "INCONCLUSIVE"))
    assert (values.decision.to_numpy() == expected).all()
    assert not values.contrast.isin(["H0_H10", "H10_H20"]).any()


def test_rq3_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    commit = "a" * 40
    generate_all(ROOT, artifact_root=first, created_from_commit=commit)
    generate_all(ROOT, artifact_root=second, created_from_commit=commit)
    assert _tree_hashes(first) == _tree_hashes(second)
    assert len(_tree_hashes(first)) == 10
    assert {m["figure_id"] for m in validate_all(ROOT, artifact_root=first)} == {HISTORY_ID, BOOTSTRAP_ID}
    for figure_id in [HISTORY_ID, BOOTSTRAP_ID]:
        manifest = json.loads((first / MANIFESTS / f"{figure_id}_manifest.json").read_text(encoding="utf-8"))
        assert manifest["created_from_commit"] == commit
        for key in ["canonical_source_artifacts", "figure_values_csv", "generator", "output_png", "output_pdf", "output_svg"]:
            paths = manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
            assert all(not Path(path).is_absolute() and ":" not in path for path in paths)


def test_committed_rq3_artifacts_match_manifests() -> None:
    manifests = validate_all(ROOT)
    assert {m["figure_id"] for m in manifests} == {HISTORY_ID, BOOTSTRAP_ID}
    assert all(m["created_from_commit"] == STARTING_COMMIT for m in manifests)

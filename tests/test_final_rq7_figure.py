"""Provenance and rendering checks for the final G4/RQ7 figure."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.final_figures.rq7 import (
    ROOT, FIGURE_ID, SOURCES, DATASETS, MODELS, PROMPTS, EXPECTED_RECALL,
    _pool_contract, figure_values, generate, render, validate, validate_sources,
)


def _hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_canonical_source_hashes_and_shared_twenty_item_pools() -> None:
    hashes = validate_sources(ROOT)
    assert set(hashes) == set(SOURCES)
    assert all(hashlib.sha256((ROOT / key).read_bytes()).hexdigest() == value for key, value in hashes.items())
    assert _pool_contract(ROOT) == EXPECTED_RECALL


def test_all_thirty_cells_have_exact_denominators_and_decomposition() -> None:
    values = figure_values(ROOT)
    assert len(values) == 30
    assert not values.duplicated(["dataset", "model", "prompt"]).any()
    assert set(values.dataset) == set(DATASETS)
    assert set(values.model) == set(MODELS)
    assert set(values.prompt) == set(PROMPTS)
    for row in values.itertuples(index=False):
        assert (row.candidate_present_n, row.planned_n) == EXPECTED_RECALL[row.dataset]
        assert abs(row.candidate_recall_at_20 - row.candidate_present_n / row.planned_n) < 1e-15
        assert row.conditional_population_n == row.candidate_present_n
        assert row.foodcom_diagnostic_flag == (row.dataset == "foodcom")
        assert row.unconditional_denominator_contract == "all_planned_users_invalid_terminal_zero_positive_absent_zero"
        assert row.conditional_denominator_contract == "positive_present_users_only_invalid_terminal_zero"
        assert row.terminal_invalid_policy == "zero_utility_in_both_populations"
        assert abs(row.hr_observed - row.hr_expected_from_decomposition) < 1e-14
        assert abs(row.ndcg_observed - row.ndcg_expected_from_decomposition) < 1e-14
    assert np.isclose(values.hr_decomposition_error.max(), 5.551115123125783e-17, atol=1e-18)
    assert np.isclose(values.ndcg_decomposition_error.max(), 2.7755575615628914e-17, atol=1e-18)
    assert values.loc[values.dataset.eq("foodcom"), "conditional_population_n"].eq(4).all()


def test_rendering_separates_stages_and_foodcom_scope() -> None:
    figure = render(figure_values(ROOT))
    try:
        text = " ".join(item.get_text() for item in figure.texts)
        assert "shared top-20 pool" in text
        assert "No same-pool traditional-ordering superiority test" in text
        assert "G1 vs G4 is descriptive only" in text
        assert len(figure.axes) == 12
        for axis in figure.axes[2:]:
            assert len(axis.lines) == 6
        assert "target-present n=4, diagnostic only" in figure.axes[-2].get_title(loc="left")
        assert "target-present n=4, diagnostic only" in figure.axes[-1].get_title(loc="left")
    finally:
        plt.close(figure)


def test_deterministic_regeneration_and_hash_chain(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate(ROOT, artifact_root=first, created_from_commit="a" * 40)
    generate(ROOT, artifact_root=second, created_from_commit="a" * 40)
    assert _hashes(first) == _hashes(second)
    assert len(_hashes(first)) == 5
    manifest = validate(ROOT, artifact_root=first)
    assert manifest["figure_id"] == FIGURE_ID
    assert manifest["pool_size"] == 20
    assert manifest["positive_forcing"] is False
    assert manifest["same_pool_all_models_prompts"] is True
    assert manifest["decomposition_cells"] == 30
    assert manifest["foodcom_conditional_n"] == 4
    assert manifest["g1_g4_comparability"] == "DESCRIPTIVE_ONLY"
    assert manifest["same_pool_superiority_test"] is False
    assert manifest["universal_llmrank_claim"] is False
    assert manifest["created_from_commit"] == "a" * 40
    for key in ("canonical_source_artifacts", "figure_values_csv", "generator", "output_png", "output_pdf", "output_svg"):
        paths = manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
        assert all(not Path(path).is_absolute() and ":" not in path for path in paths)


def test_saved_values_and_manifest_hashes() -> None:
    manifest = validate(ROOT)
    saved = pd.read_csv(ROOT / manifest["figure_values_csv"], dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(saved, figure_values(ROOT).astype(str), check_dtype=False, check_exact=True)

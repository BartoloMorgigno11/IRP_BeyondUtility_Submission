"""Focused checks for the two render-only final-figure corrections."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.analysis.final_figures import rq2_heatmaps, rq4
from src.analysis.final_figures.render_polish import (
    BASE_HEAD, BOOTSTRAP, BOOTSTRAP_VALUES_SHA256, CALIBRATION,
    CALIBRATION_TITLE, CALIBRATION_VALUES_SHA256, METRIC_TITLES,
    _labels, generate, validate,
)

ROOT = Path(__file__).resolve().parents[1]


def _hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_original_scientific_values_and_eligibility_are_unchanged() -> None:
    calibration = ROOT / f"results/final_figures/rq2/values/{CALIBRATION}_values.csv"
    bootstrap = ROOT / f"results/final_figures/rq4/values/{BOOTSTRAP}_values.csv"
    assert hashlib.sha256(calibration.read_bytes()).hexdigest() == CALIBRATION_VALUES_SHA256
    assert hashlib.sha256(bootstrap.read_bytes()).hexdigest() == BOOTSTRAP_VALUES_SHA256
    cells = pd.read_csv(calibration, dtype=str, keep_default_na=False)
    assert len(cells) == 30
    assert (cells.eligible_user_count.astype(int) + cells.undefined_user_count.astype(int)
            == cells.planned_users.astype(int)).all()
    assert cells.cell_annotation.str.contains(r"\ne=\d+\nu=\d+", regex=True).all()
    decisions = pd.read_csv(bootstrap, dtype=str, keep_default_na=False)
    assert len(decisions) == 90
    assert decisions.orientation.eq("FIRST_MINUS_RANDOM").all()
    assert decisions.decision.value_counts().to_dict() == {
        "POSITIVE": 85, "INCONCLUSIVE": 4, "NEGATIVE": 1,
    }
    assert _labels(decisions) == {
        "hr_at_5": "HR@5 (pos=29 / neg=1 / inc=0)",
        "ndcg_at_5": "NDCG@5 (pos=28 / neg=0 / inc=2)",
        "mrr": "MRR (pos=28 / neg=0 / inc=2)",
    }


def test_render_polish_is_deterministic_and_only_titles_change(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate(ROOT, artifact_root=first)
    generate(ROOT, artifact_root=second)
    assert _hashes(first) == _hashes(second)
    assert len(_hashes(first)) == 10  # two unchanged values copies; six renders; two manifests
    manifests = validate(ROOT, artifact_root=first)
    assert {item["figure_id"] for item in manifests} == {CALIBRATION, BOOTSTRAP}
    for manifest in manifests:
        remediation = manifest["render_only_remediation"]
        assert remediation["base_figure_head"] == BASE_HEAD
        assert remediation["figure_values_sha256_unchanged"] == manifest["figure_values_sha256"]
        assert remediation["scope"] == "TYPOGRAPHY_ONLY_NO_SCIENTIFIC_VALUE_CHANGE"


def test_committed_polish_and_unrelated_manifests_validate() -> None:
    manifests = validate(ROOT)
    by_id = {item["figure_id"]: item for item in manifests}
    assert by_id[CALIBRATION]["render_only_remediation"]["visible_title"] == CALIBRATION_TITLE
    assert by_id[BOOTSTRAP]["render_only_remediation"]["panel_titles"] == _labels(rq4.bootstrap_values(ROOT))
    assert all(item["render_only_remediation"]["previous_output_sha256"]["png"] != item["output_png_sha256"]
               for item in manifests)
    assert set(METRIC_TITLES.values()) == {"HR@5", "NDCG@5", "MRR"}
    assert len(rq2_heatmaps.validate_all(ROOT)) == 4
    assert len(rq4.validate_all(ROOT)) == 3

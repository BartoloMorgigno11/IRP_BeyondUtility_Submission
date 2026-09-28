"""Read-only integrity checks for the corrected RQ3 traditional freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "results/rq3_traditional_scientific_freeze/freeze_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rq3_traditional_scientific_freeze_integrity() -> None:
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert manifest["status"] == "VALID"
    assert manifest["history_grid"] == [1, 2, 3, 4, 5, 10, 15, 20]
    assert manifest["repair"]["corrected_duplicate_keys"] == 12
    assert manifest["repair"]["training_executed"] is False
    assert manifest["repair"]["inference_executed"] is False

    corrected_path = ROOT / manifest["outputs"]["corrected_aggregate"]["path"]
    assert corrected_path.is_file()
    assert sha256(corrected_path) == manifest["outputs"]["corrected_aggregate"]["sha256"]
    corrected = pd.read_csv(corrected_path, dtype=str, keep_default_na=False)
    key_columns = ["dataset", "h", "method", "metric"]
    assert not corrected.duplicated(key_columns).any()

    affected = corrected.loc[
        corrected["dataset"].eq("beauty")
        & corrected["h"].eq("5")
        & corrected["method"].isin(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"])
    ]
    assert len(affected) == 12
    assert set(affected["n"].astype(int)) == {5}

    per_seed = pd.read_csv(ROOT / "results/tables/rq3_traditional_metrics_per_seed.csv", dtype=str, keep_default_na=False)
    for _, row in affected.iterrows():
        source = per_seed.loc[
            per_seed["dataset"].eq(row["dataset"])
            & per_seed["h"].eq(row["h"])
            & per_seed["method"].eq(row["method"])
            & per_seed["metric"].eq(row["metric"])
        ]
        assert set(source["seed"].astype(int)) == {42, 43, 44, 45, 46}
        assert len(source) == 5

    h1_na = corrected.loc[
        corrected["h"].eq("1") & corrected["method"].isin(["GRU4Rec", "SASRec"])
    ]
    assert len(h1_na) == 24
    assert set(h1_na["status"]) == {"NOT_APPLICABLE"}

    for relative, expected_hash in manifest["source_files"].items():
        assert sha256(ROOT / relative) == expected_hash

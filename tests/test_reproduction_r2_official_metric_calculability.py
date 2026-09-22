from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.analysis import reproduction_r2_official_metric_calculability as r2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return r2.build_frames(PROJECT_ROOT)


def test_master_rows_cover_r1_and_have_frozen_statuses() -> None:
    built = frames()
    master = built["master"]
    r1_decoded = r2.read_csv(PROJECT_ROOT / "results/tables/reproduction_r1_original_repo_metric_values_decoded.csv")

    assert len(master) >= len(r1_decoded)
    assert set(master["calculability_status"]).issubset(set(r2.CALCULABILITY_STATUSES))
    assert master["calculability_status"].astype(str).str.len().gt(0).all()
    assert not master["calculability_status"].isin(["UNKNOWN", "UNINVESTIGATED", "UNCLASSIFIED"]).any()


def test_every_core_dataset_and_metric_family_is_represented() -> None:
    master = frames()["master"]

    assert set(r2.TARGET_DATASETS).issubset(set(master["dataset"]))
    assert {"utility", "novelty", "popularity", "diversity", "fairness", "hallucination"}.issubset(
        set(master["metric_family"])
    )


def test_direct_recalculation_rows_have_inputs_values_and_provenance() -> None:
    direct = frames()["master"].loc[
        frames()["master"]["calculability_status"].eq("DIRECTLY_RECALCULABLE")
    ]

    assert not direct.empty
    assert direct["required_inputs"].astype(str).str.len().gt(0).all()
    assert direct["available_inputs"].astype(str).str.len().gt(0).all()
    assert direct["official_recomputed_value"].astype(str).str.len().gt(0).all()
    assert direct["official_source_member_sha256"].astype(str).str.len().ge(64).all()


def test_official_value_only_rows_preserve_stored_values() -> None:
    stored_only = frames()["master"].loc[
        frames()["master"]["calculability_status"].eq("OFFICIAL_VALUE_AVAILABLE_ONLY")
    ]

    assert not stored_only.empty
    assert stored_only["official_stored_value"].astype(str).str.len().gt(0).all()
    assert stored_only["missing_inputs"].astype(str).str.len().gt(0).all()


def test_not_calculable_rows_have_exact_reasons() -> None:
    reasons = frames()["not_calculable"]

    assert not reasons.empty
    assert reasons["exact_missing_input"].astype(str).str.len().gt(0).all()
    assert reasons["why_missing"].astype(str).str.len().gt(0).all()
    missing_data = reasons.loc[reasons["status"].eq("NOT_CALCULABLE_MISSING_DATA")]
    if not missing_data.empty:
        assert missing_data["exact_missing_input"].astype(str).str.len().gt(0).all()
    missing_output = reasons.loc[reasons["status"].eq("NOT_CALCULABLE_MISSING_OUTPUT")]
    if not missing_output.empty:
        assert missing_output["exact_missing_input"].str.contains("output|artifact", case=False, regex=True).any()


def test_ml1m_lastfm_do_not_use_local_artifacts() -> None:
    master = frames()["master"]
    subset = master.loc[master["dataset"].isin(["ML-1M", "LastFM"])]

    assert not subset.empty
    assert not subset["official_source_member"].str.contains("local|f6|processed", case=False, regex=True).any()
    assert subset["official_recomputed_value"].eq("").all()


def test_recalculated_values_have_stored_value_comparison() -> None:
    master = frames()["master"]
    stored_vs = frames()["stored_vs_recomputed"]
    recalculated = master.loc[master["official_recomputed_value"].astype(str).str.len().gt(0)]

    assert not recalculated.empty
    assert len(stored_vs) == len(recalculated)
    assert set(stored_vs["recomputation_agreement"]).issubset(
        {
            "RECOMPUTATION_EXACT",
            "RECOMPUTATION_ROUNDING_ONLY",
            "RECOMPUTATION_DIFFERENT",
            "RECOMPUTATION_NOT_COMPARABLE",
        }
    )


def test_official_source_hashes_map_to_r1_evidence() -> None:
    master = frames()["master"]
    r1_decoded = r2.read_csv(PROJECT_ROOT / "results/tables/reproduction_r1_original_repo_metric_values_decoded.csv")
    r1_hashes = set(r1_decoded["source_member_sha256"])
    used = {
        value
        for value in master["official_source_member_sha256"].astype(str)
        if len(value) == 64
    }

    assert used.issubset(r1_hashes)


def test_canddif_rows_are_labeled_as_derived_or_partial() -> None:
    candidate = frames()["candidate_position"]

    assert not candidate.empty
    assert set(candidate["canddif_status"]).issubset(set(r2.CANDDIF_STATUSES))
    valued = candidate.loc[candidate["value"].astype(str).str.len().gt(0)]
    if not valued.empty:
        assert valued["value_origin"].eq("DERIVED_FROM_OFFICIAL_ARTIFACTS").all()


def test_numeric_equality_does_not_upgrade_semantic_mapping() -> None:
    comparison = frames()["paper_comparison"]
    equal_rows = comparison.loc[comparison["value_agreement"].isin(["exact", "rounding_match"])]

    assert not equal_rows["paper_mapping_status"].eq("direct_match").any()


def test_protected_root_artifacts_and_original_repo_are_unchanged() -> None:
    built = frames()
    protected = built["protected_hash_check"]

    assert len(protected) == 12
    assert protected["protection_status"].eq("PROTECTED_UNCHANGED").all()
    assert r2.git_output(PROJECT_ROOT / r2.ORIGINAL_ROOT, ["status", "--branch", "--short"]) == "## main...origin/main"


def test_validation_table_passes_and_decision_is_complete() -> None:
    validation = frames()["validation"]

    assert validation["status"].eq("pass").all()
    assert r2.decision_token(validation) == r2.COMPLETE_TOKEN

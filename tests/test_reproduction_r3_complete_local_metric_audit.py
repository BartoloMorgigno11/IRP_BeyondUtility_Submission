from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.analysis import reproduction_r3_complete_local_metric_audit as r3


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return r3.build_frames(PROJECT_ROOT)


@lru_cache(maxsize=1)
def catalog() -> r3.MetricCatalog:
    config = r3.load_config(PROJECT_ROOT / r3.DEFAULT_CONFIG_PATH)
    return r3.build_metric_catalog(PROJECT_ROOT, config)


def test_four_original_datasets_are_represented() -> None:
    master = frames()["all_metric_cells_master"]

    assert set(r3.TARGET_DATASETS).issubset(set(master["dataset"]))


def test_every_e1_paper_metric_is_represented_where_applicable() -> None:
    master = frames()["all_metric_cells_master"]
    represented = {r3.normalize_metric(value) for value in master["metric_name"]}

    assert catalog().paper_metric_ids.issubset(represented)


def test_every_paper_relevant_cell_has_classified_status() -> None:
    master = frames()["all_metric_cells_master"]

    assert set(master["calculability_status"]).issubset(set(r3.CALCULABILITY_STATUSES))
    assert not master["calculability_status"].isin(["UNKNOWN", "UNINVESTIGATED", "UNCLASSIFIED"]).any()
    assert master.loc[master["paper_metric"].eq("true"), "calculability_status"].astype(str).str.len().gt(0).all()


def test_numeric_values_have_provenance() -> None:
    master = frames()["all_metric_cells_master"]
    numeric = master.loc[master["metric_value"].astype(str).str.len().gt(0)]

    assert not numeric.empty
    assert numeric["source_artifact"].astype(str).str.len().gt(0).all()
    assert numeric["value_origin"].astype(str).str.len().gt(0).all()


def test_recomputed_values_have_required_raw_inputs() -> None:
    recomputed = frames()["offline_recalculated_values"]

    assert not recomputed.empty
    assert recomputed["raw_input_artifacts"].astype(str).str.len().gt(0).all()
    assert recomputed["formula"].astype(str).str.len().gt(0).all()


def test_paper_exact_canddif_requires_first_random_evidence() -> None:
    candidate = frames()["candidate_position_bias"]
    exact = candidate.loc[candidate["classification"].eq("CANDDIF_PAPER_COMPATIBLE")]

    assert exact.empty or exact["has_first_random_evidence"].eq("true").all()


def test_position_diagnostics_are_not_mislabeled_canddif() -> None:
    candidate = frames()["candidate_position_bias"]
    diagnostics = candidate.loc[candidate["classification"].str.contains("DIAGNOSTIC", case=False, na=False)]

    assert not diagnostics.empty
    assert diagnostics["classification"].ne("CANDDIF_PAPER_COMPATIBLE").all()


def test_hallucination_semantics_are_separated() -> None:
    hallucination = frames()["hallucination_audit"]

    assert {
        "PAPER_STYLE_TEXTUAL_HALLUCINATION",
        "CANDIDATE_VIOLATION",
        "STRUCTURAL_OUTPUT_INVALIDITY",
    }.issubset(set(hallucination["hallucination_semantic_type"]))
    candidate_violation = hallucination.loc[hallucination["hallucination_semantic_type"].eq("CANDIDATE_VIOLATION")]
    assert candidate_violation["paper_comparability"].eq("DIAGNOSTIC_ONLY").all()


def test_generation_performance_requires_downstream_recommendation_output() -> None:
    generation = frames()["generation_involved_performance"]
    performance = generation.loc[generation["metric"].str.contains("HR|NDCG", case=False, na=False)]

    assert not performance.empty
    assert performance["has_downstream_recommendation_output"].eq("true").all()


def test_no_official_r1_r2_values_are_substituted_as_local_values() -> None:
    master = frames()["all_metric_cells_master"]
    official = master.loc[
        master["source_scope"].eq("OFFICIAL_REFERENCE")
        & master["calculability_status"].isin(["ALREADY_MATERIALIZED", "RECALCULATED_FROM_LOCAL_OUTPUTS"])
    ]

    assert official.empty


def test_local_extensions_not_counted_as_paper_reproduction() -> None:
    master = frames()["all_metric_cells_master"]
    local_extensions = master.loc[master["metric_family"].eq("local_extension")]

    assert not local_extensions.empty
    assert local_extensions["paper_metric"].eq("false").all()


def test_protected_phases_and_original_repo_remain_untouched() -> None:
    protected = frames()["protected_artifact_hash_check"]

    assert len(protected) == 12
    assert protected["protection_status"].eq("PROTECTED_UNCHANGED").all()
    assert r3.git_output(PROJECT_ROOT / r3.ORIGINAL_ROOT, ["status", "--branch", "--short"]) == "## main...origin/main"
    assert r3.git_output(PROJECT_ROOT / r3.ORIGINAL_ROOT, ["diff", "--stat"]) == ""


def test_validation_table_passes_and_decision_is_complete() -> None:
    validation = frames()["validation"]

    assert validation["status"].eq("pass").all(), validation.loc[validation["status"].ne("pass")].to_dict("records")
    assert r3.decision_token(validation) == r3.COMPLETE_TOKEN

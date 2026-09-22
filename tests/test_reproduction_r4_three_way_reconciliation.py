from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.reproduction_r4_three_way_reconciliation import (
    BEYOND_GROUPS,
    EXPECTED_HEAD,
    PRIORITIES,
    build_r4,
    normalize_metric,
    relevant_paper_metrics,
    load_metric_catalog,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"


@pytest.fixture(scope="session", autouse=True)
def r4_outputs() -> dict[str, pd.DataFrame]:
    result = build_r4(PROJECT_ROOT, write=True)
    assert result["decision"] == "REPRODUCTION_R4_THREE_WAY_RECONCILIATION_COMPLETE"
    return result["frames"]


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name, keep_default_na=False, low_memory=False)


def git(args: list[str], cwd: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def test_every_relevant_paper_metric_appears(r4_outputs: dict[str, pd.DataFrame]) -> None:
    e1 = read_table("extension_e1_metric_contract.csv")
    catalog = load_metric_catalog(e1)
    expected = set(relevant_paper_metrics(catalog))
    master_metrics = set(r4_outputs["three_way_master"]["metric_name"])
    assert expected.issubset(master_metrics)


def test_all_four_datasets_are_reconciled(r4_outputs: dict[str, pd.DataFrame]) -> None:
    expected = {"Beauty", "Sports", "ML-1M", "LastFM"}
    assert expected.issubset(set(r4_outputs["three_way_master"]["dataset"]))
    assert expected == set(r4_outputs["beyond_summary"]["dataset"])
    assert expected.issubset(set(r4_outputs["gap_priority"]["dataset"]))


def test_value_sources_remain_distinct(r4_outputs: dict[str, pd.DataFrame]) -> None:
    master = r4_outputs["three_way_master"]
    required = {
        "paper_value",
        "official_stored_value",
        "official_recomputed_value",
        "official_reconstructed_value",
        "official_value",
        "local_stored_value",
        "local_recomputed_value",
        "local_reconstructed_value",
        "local_value",
    }
    assert required.issubset(master.columns)
    assert not master["official_value_origin"].str.contains("PAPER", case=False, na=False).any()
    assert not master["local_value_origin"].str.contains("OFFICIAL|PAPER", case=False, regex=True, na=False).any()


def test_no_missing_local_value_is_substituted(r4_outputs: dict[str, pd.DataFrame]) -> None:
    master = r4_outputs["three_way_master"]
    missing = master.loc[master["local_evidence_class"] == "MISSING"]
    assert not missing.empty
    assert missing["local_value"].eq("").all()
    assert missing["local_stored_value"].eq("").all()
    assert missing["local_recomputed_value"].eq("").all()
    assert missing["local_reconstructed_value"].eq("").all()


def test_no_official_value_is_substituted_from_paper(r4_outputs: dict[str, pd.DataFrame]) -> None:
    master = r4_outputs["three_way_master"]
    official_values = master.loc[master["official_value"].str.strip().ne("")]
    assert not official_values.empty
    assert official_values["official_value_origin"].str.startswith("OFFICIAL_").all()
    assert not official_values["official_value_origin"].str.contains("PAPER", case=False, na=False).any()


def test_adapted_and_diagnostic_cells_are_not_exact(r4_outputs: dict[str, pd.DataFrame]) -> None:
    master = r4_outputs["three_way_master"]
    adapted_exact = master.loc[
        (master["local_evidence_class"] == "ADAPTED")
        & (master["comparability_class"] == "PAPER_OFFICIAL_LOCAL_EXACT_PROTOCOL")
    ]
    diagnostic_exact = master.loc[
        (master["local_evidence_class"] == "DIAGNOSTIC_ONLY")
        & master["comparability_class"].str.contains("EXACT", na=False)
    ]
    assert adapted_exact.empty
    assert diagnostic_exact.empty


def test_candidate_position_diagnostics_are_not_called_canddif(r4_outputs: dict[str, pd.DataFrame]) -> None:
    candidate = r4_outputs["candidate_position"]
    assert candidate["local_exact_canddif"].astype(int).sum() == 0
    assert candidate["local_position_diagnostics"].astype(int).sum() == 3416
    assert candidate["local_canddif_diagnostics"].astype(int).sum() == 5
    assert candidate["remaining_canddif_gaps"].astype(int).sum() == 6
    ml1m = candidate.loc[candidate["dataset"].eq("ML-1M")].iloc[0]
    assert int(ml1m["remaining_canddif_gaps"]) == 0
    assert int(ml1m["local_canddif_diagnostics"]) == 4
    assert ml1m["new_inference_required_for_paper_compatible_local_canddif"] == "false"


def test_hallucination_semantics_remain_separated(r4_outputs: dict[str, pd.DataFrame]) -> None:
    hallucination = r4_outputs["hallucination"]
    assert {
        "local_paper_style_textual_values",
        "local_candidate_violations",
        "local_structural_invalidity",
    }.issubset(hallucination.columns)
    r3_hallucination = read_table("reproduction_r3_hallucination_audit.csv")
    semantic_types = set(r3_hallucination["hallucination_semantic_type"])
    assert {
        "PAPER_STYLE_TEXTUAL_HALLUCINATION",
        "CANDIDATE_VIOLATION",
        "STRUCTURAL_OUTPUT_INVALIDITY",
    }.issubset(semantic_types)


def test_profile_generation_support_rows_are_not_performance(r4_outputs: dict[str, pd.DataFrame]) -> None:
    generation = r4_outputs["generation"]
    assert generation["profile_only_support_rows"].astype(int).sum() == 160
    assert generation["downstream_rows"].astype(int).sum() == 200
    ml_last = generation.loc[generation["dataset"].isin(["ML-1M", "LastFM"])]
    assert ml_last["local_downstream_performance_values"].eq("no downstream local value").all()


def test_all_54_gaps_are_prioritized_with_exact_vocabulary(r4_outputs: dict[str, pd.DataFrame]) -> None:
    gaps = r4_outputs["gap_priority"]
    assert len(gaps) == 54
    assert set(gaps["priority"]).issubset(set(PRIORITIES))
    assert gaps["priority"].str.strip().ne("").all()
    assert gaps["gap_id"].is_unique


def test_high_priority_recovery_has_exact_reason_and_plan_only(r4_outputs: dict[str, pd.DataFrame]) -> None:
    gaps = r4_outputs["gap_priority"]
    high = gaps.loc[gaps["priority"].isin(["P0_MUST_CLOSE", "P1_HIGH_VALUE"])]
    proposed = r4_outputs["proposed_recovery_experiments"]
    recovery_counts = r4_outputs["recovery_request_count_validation"]
    assert len(high) == 14
    assert high["exact_reason"].str.strip().ne("").all()
    assert len(proposed) == 14
    assert proposed["reason"].str.strip().ne("").all()
    assert proposed["execution_status"].eq("PLANNED_ONLY_NOT_EXECUTED").all()
    assert not high.loc[(high["dataset"] == "ML-1M") & high["metric"].str.startswith("CandDif")].any().any()

    counts = recovery_counts.set_index("family")
    assert counts.loc["CandDif", "raw_planned_requests"] == "2400"
    assert counts.loc["CandDif", "duplicate_requests"] == "1200"
    assert counts.loc["CandDif", "unique_new_requests"] == "1200"
    assert counts.loc["History", "unique_new_requests"] == "2400"
    assert counts.loc["Generation", "unique_new_requests"] == "800"
    assert counts["count_status"].eq("PLANNING_ESTIMATE").all()
    assert counts["unique_new_requests"].astype(int).sum() == 4400


def test_no_proposed_experiment_has_existing_equivalent_local_output(r4_outputs: dict[str, pd.DataFrame]) -> None:
    proposed = r4_outputs["proposed_recovery_experiments"]
    local = read_table("reproduction_r3_all_metric_cells_master.csv")
    local["_metric"] = local["metric_name"].map(normalize_metric)
    calculable = local.loc[
        local["calculability_status"].isin(["ALREADY_MATERIALIZED", "RECALCULATED_FROM_LOCAL_OUTPUTS"])
    ]
    for row in proposed.itertuples():
        equivalent = calculable.loc[
            (calculable["dataset"] == row.dataset)
            & (calculable["_metric"] == row.metric)
            & (calculable["paper_comparability"] == "PAPER_EXACT")
        ]
        assert equivalent.empty


def test_beyond_summary_has_requested_metric_families(r4_outputs: dict[str, pd.DataFrame]) -> None:
    beyond = r4_outputs["beyond_summary"]
    assert set(BEYOND_GROUPS) == set(beyond["metric"])
    assert len(beyond) == 60
    assert not beyond["reproduction_class"].eq("EXACTLY_REPRODUCED").any()


def test_no_new_inference_and_frozen_layers_unchanged(r4_outputs: dict[str, pd.DataFrame]) -> None:
    validation = r4_outputs["validation"]
    assert validation["status"].eq("pass").all()
    protected = r4_outputs["protected_hash_check"]
    assert len(protected) == 12
    assert protected["protection_status"].eq("PROTECTED_UNCHANGED").all()
    assert git(["rev-parse", "HEAD"]) == EXPECTED_HEAD
    assert git(["diff", "--cached", "--stat"]) == ""
    original = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender"
    assert git(["diff", "--stat"], cwd=original) == ""

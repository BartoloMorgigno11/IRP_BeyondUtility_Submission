from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.extension import f1ar_offline_closure_sanity as f1ar


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _outputs() -> dict:
    return f1ar.generate_f1ar(ROOT)


def test_f1ar_validates_and_computes_original_four_ks_pairs() -> None:
    outputs = _outputs()
    recoverability = outputs["ks_recoverability_audit"]
    results = outputs["ks_original_four_results"]

    assert outputs["validation_passed"], outputs["validation"].loc[outputs["validation"]["status"].ne("PASS")].to_dict("records")
    assert set(recoverability["dataset"]) == {"beauty", "sports", "ml1m", "lastfm"}
    assert recoverability["vector_reconstructible"].eq("true").all()
    assert recoverability["final_status"].eq("COMPUTABLE_OFFLINE").all()
    assert len(results) == 8
    assert set(results["comparison"]) == {"primary_vs_eligible_population", "history200_vs_strict_l20_population"}
    assert results["computation_status"].eq("COMPUTED_OFFLINE").all()
    assert pd.to_numeric(results["sample_N"]).gt(0).all()
    assert pd.to_numeric(results["population_N"]).ge(pd.to_numeric(results["sample_N"])).all()


def test_recoverability_counts_match_frozen_primary_and_h0_cohorts() -> None:
    recoverability = _outputs()["ks_recoverability_audit"].set_index("dataset")

    assert recoverability.loc["beauty", "primary_sample_N"] == 990
    assert recoverability.loc["sports", "primary_sample_N"] == 892
    assert recoverability.loc["ml1m", "primary_sample_N"] == 1000
    assert recoverability.loc["lastfm", "primary_sample_N"] == 1000
    assert recoverability["history_sample_N"].astype(int).eq(200).all()
    assert recoverability.loc["beauty", "history_population_N"] == 1105
    assert recoverability.loc["sports", "history_population_N"] == 1071
    assert recoverability.loc["ml1m", "history_population_N"] == 5954
    assert recoverability.loc["lastfm", "history_population_N"] == 1720
    assert recoverability.loc["lastfm", "primary_population_N"] == 1859
    assert (recoverability["history_cohort_user_hash"] == recoverability["h0_expected_history_cohort_hash"]).all()
    assert recoverability["missing_material_if_any"].eq("").all()


def test_ks_values_are_stable_and_non_overclaimed() -> None:
    results = _outputs()["ks_original_four_results"]
    keyed = {(row.dataset, row.comparison): row for row in results.itertuples(index=False)}

    assert float(keyed[("beauty", "primary_vs_eligible_population")].ks_statistic) == 0.0
    assert float(keyed[("sports", "primary_vs_eligible_population")].ks_statistic) == 0.0
    assert float(keyed[("ml1m", "primary_vs_eligible_population")].ks_statistic) == 0.0
    assert float(keyed[("lastfm", "primary_vs_eligible_population")].ks_statistic) > 0.0
    assert results["evidence_class"].eq("LOCAL_EXTENSION;PAPER_ALIGNED_RECONSTRUCTED_TARGET").all()
    assert results["notes"].str.contains("not paper-exact", case=False).all()


def test_final_blockers_reflect_current_recovery_state() -> None:
    outputs = _outputs()
    blockers = outputs["final_metric_blockers"]

    if (ROOT / "results/manifests/f1arr_popreo_dpd_recovery.json").exists():
        assert blockers.empty
        assert outputs["manifest"]["final_metric_blockers"] == []
    else:
        assert blockers["metric"].tolist() == ["popreo_at_5", "dpd_at_5"]
        assert blockers["blocker_type"].eq("BLOCKED_GLOBAL_PROTOCOL").all()
        assert blockers["reason"].str.contains("zero|averaging|normalization|denominator", case=False, regex=True).all()


def test_cost_wording_preserves_scientific_distinctions() -> None:
    cost = _outputs()["cost_wording_audit"].set_index("cost_dimension")

    assert cost.loc["paid_api_marginal_monetary_cost", "f1a_status"] == "NOT_APPLICABLE"
    assert "No paid API provider" in cost.loc["paid_api_marginal_monetary_cost", "wording"]
    assert cost.loc["total_computational_cost", "f1a_status"] == "NOT_ZERO_BUT_UNPRICED"
    assert cost.loc["hardware_electricity_cost", "f1a_status"] == "UNMEASURED"


def test_f1ar_preserves_f0_budget_and_zero_execution_counters() -> None:
    outputs = _outputs()
    manifest = outputs["manifest"]
    validation = outputs["validation"]

    assert validation.loc[validation["check"].eq("f0_request_budget_unchanged"), "status"].iloc[0] == "PASS"
    assert manifest["avoidable_ks_blockers"] == 0
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["training_runs_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0


def test_f1ar_source_has_no_model_network_or_training_imports() -> None:
    source = (ROOT / "src/extension/f1ar_offline_closure_sanity.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_roots = {"requests", "urllib", "socket", "httpx", "openai", "anthropic", "ollama", "torch", "tensorflow"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(banned_roots)
    assert "phase_snapshot" not in source.lower()

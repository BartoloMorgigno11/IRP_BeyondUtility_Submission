from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.extension import g2_history_length_sensitivity_protocol as g2
from src.extension.e0_prompts import render_extension_prompt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g2_history_length_sensitivity.yaml"


@lru_cache(maxsize=1)
def _outputs() -> dict:
    return g2.build_protocol_freeze(ROOT, CONFIG)


def _table(name: str) -> pd.DataFrame:
    return _outputs()["tables"][name]


def test_g2_config_freezes_no_inference_dimensions() -> None:
    config = g2.load_config(ROOT, CONFIG)

    assert config["models"] == list(g2.MODEL_ORDER)
    assert config["datasets"] == list(g2.DATASET_ORDER)
    assert config["prompt_strategies"] == list(g2.PROMPT_ORDER)
    assert config["history_lengths"] == list(g2.HISTORY_ORDER)
    assert config["phase"]["allow_commit"] is False

    forbidden = [
        "allow_local_ollama_inference",
        "allow_local_ollama_metadata",
        "allow_external_api_calls",
        "allow_downloads",
        "allow_network_downloads",
        "allow_training",
        "allow_tuning",
        "allow_g3",
        "allow_g4",
        "allow_g5",
    ]
    assert all(config["execution"][key] is False for key in forbidden)


def test_history_truncation_is_suffix_nested_and_target_excluded() -> None:
    cohort = _table("cohort")
    row = cohort.loc[cohort["b2_long_history_ge20_eligible"].eq("true")].iloc[0]

    h0 = json.loads(row["history_h0_item_ids"])
    h1 = json.loads(row["history_h1_item_ids"])
    h2 = json.loads(row["history_h2_item_ids"])
    h5 = json.loads(row["history_h5_item_ids"])
    h10 = json.loads(row["history_h10_item_ids"])
    h20 = json.loads(row["history_h20_item_ids"])

    assert h0 == []
    assert h1 == h2[-1:]
    assert h2 == h5[-2:]
    assert h5 == h10[-5:]
    assert h10 == h20[-10:]
    assert row["target_excluded_from_full_train_history"] == "true"
    assert row["g1_h10_history_hash_match"] == "true"


def test_same_candidate_order_and_target_across_history_lengths() -> None:
    universe = _table("request_universe")
    primary = universe.loc[
        (universe["cohort_type"].eq(g2.PRIMARY_COHORT_TYPE))
        & (universe["dataset"].eq("ml1m"))
        & (universe["prompt_strategy"].eq("BASE"))
        & (universe["model"].eq("qwen2:7b-instruct"))
    ]
    user_id = primary.iloc[0]["user_id"]
    rows = primary.loc[primary["user_id"].eq(user_id)]

    assert set(rows["history_length"].astype(int)) == set(g2.HISTORY_ORDER)
    assert rows["candidate_hash"].nunique() == 1
    assert rows["candidate_items"].nunique() == 1
    assert rows["positive_item_id"].nunique() == 1
    assert rows["target_excluded_from_history"].eq("true").all()


def test_llmrank_h0_is_frozen_local_equivalent_without_empty_last_item() -> None:
    universe = _table("request_universe")
    h0_llmrank = universe.loc[
        universe["prompt_strategy"].eq("LLMRANK") & universe["history_length"].astype(int).eq(0)
    ]

    assert not h0_llmrank.empty
    assert h0_llmrank["llmrank_h0_status"].eq(g2.LLMRANK_H0_STATUS).all()

    candidate_items = [{"candidate_position": i, "item_id": str(i), "title": f"Item {i}"} for i in range(1, 21)]
    bundle = render_extension_prompt(
        prompt_strategy="LLMRANK",
        dataset_label="Test",
        candidate_items=candidate_items,
        history_items=[],
        history_length=0,
    )
    assert "No recency signal is available because no user history is visible." in bundle.user_message
    assert "most recent visible interaction:" not in bundle.user_message


def test_foodcom_controlled_and_true_zero_conditions_are_separate() -> None:
    universe = _table("request_universe")
    applicability = _table("applicability")

    food_h0 = universe.loc[universe["dataset"].eq("foodcom") & universe["history_length"].astype(int).eq(0)]
    assert {g2.TRUE_ZERO_LABEL, "WARM_GE10"} <= set(food_h0["population_label"])
    assert {
        g2.TRUE_ZERO_LABEL,
        f"{g2.CONTROLLED_PREFIX}0",
    } <= set(food_h0["history_condition_label"])

    true_zero_universe = universe.loc[universe["cohort_type"].eq(g2.TRUE_ZERO_COHORT_TYPE)]
    assert set(true_zero_universe["history_length"].astype(int)) == {0}
    true_zero_not_applicable = applicability.loc[
        applicability["cohort_type"].eq(g2.TRUE_ZERO_COHORT_TYPE)
        & applicability["history_length"].astype(int).gt(0)
    ]
    assert len(true_zero_not_applicable) == 30
    assert true_zero_not_applicable["applicability_status"].eq(g2.NOT_APPLICABLE).all()


def test_primary_cohort_counts_are_deterministic() -> None:
    eligibility = _table("eligibility").set_index("dataset")

    assert eligibility["users_ge_10"].astype(int).to_dict() == g2.EXPECTED_B1_USERS_BY_DATASET
    assert eligibility["users_ge_20"].astype(int).to_dict() == g2.EXPECTED_B2_USERS_BY_DATASET
    assert int(eligibility.loc["foodcom", "foodcom_true_zero_history_users"]) == g2.EXPECTED_TRUE_ZERO_FOODCOM_USERS


def test_nested_membership_and_h10_identity_are_deduplicated() -> None:
    universe = _table("request_universe")
    warm_h10 = universe.loc[
        universe["cohort_type"].eq(g2.PRIMARY_COHORT_TYPE)
        & universe["history_length"].astype(int).eq(10)
    ]
    ge20_h10 = warm_h10.loc[warm_h10["b2_long_history_ge20_member"].eq("true")]

    identity_columns = ["dataset", "user_id", "history_length", "prompt_strategy", "model", "population_label"]
    assert not universe.duplicated(identity_columns).any()
    assert not ge20_h10.empty
    assert ge20_h10["reporting_view_memberships"].str.contains(g2.PRIMARY_VIEW, regex=False).all()
    assert ge20_h10["reporting_view_memberships"].str.contains(g2.LONG_VIEW, regex=False).all()
    assert ge20_h10["reporting_view_memberships"].str.contains(g2.SECONDARY_VIEW, regex=False).all()


def test_secondary_full_curve_creates_no_new_logical_requests() -> None:
    universe = _table("request_universe")
    secondary = universe.loc[universe["secondary_full_curve_ge20_member"].eq("true")]

    assert set(secondary["history_length"].astype(int)) == set(g2.HISTORY_ORDER)
    assert secondary.loc[secondary["history_length"].astype(int).isin([0, 1, 2, 5]), "b1_primary_ge10_member"].eq("true").all()
    assert secondary.loc[secondary["history_length"].astype(int).isin([10, 20]), "b2_long_history_ge20_member"].eq("true").all()


def test_g1_exact_reuse_is_only_b1_h10_and_foodcom_true_zero_h0() -> None:
    universe = _table("request_universe")
    reusable = universe.loc[universe["planned_action"].eq(g2.G1_REUSE_ACTION)]

    assert len(reusable) == 18660
    assert len(universe.loc[universe["planned_action"].eq(g2.NEW_CALL_ACTION)]) == 69390
    assert reusable.loc[reusable["cohort_type"].eq(g2.PRIMARY_COHORT_TYPE), "history_length"].astype(int).eq(10).all()
    true_zero = reusable.loc[reusable["cohort_type"].eq(g2.TRUE_ZERO_COHORT_TYPE)]
    assert true_zero["history_length"].astype(int).eq(0).all()
    assert true_zero["population_label"].eq(g2.TRUE_ZERO_LABEL).all()


def test_prior_history_evidence_is_comparative_not_reusable() -> None:
    prior = _table("prior_evidence_audit")

    assert prior["exists"].eq("true").all()
    assert prior["reuse_status"].eq(g2.PRIOR_REUSE_STATUS).all()
    assert prior["reusable_g2_terminal_evidence"].eq("false").all()
    r5c = prior.loc[prior["source_id"].eq("R5C_HISTORY_LENGTH_CORRECTED")].iloc[0]
    assert "r5c_ranked_index_json_top5_v1" in r5c["parser_contracts_observed"]
    assert "parser_contract_differs_from_g2_strict_top10" in r5c["rejection_reasons"]


def test_request_accounting_totals_are_frozen() -> None:
    account = _table("request_accounting")
    grand = account.loc[account["accounting_scope"].eq("GRAND_TOTAL")].iloc[0]

    assert int(grand["planned_applicable_requests"]) == 88050
    assert int(grand["g1_exact_reuse_requests"]) == 18660
    assert int(grand["new_future_ollama_calls_required"]) == 69390
    assert int(grand["not_applicable_user_conditions"]) == 21810

    by_dataset = {
        row["dataset"]: {
            "total": int(row["planned_applicable_requests"]),
            "reuse": int(row["g1_exact_reuse_requests"]),
            "new": int(row["new_future_ollama_calls_required"]),
        }
        for row in account.loc[account["accounting_scope"].eq("DEDUP_BY_DATASET")].to_dict("records")
    }
    assert by_dataset == g2.EXPECTED_TOTALS_BY_DATASET

    by_model = account.loc[account["accounting_scope"].eq("DEDUP_BY_MODEL")]
    assert set(zip(by_model["planned_applicable_requests"].astype(int), by_model["g1_exact_reuse_requests"].astype(int), by_model["new_future_ollama_calls_required"].astype(int))) == {
        (29350, 6220, 23130)
    }

    by_prompt = account.loc[account["accounting_scope"].eq("DEDUP_BY_PROMPT")]
    assert set(zip(by_prompt["planned_applicable_requests"].astype(int), by_prompt["g1_exact_reuse_requests"].astype(int), by_prompt["new_future_ollama_calls_required"].astype(int))) == {
        (44025, 9330, 34695)
    }


def test_protocol_source_has_no_inference_or_network_entrypoint() -> None:
    source = (ROOT / "src" / "extension" / "g2_history_length_sensitivity_protocol.py").read_text(encoding="utf-8")
    manifest = _outputs()["manifest"]

    assert "call_ollama_generate(" not in source
    assert "urllib.request" not in source
    assert manifest["no_inference_guards"]["llm_inference_calls_executed"] == 0
    assert manifest["no_inference_guards"]["local_ollama_metadata_calls_executed"] == 0
    assert manifest["no_inference_guards"]["external_api_calls_executed"] == 0
    assert manifest["no_inference_guards"]["metrics_recomputed"] is False

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest

from src.extension import g3_candidate_position_execution as g3


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _outputs() -> dict:
    config = g3.load_config(ROOT, g3.CONFIG_PATH)
    return g3.build_all(ROOT, config)


def test_import_has_no_side_effects_or_remote_clients() -> None:
    source = Path(g3.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert {"openai", "anthropic", "requests", "socket"}.isdisjoint(imports)


def test_protocol_manifest_and_accounting_are_frozen() -> None:
    outputs = _outputs()
    manifest = outputs["protocol_manifest"]
    accounting = outputs["tables"]["request_accounting"]
    grand = accounting.loc[accounting["accounting_scope"].eq("GRAND_TOTAL")].iloc[0]

    assert manifest["decision_token"] == g3.DECISION_FROZEN
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["g3_execution_started"] is False
    assert manifest["llmsrec_excluded"] is True
    assert int(grand["logical_requests"]) == 12000
    assert int(grand["new_ollama_calls_required"]) == 12000
    assert int(grand["exact_g1_g2_reuse"]) == 0


def test_cohort_is_200_users_per_dataset_and_shared() -> None:
    cohort = _outputs()["tables"]["cohort"]
    assert cohort.groupby("dataset")["user_id"].nunique().to_dict() == {dataset: 200 for dataset in g3.DATASET_ORDER}
    assert cohort["same_users_across_models_prompts_orders"].eq("true").all()
    assert cohort["outcome_dependent_filtering"].eq("false").all()
    assert cohort["candidate_count"].astype(int).eq(20).all()
    assert cohort["target_excluded_from_history"].eq("true").all()


def test_request_universe_dimensions_and_contracts() -> None:
    universe = _outputs()["tables"]["request_universe"]
    assert len(universe) == 5 * 3 * 2 * 200 * 2
    assert universe["canonical_request_id"].is_unique
    assert set(universe["dataset"]) == set(g3.DATASET_ORDER)
    assert set(universe["model"]) == set(g3.MODEL_ORDER)
    assert set(universe["prompt_strategy"]) == set(g3.PROMPT_ORDER)
    assert set(universe["candidate_order_condition"]) == set(g3.ORDER_CONDITIONS)
    assert set(universe["parser_contract"]) == {g3.PARSER_CONTRACT_ID}
    assert set(universe["requested_output_length"].astype(int)) == {10}
    assert set(universe["parser_expected_length"].astype(int)) == {10}
    assert universe["exact_g1_g2_reuse"].eq("false").all()
    assert universe["planned_action"].eq("NEW_G3_OLLAMA_CALL_REQUIRED").all()


def test_first_and_random_candidate_order_contracts() -> None:
    universe = _outputs()["tables"]["request_universe"]
    first = universe.loc[universe["candidate_order_condition"].eq(g3.ORDER_FIRST)]
    random = universe.loc[universe["candidate_order_condition"].eq(g3.ORDER_RANDOM)]
    assert first["positive_candidate_position"].astype(int).eq(1).all()
    assert random["positive_candidate_position"].astype(int).between(2, 20).all()
    assert random["candidate_order_algorithm"].eq(g3.RANDOM_ALGORITHM).all()
    assert random["candidate_order_seed"].eq(g3.RANDOM_SEED).all()


def test_pair_integrity_says_only_candidate_order_differs() -> None:
    pair_integrity = _outputs()["tables"]["pair_integrity"]
    assert len(pair_integrity) == 5 * 3 * 2 * 200
    assert pair_integrity["has_exactly_first_and_random"].eq("true").all()
    assert pair_integrity["same_candidate_membership"].eq("true").all()
    assert pair_integrity["same_history"].eq("true").all()
    assert pair_integrity["same_target"].eq("true").all()
    assert pair_integrity["candidate_order_differs"].eq("true").all()
    assert pair_integrity["only_candidate_order_differs"].eq("true").all()


def test_statistical_plan_reuses_frozen_bootstrap_convention() -> None:
    plan = _outputs()["tables"]["statistical_plan"]
    assert len(plan) == 5 * 3 * 2 * 3
    assert set(plan["metric"]) == {"hr_at_5", "ndcg_at_5", "mrr"}
    assert set(plan["condition_a"]) == {g3.ORDER_RANDOM}
    assert set(plan["condition_b"]) == {g3.ORDER_FIRST}
    assert set(plan["contrast"]) == {"FIRST_minus_RANDOM"}
    assert set(plan["bootstrap_resamples"].astype(int)) == {10000}
    assert set(plan["bootstrap_seed"].astype(int)) == {20260901}
    assert set(plan.loc[plan["metric"].isin(["hr_at_5", "ndcg_at_5"]), "canddif_formula"]) == {
        "log((1-Acc_random)/(1-Acc_first))"
    }
    assert set(plan.loc[plan["metric"].eq("mrr"), "canddif_metric"]) == {"not_applicable"}


def test_execution_freeze_is_zero_call_dry_run_ready() -> None:
    outputs = _outputs()
    execution = outputs["execution_manifest"]
    order = outputs["tables"]["execution_order"]
    accounting = outputs["tables"]["execution_accounting"].iloc[0]
    assert execution["decision_token"] == g3.EXECUTION_DECISION_FROZEN
    assert execution["llm_inference_calls_executed"] == 0
    assert execution["zero_inference"] is True
    assert len(order) == 12000
    assert int(accounting["existing_g3_terminal_requests"]) == 0
    assert int(accounting["pending_new_requests"]) == 12000
    assert int(accounting["new_ollama_calls_in_dry_run"]) == 0


def test_validation_all_pass_and_llmsrec_absent() -> None:
    validation = _outputs()["tables"]["validation"]
    universe = _outputs()["tables"]["request_universe"]
    assert validation["status"].eq("PASS").all(), validation.loc[validation["status"].eq("FAIL")].to_dict("records")
    assert not universe.astype(str).apply(lambda col: col.str.contains("LLMSRec|llmsrec", regex=True).any()).any()


def test_live_execution_requires_explicit_api_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert g3.main(["--config", str(g3.CONFIG_PATH), "--no-write", "--max-new-calls", "0"]) == 1
    captured = capsys.readouterr()
    assert "--allow-api-calls is required" in captured.out

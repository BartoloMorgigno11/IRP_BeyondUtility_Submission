from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f6d3_lastfm_llm_preflight import (
    COHORT_POLICY_COLUMNS,
    EVALUATION_POLICY_COLUMNS,
    GO_NO_GO_COLUMNS,
    LOCAL_LLM_COLUMNS,
    MANIFEST_COLUMNS,
    MODEL_ADEQUACY_COLUMNS,
    OUTPUT_TABLES,
    POSITIVE_COLUMNS,
    PROMPT_CONTRACT_COLUMNS,
    RERANKING_PROTOCOL_COLUMNS,
    TRAINING_ADEQUACY_COLUMNS,
    UPPER_BOUND_COLUMNS,
    build_cohort_policy,
    build_evaluation_policy,
    build_go_no_go,
    build_local_llm_inventory,
    build_model_adequacy_summary,
    build_pool_composition,
    build_positive_inclusion_summary,
    build_prompt_contract,
    build_request_manifests,
    build_reranking_protocol,
    build_training_adequacy,
    build_upper_bounds,
    frame_sha256,
    no_absolute_paths,
    previous_phase_input_status,
    protected_hash_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_inputs() -> dict[str, pd.DataFrame]:
    manifest = pd.DataFrame(
        [
            {
                "execution_id": "F6D2-RUN-BPRMF-SEED42",
                "model": "bprmf",
                "seed": "42",
                "epoch_limit": "1",
                "epochs_completed": "1",
                "best_epoch": "1",
                "best_validation_metric": "0.1",
                "checkpoint_file": "tmp/checkpoint.pt",
            }
        ]
    )
    curves = pd.DataFrame(
        [
            {
                "execution_id": "F6D2-RUN-BPRMF-SEED42",
                "model": "bprmf",
                "seed": "42",
                "epoch": "1",
                "train_loss": "0.6",
                "validation_metric": "0.1",
                "is_best_epoch": "true",
            }
        ]
    )
    source = pd.DataFrame(
        [
            {
                "model": "bprmf",
                "seed": "42",
                "selection_metric": "validation_hr_at_5_full_catalog",
                "selection_reason": "Highest validation HR@5; tie broken by lower seed; no test metric used.",
                "selected_for_adapted_pool": "true",
            }
        ]
    )
    train = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1"},
            {"user_id": "u1", "item_id": "i2"},
            {"user_id": "u2", "item_id": "i2"},
            {"user_id": "u2", "item_id": "i3"},
        ]
    )
    catalog = pd.DataFrame({"item_id": ["i1", "i2", "i3", "i4"], "title": ["A", "B", "C", "D"]})
    ranking = pd.DataFrame(
        [
            {"user_id": "u1", "candidate_item_id": "i3", "ground_truth_item_id": "i4", "candidate_position": "1"},
            {"user_id": "u1", "candidate_item_id": "i4", "ground_truth_item_id": "i4", "candidate_position": "2"},
            {"user_id": "u2", "candidate_item_id": "i1", "ground_truth_item_id": "i4", "candidate_position": "1"},
            {"user_id": "u2", "candidate_item_id": "i4", "ground_truth_item_id": "i4", "candidate_position": "2"},
        ]
    )
    pool = pd.DataFrame(
        [
            {"user_id": "u1", "candidate_item_id": "i3", "candidate_position": "1", "source_models": "bprmf", "source_ranks": "1", "source_scores": "1.0", "source_seed": "42", "is_positive": "false", "positive_item_id": "i4"},
            {"user_id": "u1", "candidate_item_id": "i4", "candidate_position": "2", "source_models": "gru4rec;sasrec", "source_ranks": "2;3", "source_scores": "0.9;0.8", "source_seed": "42;43", "is_positive": "true", "positive_item_id": "i4"},
            {"user_id": "u2", "candidate_item_id": "i1", "candidate_position": "1", "source_models": "bprmf", "source_ranks": "1", "source_scores": "1.0", "source_seed": "42", "is_positive": "false", "positive_item_id": "i4"},
            {"user_id": "u2", "candidate_item_id": "i2", "candidate_position": "2", "source_models": "lightgcn", "source_ranks": "2", "source_scores": "0.9", "source_seed": "42", "is_positive": "false", "positive_item_id": "i4"},
        ]
    )
    prompt = pd.DataFrame(
        [
            {"prompt_contract_id": "llm_rank_index_json_top5_v1", "paper_source": "paper", "official_code_source": "code", "local_source": "local", "history_item_representation": "title", "candidate_representation": "index", "candidate_order_policy": "preserve", "requested_output_representation": "json"},
            {"prompt_contract_id": "llm_rerank_index_json_full20_v1", "paper_source": "paper", "official_code_source": "code", "local_source": "local", "history_item_representation": "title", "candidate_representation": "index", "candidate_order_policy": "preserve", "requested_output_representation": "json"},
        ]
    )
    parser = pd.DataFrame(
        [
            {"parser_contract_id": "llm_rank_strict_primary_parser_v1", "invalid_output_policy": "mark_invalid"},
            {"parser_contract_id": "llm_rerank_strict_primary_parser_v1", "invalid_output_policy": "mark_invalid"},
        ]
    )
    execution = pd.DataFrame(
        [
            {"model_family": family, "task_type": task, "temperature": "0", "top_p": "", "seed": "", "max_output_tokens": "256", "infrastructure_retry_count": "0", "scientific_retry_count": "0"}
            for family in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]
            for task in ["ranking", "reranking"]
        ]
    )
    model_inventory = pd.DataFrame(
        [
            {"local_model_tag": "qwen2:7b-instruct"},
            {"local_model_tag": "mistral:7b-instruct"},
        ]
    )
    return {
        "f6d2_manifest": manifest,
        "f6d2_curves": curves,
        "f6d2_source_selection": source,
        "train": train,
        "catalog": catalog,
        "ranking_candidates": ranking,
        "f6d2_pool": pool,
        "prompt_contract": prompt,
        "parser_contract": parser,
        "execution_contract": execution,
        "model_inventory": model_inventory,
        "provenance": pd.DataFrame([{"expected_request_count": "10"}]),
    }


def test_training_curve_parsing_epoch_and_best_epoch_validation(tmp_path: Path) -> None:
    inputs = synthetic_inputs()
    ckpt_dir = tmp_path / "tmp"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint.pt").write_bytes(b"placeholder")
    (ckpt_dir / "metadata.json").write_text('{"model":"bprmf","seed":42,"epoch":1}', encoding="utf-8")
    training = build_training_adequacy(tmp_path, inputs)
    assert list(training.columns) == TRAINING_ADEQUACY_COLUMNS
    assert training["epochs_completed"].iloc[0] == 1
    assert training["best_epoch"].iloc[0] == 1
    assert training["adequacy_status"].iloc[0] == "configuration_limited"
    assert training["smoke_configuration_contamination"].iloc[0] == "false"


def test_model_summary_marks_configuration_limited_not_invalid(tmp_path: Path) -> None:
    inputs = synthetic_inputs()
    ckpt_dir = tmp_path / "tmp"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint.pt").write_bytes(b"placeholder")
    (ckpt_dir / "metadata.json").write_text('{"model":"bprmf","seed":42,"epoch":1}', encoding="utf-8")
    training = build_training_adequacy(tmp_path, inputs)
    summary = build_model_adequacy_summary(training, inputs["f6d2_source_selection"])
    assert list(summary.columns) == MODEL_ADEQUACY_COLUMNS
    assert summary["model_adequacy_status"].iloc[0] == "configuration_limited"
    assert summary["can_use_for_reranking_pool"].iloc[0] == "diagnostic_only_not_primary"


def test_reranking_protocol_freezes_unknowns_without_inventing_paper_policy() -> None:
    protocol = build_reranking_protocol()
    assert list(protocol.columns) == RERANKING_PROTOCOL_COLUMNS
    assert protocol["is_paper_value_known"].isin(["true", "false"]).all()
    assert protocol.loc[protocol["aspect"].eq("union method"), "is_paper_value_known"].iloc[0] == "false"
    assert protocol.loc[protocol["aspect"].eq("union method"), "recommended_policy"].iloc[0] == "adapted round-robin"


def test_pool_composition_and_positive_inclusion() -> None:
    inputs = synthetic_inputs()
    composition = build_pool_composition(inputs["f6d2_pool"])
    positive = build_positive_inclusion_summary(inputs["f6d2_pool"], inputs["train"])
    assert composition.loc[composition["summary_key"].eq("users"), "value"].iloc[0] == 2
    assert list(positive.columns) == POSITIVE_COLUMNS
    present = positive.loc[positive["summary_key"].eq("positive_present")].iloc[0]
    absent = positive.loc[positive["summary_key"].eq("positive_absent")].iloc[0]
    assert present.users == 1
    assert absent.users == 1


def test_upper_bounds_distinguish_all_user_and_positive_present() -> None:
    upper = build_upper_bounds(synthetic_inputs()["f6d2_pool"])
    assert list(upper.columns) == UPPER_BOUND_COLUMNS
    all_user = upper.loc[(upper["evaluation_population"].eq("all_user_denominator")) & (upper["metric"].eq("maximum_HR@5"))].iloc[0]
    subset = upper.loc[(upper["evaluation_population"].eq("positive_present_subset")) & (upper["metric"].eq("maximum_HR@5"))].iloc[0]
    assert float(all_user.upper_bound) == 0.5
    assert float(subset.upper_bound) == 1.0


def test_evaluation_and_cohort_policy_schemas() -> None:
    inputs = synthetic_inputs()
    policy = build_evaluation_policy()
    cohort = build_cohort_policy(inputs)
    assert list(policy.columns) == EVALUATION_POLICY_COLUMNS
    assert list(cohort.columns) == COHORT_POLICY_COLUMNS
    assert policy.loc[policy["policy_role"].eq("adapted_primary"), "evaluation_population"].iloc[0] == "all_user_denominator"
    assert policy.loc[policy["policy_role"].eq("adapted_diagnostic"), "evaluation_population"].iloc[0] == "positive_present_subset"


def test_local_llm_inventory_prompt_contract_and_manifests() -> None:
    inputs = synthetic_inputs()
    inventory = build_local_llm_inventory(inputs)
    prompt = build_prompt_contract(inputs)
    ranking, reranking = build_request_manifests(inputs, prompt)
    assert list(inventory.columns) == LOCAL_LLM_COLUMNS
    assert set(inventory["local_model_identifier"]) == {"qwen2:7b-instruct", "mistral:7b-instruct"}
    assert list(prompt.columns) == PROMPT_CONTRACT_COLUMNS
    assert len(prompt) == 4
    assert list(ranking.columns) == MANIFEST_COLUMNS
    assert list(reranking.columns) == MANIFEST_COLUMNS
    assert len(ranking) == 4
    assert len(reranking) == 4
    assert reranking["positive_present"].tolist().count("false") == 2


def test_go_no_go_logic_marks_reranking_diagnostic_when_training_limited() -> None:
    inputs = synthetic_inputs()
    inventory = build_local_llm_inventory(inputs)
    limited = pd.DataFrame(
        [
            {"model": model, "model_adequacy_status": "configuration_limited"}
            for model in ["bprmf", "gru4rec", "sasrec", "lightgcn"]
        ]
    )
    go = build_go_no_go(inventory, limited)
    assert list(go.columns) == GO_NO_GO_COLUMNS
    assert set(go.loc[go["task"].eq("ranking"), "decision"]) == {"go_adapted"}
    assert set(go.loc[go["task"].eq("reranking"), "decision"]) == {"go_diagnostic_only"}


def test_idempotence_and_no_absolute_paths() -> None:
    protocol = build_reranking_protocol()
    assert frame_sha256(protocol) == frame_sha256(protocol.copy())
    assert no_absolute_paths(protocol)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_real_outputs_schema_when_materialized() -> None:
    validation_path = PROJECT_ROOT / OUTPUT_TABLES["validation"]
    if not validation_path.exists():
        return
    validation = pd.read_csv(validation_path)
    ranking = pd.read_csv(PROJECT_ROOT / OUTPUT_TABLES["ranking_manifest"])
    reranking = pd.read_csv(PROJECT_ROOT / OUTPUT_TABLES["reranking_manifest"])
    assert validation["status"].eq("pass").all()
    assert len(ranking) == 3718
    assert len(reranking) == 3718


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_protected_and_previous_phase_input_status() -> None:
    assert protected_hash_status(PROJECT_ROOT).endswith("changed=0")
    assert previous_phase_input_status(PROJECT_ROOT).endswith("changed=0")

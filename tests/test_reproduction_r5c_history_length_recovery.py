from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.reproduction_r5c_history_length_recovery import (
    R5COllamaResult,
    build_context_preflight,
    build_group_validation,
    build_result_payload,
    build_synthetic_tests,
    load_config,
    load_r5c_manifest,
    parse_output_record,
    select_first_n_per_cell,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "reproduction_r5c_history_length_recovery.yaml"
PROTOCOL_HASH = "B" * 64


def r5c_frames() -> tuple[dict, pd.DataFrame, str]:
    config = load_config(CONFIG_PATH)
    _, history, manifest_sha = load_r5c_manifest(PROJECT_ROOT, config)
    return config, history, manifest_sha


def test_r5c_scope_is_history_length_only() -> None:
    config, history, _ = r5c_frames()
    assert len(history) == 2400
    assert set(history["recovery_family"]) == {"HISTORY_LENGTH_RECOVERY"}
    assert set(history["task"]) == {"ranking"}
    assert set(history["dataset"]) == set(config["scope"]["datasets"])
    assert set(history["model_name"]) == set(config["scope"]["models"])
    assert not history["original_output_destination"].astype(str).str.contains("r5c_history_length", regex=False).any()
    assert history["output_destination"].astype(str).str.startswith("results/recovery/r5c_history_length/").all()


def test_r5c_prompt_contains_exactly_5_requirement() -> None:
    _, history, _ = r5c_frames()
    prompts = history["prompt_render"].astype(str)
    assert prompts.str.contains("EXACTLY 5", regex=False).all()
    assert prompts.str.contains("Do not return more than 5", regex=False).all()
    assert prompts.str.contains("Do not return fewer than 5", regex=False).all()
    assert prompts.str.contains("Do not include rationale", regex=False).all()


def test_history_length_group_validation_preserves_candidates_and_positive_item() -> None:
    config, history, _ = r5c_frames()
    groups = build_group_validation(history, config)
    assert len(groups) == 800
    assert groups["group_valid"].eq("true").all()
    assert groups["same_candidate_list_hash"].eq("true").all()
    assert groups["same_unordered_candidate_set_hash"].eq("true").all()
    assert groups["same_positive_item"].eq("true").all()


def test_canary_selects_one_row_per_model_dataset_condition_cell() -> None:
    _, history, _ = r5c_frames()
    canary = select_first_n_per_cell(history, 1)
    cells = {(row["model_name"], row["dataset"], row["experimental_condition"]) for row in canary}
    assert len(canary) == 24
    assert len(cells) == 24
    assert any(row["experimental_condition"] == "HISTORY_LENGTH_20" for row in canary)


def test_valid_five_index_output_parses_and_scores() -> None:
    _, history, _ = r5c_frames()
    row = history.iloc[0].to_dict()
    parsed = parse_output_record(row, '{"ranked_candidate_indices":[1,2,3,4,5]}')
    assert parsed["parse_valid"] is True
    assert parsed["validity_class"] == "VALID"
    assert parsed["ranked_candidate_indices"] == [1, 2, 3, 4, 5]
    assert len(parsed["ranked_item_ids"]) == 5
    assert parsed["output_length"] == 5
    assert parsed["expected_output_count"] == 5


def test_wrong_lengths_and_prose_are_invalid() -> None:
    _, history, _ = r5c_frames()
    row = history.iloc[0].to_dict()
    four = parse_output_record(row, '{"ranked_candidate_indices":[1,2,3,4]}')
    six = parse_output_record(row, '{"ranked_candidate_indices":[1,2,3,4,5,6]}')
    prose = parse_output_record(row, 'Here: {"ranked_candidate_indices":[1,2,3,4,5]}')
    assert four["parse_status"] == "invalid_length"
    assert four["validity_class"] == "WRONG_LENGTH"
    assert six["parse_status"] == "invalid_length"
    assert six["validity_class"] == "WRONG_LENGTH"
    assert prose["parse_status"] == "parse_failed"
    assert prose["validity_class"] == "PARSE_FAILURE"


def test_incomplete_ollama_response_is_not_succeeded_or_metric_ready() -> None:
    _, history, manifest_sha = r5c_frames()
    row = history.iloc[0].to_dict()
    runtime = R5COllamaResult(
        response_received=True,
        backend_complete=False,
        response_text='{"ranked_candidate_indices":[1,2,3',
        latency_seconds=0.1,
        raw_response={"response": '{"ranked_candidate_indices":[1,2,3', "done": False},
        error="Ollama done flag is not true",
        error_category="backend_incomplete",
        runtime_model_id=str(row["model_name"]),
        done_reason="",
    )
    payload = build_result_payload(row, manifest_sha, PROTOCOL_HASH, runtime, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "test")
    assert payload["execution_status"] == "BACKEND_INCOMPLETE"
    assert payload["validity_class"] == "BACKEND_INCOMPLETE"
    assert payload["parse_valid"] is False
    assert payload["metric_ready"] is False
    assert payload["hr_at_5"] == 0.0
    assert payload["ndcg_at_5"] == 0.0


def test_synthetic_suite_covers_required_parser_edges() -> None:
    _, history, _ = r5c_frames()
    synthetic = build_synthetic_tests(history)
    expected_cases = {
        "valid_json_object_indices",
        "k_minus_1",
        "k_plus_1",
        "numbered_ranking",
        "plain_json_array_candidate_ids",
        "leading_prose",
        "trailing_prose",
        "duplicate_index",
        "candidate_set_violation",
        "truncated_output",
        "backend_incomplete_response",
        "empty_history_row_valid",
    }
    assert expected_cases.issubset(set(synthetic["case_id"]))
    assert synthetic["pass"].eq("true").all()


def test_context_preflight_has_positive_margin_for_all_cells_without_runtime_tags() -> None:
    config, history, _ = r5c_frames()
    context = build_context_preflight(history, config, {})
    assert len(context.loc[context["scope"].eq("model_dataset_condition")]) == 24
    assert context["status"].eq("pass").all()

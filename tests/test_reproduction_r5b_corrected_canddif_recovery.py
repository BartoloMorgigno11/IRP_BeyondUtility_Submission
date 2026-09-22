from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.reproduction_r5b_corrected_canddif_recovery import (
    CorrectedOllamaResult,
    build_pair_validation,
    build_result_payload,
    load_config,
    load_corrected_manifest,
    parse_output_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "reproduction_r5b_corrected_canddif_recovery.yaml"
PROTOCOL_HASH = "A" * 64


def corrected_frames() -> tuple[dict, pd.DataFrame, str]:
    config = load_config(CONFIG_PATH)
    _, canddif, manifest_sha = load_corrected_manifest(PROJECT_ROOT, config)
    return config, canddif, manifest_sha


def test_corrected_prompt_contains_exactly_5_requirement() -> None:
    _, canddif, _ = corrected_frames()
    prompts = canddif["prompt_render"].astype(str)
    assert prompts.str.contains("EXACTLY 5", regex=False).all()
    assert prompts.str.contains("Do not return more than 5", regex=False).all()
    assert prompts.str.contains("Do not return fewer than 5", regex=False).all()
    assert prompts.str.contains("Do not include rationale", regex=False).all()


def test_incomplete_ollama_response_is_not_succeeded_or_metric_ready() -> None:
    _, canddif, manifest_sha = corrected_frames()
    row = canddif.iloc[0].to_dict()
    runtime = CorrectedOllamaResult(
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


def test_valid_five_index_output_parses_and_validates_without_alteration() -> None:
    _, canddif, _ = corrected_frames()
    row = canddif.iloc[0].to_dict()
    parsed = parse_output_record(row, '{"ranked_candidate_indices":[1,2,3,4,5]}')
    assert parsed["parse_valid"] is True
    assert parsed["validity_class"] == "VALID"
    assert parsed["ranked_candidate_indices"] == [1, 2, 3, 4, 5]
    assert len(parsed["ranked_item_ids"]) == 5
    assert parsed["output_length"] == 5
    assert parsed["expected_output_count"] == 5


def test_full_candidate_set_is_wrong_length() -> None:
    _, canddif, _ = corrected_frames()
    row = canddif.iloc[0].to_dict()
    parsed = parse_output_record(row, '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}')
    assert parsed["parse_valid"] is False
    assert parsed["parse_status"] == "invalid_length"
    assert parsed["validity_class"] == "WRONG_LENGTH"
    assert parsed["output_length"] == 20


def test_first_random_pairing_remains_unchanged() -> None:
    _, canddif, _ = corrected_frames()
    pairs = build_pair_validation(canddif)
    assert len(pairs) == 600
    assert pairs["pair_valid"].eq("true").all()
    assert pairs["same_candidate_items"].eq("true").all()
    assert pairs["different_order_as_expected"].eq("true").all()
    assert pairs["same_history"].eq("true").all()

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.extension import g2_history_length_execution as g2_exec
from src.extension.g0_canonical_llm_canary import RequestResult
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g2_history_length_execution.yaml"


@lru_cache(maxsize=1)
def _outputs() -> dict:
    config = g2_exec.load_config(ROOT, CONFIG_PATH)
    return g2_exec.build_execution_freeze(ROOT, config)


def _table(name: str) -> pd.DataFrame:
    return _outputs()["tables"][name]


def _config_for_tmp(tmp_path: Path) -> dict:
    config = g2_exec.load_config(ROOT, CONFIG_PATH)
    config = dict(config)
    config["raw_evidence"] = dict(config["raw_evidence"])
    config["raw_evidence"]["request_jsonl"] = "raw/requests.jsonl"
    config["raw_evidence"]["response_jsonl"] = "raw/responses.jsonl"
    config["raw_evidence"]["event_jsonl"] = "raw/events.jsonl"
    config["raw_evidence"]["response_dir"] = "raw/responses"
    config["raw_evidence"]["raw_dir"] = "raw"
    return config


def _result(success: bool, text: str = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}') -> RequestResult:
    return RequestResult(success, text if success else "", {"done": success}, "URLError" if not success else "", "x" if not success else "", "start", "end", 0.1)


def test_runner_dry_freeze_accounting_matches_nested_protocol() -> None:
    manifest = _outputs()["manifest"]

    assert manifest["decision_token"] == g2_exec.DECISION_FROZEN
    assert manifest["protocol_decision"] == g2_exec.PROTOCOL_DECISION
    assert manifest["logical_applicable_requests"] == 88050
    assert manifest["g1_reusable_requests"] == 18660
    assert manifest["future_new_calls_required"] == 69390
    assert manifest["dry_run_generation_calls"] == 0


def test_canonical_identity_unique_and_contains_required_fields() -> None:
    order = _table("execution_order")

    assert order["canonical_request_id"].nunique() == len(order)
    assert order["request_identity_hash"].nunique() == len(order)
    required = {
        "dataset",
        "user_id",
        "model",
        "prompt_strategy",
        "history_length",
        "history_hash",
        "candidate_hash",
        "positive_item_id",
        "generation_config_hash",
        "parser_contract_hash",
        "model_digest",
    }
    assert required.issubset(order.columns)


def test_b1_b2_h10_overlap_has_one_execution_identity() -> None:
    order = _table("execution_order")
    overlap = order.loc[
        order["history_length"].astype(int).eq(10)
        & order["reporting_view_memberships"].str.contains(g2_exec.PRIMARY_VIEW, regex=False)
        & order["reporting_view_memberships"].str.contains(g2_exec.LONG_VIEW, regex=False)
    ]

    assert not overlap.empty
    assert not overlap["canonical_request_id"].duplicated().any()
    assert overlap["reporting_view_memberships"].str.contains(g2_exec.SECONDARY_VIEW, regex=False).all()


def test_g1_reuse_is_terminal_zero_call_and_skipped() -> None:
    order = _table("execution_order")
    reuse = _table("g1_execution_reuse")
    reuse_order = order.loc[order["execution_category"].eq("G1_REUSE_TERMINAL_ZERO_CALL")]

    assert len(reuse) == 18660
    assert len(reuse_order) == 18660
    assert reuse_order["ollama_call_required"].eq("false").all()
    assert reuse["ollama_call_required"].eq("false").all()


def test_max_new_calls_counts_only_new_execution_rows() -> None:
    order = _table("execution_order")

    limited = g2_exec.select_pending_execution_rows(order, max_new_calls=7, resume=True)
    assert len(limited) == 7
    assert set(limited["execution_category"]).issubset({"NEW_G2_CALL_PENDING", "INTERRUPTED_NONTERMINAL_RESUMABLE"})
    assert not limited["execution_category"].eq("G1_REUSE_TERMINAL_ZERO_CALL").any()


def test_resume_skips_already_terminal_g2_rows() -> None:
    order = _table("execution_order").head(4).copy()
    order.loc[order.index[0], "execution_category"] = "G2_TERMINAL_SKIP_ZERO_CALL"
    order.loc[order.index[1], "execution_category"] = "INTERRUPTED_NONTERMINAL_RESUMABLE"
    order.loc[order.index[2], "execution_category"] = "G1_REUSE_TERMINAL_ZERO_CALL"
    order.loc[order.index[3], "execution_category"] = "NEW_G2_CALL_PENDING"

    selected = g2_exec.select_pending_execution_rows(order, max_new_calls=None, resume=True)
    assert selected["execution_category"].tolist() == ["INTERRUPTED_NONTERMINAL_RESUMABLE", "NEW_G2_CALL_PENDING"]


def test_raw_first_persistence_and_terminal_valid_parse(tmp_path: Path) -> None:
    row = _table("execution_order").loc[_table("execution_order")["execution_category"].eq("NEW_G2_CALL_PENDING")].iloc[0].to_dict()
    raw = g2_exec.build_unparsed_raw_record(row, _result(True), [{"backend_attempt_number": 1, "backend_success": True}])
    raw_path = tmp_path / "raw.json"
    g2_exec.write_raw_response_file(raw_path, raw)
    terminal = g2_exec.build_terminal_record(row, raw)

    assert raw_path.exists()
    assert raw["raw_response_recorded_before_parse"] is True
    assert terminal["raw_record_hash"] == raw["raw_record_hash"]
    assert terminal["terminal_state"] == "STRICT_VALID"
    assert terminal["strict_parse_valid"] is True


def test_structural_invalid_is_terminal_without_scientific_retry() -> None:
    row = _table("execution_order").loc[_table("execution_order")["execution_category"].eq("NEW_G2_CALL_PENDING")].iloc[0].to_dict()
    raw = g2_exec.build_unparsed_raw_record(row, _result(True, '{"ranked_candidate_indices":[1,2]}'), [{"backend_attempt_number": 1, "backend_success": True}])
    terminal = g2_exec.build_terminal_record(row, raw)

    assert terminal["terminal_state"] == "INVALID_LENGTH"
    assert terminal["structural_invalidity_retried"] is False
    assert terminal["strict_parse_valid"] is False


def test_backend_retry_stops_after_success(tmp_path: Path) -> None:
    config = _config_for_tmp(tmp_path)
    calls: list[bool] = []

    def fake_call(base_url: str, payload: dict, timeout: int) -> RequestResult:
        calls.append(True)
        return _result(len(calls) == 2)

    result, attempts = g2_exec.call_with_backend_retries(
        tmp_path,
        config,
        {"canonical_request_id": "REQ-1"},
        {"model": "qwen2:7b-instruct"},
        call_fn=fake_call,
        sleep_fn=lambda seconds: None,
    )

    assert result.backend_success is True
    assert len(attempts) == 2
    assert len(calls) == 2


def test_orphan_request_without_response_is_resumable(tmp_path: Path) -> None:
    config = _config_for_tmp(tmp_path)
    universe = pd.DataFrame(
        [
            {
                "canonical_request_id": "REQ-1",
                "planned_action": g2_exec.NEW_CALL_ACTION,
                "dataset": "beauty",
                "model": "qwen2:7b-instruct",
                "prompt_strategy": "BASE",
                "user_id": "u",
                "history_length": 0,
                "history_hash": "h",
                "candidate_hash": "c",
                "model_digest": "d",
                "generation_config_hash": "g",
                "parser_contract_hash": "p",
                "g2_future_raw_response_path": "raw/responses/REQ-1.json",
            }
        ]
    )
    append_jsonl_fsync(tmp_path / config["raw_evidence"]["request_jsonl"], {"canonical_request_id": "REQ-1"})

    state = g2_exec.inspect_existing_g2_evidence(tmp_path, config, universe)

    assert state.validation_failures == []
    assert state.interrupted_request_ids == {"REQ-1"}


def test_conflicting_terminal_raw_hash_blocks(tmp_path: Path) -> None:
    config = _config_for_tmp(tmp_path)
    universe = pd.DataFrame(
        [
            {
                "canonical_request_id": "REQ-1",
                "planned_action": g2_exec.NEW_CALL_ACTION,
                "dataset": "beauty",
                "model": "qwen2:7b-instruct",
                "prompt_strategy": "BASE",
                "user_id": "u",
                "history_length": 0,
                "history_hash": "h",
                "candidate_hash": "c",
                "model_digest": "d",
                "generation_config_hash": "g",
                "parser_contract_hash": "p",
                "g2_future_raw_response_path": "raw/responses/REQ-1.json",
            }
        ]
    )
    terminal = {
        "canonical_request_id": "REQ-1",
        "dataset": "beauty",
        "model": "qwen2:7b-instruct",
        "prompt_strategy": "BASE",
        "user_id": "u",
        "history_length": 0,
        "history_hash": "h",
        "candidate_hash": "c",
        "model_digest": "d",
        "generation_config_hash": "g",
        "parser_contract_hash": "p",
        "raw_record_hash": "ledger",
        "raw_response_hash": "same",
    }
    append_jsonl_fsync(tmp_path / config["raw_evidence"]["response_jsonl"], terminal)
    raw_path = tmp_path / "raw" / "responses" / "REQ-1.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({"raw_record_hash": "different", "raw_response_hash": "same"}), encoding="utf-8")

    state = g2_exec.inspect_existing_g2_evidence(tmp_path, config, universe)

    assert any(value.startswith("CONFLICTING_RAW_RECORD_HASH") for value in state.validation_failures)


def test_model_digest_mismatch_blocks_before_inference() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": "qwen2:7b-instruct",
                "digest_match": "false",
                "current_model_identity_status": "MODEL_DIGEST_MISMATCH",
            },
            {"model": "mistral:7b-instruct", "digest_match": "true", "current_model_identity_status": "MODEL_AVAILABLE_MATCHED"},
            {"model": "gemma2:9b", "digest_match": "true", "current_model_identity_status": "MODEL_AVAILABLE_MATCHED"},
        ]
    )

    assert "MODEL_DIGEST_MISMATCH_BLOCK_BEFORE_INFERENCE" in g2_exec.validate_model_preflight(frame)


def test_dry_run_cli_no_write_executes_zero_inference(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = g2_exec.main(["--config", str(CONFIG_PATH), "--dry-run", "--no-write"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "new_ollama_calls_in_this_invocation=0" in captured.out


def test_retention_paths_are_classified_for_future_external_evidence() -> None:
    manifest = _outputs()["manifest"]

    assert manifest["retention"]["future_raw_evidence_class"] == "EXTERNALIZED_RAW_EVIDENCE"
    assert manifest["retention"]["future_per_request_audit_class"] == "EXTERNALIZED_PER_REQUEST_AUDIT"
    assert manifest["raw_evidence_paths"]["response_dir"].endswith("g2_history_length_sensitivity/responses")

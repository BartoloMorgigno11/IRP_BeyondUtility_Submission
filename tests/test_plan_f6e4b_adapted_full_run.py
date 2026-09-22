from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.analysis import plan_f6e4b_adapted_full_run as f6e4b
from src.analysis import plan_f6e3_exact_prompt_pilot as f6e3


def _candidate_ids() -> tuple[str, ...]:
    return tuple(f"I{index}" for index in range(1, 21))


def _candidate_titles() -> tuple[str, ...]:
    return tuple(f"Title {index}" for index in range(1, 21))


def _runtime_json() -> str:
    return f6e4b.stable_json(
        {
            "temperature": 0.0,
            "num_predict": 256,
            "stream": False,
            "timeout_seconds": 120,
            "top_p": "",
            "top_k": "",
            "seed": "",
            "context_window": "",
            "stop_sequences": "",
            "keep_alive": "",
        }
    )


def _row(index: int = 1, model: str = f6e4b.QWEN2_TAG, strategy: str | None = None) -> dict[str, str]:
    strategy = strategy or f6e4b.STRATEGY_ORDER[0]
    candidate_ids = _candidate_ids()
    runtime = _runtime_json()
    return {
        "request_id": f"REQ-{index}",
        "source_request_id": f"SRC-REQ-{index}",
        "source_context_id": f"SRCCTX-{index}",
        "source_kind": "synthetic",
        "dataset": "Amazon Beauty",
        "task": "ranking",
        "model": model,
        "expected_model_digest": f6e4b.EXPECTED_MODEL_DIGESTS[model],
        "model_equivalence": "exact_family_version_uncertain",
        "core_strategy_id": strategy,
        "adapted_strategy_id": "base_adapted_indexed_output_contract_v1",
        "adapter_classification": "adapted_indexed_output_contract",
        "candidate_count": "20",
        "candidate_order_hash": f6e4b.sha256_text(f6e4b.stable_json(candidate_ids)),
        "candidate_item_ids": f6e4b.join_pipe(candidate_ids),
        "positive_item_id": "I1",
        "positive_initial_position": "1",
        "core_prompt_hash": f6e4b.sha256_text("core"),
        "adapter_hash": f6e4b.sha256_text("adapter"),
        "complete_prompt_hash": f6e4b.sha256_text("prompt"),
        "runtime_parameters_hash": f6e4b.sha256_text(runtime),
        "runtime_parameters_json": runtime,
        "shard_id": "SHARD-1",
        "ordinal_position": str(index),
        "status": "pending",
    }


def _rows(count: int = 2) -> list[dict[str, str]]:
    return [_row(index) for index in range(1, count + 1)]


def _shard(count: int = 2) -> list[dict[str, str]]:
    return [
        {
            "shard_id": "SHARD-1",
            "model": f6e4b.QWEN2_TAG,
            "strategy_id": f6e4b.STRATEGY_ORDER[0],
            "adapted_strategy_id": "base_adapted_indexed_output_contract_v1",
            "dataset": "Amazon Beauty",
            "task": "ranking",
            "first_request_id": "REQ-1",
            "last_request_id": f"REQ-{count}",
            "request_count": str(count),
            "estimated_runtime_seconds": "1",
            "estimated_storage_bytes": "1",
            "raw_output_path": f6e4b.RAW_PATHS[f6e4b.QWEN2_TAG].as_posix(),
            "progress_path": "results/tables/phase_f6e4b_progress.csv",
            "log_path": "results/logs/phase_f6e4b_SHARD-1.log",
            "checkpoint_policy": "flush_after_each_response_checkpoint_after_shard",
            "resume_key": "request_id",
            "status": "pending_future_execution",
        }
    ]


def _bundle(_: dict[str, str]) -> f6e4b.PromptBundle:
    return f6e4b.PromptBundle("prompt", _candidate_ids(), _candidate_titles())


def _result(text: str = '{"ranking":[1,2,3,4,5,6,7,8,9,10]}', success: bool = True) -> f6e3.RequestResult:
    return f6e3.RequestResult(
        success=success,
        response_text=text if success else "",
        raw_backend_response={
            "prompt_eval_count": 20,
            "eval_count": 10,
            "prompt_eval_duration": 1,
            "eval_duration": 1,
            "total_duration": 2,
            "done": success,
            "done_reason": "stop" if success else "",
        },
        error_type="" if success else "URLError",
        error_message="" if success else "connection refused",
        started_at_utc="2026-07-26T00:00:00Z",
        finished_at_utc="2026-07-26T00:00:01Z",
        duration_seconds=1.0,
    )


def _gate_pass() -> list[dict[str, str]]:
    return [{"check_name": "api_version", "status": "pass", "severity": "critical"}]


def _patch_execution(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, str]], shards: list[dict[str, str]]) -> None:
    monkeypatch.setattr(f6e4b, "verify_frozen_inputs", lambda root: {})
    monkeypatch.setattr(f6e4b, "load_request_manifest", lambda root: rows)
    monkeypatch.setattr(f6e4b, "load_shard_manifest", lambda root: shards)
    monkeypatch.setattr(f6e4b, "assert_manifest_valid", lambda request_rows, shard_rows, validate_expected=True: None)
    monkeypatch.setattr(f6e4b, "validate_existing_raw_records", lambda root, request_rows: None)


def test_import_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    importlib.reload(f6e4b)
    after = sorted(tmp_path.rglob("*"))
    assert before == after


def test_manifest_hash_constants_match_existing_files() -> None:
    root = Path.cwd()
    assert f6e4b.file_sha256_or_missing(root / f6e4b.F6E4A_FREEZE_MANIFEST) == f6e4b.EXPECTED_F6E4A_FREEZE_HASH
    assert f6e4b.file_sha256_or_missing(root / f6e4b.REQUEST_MANIFEST) == f6e4b.EXPECTED_F6E4A_REQUEST_HASH
    assert f6e4b.file_sha256_or_missing(root / f6e4b.SHARD_MANIFEST) == f6e4b.EXPECTED_F6E4A_SHARD_HASH


def test_frozen_input_verification_passes() -> None:
    observed = f6e4b.verify_frozen_inputs(Path.cwd())
    assert observed[f6e4b.REQUEST_MANIFEST.as_posix()] == f6e4b.EXPECTED_F6E4A_REQUEST_HASH


def test_full_request_manifest_count() -> None:
    assert len(f6e4b.load_request_manifest(Path.cwd())) == 30492


def test_full_shard_manifest_count() -> None:
    assert len(f6e4b.load_shard_manifest(Path.cwd())) == 64


def test_full_manifest_request_ids_are_unique() -> None:
    rows = f6e4b.load_request_manifest(Path.cwd())
    assert len({row["request_id"] for row in rows}) == len(rows)


def test_four_cells_only() -> None:
    rows = f6e4b.load_request_manifest(Path.cwd())
    assert {(row["model"], row["core_strategy_id"]) for row in rows} == {
        (model, strategy) for model in f6e4b.MODEL_ORDER for strategy in f6e4b.STRATEGY_ORDER
    }


def test_lastfm_reranking_is_absent() -> None:
    assert not any(row["dataset"] == "LastFM" and row["task"] == "reranking" for row in f6e4b.load_request_manifest(Path.cwd()))


def test_shards_have_max_500_requests() -> None:
    assert max(int(row["request_count"]) for row in f6e4b.load_shard_manifest(Path.cwd())) <= 500


def test_shard_coverage_is_complete() -> None:
    f6e4b.assert_manifest_valid(f6e4b.load_request_manifest(Path.cwd()), f6e4b.load_shard_manifest(Path.cwd()))


def test_candidate_order_invariance() -> None:
    for row in f6e4b.load_request_manifest(Path.cwd())[:20]:
        assert row["candidate_order_hash"] == f6e4b.sha256_text(f6e4b.stable_json(tuple(f6e4b.split_pipe(row["candidate_item_ids"]))))


def test_prompt_hash_invariance_for_first_row() -> None:
    root = Path.cwd()
    row = f6e4b.load_request_manifest(root)[0]
    bundle = f6e4b.prompt_bundle_for_row(root, row, f6e4b.source_context_map(root))
    assert f6e4b.sha256_text(bundle.prompt) == row["complete_prompt_hash"]


def test_runtime_parameters_are_frozen() -> None:
    params = f6e4b.runtime_parameters(f6e4b.load_request_manifest(Path.cwd())[0])
    assert params["temperature"] == 0.0
    assert params["num_predict"] == 256
    assert params["stream"] is False


def test_raw_paths_are_model_separated() -> None:
    assert f6e4b.raw_path_for_model(f6e4b.QWEN2_TAG) != f6e4b.raw_path_for_model(f6e4b.MISTRAL_TAG)


def test_payload_has_stream_false_and_runtime_options() -> None:
    payload = f6e4b.make_payload(_row(), "prompt")
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0


def test_strict_json_parser_accepts_valid_output() -> None:
    parsed = f6e3b_parse('{"ranking":[1,2,3,4,5,6,7,8,9,10]}')
    assert parsed["strict_valid"] is True


def f6e3b_parse(text: str) -> dict[str, object]:
    from src.analysis import plan_f6e3b_adapted_output_contract_pilot as f6e3b

    return f6e3b.parse_adapted_indexed_response(text)


def test_strict_json_parser_rejects_duplicate_output() -> None:
    parsed = f6e3b_parse('{"ranking":[1,1,3,4,5,6,7,8,9,10]}')
    assert parsed["strict_valid"] is False
    assert parsed["parse_status"] == "duplicate_indices"


def test_make_raw_record_contains_required_contract_fields() -> None:
    record = f6e4b.make_raw_record(_row(), _result(), _bundle(_row()))
    assert record["phase"] == "phase_f6e4b"
    assert record["strict_valid"] is True
    assert record["parsed_item_ids"][0] == "I1"


def test_analyze_instances_marks_missing_raw() -> None:
    instances, parsers, invalids = f6e4b.analyze_instances(Path.cwd(), [_row()], {})
    assert instances[0]["transport_status"] == "missing"
    assert parsers[0]["parse_status"] == "missing_raw_response"
    assert invalids == []


def test_analyze_instances_marks_strict_valid_raw() -> None:
    record = f6e4b.make_raw_record(_row(), _result(), _bundle(_row()))
    instances, _, invalids = f6e4b.analyze_instances(Path.cwd(), [_row()], {"REQ-1": record})
    assert instances[0]["strict_valid"] == "true"
    assert instances[0]["hr_at5_operational"] == "1"
    assert invalids == []


def test_conditional_utility_only_for_strict_valid() -> None:
    invalid_record = f6e4b.make_raw_record(_row(), _result('{"ranking":[1]}'), _bundle(_row()))
    instances, _, _ = f6e4b.analyze_instances(Path.cwd(), [_row()], {"REQ-1": invalid_record})
    assert instances[0]["strict_valid"] == "false"
    assert instances[0]["hr_at5_conditional"] == ""


def test_invalid_output_audit_preserves_invalid_response() -> None:
    invalid_record = f6e4b.make_raw_record(_row(), _result('{"ranking":[1]}'), _bundle(_row()))
    _, _, invalids = f6e4b.analyze_instances(Path.cwd(), [_row()], {"REQ-1": invalid_record})
    assert invalids[0]["invalid_reason"] == "invalid_length"


def test_append_jsonl_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    f6e4b.append_jsonl_fsync(path, {"request_id": "A"})
    f6e4b.append_jsonl_fsync(path, {"request_id": "B"})
    assert [row["request_id"] for row in f6e4b.read_jsonl(path)] == ["A", "B"]


def test_duplicate_final_raw_records_raise(tmp_path: Path) -> None:
    path = tmp_path / f6e4b.RAW_PATHS[f6e4b.QWEN2_TAG]
    f6e4b.append_jsonl_fsync(path, {"phase": "phase_f6e4b", "request_id": "REQ-1", "is_final_attempt": True})
    f6e4b.append_jsonl_fsync(path, {"phase": "phase_f6e4b", "request_id": "REQ-1", "is_final_attempt": True})
    with pytest.raises(RuntimeError, match="duplicate_final"):
        f6e4b.final_raw_by_request(tmp_path)


def test_validate_existing_raw_detects_prompt_mismatch(tmp_path: Path) -> None:
    row = _row()
    bad = f6e4b.make_raw_record(row, _result(), _bundle(row))
    bad["complete_prompt_hash"] = "bad"
    f6e4b.append_jsonl_fsync(tmp_path / f6e4b.RAW_PATHS[f6e4b.QWEN2_TAG], bad)
    with pytest.raises(RuntimeError, match="prompt_hash_mismatch"):
        f6e4b.validate_existing_raw_records(tmp_path, [row])


def test_progress_atomic_writer(tmp_path: Path) -> None:
    f6e4b.write_progress_only(tmp_path, request_count=2, raw_by_id={}, decision="partial", last_request_id="", last_shard_id="")
    rows = f6e4b.read_csv_rows(tmp_path / f6e4b.OUTPUT_TABLES["progress"])
    assert rows[0]["missing_requests"] == "2"


def test_ollama_unreachable_gate_mapping() -> None:
    rows = [{"check_name": "api_version", "status": "fail", "severity": "critical"}]
    assert f6e4b.gate_decision_from_ollama(rows) == "f6e4b_blocked_ollama_unreachable"


def test_digest_mismatch_gate_mapping() -> None:
    rows = [{"check_name": "digest_qwen2", "status": "fail", "severity": "critical"}]
    assert f6e4b.gate_decision_from_ollama(rows) == "f6e4b_blocked_model_digest_mismatch"


def test_execute_requires_explicit_allow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(1)
    shards = _shard(1)
    _patch_execution(monkeypatch, rows, shards)
    with pytest.raises(PermissionError):
        f6e4b.execute_phase(tmp_path, allow_ollama_generate=False, gate_rows=_gate_pass(), prompt_provider=_bundle)


def test_execute_appends_one_raw_with_fake_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(1)
    shards = _shard(1)
    _patch_execution(monkeypatch, rows, shards)
    sent = f6e4b.execute_phase(
        tmp_path,
        allow_ollama_generate=True,
        max_new_requests=1,
        request_executor=lambda payload, timeout: _result(),
        prompt_provider=_bundle,
        gate_rows=_gate_pass(),
    )
    assert sent.sent_requests == 1
    assert len(f6e4b.read_jsonl(tmp_path / f6e4b.RAW_PATHS[f6e4b.QWEN2_TAG])) == 1


def test_resume_skips_completed_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(1)
    shards = _shard(1)
    _patch_execution(monkeypatch, rows, shards)
    f6e4b.append_jsonl_fsync(tmp_path / f6e4b.RAW_PATHS[f6e4b.QWEN2_TAG], f6e4b.make_raw_record(rows[0], _result(), _bundle(rows[0])))
    result = f6e4b.execute_phase(
        tmp_path,
        allow_ollama_generate=True,
        max_new_requests=1,
        request_executor=lambda payload, timeout: pytest.fail("should not execute completed request"),
        prompt_provider=_bundle,
        gate_rows=_gate_pass(),
    )
    assert result.sent_requests == 0


def test_invalid_output_is_not_retried_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(1)
    shards = _shard(1)
    _patch_execution(monkeypatch, rows, shards)
    invalid = f6e4b.make_raw_record(rows[0], _result('{"ranking":[1]}'), _bundle(rows[0]))
    f6e4b.append_jsonl_fsync(tmp_path / f6e4b.RAW_PATHS[f6e4b.QWEN2_TAG], invalid)
    result = f6e4b.execute_phase(
        tmp_path,
        allow_ollama_generate=True,
        max_new_requests=1,
        request_executor=lambda payload, timeout: pytest.fail("invalid output must not be retried"),
        prompt_provider=_bundle,
        gate_rows=_gate_pass(),
    )
    assert result.sent_requests == 0


def test_transport_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(1)
    shards = _shard(1)
    _patch_execution(monkeypatch, rows, shards)
    f6e4b.execute_phase(
        tmp_path,
        allow_ollama_generate=True,
        max_new_requests=1,
        request_executor=lambda payload, timeout: _result(success=False),
        prompt_provider=_bundle,
        gate_rows=_gate_pass(),
    )
    raw = f6e4b.read_jsonl(tmp_path / f6e4b.RAW_PATHS[f6e4b.QWEN2_TAG])[0]
    assert raw["transport_status"] == "failure"


def test_build_all_tables_contains_required_schemas(tmp_path: Path) -> None:
    row = _row()
    f6e4b.append_jsonl_fsync(tmp_path / f6e4b.RAW_PATHS[f6e4b.QWEN2_TAG], f6e4b.make_raw_record(row, _result(), _bundle(row)))
    tables = f6e4b.build_all_tables(tmp_path, [row], _shard(1))
    assert set(f6e4b.OUTPUT_TABLES).issubset(tables)
    assert tables["cell_summary"][0]["strict_valid"] == "1"


def test_analysis_only_does_not_query_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(1)
    shards = _shard(1)
    _patch_execution(monkeypatch, rows, shards)
    monkeypatch.setattr(f6e4b, "inspect_ollama_gate", lambda: pytest.fail("analysis-only must not inspect Ollama"))
    written = f6e4b.run_analysis_only(tmp_path)
    assert f6e4b.OUTPUT_TABLES["progress"].as_posix() in written


def test_runtime_gate_only_records_blocker_without_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows(1)
    shards = _shard(1)
    _patch_execution(monkeypatch, rows, shards)
    monkeypatch.setattr(f6e4b, "inspect_ollama_gate", lambda: [{"gate_id": "G", "check_name": "api_version", "expected": "reachable", "observed": "no", "status": "fail", "severity": "critical", "source": "local", "notes": ""}])
    written = f6e4b.run_runtime_gate_only(tmp_path)
    assert f6e4b.FREEZE_MANIFEST_PATH.as_posix() in written


def test_final_decision_complete_with_invalid_outputs() -> None:
    strict_record = f6e4b.make_raw_record(_row(1), _result(), _bundle(_row(1)))
    invalid_record = f6e4b.make_raw_record(_row(2), _result('{"ranking":[1]}'), _bundle(_row(2)))
    instances, _, _ = f6e4b.analyze_instances(Path.cwd(), _rows(2), {"REQ-1": strict_record, "REQ-2": invalid_record})
    assert f6e4b.final_decision(_rows(2), instances) == "f6e4b_adapted_full_run_complete_with_invalid_outputs"


def test_final_decision_interrupted_resumable() -> None:
    instances, _, _ = f6e4b.analyze_instances(Path.cwd(), _rows(2), {})
    assert f6e4b.final_decision(_rows(2), instances) == "f6e4b_adapted_full_run_interrupted_resumable"


def test_no_plan_e_output_paths_are_written(tmp_path: Path) -> None:
    tables = f6e4b.build_all_tables(tmp_path, [_row()], _shard(1))
    manifest = f6e4b.build_manifest(tmp_path, [_row()], _shard(1), tables, "partial")
    written = f6e4b.write_outputs(tmp_path, tables, manifest, "report")
    assert all("plan_e_" not in path for path in written)

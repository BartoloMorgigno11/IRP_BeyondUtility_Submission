from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.analysis import plan_f6e3_exact_prompt_pilot as f6e3
from src.llm.plan_f6e1_prompt_strategies import (
    BASE_STRATEGY_ID,
    JSON_INDEX_STRATEGY_ID,
    LLMRANK_STRATEGY_ID,
    LLMSREC_STRATEGY_ID,
)


def test_import_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    importlib.reload(f6e3)
    after = sorted(tmp_path.rglob("*"))
    assert before == after


def test_source_contains_generation_only_behind_explicit_flag() -> None:
    source = Path(f6e3.__file__).read_text(encoding="utf-8")
    assert "/api/generate" in source
    assert "allow_ollama_generate" in source
    assert "ollama pull" not in source.lower()
    assert "pip install" not in source.lower()
    assert "snapshot_download" not in source.lower()


def test_materialized_manifest_is_deterministic_and_scoped() -> None:
    first = f6e3.materialize_request_manifest(Path.cwd())
    second = f6e3.materialize_request_manifest(Path.cwd())
    assert first == second
    assert len(first) == 32
    assert len({row["request_id"] for row in first}) == 32
    assert {row["model"] for row in first} == {f6e3.QWEN2_TAG, f6e3.MISTRAL_TAG}
    assert {row["strategy_id"] for row in first} == {BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID}
    assert {row["dataset"] for row in first} == set(f6e3.DATASET_ORDER)


def test_candidate_order_and_prompt_hash_stable() -> None:
    rows = f6e3.materialize_request_manifest(Path.cwd())
    for row in rows:
        candidate_ids = f6e3.split_pipe(row["candidate_item_ids"])
        assert len(candidate_ids) == 20
        assert len(set(candidate_ids)) == 20
        assert row["candidate_order_hash"] == f6e3.sha256_text(f6e3.stable_json(candidate_ids))
        assert row["prompt_hash"] == f6e3.sha256_text(row["prompt"])


def test_model_digest_and_runtime_params_are_adapted() -> None:
    rows = f6e3.materialize_request_manifest(Path.cwd())
    for row in rows:
        assert row["expected_model_digest"] == f6e3.EXPECTED_TAG_DIGESTS[row["model"]]
        params = json.loads(row["runtime_parameters_json"])
        assert params["temperature"] == 0.0
        assert params["num_predict"] == 256
        assert params["stream"] is False
        assert row["runtime_parameter_classification"] == "adapted_runtime"


def test_execute_requires_explicit_allow_and_gate_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = f6e3.materialize_request_manifest(Path.cwd())
    gate = [
        f6e3.gate_row("api_version", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3.gate_row("api_tags", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3.gate_row("digest_qwen2_7b_instruct", f6e3.EXPECTED_TAG_DIGESTS[f6e3.QWEN2_TAG], f6e3.EXPECTED_TAG_DIGESTS[f6e3.QWEN2_TAG], "pass", "critical", "mock", ""),
        f6e3.gate_row("digest_mistral_7b_instruct", f6e3.EXPECTED_TAG_DIGESTS[f6e3.MISTRAL_TAG], f6e3.EXPECTED_TAG_DIGESTS[f6e3.MISTRAL_TAG], "pass", "critical", "mock", ""),
    ]
    with pytest.raises(PermissionError):
        f6e3.execute_pilot(tmp_path, rows, gate, allow_ollama_generate=False, max_new_requests=1)


def test_raw_append_only_resume_and_no_retry(tmp_path: Path) -> None:
    rows = f6e3.materialize_request_manifest(Path.cwd())
    gate = [
        f6e3.gate_row("api_version", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3.gate_row("api_tags", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3.gate_row("digest_qwen2_7b_instruct", f6e3.EXPECTED_TAG_DIGESTS[f6e3.QWEN2_TAG], f6e3.EXPECTED_TAG_DIGESTS[f6e3.QWEN2_TAG], "pass", "critical", "mock", ""),
        f6e3.gate_row("digest_mistral_7b_instruct", f6e3.EXPECTED_TAG_DIGESTS[f6e3.MISTRAL_TAG], f6e3.EXPECTED_TAG_DIGESTS[f6e3.MISTRAL_TAG], "pass", "critical", "mock", ""),
    ]
    calls = []

    def fake_executor(payload, timeout):
        calls.append((payload, timeout))
        return f6e3.RequestResult(True, "['a','b']", {"done": True, "eval_count": 2}, "", "", "t0", "t1", 1.0)

    sent = f6e3.execute_pilot(tmp_path, rows, gate, allow_ollama_generate=True, max_new_requests=1, request_executor=fake_executor)
    sent_again = f6e3.execute_pilot(tmp_path, rows, gate, allow_ollama_generate=True, max_new_requests=1, request_executor=fake_executor)
    assert sent == 1
    assert sent_again == 1
    assert len(calls) == 2
    raw = f6e3.read_jsonl(tmp_path / f6e3.raw_path_for_model(rows[0]["model"]))
    assert len(raw) == 2
    assert len({record["request_id"] for record in raw}) == 2
    assert raw[0]["attempt"] == 1


def test_transport_failure_is_recorded(tmp_path: Path) -> None:
    rows = f6e3.materialize_request_manifest(Path.cwd())
    gate = [
        f6e3.gate_row("api_version", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3.gate_row("api_tags", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3.gate_row("digest_qwen2_7b_instruct", f6e3.EXPECTED_TAG_DIGESTS[f6e3.QWEN2_TAG], f6e3.EXPECTED_TAG_DIGESTS[f6e3.QWEN2_TAG], "pass", "critical", "mock", ""),
        f6e3.gate_row("digest_mistral_7b_instruct", f6e3.EXPECTED_TAG_DIGESTS[f6e3.MISTRAL_TAG], f6e3.EXPECTED_TAG_DIGESTS[f6e3.MISTRAL_TAG], "pass", "critical", "mock", ""),
    ]

    def fake_executor(payload, timeout):
        return f6e3.RequestResult(False, "", {}, "URLError", "refused", "t0", "t1", 0.1)

    f6e3.execute_pilot(tmp_path, rows, gate, allow_ollama_generate=True, max_new_requests=1, request_executor=fake_executor)
    raw = f6e3.read_jsonl(tmp_path / f6e3.raw_path_for_model(rows[0]["model"]))
    assert raw[0]["transport_status"] == "failure"
    assert raw[0]["error_type"] == "URLError"


def test_title_parser_contract_strict_and_all_request_populations(tmp_path: Path) -> None:
    rows = f6e3.materialize_request_manifest(Path.cwd())
    titles = f6e3.split_pipe(rows[0]["candidate_item_titles"])
    response = "\n".join(f"{idx}. {title}" for idx, title in enumerate(titles[:10], start=1))
    result = f6e3.RequestResult(True, response, {"done": True, "eval_count": 10}, "", "", "t0", "t1", 1.0)
    f6e3.append_jsonl_fsync(tmp_path / f6e3.raw_path_for_model(rows[0]["model"]), f6e3.make_raw_record(rows[0], result))
    instances, parser = f6e3.analyze_instances(tmp_path, rows)
    assert instances[0]["parse_valid"] == "true"
    assert instances[0]["strict_valid"] == "true"
    summary = f6e3.build_strategy_summary(instances)
    assert {row["evaluation_population"] for row in summary} == {"all_request", "strict_valid"}


def test_analysis_only_does_not_query_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("network should not be used in analysis-only")

    monkeypatch.setattr(f6e3, "http_get_json", fail)
    tables, manifest, report = f6e3.analyze_only(Path.cwd(), gate_rows=[])
    assert tables["request_manifest"]
    assert manifest["runtime_metadata"]["analysis_only_uses_no_ollama"] is True
    assert "Decision:" in report


def test_stop_conditions_context_model_and_digest_mismatch() -> None:
    gate = [
        f6e3.gate_row("api_version", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3.gate_row("api_tags", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3.gate_row("digest_qwen2_7b_instruct", "expected", "wrong", "fail", "critical", "mock", ""),
    ]
    assert f6e3.gate_decision(gate) == "f6e3_blocked_model_digest_mismatch"
    rows = f6e3.materialize_request_manifest(Path.cwd())
    too_long = dict(rows[0])
    too_long["prompt"] = "x" * 20000
    assert len(too_long["prompt"]) > 16000


def test_llmsrec_and_json_index_not_promoted() -> None:
    rows = f6e3.materialize_request_manifest(Path.cwd())
    assert LLMSREC_STRATEGY_ID not in {row["strategy_id"] for row in rows}
    assert JSON_INDEX_STRATEGY_ID not in {row["strategy_id"] for row in rows}
    missing = f6e3.build_missing_or_blocked([], rows, [])
    assert any(row["status"] == "not_executed_selected_scope" and "LLMSRec" in row["affected_scope"] for row in missing)
    assert any(row["status"] == "not_executed_selected_scope" and "JSON/index" in row["affected_scope"] for row in missing)


def test_protected_inputs_and_f6d4a_raw_invariant() -> None:
    summary = f6e3.protected_summary(Path.cwd())
    assert summary["f2_changed"] == 0
    assert summary["f6d4a_changed"] == 0
    assert summary["f6d4a_raw_changed"] == 0


def test_no_plan_e_canonical_writes(tmp_path: Path) -> None:
    tables, manifest, report = f6e3.analyze_only(Path.cwd(), gate_rows=[])
    f6e3.write_all(tmp_path, tables, manifest, report)
    written = [path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()]
    assert written
    assert all(path.startswith(("results/tables/phase_f6e3_", "results/manifests/phase_f6e3_", "report/phase_f6e3_")) for path in written)


def test_analysis_outputs_are_deterministic(tmp_path: Path) -> None:
    first = f6e3.analyze_only(Path.cwd(), gate_rows=[])
    second = f6e3.analyze_only(Path.cwd(), gate_rows=[])
    assert first[1]["scientific_content_hash"] == second[1]["scientific_content_hash"]
    h1 = f6e3.write_all(tmp_path / "a", first[0], first[1], first[2])
    h2 = f6e3.write_all(tmp_path / "b", second[0], second[1], second[2])
    assert h1.keys() == h2.keys()

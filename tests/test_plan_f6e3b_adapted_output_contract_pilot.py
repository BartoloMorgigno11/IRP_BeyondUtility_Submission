from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from src.analysis import plan_f6e3b_adapted_output_contract_pilot as f6e3b
from src.llm.plan_f6e1_prompt_strategies import BASE_STRATEGY_ID, JSON_INDEX_STRATEGY_ID, LLMRANK_STRATEGY_ID, LLMSREC_STRATEGY_ID


def gate_pass_rows() -> list[dict[str, str]]:
    return [
        f6e3b.gate_row("api_version", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3b.gate_row("api_tags", "reachable", "{}", "pass", "critical", "mock", ""),
        f6e3b.gate_row("digest_qwen2_7b_instruct", f6e3b.EXPECTED_TAG_DIGESTS[f6e3b.QWEN2_TAG], f6e3b.EXPECTED_TAG_DIGESTS[f6e3b.QWEN2_TAG], "pass", "critical", "mock", ""),
        f6e3b.gate_row("digest_mistral_7b_instruct", f6e3b.EXPECTED_TAG_DIGESTS[f6e3b.MISTRAL_TAG], f6e3b.EXPECTED_TAG_DIGESTS[f6e3b.MISTRAL_TAG], "pass", "critical", "mock", ""),
    ]


def test_import_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    importlib.reload(f6e3b)
    after = sorted(tmp_path.rglob("*"))
    assert before == after


def test_source_contains_generation_only_behind_explicit_flag() -> None:
    source = Path(f6e3b.__file__).read_text(encoding="utf-8")
    assert "allow_ollama_generate" in source
    assert "ollama pull" not in source.lower()
    assert "pip install" not in source.lower()
    assert "snapshot_download" not in source.lower()


def test_materialized_manifest_is_deterministic_scoped_and_adapted() -> None:
    first = f6e3b.materialize_request_manifest(Path.cwd())
    second = f6e3b.materialize_request_manifest(Path.cwd())
    assert first == second
    assert len(first) == 32
    assert len({row["request_id"] for row in first}) == 32
    assert len({row["f6e3_request_id"] for row in first}) == 32
    assert {row["model"] for row in first} == {f6e3b.QWEN2_TAG, f6e3b.MISTRAL_TAG}
    assert {row["strategy_id"] for row in first} == {BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID}
    assert {row["dataset"] for row in first} == set(f6e3b.DATASET_ORDER)
    assert {row["strategy_classification"] for row in first} == {"adapted_indexed_output_contract"}
    assert JSON_INDEX_STRATEGY_ID not in {row["strategy_id"] for row in first}
    assert LLMSREC_STRATEGY_ID not in {row["strategy_id"] for row in first}


def test_prompt_hash_and_candidate_order_are_stable() -> None:
    rows = f6e3b.materialize_request_manifest(Path.cwd())
    for row in rows:
        candidate_ids = f6e3b.split_pipe(row["candidate_item_ids"])
        assert len(candidate_ids) == 20
        assert len(set(candidate_ids)) == 20
        assert row["candidate_order_hash"] == f6e3b.sha256_text(f6e3b.stable_json(candidate_ids))
        assert row["prompt_hash"] == f6e3b.sha256_text(row["prompt"])
        assert row["candidate_serialization_hash"]
        assert row["output_adapter_hash"]
        assert '{"ranking":[1,2,3,4,5,6,7,8,9,10]}' in row["prompt"]
        assert "ranked_candidate_indices" not in row["prompt"]


def test_frozen_f6e3_and_f6e3a_inputs_are_verified() -> None:
    summary = f6e3b.verify_frozen_f6e3_inputs(Path.cwd())
    assert summary["f6e3a_scientific_content_hash"] == f6e3b.EXPECTED_F6E3A_SCIENTIFIC_HASH
    assert summary["f6e3_raw_hashes"] == f6e3b.EXPECTED_F6E3_RAW_HASHES


def test_strict_parser_accepts_one_key_ranking_json() -> None:
    parsed = f6e3b.parse_adapted_indexed_response('{"ranking":[1,2,3,4,5,6,7,8,9,10]}')
    assert parsed["strict_valid"] is True
    assert parsed["output_indices"] == tuple(range(1, 11))


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ('Here is {"ranking":[1,2,3,4,5,6,7,8,9,10]}', "extra_text_or_code_fence"),
        ('```json\n{"ranking":[1,2,3,4,5,6,7,8,9,10]}\n```', "extra_text_or_code_fence"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9]}', "invalid_length"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9,9]}', "duplicate_indices"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9,21]}', "out_of_range_indices"),
        ('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}', "missing_ranking_key"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9,10],"notes":""}', "extra_keys"),
        ("not-json", "extra_text_or_code_fence"),
    ],
)
def test_strict_parser_rejects_invalid_contracts(payload: str, status: str) -> None:
    parsed = f6e3b.parse_adapted_indexed_response(payload)
    assert parsed["strict_valid"] is False
    assert parsed["parse_status"] == status


def test_execute_requires_explicit_allow_and_gate_pass(tmp_path: Path) -> None:
    rows = f6e3b.materialize_request_manifest(Path.cwd())
    with pytest.raises(PermissionError):
        f6e3b.execute_pilot(tmp_path, rows, gate_pass_rows(), allow_ollama_generate=False, max_new_requests=1)
    bad_gate = [f6e3b.gate_row("api_version", "reachable", "refused", "fail", "critical", "mock", "")]
    with pytest.raises(RuntimeError, match="f6e3b_blocked_ollama_unreachable"):
        f6e3b.execute_pilot(tmp_path, rows, bad_gate, allow_ollama_generate=True, max_new_requests=1)


def test_raw_append_only_resume_and_no_retry(tmp_path: Path) -> None:
    rows = f6e3b.materialize_request_manifest(Path.cwd())
    calls: list[dict[str, object]] = []

    def fake_executor(payload, timeout):
        calls.append({"payload": payload, "timeout": timeout})
        return f6e3b.f6e3.RequestResult(True, '{"ranking":[1,2,3,4,5,6,7,8,9,10]}', {"done": True, "eval_count": 10}, "", "", "t0", "t1", 1.0)

    original_verify = f6e3b.verify_frozen_f6e3_inputs
    f6e3b.verify_frozen_f6e3_inputs = lambda root: {}
    try:
        sent = f6e3b.execute_pilot(tmp_path, rows, gate_pass_rows(), allow_ollama_generate=True, max_new_requests=1, request_executor=fake_executor)
        sent_again = f6e3b.execute_pilot(tmp_path, rows, gate_pass_rows(), allow_ollama_generate=True, max_new_requests=1, request_executor=fake_executor)
    finally:
        f6e3b.verify_frozen_f6e3_inputs = original_verify
    assert sent == 1
    assert sent_again == 1
    assert len(calls) == 2
    raw = f6e3b.read_jsonl(tmp_path / f6e3b.raw_path_for_model(rows[0]["model"]))
    assert len(raw) == 2
    assert len({record["request_id"] for record in raw}) == 2
    assert raw[0]["parser_version"] == "json_ranking_1based_top10_strict_v1"


def test_transport_failure_is_recorded_and_not_retried_for_same_request(tmp_path: Path) -> None:
    rows = f6e3b.materialize_request_manifest(Path.cwd())
    calls = 0

    def fake_executor(payload, timeout):
        nonlocal calls
        calls += 1
        return f6e3b.f6e3.RequestResult(False, "", {}, "URLError", "refused", "t0", "t1", 0.1)

    original_verify = f6e3b.verify_frozen_f6e3_inputs
    f6e3b.verify_frozen_f6e3_inputs = lambda root: {}
    try:
        f6e3b.execute_pilot(tmp_path, rows, gate_pass_rows(), allow_ollama_generate=True, max_new_requests=1, request_executor=fake_executor)
        f6e3b.execute_pilot(tmp_path, rows, gate_pass_rows(), allow_ollama_generate=True, max_new_requests=1, request_executor=fake_executor)
    finally:
        f6e3b.verify_frozen_f6e3_inputs = original_verify
    raw = f6e3b.read_jsonl(tmp_path / f6e3b.raw_path_for_model(rows[0]["model"]))
    assert calls == 2
    assert len(raw) == 2
    assert raw[0]["request_id"] != raw[1]["request_id"]
    assert raw[0]["transport_status"] == "failure"


def test_analyze_instances_maps_indices_to_candidates(tmp_path: Path) -> None:
    rows = f6e3b.materialize_request_manifest(Path.cwd())
    result = f6e3b.f6e3.RequestResult(True, '{"ranking":[1,2,3,4,5,6,7,8,9,10]}', {"done": True, "eval_count": 10}, "", "", "t0", "t1", 1.0)
    f6e3b.f6e3.append_jsonl_fsync(tmp_path / f6e3b.raw_path_for_model(rows[0]["model"]), f6e3b.make_raw_record(rows[0], result))
    instances, parser = f6e3b.analyze_instances(tmp_path, rows)
    assert instances[0]["strict_valid"] == "true"
    assert instances[0]["parsed_ranking_item_ids"] == "|".join(f6e3b.split_pipe(rows[0]["candidate_item_ids"])[:10])
    assert parser[0]["parse_status"] == "parsed_valid"


def test_analysis_only_does_not_query_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("network should not be used in analysis-only")

    monkeypatch.setattr(f6e3b, "inspect_ollama_gate", fail)
    tables, manifest, report = f6e3b.analyze_only(Path.cwd(), gate_rows=[])
    assert tables["request_manifest"]
    assert manifest["runtime_metadata"]["analysis_only_uses_no_ollama"] is True
    assert "Decision:" in report


def test_exact_vs_adapted_rows_are_pairwise() -> None:
    rows = f6e3b.materialize_request_manifest(Path.cwd())
    instances, _ = f6e3b.analyze_instances(Path.cwd(), rows)
    comparison = f6e3b.build_exact_vs_adapted(Path.cwd(), instances)
    assert len(comparison) == 32
    assert len({row["f6e3_request_id"] for row in comparison}) == 32
    assert {row["metric_comparability"] for row in comparison}


def test_protected_inputs_include_f6e3_and_f6e3a_invariance() -> None:
    summary = f6e3b.protected_summary(Path.cwd())
    assert summary["f2_changed"] == 0
    assert summary["f6d4a_changed"] == 0
    assert summary["f6d4a_raw_changed"] == 0
    assert summary["f6e3_raw_changed"] == 0
    assert summary["f6e3a_changed"] == 0


def test_no_plan_e_or_previous_phase_writes(tmp_path: Path) -> None:
    tables, manifest, report = f6e3b.analyze_only(Path.cwd(), gate_rows=[])
    written = f6e3b.write_all(tmp_path, tables, manifest, report)
    assert written
    assert all(path.startswith(("results/tables/phase_f6e3b_", "results/manifests/phase_f6e3b_", "report/phase_f6e3b_")) for path in written)


def test_analysis_outputs_are_deterministic(tmp_path: Path) -> None:
    first = f6e3b.analyze_only(Path.cwd(), gate_rows=[])
    second = f6e3b.analyze_only(Path.cwd(), gate_rows=[])
    assert first[1]["scientific_content_hash"] == second[1]["scientific_content_hash"]
    h1 = f6e3b.write_all(tmp_path / "a", first[0], first[1], first[2])
    h2 = f6e3b.write_all(tmp_path / "b", second[0], second[1], second[2])
    assert h1.keys() == h2.keys()
    for path, digest in h1.items():
        assert digest == h2[path]

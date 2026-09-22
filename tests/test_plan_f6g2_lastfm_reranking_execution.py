from __future__ import annotations

import hashlib
import importlib
import csv
import json
import sys
from pathlib import Path

import pytest

from src.analysis import plan_f6g2_lastfm_reranking_execution_analysis as f6g2
from src.llm import plan_f6g2_lastfm_reranking_runner as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(root: Path, rel: str, rows: list[dict[str, object]]) -> dict[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return {"path": rel, "sha256": _sha(path), "phase": "test"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: "" if value is None else value for key, value in row.items()} for row in csv.DictReader(handle)]


def _write_json(root: Path, rel: str, payload: dict[str, object]) -> dict[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f6g2.stable_json(payload) + "\n", encoding="utf-8")
    return {"path": rel, "sha256": _sha(path), "phase": "test"}


def _request_rows(output_length: int = 20) -> list[dict[str, object]]:
    candidates = "|".join(str(value) for value in range(1, 21))
    labels = "|".join(f"{value}:artist {value}" for value in range(1, 21))
    schema = {"type": "object", "required": ["ranking"], "properties": {"ranking": {"type": "array", "items": {"type": "integer"}, "minItems": output_length, "maxItems": output_length}}}
    rows: list[dict[str, object]] = []
    request_id = 1
    for user_id, positive, positive_present in [("1", "3", "true"), ("2", "99", "false")]:
        for model in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
            rows.append(
                {
                    "request_id": f"REQ-{request_id:03d}",
                    "logical_instance_id": f"LOG-{user_id}",
                    "user_id": user_id,
                    "dataset": "LastFM",
                    "task": "reranking",
                    "model": model,
                    "strategy": "base_adapted_indexed",
                    "history_item_ids": "10|11|12",
                    "candidate_item_ids": candidates,
                    "indexed_candidate_labels": labels,
                    "candidate_order_hash": f"hash-{user_id}",
                    "positive_item_id": positive,
                    "positive_initial_position": "3" if positive_present == "true" else "",
                    "positive_present": positive_present,
                    "pool_provenance_hash": "pool",
                    "expected_json_schema": json.dumps(schema, separators=(",", ":")),
                    "required_output_length": str(output_length),
                    "parser_contract": "indexed_json_top20_unique_candidates" if output_length == 20 else "indexed_json_top10_unique_candidates",
                    "invalid_output_policy": "invalid_zero",
                    "operational_utility_policy": "all_user",
                    "conditional_utility_policy": "strict_valid_positive_present",
                    "title_mapping_hash": "title",
                    "request_hash": f"rh-{request_id}",
                    "request_status": "planned_not_executed",
                    "block_reason": "",
                }
            )
            request_id += 1
    return rows


def _logical_rows(output_length: int = 20) -> list[dict[str, object]]:
    candidates = "|".join(str(value) for value in range(1, 21))
    titles = "|".join(f"artist {value}" for value in range(1, 21))
    return [
        {
            "logical_instance_id": "LOG-1",
            "user_id": "1",
            "dataset": "LastFM",
            "task": "reranking",
            "cohort_name": "cohort",
            "history_item_ids": "10|11|12",
            "history_item_titles": "a|b|c",
            "candidate_item_ids": candidates,
            "candidate_item_titles": titles,
            "candidate_order_hash": "hash-1",
            "positive_item_id": "3",
            "positive_initial_position": "3",
            "positive_present": "true",
            "pool_provenance_hash": "pool",
            "expected_output_length": str(output_length),
            "parser_contract": "indexed_json_top20_unique_candidates" if output_length == 20 else "indexed_json_top10_unique_candidates",
            "operational_utility_policy": "all_user",
            "conditional_utility_policy": "strict_valid_positive_present",
            "request_ready": "true",
            "block_reason": "",
        },
        {
            "logical_instance_id": "LOG-2",
            "user_id": "2",
            "dataset": "LastFM",
            "task": "reranking",
            "cohort_name": "cohort",
            "history_item_ids": "13|14|15",
            "history_item_titles": "d|e|f",
            "candidate_item_ids": candidates,
            "candidate_item_titles": titles,
            "candidate_order_hash": "hash-2",
            "positive_item_id": "99",
            "positive_initial_position": "",
            "positive_present": "false",
            "pool_provenance_hash": "pool",
            "expected_output_length": str(output_length),
            "parser_contract": "indexed_json_top20_unique_candidates" if output_length == 20 else "indexed_json_top10_unique_candidates",
            "operational_utility_policy": "all_user",
            "conditional_utility_policy": "strict_valid_positive_present",
            "request_ready": "true",
            "block_reason": "",
        },
    ]


def _fixture_config(root: Path, output_length: int = 20) -> dict[str, object]:
    logical_hash = _write_csv(root, "inputs/f6g1_logical.csv", _logical_rows(output_length))
    request_hash = _write_csv(root, "inputs/f6g1_requests.csv", _request_rows(output_length))
    shard_hash = _write_csv(
        root,
        "inputs/f6g1_shards.csv",
        [
            {"shard_id": "SHARD-1", "model": "qwen2:7b-instruct", "strategy": "base_adapted_indexed", "shard_index": "1", "request_count": "2", "first_request_id": "REQ-001", "last_request_id": "REQ-002", "request_manifest_hash": "x", "shard_hash": "x", "status": "planned_not_executed", "notes": ""},
            {"shard_id": "SHARD-2", "model": "mistral:7b-instruct", "strategy": "base_adapted_indexed", "shard_index": "2", "request_count": "2", "first_request_id": "REQ-003", "last_request_id": "REQ-004", "request_manifest_hash": "x", "shard_hash": "x", "status": "planned_not_executed", "notes": ""},
        ],
    )
    inputs: dict[str, dict[str, str]] = {
        "f6g1_future_logical_instances": logical_hash,
        "f6g1_future_request_manifest": request_hash,
        "f6g1_future_shard_manifest": shard_hash,
        "f6g1_paper_pool_contract": _write_csv(root, "inputs/pool_contract.csv", [{"component": "target_pool_size", "current_local_value": "20"}]),
        "f6e3b_request_manifest": _write_csv(root, "inputs/f6e3b.csv", [{"output_contract_id": "json_ranking_1based_top10_strict_v1"}]),
        "f6e4a_request_manifest": _write_csv(root, "inputs/f6e4a.csv", [{"expected_schema": '{"ranking":[1,2,3,4,5,6,7,8,9,10]}'}]),
        "reranking_protocol_decision": _write_csv(root, "inputs/protocol.csv", [{"selected_protocol_option_id": "option_b_strict_top10_confirmation", "recommended_decision": "approve_paper_aligned_topk_reranking_confirmation_pilot"}]),
        "top10_prompt_contract": _write_csv(root, "inputs/top10_prompt.csv", [{"prompt_contract_id": "llm_rerank_index_json_top10_officialcode_v1", "requested_output_count": "10"}]),
        "top10_parser_contract": _write_csv(root, "inputs/top10_parser.csv", [{"parser_contract_id": "llm_rerank_top10_strict_primary_parser_v1", "expected_output_count": "10"}]),
        "full20_prompt_v2_contract": _write_csv(root, "inputs/full20.csv", [{"prompt_contract_id": "llm_rerank_index_json_full20_v2", "requested_output_count": "20"}]),
        "f2_protected_hashes": _write_csv(root, "inputs/protected.csv", [{"path": "x", "sha256": "y"}]),
    }
    f6g1_hash = "f6g1hash"
    f6f2_hash = "f6f2hash"
    f6f1_hash = "f6f1hash"
    f6e4c_hash = "f6e4chash"
    inputs["f6g1_manifest"] = _write_json(
        root,
        "inputs/f6g1_manifest.json",
        {
            "decision": "f6g1_ok",
            "scientific_content_hash": f6g1_hash,
            "future_logical_instance_hash": logical_hash["sha256"],
            "future_request_manifest_hash": request_hash["sha256"],
            "future_shard_manifest_hash": shard_hash["sha256"],
        },
    )
    inputs["f6f2_manifest"] = _write_json(root, "inputs/f6f2_manifest.json", {"decision": "f6f2_ok", "scientific_content_hash": f6f2_hash})
    inputs["f6f1_manifest"] = _write_json(root, "inputs/f6f1_manifest.json", {"decision": "f6f1_ok", "scientific_content_hash": f6f1_hash})
    inputs["f6e4c_manifest"] = _write_json(root, "inputs/f6e4c_manifest.json", {"decision": "f6e4c_ok", "scientific_content_hash": f6e4c_hash})
    tables = {name: f"results/tables/phase_f6g2_{name}.csv" for name in f6g2.TABLE_COLUMNS}
    return {
        "phase": {
            "id": "phase_f6g2",
            "expected_branch": "",
            "expected_head": "",
            "interpreter_path": str(Path(sys.executable)),
            "environment_classification": "test",
            "expected_f6g1_decision": "f6g1_ok",
            "expected_f6g1_scientific_hash": f6g1_hash,
            "expected_f6f2_decision": "f6f2_ok",
            "expected_f6f2_scientific_hash": f6f2_hash,
            "expected_f6f1_decision": "f6f1_ok",
            "expected_f6f1_scientific_hash": f6f1_hash,
            "expected_f6e4c_decision": "f6e4c_ok",
            "expected_f6e4c_scientific_hash": f6e4c_hash,
            "blocked_contract_decision": f6g2.DECISION_CONTRACT,
            "blocked_contract_recommendation": f6g2.RECOMMEND_CONTRACT,
        },
        "execution_contract": {
            "dataset": "LastFM",
            "task": "reranking",
            "expected_logical_instances": 2,
            "expected_requests": 4,
            "expected_shards": 2,
            "candidate_pool_size": 20,
            "required_requested_ranking_length": 10,
            "required_parser_expected_length": 10,
            "evaluation_cutoffs": [1, 5, 10],
            "blocked_f6g1_parser_contracts": ["indexed_json_top20_unique_candidates"],
            "blocked_f6g1_output_lengths": [20],
            "no_raw_on_contract_block": True,
        },
        "runtime": {
            "models": ["qwen2:7b-instruct", "mistral:7b-instruct"],
            "model_equivalence_classification": "exact_family_version_uncertain",
            "runtime_parameters": {"timeout_seconds": 120},
        },
        "input_hashes": inputs,
        "outputs": {
            "report": "report/phase_f6g2_lastfm_reranking_execution.md",
            "freeze_manifest": "results/manifests/phase_f6g2_lastfm_reranking_execution_freeze.json",
            "raw": {
                "qwen2": "results/raw/phase_f6g2_lastfm_reranking_qwen2.jsonl",
                "mistral": "results/raw/phase_f6g2_lastfm_reranking_mistral.jsonl",
            },
            "tables": tables,
        },
        "validation": {"run_suites": False, "unsafe_exclusions": ["ollama_generation", "network", "training", "cuda"]},
    }


def test_import_has_no_side_effects(tmp_path: Path) -> None:
    before = set(tmp_path.rglob("*"))
    importlib.reload(f6g2)
    importlib.reload(runner)
    after = set(tmp_path.rglob("*"))
    assert before == after


def test_no_unsafe_remote_or_training_imports() -> None:
    source = Path(f6g2.__file__).read_text(encoding="utf-8") + Path(runner.__file__).read_text(encoding="utf-8")
    blocked_tokens = ["import requests", "from requests", "urllib.request", "openai", "anthropic", "torch", "tensorflow", "ollama pull", "git reset"]
    assert not any(token in source for token in blocked_tokens)


def test_interpreter_and_input_hash_gates(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    assert f6g2.verify_interpreter(tmp_path, config)["status"] == "pass"
    rows = f6g2.verify_input_integrity(tmp_path, config)
    assert {row["status"] for row in rows} == {"pass"}
    (tmp_path / config["input_hashes"]["top10_parser_contract"]["path"]).write_text("changed\n", encoding="utf-8")
    changed = f6g2.verify_input_integrity(tmp_path, config)
    assert any(row["artifact"] == "top10_parser_contract" and row["changed"] == "true" for row in changed)


def test_contract_conflict_blocks_before_inference_and_raw_creation(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, output_length=20)
    manifest = f6g2.run_phase(tmp_path, config, mode="preflight-only")
    assert manifest["decision"] == f6g2.DECISION_CONTRACT
    assert manifest["recommended_next_action"] == f6g2.RECOMMEND_CONTRACT
    assert manifest["contract_reconciliation"]["candidate_pool_sizes"] == ["20"]
    assert manifest["contract_reconciliation"]["request_output_lengths"] == ["20"]
    assert manifest["contract_reconciliation"]["top10_parser_expected_counts"] == ["10"]
    assert not (tmp_path / config["outputs"]["raw"]["qwen2"]).exists()
    requests = _read_csv(tmp_path / config["outputs"]["tables"]["request_execution"])
    assert {row["execution_status"] for row in requests} == {"blocked_pre_inference_contract_conflict"}


def test_request_order_shard_assignment_and_counts(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, output_length=20)
    f6g2.run_phase(tmp_path, config, mode="preflight-only")
    requests = _read_csv(tmp_path / config["outputs"]["tables"]["request_execution"])
    shards = _read_csv(tmp_path / config["outputs"]["tables"]["shard_status"])
    assert [row["request_order"] for row in requests] == ["1", "2", "3", "4"]
    assert [row["shard_id"] for row in requests] == ["SHARD-1", "SHARD-1", "SHARD-2", "SHARD-2"]
    assert [row["blocked_request_count"] for row in shards] == ["2", "2"]
    assert [row["status"] for row in shards] == ["blocked_integrity", "blocked_integrity"]


def test_strict_parser_accepts_only_exact_index_json() -> None:
    parsed = f6g2.parse_strict_indexed_response('{"ranking":[1,2,3,4,5,6,7,8,9,10]}', candidate_count=20, expected_count=10)
    assert parsed["strict_valid"] is True
    cases = [
        ('Here is {"ranking":[1,2,3,4,5,6,7,8,9,10]}', "extra_text_or_code_fence"),
        ('```json\n{"ranking":[1,2,3,4,5,6,7,8,9,10]}\n```', "extra_text_or_code_fence"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}', "invalid_length"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9,9]}', "duplicate_indices"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9,21]}', "out_of_range_indices"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9,true]}', "invalid_schema"),
        ('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}', "missing_ranking_key"),
        ('{"ranking":[1,2,3,4,5,6,7,8,9,10],"notes":""}', "extra_keys"),
    ]
    for payload, reason in cases:
        assert f6g2.parse_strict_indexed_response(payload, candidate_count=20, expected_count=10)["parse_status"] == reason


def test_parser_supports_alternate_frozen_key_when_configured() -> None:
    parsed = f6g2.parse_strict_indexed_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
        candidate_count=20,
        expected_count=10,
        ranking_key="ranked_candidate_indices",
    )
    assert parsed["strict_valid"] is True


def test_resume_planning_and_duplicate_raw_rejection(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, output_length=10)
    raw_path = tmp_path / config["outputs"]["raw"]["qwen2"]
    runner.append_raw_record(raw_path, {"request_id": "REQ-001", "raw_response": '{"ranking":[1,2,3,4,5,6,7,8,9,10]}', "transport_status": "success"})
    runner.append_raw_record(raw_path, {"request_id": "REQ-001", "raw_response": '{"ranking":[1,2,3,4,5,6,7,8,9,10]}', "transport_status": "success"})
    raw, duplicates, errors = f6g2.load_raw_records(tmp_path, config)
    assert duplicates == ["REQ-001"]
    assert errors == []
    assert f6g2.plan_resume(["REQ-001", "REQ-002"], raw) == ["REQ-002"]


def test_allowed_top10_contract_parses_raw_and_separates_populations(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, output_length=10)
    for request_id, model in [("REQ-001", "qwen2"), ("REQ-002", "mistral"), ("REQ-003", "qwen2"), ("REQ-004", "mistral")]:
        runner.append_raw_record(
            tmp_path / config["outputs"]["raw"][model],
            {"request_id": request_id, "raw_response": '{"ranking":[1,2,3,4,5,6,7,8,9,10]}', "transport_status": "success", "duration_seconds": "1"},
        )
    manifest = f6g2.run_phase(tmp_path, config, mode="analysis-only")
    assert manifest["decision"] == f6g2.DECISION_COMPLETE
    parser = _read_csv(tmp_path / config["outputs"]["tables"]["parser_validity"])
    assert {row["strict_valid"] for row in parser} == {"true"}
    assert next(row for row in parser if row["request_id"] == "REQ-001")["positive_final_rank"] == "3"
    cell = _read_csv(tmp_path / config["outputs"]["tables"]["cell_execution_summary"])
    assert {row["population"] for row in cell} == {
        "all_user_operational",
        "all_user_strict_valid_conditional",
        "positive_present_operational_diagnostic",
        "positive_present_strict_valid_conditional_diagnostic",
    }


def test_manifest_scientific_hash_is_deterministic(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path, output_length=20)
    first = f6g2.run_phase(tmp_path, config, mode="preflight-only")
    first_hashes = dict(first["table_hashes"])
    second = f6g2.run_phase(tmp_path, config, mode="preflight-only")
    assert second["scientific_content_hash"] == first["scientific_content_hash"]
    assert second["table_hashes"] == first_hashes
    assert second["runtime_metadata"]["analysis_only_determinism_result"] == "stable_with_previous_manifest"


def test_runner_cli_execute_does_not_generate_when_contract_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _fixture_config(tmp_path, output_length=20)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_dump(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert runner.main(["--config", str(config_path), "--execute"]) == 0
    manifest = f6g2.load_json(tmp_path / config["outputs"]["freeze_manifest"])
    assert manifest["decision"] == f6g2.DECISION_CONTRACT
    assert not (tmp_path / config["outputs"]["raw"]["qwen2"]).exists()


def yaml_dump(value: object) -> str:
    import yaml

    return yaml.safe_dump(value, sort_keys=False)

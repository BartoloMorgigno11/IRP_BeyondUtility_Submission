from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path

from src.analysis import plan_f6g2r_runtime_recovery as f6g2r


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_import_has_no_subprocess_side_effects(monkeypatch):
    calls: list[object] = []

    def fake_run(*args, **kwargs):  # pragma: no cover - should never be called
        calls.append((args, kwargs))
        raise AssertionError("subprocess.run called during import")

    monkeypatch.setattr(f6g2r.subprocess, "run", fake_run)
    importlib.reload(f6g2r)
    assert calls == []


def test_no_unsafe_runtime_imports():
    tree = ast.parse(Path(f6g2r.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "requests" not in imported
    assert "ollama" not in imported
    assert "torch" not in imported
    assert "tensorflow" not in imported


def test_python_candidate_classifies_windowsapps_stub():
    result = f6g2r.classify_python_candidate(
        Path("C:/Program Files/WindowsApps/PythonSoftwareFoundation.Python.3.12/python.exe"),
        size_bytes=0,
    )
    assert result["windowsapps_related"] == "true"
    assert result["alias_or_stub_classification"] == "windowsapps_app_execution_alias_or_stub"


def test_dependency_lock_requires_exact_wheel_hash(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "example-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")
    config = {
        "dependencies": {
            "wheelhouse": "wheelhouse",
            "pip_index_policy": "official_pypi_binary_wheels_only",
            "pins": [
                {
                    "package": "example",
                    "version": "1.0.0",
                    "wheel": wheel.name,
                    "sha256": _sha(wheel),
                }
            ],
        }
    }
    rows = f6g2r.build_dependency_lock(config, tmp_path)
    assert rows[0]["status"] == "pass"
    assert rows[0]["wheel_tag"] == "py3-none-any"

    config["dependencies"]["pins"][0]["sha256"] = "0" * 64
    rows = f6g2r.build_dependency_lock(config, tmp_path)
    assert rows[0]["status"] == "fail"


def _contract_config() -> dict[str, object]:
    return {
        "f6g2_contract": {
            "expected_logical_instances": 2,
            "expected_requests": 4,
            "expected_shards": 2,
            "top10_required_requested_ranking_length": 10,
            "top10_required_parser_expected_length": 10,
        }
    }


def _logical_rows(output_length: str) -> list[dict[str, str]]:
    return [
        {"logical_instance_id": "L1", "expected_output_length": output_length, "parser_contract": f"indexed_json_top{output_length}_unique_candidates"},
        {"logical_instance_id": "L2", "expected_output_length": output_length, "parser_contract": f"indexed_json_top{output_length}_unique_candidates"},
    ]


def _request_rows(output_length: str) -> list[dict[str, str]]:
    candidates = "|".join(str(index) for index in range(1, 21))
    rows: list[dict[str, str]] = []
    for index in range(4):
        rows.append(
            {
                "request_id": f"R{index}",
                "candidate_item_ids": candidates,
                "required_output_length": output_length,
                "parser_contract": f"indexed_json_top{output_length}_unique_candidates",
            }
        )
    return rows


def test_contract_summary_blocks_top20_against_top10_adapter():
    summary = f6g2r.summarize_contract(
        _contract_config(),
        _logical_rows("20"),
        _request_rows("20"),
        [{"shard_id": "S1"}, {"shard_id": "S2"}],
        [{"requested_ranking_length": "20", "parser_expected_length": "20"}],
    )
    assert summary["contract_status"] == "blocked_by_top20_top10_conflict"
    assert summary["execution_allowed"] is False
    assert summary["candidate_pool_sizes"] == ["20"]


def test_contract_summary_allows_internally_correct_top10_requests():
    summary = f6g2r.summarize_contract(
        _contract_config(),
        _logical_rows("10"),
        _request_rows("10"),
        [{"shard_id": "S1"}, {"shard_id": "S2"}],
        [{"requested_ranking_length": "10", "parser_expected_length": "10"}],
    )
    assert summary["contract_status"] == "pass"
    assert summary["execution_allowed"] is True


def test_parse_ollama_list_short_ids():
    stdout = """NAME                 ID              SIZE      MODIFIED
mistral:7b-instruct  6577803aa9a0    4.4 GB    3 weeks ago
qwen2:7b-instruct    dd314f039b9d    4.4 GB    3 weeks ago
"""
    parsed = f6g2r.parse_ollama_list(stdout)
    assert parsed["mistral:7b-instruct"] == "6577803aa9a0"
    assert parsed["qwen2:7b-instruct"] == "dd314f039b9d"


def _rows(status: str = "pass") -> list[dict[str, object]]:
    return [{"status": status, "changed": "false", "missing": "false"}]


def test_decision_priority_contract_after_runtime_gates_pass():
    decision, recommendation = f6g2r.select_final_decision(
        _rows(),
        _rows(),
        _rows(),
        _rows(),
        _rows(),
        _rows(),
        _rows(),
        [{"status": "pass", "passed": 3, "failed": 0, "skipped": 0}],
        [{"status": "blocked"}],
    )
    assert decision == f6g2r.DECISION_CONTRACT
    assert recommendation == f6g2r.RECOMMEND_CONTRACT


def test_decision_priority_input_invariance_first():
    decision, recommendation = f6g2r.select_final_decision(
        [{"status": "changed", "changed": "true", "missing": "false"}],
        _rows("fail"),
        _rows("fail"),
        _rows("fail"),
        _rows("fail"),
        _rows("fail"),
        _rows("fail"),
        [{"status": "fail", "passed": 0, "failed": 1, "skipped": 0}],
        [{"status": "blocked"}],
    )
    assert decision == f6g2r.DECISION_INPUTS
    assert recommendation == f6g2r.RECOMMEND_BLOCKED


def test_manifest_scientific_hash_is_runtime_timestamp_independent(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    config = {"validation": {"unsafe_exclusions": ["network"]}}
    kwargs = {
        "config": config,
        "root": tmp_path,
        "decision": f6g2r.DECISION_CONTRACT,
        "recommendation": f6g2r.RECOMMEND_CONTRACT,
        "repository_state": {"head": "abc"},
        "identity": {"architecture": "64bit", "python_sha256": "py", "python_version": "3.12.10", "executable": "venv/python"},
        "table_hashes": {"table": "hash"},
        "environment_hashes": {"pip_freeze": "freeze"},
        "report_hash": "report",
        "summary": {
            "contract_status": "blocked_by_top20_top10_conflict",
            "logical_count": 2,
            "request_count": 4,
            "shard_count": 2,
            "candidate_pool_sizes": ["20"],
            "request_output_lengths": ["20"],
            "f6g2_requested_lengths": ["20"],
            "f6g2_parser_lengths": ["20"],
            "execution_allowed": False,
        },
        "test_rows": [{"passed": 1, "failed": 0, "skipped": 0}],
        "input_rows": [{"path": "input.csv", "actual_sha256": "abc", "changed": "false", "missing": "false"}],
        "manifest_path": manifest_path,
    }
    first = f6g2r.build_manifest(**kwargs)
    second = f6g2r.build_manifest(**kwargs)
    assert first["scientific_content_hash"] == second["scientific_content_hash"]

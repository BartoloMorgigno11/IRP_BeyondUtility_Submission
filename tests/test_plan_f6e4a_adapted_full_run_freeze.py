from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.analysis import plan_f6e4a_adapted_full_run_freeze as f6e4a
from src.llm.plan_f6e1_prompt_strategies import BASE_STRATEGY_ID, JSON_INDEX_STRATEGY_ID, LLMRANK_STRATEGY_ID, LLMSREC_STRATEGY_ID


def test_import_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    importlib.reload(f6e4a)
    after = sorted(tmp_path.rglob("*"))
    assert before == after


def test_source_contains_no_runtime_or_inference_calls() -> None:
    source = Path(f6e4a.__file__).read_text(encoding="utf-8").lower()
    forbidden = [
        "inspect_ollama_gate",
        "execute_ollama_request",
        "/api/generate",
        "/api/version",
        "/api/tags",
        "ollama pull",
        "allow_ollama_generate",
        "urlopen",
    ]
    assert not any(token in source for token in forbidden)


def test_frozen_raw_and_f6e3b_hashes_are_verified() -> None:
    observed = f6e4a.verify_expected_hashes(Path.cwd())
    for path, expected in {**f6e4a.F6D4A_RAW_HASHES, **f6e4a.F6E3_RAW_HASHES, **f6e4a.F6E3B_RAW_HASHES}.items():
        assert observed[path] == expected


def test_source_context_counts_and_lastfm_reranking_exclusion() -> None:
    contexts = f6e4a.build_source_contexts(Path.cwd())
    assert len(contexts) == 7623
    counts = {}
    for context in contexts:
        counts[(context.dataset, context.task)] = counts.get((context.dataset, context.task), 0) + 1
    assert counts[("Amazon Beauty", "ranking")] == 990
    assert counts[("Amazon Beauty", "reranking")] == 990
    assert counts[("Amazon Sports", "ranking")] == 892
    assert counts[("Amazon Sports", "reranking")] == 892
    assert counts[("ML-1M", "ranking")] == 1000
    assert counts[("ML-1M", "reranking")] == 1000
    assert counts[("LastFM", "ranking")] == 1859
    assert ("LastFM", "reranking") not in counts


def test_request_manifest_has_only_four_allowed_cells_and_expected_size() -> None:
    rows = f6e4a.build_request_manifest(Path.cwd(), f6e4a.build_source_contexts(Path.cwd()))
    assert len(rows) == 30492
    assert len({row["request_id"] for row in rows}) == len(rows)
    assert {row["model"] for row in rows} == {f6e4a.QWEN2_TAG, f6e4a.MISTRAL_TAG}
    assert {row["core_strategy_id"] for row in rows} == {BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID}
    assert {row["adapter_classification"] for row in rows} == {"adapted_indexed_output_contract"}
    assert JSON_INDEX_STRATEGY_ID not in {row["core_strategy_id"] for row in rows}
    assert LLMSREC_STRATEGY_ID not in {row["core_strategy_id"] for row in rows}
    assert all(row["model_equivalence"] == "exact_family_version_uncertain" for row in rows)


def test_prompt_and_candidate_hashes_are_stable() -> None:
    context = f6e4a.build_source_contexts(Path.cwd())[0]
    rows = f6e4a.build_request_manifest(Path.cwd(), [context], validate_expected=False)
    row = rows[0]
    core, adapter, complete = f6e4a.render_complete_prompt(Path.cwd(), row["core_strategy_id"], context)
    assert row["candidate_order_hash"] == f6e4a.sha256_text(f6e4a.stable_json(context.candidate_ids))
    assert row["core_prompt_hash"] == f6e4a.sha256_text(core)
    assert row["adapter_hash"] == f6e4a.sha256_text(adapter)
    assert row["complete_prompt_hash"] == f6e4a.sha256_text(complete)
    assert row["expected_schema"] == '{"ranking":[1,2,3,4,5,6,7,8,9,10]}'


def test_sharding_is_deterministic_complete_and_non_overlapping() -> None:
    rows = f6e4a.build_request_manifest(Path.cwd(), f6e4a.build_source_contexts(Path.cwd()))
    shards = f6e4a.build_shard_manifest(Path.cwd(), rows)
    assert len(shards) == 64
    assert max(int(row["request_count"]) for row in shards) <= 500
    assert sum(int(row["request_count"]) for row in shards) == len(rows)
    assert all(row["resume_key"] == "request_id" for row in shards)
    by_shard = {}
    for row in rows:
        by_shard.setdefault(row["shard_id"], []).append(row["request_id"])
    assert set(by_shard) == {row["shard_id"] for row in shards}
    assert len({request_id for ids in by_shard.values() for request_id in ids}) == len(rows)


def test_runtime_policy_stop_conditions_and_future_raw_contract() -> None:
    runtime = f6e4a.build_runtime_parameters()
    assert {row["classification"] for row in runtime} >= {"adapted_runtime", "default_empty_unset"}
    assert any(row["parameter_name"] == "num_predict" and row["value"] == "256" for row in runtime)
    stops = f6e4a.build_stop_conditions()
    names = {row["condition_name"] for row in stops}
    assert {"ollama_unavailable", "digest_mismatch", "raw_write_failure", "protected_input_change", "systematic_non_json"}.issubset(names)
    assert all(row["retry_allowed"] == "no" for row in stops)
    assert f6e4a.FUTURE_RAW_PATHS[f6e4a.QWEN2_TAG].as_posix().endswith("_qwen2_adapted_full_run_raw.jsonl")
    assert f6e4a.FUTURE_RAW_PATHS[f6e4a.MISTRAL_TAG].as_posix().endswith("_mistral_adapted_full_run_raw.jsonl")


def test_build_all_has_expected_schemas_decision_and_budget() -> None:
    outputs = f6e4a.build_all(Path.cwd())
    assert outputs.manifest["decision"] == "f6e4a_adapted_full_run_freeze_complete"
    assert outputs.manifest["total_source_contexts"] == 7623
    assert outputs.manifest["total_expected_calls"] == 30492
    assert outputs.manifest["total_shards"] == 64
    assert len(outputs.tables["execution_matrix"]) == 28
    assert any(row["scope"] == "global_total" for row in outputs.tables["time_storage_budget"])
    assert outputs.manifest["protected_summary"]["f2_changed"] == 0
    assert outputs.manifest["protected_summary"]["f6d4a_changed"] == 0


def test_no_plan_e_or_previous_phase_writes(tmp_path: Path) -> None:
    written = f6e4a.write_outputs(f6e4a.build_all(Path.cwd()), tmp_path)
    assert written
    assert all(path.startswith(("results/tables/phase_f6e4a_", "results/manifests/phase_f6e4a_", "report/phase_f6e4a_")) for path in written)


def test_outputs_are_deterministic(tmp_path: Path) -> None:
    first = f6e4a.build_all(Path.cwd())
    second = f6e4a.build_all(Path.cwd())
    assert first.manifest["scientific_content_hash"] == second.manifest["scientific_content_hash"]
    h1 = f6e4a.write_outputs(first, tmp_path / "a")
    h2 = f6e4a.write_outputs(second, tmp_path / "b")
    assert h1.keys() == h2.keys()
    for path, digest in h1.items():
        assert digest == h2[path]

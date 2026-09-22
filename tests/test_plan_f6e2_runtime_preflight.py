from __future__ import annotations

import csv
import importlib
import json
import shutil
from pathlib import Path

import pytest

from src.analysis import plan_f6e2_runtime_preflight as f6e2
from src.llm.plan_f6e1_prompt_strategies import (
    BASE_STRATEGY_ID,
    JSON_INDEX_STRATEGY_ID,
    LLMRANK_STRATEGY_ID,
    LLMSREC_STRATEGY_ID,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_import_has_no_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = set(tmp_path.rglob("*"))
    monkeypatch.chdir(tmp_path)
    importlib.reload(f6e2)
    after = set(tmp_path.rglob("*"))
    assert before == after


def test_source_contains_no_generation_or_download_commands() -> None:
    source = Path(f6e2.__file__).read_text(encoding="utf-8").lower()
    forbidden = ["api/generate", "ollama pull", "pip install", "git clone", "urlopen", "import requests", "from requests"]
    assert not any(token in source for token in forbidden)


def test_prompt_rendering_and_token_estimation_are_local_and_deterministic() -> None:
    context = f6e2.PromptRenderContext(
        dataset="Amazon Beauty",
        user_name="u1",
        user_history_titles=tuple(f"History {idx}" for idx in range(10)),
        candidate_titles=tuple(f"Candidate {idx}" for idx in range(20)),
    )
    prompt_a = f6e2.render_strategy_prompt(BASE_STRATEGY_ID, context, root=Path.cwd())
    prompt_b = f6e2.render_strategy_prompt(BASE_STRATEGY_ID, context, root=Path.cwd())
    assert prompt_a == prompt_b
    assert f6e2.estimate_tokens(prompt_a) == f6e2.estimate_tokens(prompt_b)
    assert f6e2.estimate_tokens(prompt_a) > 0


def test_strategy_classifications_are_preserved() -> None:
    assert f6e2.STRATEGY_CLASSIFICATION[BASE_STRATEGY_ID] == "exact_parametric"
    assert f6e2.STRATEGY_CLASSIFICATION[LLMRANK_STRATEGY_ID] == "exact_parametric"
    assert f6e2.STRATEGY_CLASSIFICATION[LLMSREC_STRATEGY_ID] == "reconstructed"
    assert f6e2.STRATEGY_CLASSIFICATION[JSON_INDEX_STRATEGY_ID] == "local_ablation"


def test_build_outputs_distinguish_observed_from_estimated() -> None:
    outputs = f6e2.build_all(Path.cwd())
    runtime_rows = outputs.tables["runtime_estimates"]
    assert runtime_rows
    assert any(row["estimate_basis"] == "estimate_from_same_model" for row in runtime_rows)
    assert any(row["estimate_basis"] == "not_estimable" for row in runtime_rows)


def test_model_unavailable_and_llama70b_memory_blocks() -> None:
    outputs = f6e2.build_all(Path.cwd())
    models = {row["model_tag"]: row for row in outputs.tables["model_inventory"]}
    assert models["llama3:70b-instruct"]["availability_status"] == "missing"
    scenarios = [row for row in outputs.tables["memory_disk_feasibility"] if row["model_tag"] == "llama3:70b-instruct"]
    assert scenarios
    assert any(row["scenario_status"] == "blocked_by_ram" for row in scenarios)
    assert any(row["scenario_status"] == "blocked_by_vram" for row in scenarios)
    assert any(row["scenario_status"] == "technically_possible_but_impractical" for row in scenarios)


def test_execution_matrix_has_single_allowed_status_and_no_exact_llmsrec() -> None:
    outputs = f6e2.build_all(Path.cwd())
    allowed = {
        "ready_local_exact_prompt",
        "ready_local_reconstructed_prompt",
        "ready_local_extension",
        "conditional_missing_model",
        "conditional_context_risk",
        "conditional_runtime_risk",
        "blocked_hardware",
        "blocked_missing_manifest",
        "blocked_missing_prompt_evidence",
        "blocked_closed_source",
        "blocked_retired_model",
        "not_applicable",
    }
    assert len(outputs.tables["execution_matrix"]) == 11
    for row in outputs.tables["execution_matrix"]:
        assert row["preflight_status"] in allowed
        if row["strategy_id"] == LLMSREC_STRATEGY_ID:
            assert row["preflight_status"] != "ready_local_exact_prompt"


def test_missing_manifest_classification_helper() -> None:
    row = f6e2.build_lastfm_workload_row(Path("missing-root"), "ranking", Path("missing.csv"))
    assert row["manifest_sha256"] == "missing"
    assert row["logical_instances"] == "0"


def test_request_multiplication_and_resume_workload() -> None:
    outputs = f6e2.build_all(Path.cwd())
    lastfm_ranking = [
        row for row in outputs.tables["request_workload"] if row["dataset"] == "LastFM" and row["task"] == "ranking"
    ][0]
    assert lastfm_ranking["logical_instances"] == "1859"
    llmsrec_rows = [row for row in outputs.tables["runtime_estimates"] if row["strategy_id"] == LLMSREC_STRATEGY_ID and row["model_tag"] == "qwen2:7b-instruct"]
    assert llmsrec_rows
    assert all(float(row["calls_per_instance_max"]) == 2.0 for row in llmsrec_rows)
    assert all(row["resume_possible"] == "true" for row in llmsrec_rows)


def test_context_overflow_and_disk_evidence_are_represented() -> None:
    outputs = f6e2.build_all(Path.cwd())
    tokens = outputs.tables["prompt_token_estimates"]
    assert tokens
    assert all("above_4k_percent" in row for row in tokens)
    storage = [row for row in outputs.tables["hardware_inventory"] if row["component_type"] == "storage"]
    assert storage and "185582936064" in storage[0]["observed_value"]


def test_output_determinism(tmp_path: Path) -> None:
    first = f6e2.build_all(Path.cwd())
    first_hash = first.manifest["scientific_content_hash"]
    second = f6e2.build_all(Path.cwd())
    assert first_hash == second.manifest["scientific_content_hash"]
    h1 = f6e2.write_outputs(first, tmp_path)
    h2 = f6e2.write_outputs(second, tmp_path)
    assert h1 == h2


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_raw_f6d4a_hashes_invariant() -> None:
    summary = f6e2.protected_input_summary(Path.cwd())
    assert summary["qwen2_raw_hash"] == f6e2.QWEN2_RAW_SHA256
    assert summary["mistral_raw_hash"] == f6e2.MISTRAL_RAW_SHA256
    assert summary["raw_changed"] == 0


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_protected_inputs_invariant() -> None:
    summary = f6e2.protected_input_summary(Path.cwd())
    assert int(summary["f2_checked"]) >= 1144
    assert int(summary["f2_changed"]) == 0
    assert int(summary["f6d4a_changed"]) == 0


def test_no_plan_e_canonical_writes(tmp_path: Path) -> None:
    outputs = f6e2.build_all(Path.cwd())
    f6e2.write_outputs(outputs, tmp_path)
    written = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert written
    assert all("plan_e_" not in path for path in written)
    assert all(path.startswith(("results/tables/phase_f6e2_", "results/manifests/phase_f6e2_", "report/phase_f6e2_")) for path in written)


def test_manifest_contains_required_hash_blocks() -> None:
    outputs = f6e2.build_all(Path.cwd())
    manifest = outputs.manifest
    assert manifest["prompt_freeze_hash"] == f6e2.F6E1_SCIENTIFIC_HASH
    assert "hardware_fingerprint" in manifest
    assert "runtime_versions" in manifest
    assert "local_model_digests" in manifest
    assert "workload_hashes" in manifest
    assert manifest["decision"] == "f6e2_runtime_preflight_complete_with_documented_blocks"


def test_written_artifact_schemas(tmp_path: Path) -> None:
    outputs = f6e2.build_all(Path.cwd())
    f6e2.write_outputs(outputs, tmp_path)
    for name, rel_path in f6e2.OUTPUT_TABLES.items():
        rows = read_rows(tmp_path / rel_path)
        assert rows, name
    manifest = json.loads((tmp_path / f6e2.MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest["phase"] == "F6E2"
    assert (tmp_path / f6e2.REPORT_PATH).exists()

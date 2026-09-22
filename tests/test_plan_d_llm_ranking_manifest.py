from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_llm_prompt_templates_and_ranking_manifest.md"

REQUIRED_COLUMNS = {
    "plan",
    "phase",
    "dataset",
    "user_id",
    "instance_id",
    "prompt_strategy",
    "model",
    "backend",
    "top_k",
    "candidate_pool_size",
    "history_length",
    "candidate_count",
    "positive_item_id",
    "positive_in_candidates",
    "history_preview",
    "candidate_ids",
    "candidate_titles_preview",
    "prompt_char_length",
    "manifest_status",
    "notes",
}


def test_manifest_and_report_exist() -> None:
    assert MANIFEST_PATH.exists()
    assert REPORT_PATH.exists()


def test_manifest_has_required_columns_and_expected_scope() -> None:
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")

    assert REQUIRED_COLUMNS.issubset(manifest.columns)
    assert 0 < len(manifest) <= 20
    assert len(manifest) == 20
    assert set(manifest["plan"]) == {"Plan D"}
    assert set(manifest["phase"]) == {"D5E_qwen2_ml1m_ranking_diagnostic"}
    assert set(manifest["dataset"]) == {"ml1m"}
    assert set(manifest["model"]) == {"qwen2:7b-instruct"}
    assert set(manifest["backend"]) == {"ollama"}
    assert set(manifest["prompt_strategy"]) == {"openp5_base"}
    assert set(manifest["top_k"].astype(int)) == {5}


def test_ready_manifest_rows_have_20_candidates_and_positive_targets() -> None:
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    ready = manifest.loc[manifest["manifest_status"] == "ready_for_llm_inference"].copy()

    assert len(ready) == 20
    assert set(ready["candidate_count"].astype(int)) == {20}
    assert set(ready["candidate_pool_size"].astype(int)) == {20}
    assert ready["history_length"].astype(int).min() > 0
    assert ready["prompt_char_length"].astype(int).min() > 0
    assert set(ready["positive_in_candidates"].str.lower()) == {"true"}
    assert ready["positive_item_id"].str.len().min() > 0

    for encoded_ids in ready["candidate_ids"]:
        candidate_ids = json.loads(encoded_ids)
        assert len(candidate_ids) == 20
        assert all(str(item_id).strip() for item_id in candidate_ids)


def test_manifest_previews_are_non_empty() -> None:
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")

    assert manifest["history_preview"].str.len().min() > 0
    assert manifest["candidate_titles_preview"].str.len().min() > 0


def test_manifest_report_documents_no_inference_or_external_api() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8").lower()

    assert "no llm inference was run" in report
    assert "no external api was called" in report
    assert "ml-1m 20-user diagnostic manifest" in report
    assert "candidate pool setup" in report
    assert "positives in candidates" in report


def test_manifest_tests_do_not_call_ollama_or_external_apis() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    string_constants = [
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]

    assert not {"requests", "urllib", "subprocess", "socket", "openai", "anthropic"} & imported_modules
    assert not any(("ollama " + "run") in value for value in string_constants)
    assert not any(("localhost:" + "11434") in value for value in string_constants)

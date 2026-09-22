from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.llm.plan_d_mistral_smoke_test import (
    DEFAULT_MODEL,
    METRIC_COLUMNS,
    SUMMARY_COLUMNS,
    LocalModelInventory,
    blocked_evaluation,
    build_mistral_smoke_prompt,
    detect_mistral_candidates,
    evaluate_smoke_response,
    metric_rows,
    parse_ollama_list_output,
    select_local_mistral_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = PROJECT_ROOT / "src/llm/plan_d_mistral_smoke_test.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/run_plan_d_mistral_smoke_test.py"
SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_mistral_smoke_test_summary.csv"
METRICS_PATH = PROJECT_ROOT / "results/metrics/plan_d_mistral_smoke_test_metrics.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_mistral_smoke_test.md"
REPRO_MATRIX_PATH = PROJECT_ROOT / "report/plan_d_reproducibility_matrix.md"
RAW_OUTPUT_PATH = PROJECT_ROOT / "results/raw_outputs/plan_d_mistral_smoke_test_raw.jsonl"


def test_parse_ollama_list_output_from_tiny_synthetic_output() -> None:
    output = """NAME                         ID              SIZE      MODIFIED
qwen2:7b-instruct            abc123          4.4 GB    2 days ago
mistral:7b-instruct          def456          4.1 GB    1 day ago
"""

    assert parse_ollama_list_output(output) == ["qwen2:7b-instruct", "mistral:7b-instruct"]


def test_detect_mistral_candidates_and_select_exact_requested_model() -> None:
    models = ["qwen2:7b-instruct", "mistral:7b-instruct", "mistral:latest"]

    assert detect_mistral_candidates(models) == ["mistral:7b-instruct", "mistral:latest"]
    assert select_local_mistral_model(models, "mistral:7b-instruct") == "mistral:7b-instruct"
    assert select_local_mistral_model(models, "mistral:7b-instruct-v0.2") is None


def test_smoke_prompt_contains_candidate_id_only_constraints() -> None:
    prompt = build_mistral_smoke_prompt(["101", "102", "103", "104", "105"], top_k=3).lower()

    assert "return only valid json" in prompt
    assert "select exactly 3 item ids" in prompt
    assert "copied exactly from the allowed candidate id list" in prompt
    assert "do not invent ids" in prompt
    assert "do not output explanations" in prompt
    for item_id in ("101", "102", "103", "104", "105"):
        assert item_id in prompt


def test_strict_json_parsing_of_valid_tiny_response() -> None:
    parsed, metrics = evaluate_smoke_response(
        '{"ranked_item_ids":["101","103","105"],"status":"ok"}',
        response_received=True,
    )

    assert parsed.parse_success is True
    assert parsed.schema_valid is True
    assert metrics["response_received"] is True
    assert metrics["parse_success"] is True
    assert metrics["schema_valid"] is True
    assert metrics["exact_top_k"] is True
    assert metrics["candidate_validity_rate"] == 1.0
    assert metrics["hallucination_rate"] == 0.0


def test_candidate_validity_and_hallucination_metrics_on_synthetic_response() -> None:
    parsed, metrics = evaluate_smoke_response(
        '{"ranked_item_ids":["101","999","101"],"status":"ok"}',
        response_received=True,
    )

    assert parsed.hallucinated_count == 1
    assert parsed.duplicate_count == 1
    assert metrics["candidate_validity_rate"] == 2 / 3
    assert metrics["hallucination_rate"] == 1 / 3
    assert metrics["duplicate_rate"] == 1 / 3


def test_blocked_path_status_and_blank_metric_values() -> None:
    inventory = LocalModelInventory(
        command_available=True,
        model_names=["qwen2:7b-instruct"],
        raw_output="",
        error_message="",
    )
    evaluation = blocked_evaluation(requested_model=DEFAULT_MODEL, inventory=inventory)
    rows = metric_rows(evaluation)

    assert evaluation.status == "blocked_model_not_available"
    assert evaluation.local_request_count == 0
    assert evaluation.local_ollama_used is False
    assert all(row["status"] == "blocked_model_not_available" for row in rows)
    assert all(row["value"] == "" for row in rows)


def test_raw_output_path_is_under_results_raw_outputs() -> None:
    raw_path = RAW_OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()

    assert raw_path.startswith("results/raw_outputs/")
    assert raw_path.endswith(".jsonl")


def test_d5h_artifacts_if_present_have_expected_structure() -> None:
    if not SUMMARY_PATH.exists():
        pytest.skip("D5H outputs are not generated before the local availability/smoke run")
    summary = pd.read_csv(SUMMARY_PATH, dtype=str).fillna("")
    metrics = pd.read_csv(METRICS_PATH, dtype=str).fillna("")

    assert set(SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(METRIC_COLUMNS).issubset(metrics.columns)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["backend"] == "ollama"
    assert row["requested_model"]
    assert row["status"] in {
        "blocked_model_not_available",
        "smoke_test_passed",
        "smoke_test_failed_backend",
        "smoke_test_failed_parse",
        "smoke_test_failed_schema",
        "smoke_test_failed_candidate_validity",
    }
    if row["status"] == "blocked_model_not_available":
        assert row["local_request_count"] == "0"
        assert row["response_received"] == ""
    else:
        assert row["local_request_count"] == "1"
        assert row["response_received"] in {"true", "false"}
        assert row["raw_output_path"].startswith("results/raw_outputs/")


def test_d5h_report_contains_no_api_no_download_no_training_statements_if_present() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5H report is not generated before the local availability/smoke run")
    report = REPORT_PATH.read_text(encoding="utf-8").lower()
    matrix = REPRO_MATRIX_PATH.read_text(encoding="utf-8").lower()

    assert "not exact reproduction" in report
    assert "not full llm reproduction" in report
    assert "no openai api call was made" in report
    assert "no anthropic api call was made" in report
    assert "no external api call was made" in report
    assert "no recommender training was performed" in report
    assert "plan c was not modified" in report
    assert "d5h does not provide ml-1m ranking metrics" in report
    assert "mistral" in matrix
    assert "not exact reproduction" in matrix or "conditional_manual_pull" in matrix or "smoke_test_completed" in matrix


def test_d5h_tests_do_not_call_ollama_or_external_apis() -> None:
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
    assert HELPER_PATH.exists()
    assert RUNNER_PATH.exists()

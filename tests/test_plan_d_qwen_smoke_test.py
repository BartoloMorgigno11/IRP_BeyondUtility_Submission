from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = PROJECT_ROOT / "src/llm/plan_d_ollama_client.py"
PARSER_PATH = PROJECT_ROOT / "src/llm/plan_d_llm_json_parser.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/run_plan_d_qwen_smoke_test.py"
SMOKE_CSV_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ollama_smoke_test.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_qwen2_ollama_smoke_test.md"

REQUIRED_COLUMNS = {
    "plan",
    "phase",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "response_received",
    "parse_success",
    "schema_valid",
    "selected_count",
    "candidate_validity_rate",
    "duplicate_count",
    "hallucinated_count",
    "latency_seconds",
    "raw_output_path",
    "status",
    "notes",
}
ALLOWED_STATUSES = {
    "smoke_test_passed",
    "smoke_test_failed_parse",
    "smoke_test_failed_schema",
    "smoke_test_failed_backend",
    "smoke_test_failed_candidate_validity",
}
BOOLEAN_LIKE = {"true", "false", "True", "False", True, False}


def test_plan_d_qwen_smoke_files_exist() -> None:
    assert CLIENT_PATH.exists()
    assert PARSER_PATH.exists()
    assert RUNNER_PATH.exists()
    assert SMOKE_CSV_PATH.exists()
    assert REPORT_PATH.exists()


def test_plan_d_qwen_smoke_csv_schema_and_single_row() -> None:
    frame = pd.read_csv(SMOKE_CSV_PATH)

    assert len(frame) == 1
    assert REQUIRED_COLUMNS.issubset(frame.columns)
    row = frame.iloc[0]
    assert row["plan"] == "Plan D"
    assert row["backend"] == "ollama"
    assert row["model"] == "qwen2:7b-instruct"
    assert float(row["temperature"]) == 0.0
    assert row["status"] in ALLOWED_STATUSES


def test_plan_d_qwen_smoke_csv_values_are_machine_readable() -> None:
    row = pd.read_csv(SMOKE_CSV_PATH).iloc[0]

    assert row["response_received"] in BOOLEAN_LIKE
    assert row["parse_success"] in BOOLEAN_LIKE
    assert row["schema_valid"] in BOOLEAN_LIKE
    assert int(row["selected_count"]) >= 0
    assert int(row["duplicate_count"]) >= 0
    assert int(row["hallucinated_count"]) >= 0
    assert float(row["latency_seconds"]) >= 0.0
    candidate_validity_rate = float(row["candidate_validity_rate"])
    if str(row["response_received"]).lower() == "true":
        assert 0.0 <= candidate_validity_rate <= 1.0


def test_plan_d_qwen_smoke_report_documents_scope_limits() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8").lower()

    assert "no external api was called" in report
    assert "not a dataset ranking experiment" in report
    assert "not an exact paper llm reproduction" in report
    assert "no model was pulled or downloaded" in report
    assert "no recommender model was trained" in report
    assert "no plan c output was modified" in report


def test_plan_d_qwen_smoke_tests_do_not_call_ollama_or_network() -> None:
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

    assert not {"requests", "urllib", "subprocess", "socket"} & imported_modules
    assert not any(("localhost:" + "11434") in value for value in string_constants)
    assert not any(("ollama " + "run") in value for value in string_constants)
    assert not any(("ollama " + "pull") in value for value in string_constants)

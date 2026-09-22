from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT_ROOT / "src/llm/plan_d_llm_ranking_evaluator.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/run_plan_d_qwen2_ml1m_ranking_diagnostic.py"
METRICS_PATH = PROJECT_ROOT / "results/metrics/plan_d_qwen2_ml1m_ranking_diagnostic_metrics.csv"
INSTANCE_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv"
AGGREGATE_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_summary.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_qwen2_ml1m_ranking_diagnostic.md"

METRICS_COLUMNS = {
    "plan",
    "phase",
    "dataset",
    "scope",
    "method",
    "backend",
    "model",
    "prompt_strategy",
    "metric",
    "k",
    "value",
    "num_instances",
    "notes",
}

INSTANCE_COLUMNS = {
    "plan",
    "phase",
    "dataset",
    "scope",
    "user_id",
    "instance_id",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "top_k",
    "candidate_pool_size",
    "positive_item_id",
    "response_received",
    "parse_success",
    "schema_valid",
    "selected_count",
    "candidate_validity_rate",
    "duplicate_count",
    "hallucinated_count",
    "positive_rank",
    "hit_at_5",
    "ndcg_at_5",
    "latency_seconds",
    "parse_status",
    "status",
    "notes",
}

AGGREGATE_COLUMNS = {
    "plan",
    "phase",
    "dataset",
    "scope",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "num_instances",
    "num_responses_received",
    "response_received_rate",
    "parse_success_rate",
    "schema_valid_rate",
    "mean_candidate_validity_rate",
    "hallucination_rate",
    "duplicate_rate",
    "exact_top_k_rate",
    "empty_output_rate",
    "hit_rate_at_5",
    "ndcg_at_5",
    "mean_latency_seconds",
    "total_runtime_seconds",
    "raw_output_path",
    "status",
    "notes",
}

RATE_COLUMNS = {
    "response_received_rate",
    "parse_success_rate",
    "schema_valid_rate",
    "mean_candidate_validity_rate",
    "hallucination_rate",
    "duplicate_rate",
    "exact_top_k_rate",
    "empty_output_rate",
    "hit_rate_at_5",
    "ndcg_at_5",
}


def test_d5e_runner_and_evaluator_exist() -> None:
    assert EVALUATOR_PATH.exists()
    assert RUNNER_PATH.exists()


def test_d5e_output_files_exist() -> None:
    assert METRICS_PATH.exists()
    assert INSTANCE_SUMMARY_PATH.exists()
    assert AGGREGATE_SUMMARY_PATH.exists()
    assert REPORT_PATH.exists()


def test_d5e_csvs_have_required_columns() -> None:
    metrics = pd.read_csv(METRICS_PATH, dtype=str).fillna("")
    instances = pd.read_csv(INSTANCE_SUMMARY_PATH, dtype=str).fillna("")
    aggregate = pd.read_csv(AGGREGATE_SUMMARY_PATH, dtype=str).fillna("")

    assert METRICS_COLUMNS.issubset(metrics.columns)
    assert INSTANCE_COLUMNS.issubset(instances.columns)
    assert AGGREGATE_COLUMNS.issubset(aggregate.columns)


def test_d5e_instance_summary_scope_and_configuration() -> None:
    instances = pd.read_csv(INSTANCE_SUMMARY_PATH, dtype=str).fillna("")

    assert len(instances) == 20
    assert set(instances["plan"]) == {"Plan D"}
    assert set(instances["phase"]) == {"D5E_qwen2_ml1m_ranking_diagnostic"}
    assert set(instances["dataset"]) == {"ml1m"}
    assert set(instances["scope"]) == {"diagnostic_subset"}
    assert set(instances["model"]) == {"qwen2:7b-instruct"}
    assert set(instances["backend"]) == {"ollama"}
    assert set(instances["prompt_strategy"]) == {"openp5_base"}
    assert set(instances["candidate_pool_size"].astype(int)) == {20}
    assert set(instances["top_k"].astype(int)) == {5}
    assert set(instances["temperature"].astype(float)) == {0.0}


def test_d5e_instance_metrics_are_machine_readable() -> None:
    instances = pd.read_csv(INSTANCE_SUMMARY_PATH, dtype=str).fillna("")

    for column in ("hit_at_5", "ndcg_at_5", "candidate_validity_rate", "latency_seconds"):
        values = pd.to_numeric(instances[column], errors="raise")
        assert values.notna().all()
    assert pd.to_numeric(instances["hit_at_5"]).between(0, 1).all()
    assert pd.to_numeric(instances["ndcg_at_5"]).between(0, 1).all()
    assert pd.to_numeric(instances["candidate_validity_rate"]).between(0, 1).all()
    assert (pd.to_numeric(instances["selected_count"]) >= 0).all()
    assert (pd.to_numeric(instances["duplicate_count"]) >= 0).all()
    assert (pd.to_numeric(instances["hallucinated_count"]) >= 0).all()


def test_d5e_aggregate_summary_rates_and_raw_path() -> None:
    aggregate = pd.read_csv(AGGREGATE_SUMMARY_PATH, dtype=str).fillna("")

    assert len(aggregate) == 1
    row = aggregate.iloc[0]
    assert row["dataset"] == "ml1m"
    assert row["model"] == "qwen2:7b-instruct"
    assert row["backend"] == "ollama"
    assert row["prompt_strategy"] == "openp5_base"
    assert int(row["num_instances"]) == 20
    for column in RATE_COLUMNS:
        value = float(row[column])
        assert 0.0 <= value <= 1.0
    assert float(row["mean_latency_seconds"]) >= 0.0
    assert float(row["total_runtime_seconds"]) >= 0.0
    raw_path = row["raw_output_path"].replace("\\", "/")
    assert raw_path.startswith("results/raw_outputs/")
    assert raw_path.endswith(".jsonl")


def test_d5e_metrics_csv_contains_required_metric_rows() -> None:
    metrics = pd.read_csv(METRICS_PATH, dtype=str).fillna("")
    metric_names = set(metrics["metric"])

    assert "hit_rate" in metric_names
    assert "ndcg" in metric_names
    assert "response_received_rate" in metric_names
    assert "parse_success_rate" in metric_names
    assert set(metrics["method"]) == {"qwen2_ollama_openp5"}
    assert set(metrics["scope"]) == {"diagnostic_subset"}
    assert set(metrics["num_instances"].astype(int)) == {20}
    assert pd.to_numeric(metrics["value"], errors="raise").notna().all()


def test_d5e_report_documents_scope_and_api_limits() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8").lower()

    assert "local-only ollama inference" in report
    assert "no openai, anthropic, or remote api was called" in report
    assert "not exact paper reproduction" in report
    assert "no model was pulled or downloaded" in report
    assert "no recommender model was trained" in report
    assert "only 20 users" in report


def test_d5e_tests_do_not_call_ollama_or_external_apis() -> None:
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
    assert not any(("ollama " + "pull") in value for value in string_constants)
    assert not any(("localhost:" + "11434") in value for value in string_constants)

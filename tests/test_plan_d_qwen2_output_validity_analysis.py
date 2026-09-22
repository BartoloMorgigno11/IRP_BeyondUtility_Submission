from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "src/analysis/plan_d_qwen2_output_validity_analysis.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/analyze_plan_d_qwen2_output_validity.py"
BREAKDOWN_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_output_validity_breakdown.csv"
VALID_ONLY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_valid_only_utility_summary.csv"
PROFILE_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_hallucination_instance_profile.csv"
PATTERNS_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_invalid_item_patterns.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_qwen2_output_validity_and_hallucination_analysis.md"

BREAKDOWN_COLUMNS = {
    "plan",
    "phase",
    "dataset",
    "scope",
    "backend",
    "model",
    "prompt_strategy",
    "num_instances",
    "num_responses_received",
    "num_parse_success",
    "num_schema_valid",
    "num_exact_top_k",
    "num_empty_outputs",
    "total_selected_items",
    "total_valid_selected_items",
    "total_hallucinated_items",
    "total_duplicate_items",
    "response_received_rate",
    "parse_success_rate",
    "schema_valid_rate",
    "exact_top_k_rate",
    "empty_output_rate",
    "mean_candidate_validity_rate",
    "micro_candidate_validity_rate",
    "hallucination_rate",
    "duplicate_rate",
    "status",
    "notes",
}

VALID_ONLY_COLUMNS = {
    "plan",
    "phase",
    "dataset",
    "scope",
    "backend",
    "model",
    "prompt_strategy",
    "utility_variant",
    "num_instances",
    "hit_rate_at_5",
    "ndcg_at_5",
    "mean_positive_rank",
    "num_hits",
    "notes",
}

PROFILE_COLUMNS = {
    "plan",
    "phase",
    "dataset",
    "instance_id",
    "user_id",
    "selected_count",
    "candidate_validity_rate",
    "hallucinated_count",
    "duplicate_count",
    "parse_success",
    "schema_valid",
    "positive_item_id",
    "positive_rank",
    "hit_at_5",
    "ndcg_at_5",
    "has_hallucination",
    "hallucination_severity",
    "latency_seconds",
    "status",
    "notes",
}

PATTERN_COLUMNS = {
    "plan",
    "phase",
    "dataset",
    "model",
    "invalid_item_id",
    "frequency",
    "pattern_type",
    "example_instance_ids",
    "notes",
}

RATE_COLUMNS = {
    "response_received_rate",
    "parse_success_rate",
    "schema_valid_rate",
    "exact_top_k_rate",
    "empty_output_rate",
    "mean_candidate_validity_rate",
    "micro_candidate_validity_rate",
    "hallucination_rate",
    "duplicate_rate",
}


def test_d5e1_analysis_scripts_exist() -> None:
    assert ANALYSIS_PATH.exists()
    assert RUNNER_PATH.exists()


def test_d5e1_output_files_exist() -> None:
    assert BREAKDOWN_PATH.exists()
    assert VALID_ONLY_PATH.exists()
    assert PROFILE_PATH.exists()
    assert PATTERNS_PATH.exists()
    assert REPORT_PATH.exists()


def test_d5e1_csvs_have_required_columns() -> None:
    assert BREAKDOWN_COLUMNS.issubset(pd.read_csv(BREAKDOWN_PATH, dtype=str).columns)
    assert VALID_ONLY_COLUMNS.issubset(pd.read_csv(VALID_ONLY_PATH, dtype=str).columns)
    assert PROFILE_COLUMNS.issubset(pd.read_csv(PROFILE_PATH, dtype=str).columns)
    assert PATTERN_COLUMNS.issubset(pd.read_csv(PATTERNS_PATH, dtype=str).columns)


def test_d5e1_breakdown_has_one_valid_aggregate_row() -> None:
    breakdown = pd.read_csv(BREAKDOWN_PATH, dtype=str).fillna("")

    assert len(breakdown) == 1
    row = breakdown.iloc[0]
    assert row["plan"] == "Plan D"
    assert row["phase"] == "D5E.1_qwen2_output_validity_analysis"
    assert row["dataset"] == "ml1m"
    assert row["scope"] == "diagnostic_subset"
    assert row["backend"] == "ollama"
    assert row["model"] == "qwen2:7b-instruct"
    assert row["prompt_strategy"] == "openp5_base"
    assert int(row["total_selected_items"]) > 0
    for column in RATE_COLUMNS:
        value = float(row[column])
        assert 0.0 <= value <= 1.0


def test_d5e1_hallucination_rate_is_consistent_with_counts() -> None:
    row = pd.read_csv(BREAKDOWN_PATH, dtype=str).fillna("").iloc[0]
    total_selected = int(row["total_selected_items"])
    total_hallucinated = int(row["total_hallucinated_items"])
    expected = total_hallucinated / total_selected

    assert total_selected > 0
    assert abs(float(row["hallucination_rate"]) - expected) < 1e-6
    assert int(row["total_valid_selected_items"]) + total_hallucinated == total_selected


def test_d5e1_profile_has_20_rows_and_severity_values() -> None:
    profile = pd.read_csv(PROFILE_PATH, dtype=str).fillna("")

    assert len(profile) == 20
    assert set(profile["dataset"]) == {"ml1m"}
    assert set(profile["hallucination_severity"]).issubset({"none", "low", "medium", "high"})
    assert pd.to_numeric(profile["candidate_validity_rate"], errors="raise").between(0, 1).all()
    assert pd.to_numeric(profile["hallucinated_count"], errors="raise").ge(0).all()


def test_d5e1_valid_only_summary_contains_required_variants() -> None:
    utility = pd.read_csv(VALID_ONLY_PATH, dtype=str).fillna("")
    variants = set(utility["utility_variant"])

    assert "original_parsed_output" in variants
    assert "valid_only_output" in variants
    assert "invalid_as_failure" in variants
    original = utility.loc[utility["utility_variant"] == "original_parsed_output"].iloc[0]
    assert int(original["num_instances"]) == 20
    assert 0.0 <= float(original["hit_rate_at_5"]) <= 1.0
    assert 0.0 <= float(original["ndcg_at_5"]) <= 1.0


def test_d5e1_report_documents_no_inference_and_candidate_validity() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8").lower()

    assert "no new llm inference was run" in report
    assert "no openai api call" in report
    assert "no anthropic api call" in report
    assert "hallucination analysis" in report
    assert "candidate adherence was the main failure mode" in report
    assert "candidate validity" in report


def test_d5e1_tests_do_not_call_ollama_or_external_apis() -> None:
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

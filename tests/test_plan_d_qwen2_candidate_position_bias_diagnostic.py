from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_candidate_position_bias_analysis import (
    PHASE,
    VARIANT_SUMMARY_COLUMNS,
    build_pairwise_delta,
    build_selected_position_distribution,
    build_variant_summary,
    candidate_order_for_variant,
    candidate_order_signature,
    selected_position_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "src/analysis/plan_d_candidate_position_bias_analysis.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/run_plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.py"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic.md"
REPRO_MATRIX_PATH = PROJECT_ROOT / "report/plan_d_reproducibility_matrix.md"
RAW_OUTPUT_PATH = PROJECT_ROOT / "results/raw_outputs/plan_d_qwen2_ml1m_candidate_position_bias_diagnostic_raw.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_candidate_position_bias_manifest.csv"
INSTANCE_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_candidate_position_bias_instance_summary.csv"
VARIANT_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_candidate_position_bias_variant_summary.csv"
DISTRIBUTION_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_candidate_position_bias_selected_position_distribution.csv"


def test_variant_transforms_preserve_candidate_set_and_expected_order() -> None:
    candidate_ids = [f"i{index}" for index in range(1, 21)]
    positive = "i7"

    reverse = candidate_order_for_variant(candidate_ids, positive, "reverse_order")
    first = candidate_order_for_variant(candidate_ids, positive, "positive_first")
    last = candidate_order_for_variant(candidate_ids, positive, "positive_last")

    assert reverse == list(reversed(candidate_ids))
    assert first[0] == positive
    assert first[1:] == [item_id for item_id in candidate_ids if item_id != positive]
    assert last[-1] == positive
    assert last[:-1] == [item_id for item_id in candidate_ids if item_id != positive]
    assert set(reverse) == set(first) == set(last) == set(candidate_ids)


def test_candidate_order_signature_changes_with_order_but_preserves_set_identity() -> None:
    candidate_ids = [f"i{index}" for index in range(1, 21)]
    positive = "i7"
    variants = [
        candidate_order_for_variant(candidate_ids, positive, "reverse_order"),
        candidate_order_for_variant(candidate_ids, positive, "positive_first"),
        candidate_order_for_variant(candidate_ids, positive, "positive_last"),
    ]

    assert len({candidate_order_signature(items) for items in variants}) == 3
    assert all(set(items) == set(candidate_ids) for items in variants)


def test_selected_position_distribution_on_tiny_synthetic_example() -> None:
    frame = pd.DataFrame(
        [
            {"variant": "reverse_order", "valid_selected_input_positions": "[1,11,20]"},
            {"variant": "reverse_order", "valid_selected_input_positions": "[1,2]"},
        ]
    )
    distribution = build_selected_position_distribution(frame)
    reverse = distribution.loc[distribution["variant"] == "reverse_order"]

    assert int(reverse.loc[reverse["input_position"] == "1", "selected_count"].iloc[0]) == 2
    assert float(reverse.loc[reverse["input_position"] == "1", "selected_rate"].iloc[0]) == 0.2
    assert int(reverse.loc[reverse["input_position"] == "20", "selected_count"].iloc[0]) == 1
    assert float(reverse.loc[reverse["input_position"] == "20", "selected_rate"].iloc[0]) == 0.1


def test_selected_position_metrics_front_back_rates() -> None:
    metrics = selected_position_metrics([1, 2, 11, 20])

    assert metrics["mean_selected_input_position"] == 8.5
    assert metrics["median_selected_input_position"] == 6.5
    assert metrics["front_half_selection_rate"] == 0.5
    assert metrics["back_half_selection_rate"] == 0.5
    assert metrics["position_1_selection_rate"] == 0.25
    assert metrics["position_20_selection_rate"] == 0.25


def test_pairwise_hr_ndcg_deltas_are_computed_correctly() -> None:
    summary = pd.DataFrame(
        [
            {
                "variant": "reverse_order",
                "hr_at_5": "0.300000",
                "ndcg_at_5": "0.200000",
                "hallucination_rate": "0.100000",
                "mean_candidate_validity_rate": "0.900000",
                "micro_candidate_validity_rate": "0.900000",
                "exact_top_k_rate": "1.000000",
            }
        ]
    )
    original = {
        "hit_rate_at_5": "0.200000",
        "ndcg_at_5": "0.250000",
        "hallucination_rate": "0.200000",
        "mean_candidate_validity_rate": "0.800000",
        "micro_candidate_validity_rate": "0.800000",
        "exact_top_k_rate": "1.000000",
    }
    delta = build_pairwise_delta(summary, original)

    hr = delta.loc[delta["metric_name"] == "HR@5"].iloc[0]
    ndcg = delta.loc[delta["metric_name"] == "NDCG@5"].iloc[0]
    assert float(hr["delta"]) == 0.1
    assert float(ndcg["delta"]) == -0.05


def test_variant_summary_hallucination_and_candidate_validity_metrics() -> None:
    rows = []
    for variant in ("reverse_order", "positive_first", "positive_last"):
        rows.append(
            {
                "variant": variant,
                "response_received": "true",
                "parse_success": "true",
                "schema_valid": "true",
                "selected_count": "5",
                "candidate_validity_rate": "0.800000",
                "duplicate_count": "0",
                "hallucinated_count": "1",
                "hit_at_5": "1.000000",
                "ndcg_at_5": "1.000000",
                "positive_rank": "1",
                "valid_selected_input_positions": "[1,2,3,4]",
                "valid_selected_item_ids": '["a","b","c","d"]',
                "variant_positive_position": "1",
                "original_instance_id": "orig_1",
            }
        )
    frame = pd.DataFrame(rows)
    original = {
        "hit_rate_at_5": "0.000000",
        "ndcg_at_5": "0.000000",
        "hallucination_rate": "0.000000",
        "mean_candidate_validity_rate": "1.000000",
        "micro_candidate_validity_rate": "1.000000",
        "exact_top_k_rate": "1.000000",
    }
    summary = build_variant_summary(
        frame,
        original_reference=original,
        original_valid_items_by_instance={"orig_1": {"a", "z"}},
    )

    assert set(VARIANT_SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(summary["hallucination_rate"]) == {"0.200000"}
    assert set(summary["mean_candidate_validity_rate"]) == {"0.800000"}


def test_raw_output_path_is_under_results_raw_outputs() -> None:
    raw_path = RAW_OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()

    assert raw_path.startswith("results/raw_outputs/")
    assert raw_path.endswith(".jsonl")


def test_d5f_artifacts_if_present_have_expected_scope() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("D5F outputs are not generated before the local diagnostic run")
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    instances = pd.read_csv(INSTANCE_SUMMARY_PATH, dtype=str).fillna("")
    summary = pd.read_csv(VARIANT_SUMMARY_PATH, dtype=str).fillna("")
    distribution = pd.read_csv(DISTRIBUTION_PATH, dtype=str).fillna("")

    assert len(manifest) == 60
    assert len(instances) == 60
    assert set(manifest["variant"]) == {"reverse_order", "positive_first", "positive_last"}
    assert set(instances["phase"]) == {PHASE}
    assert set(summary["scope"]) == {"diagnostic_subset"}
    assert set(summary["variant"]) == {"reverse_order", "positive_first", "positive_last"}
    assert len(distribution) == 60


def test_d5f_report_and_matrix_label_diagnostic_not_exact_if_present() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5F report is not generated before the local diagnostic run")
    report = REPORT_PATH.read_text(encoding="utf-8").lower()
    matrix = REPRO_MATRIX_PATH.read_text(encoding="utf-8").lower()

    assert "diagnostic_subset robustness diagnostic" in report
    assert "not exact reproduction" in report
    assert "not full llm reproduction" in report
    assert "no openai api call was made" in report
    assert "no anthropic api call was made" in report
    assert "no external api call was made" in report
    assert "no download was performed" in report
    assert "no model pull was performed" in report
    assert "no recommender training was performed" in report
    assert "candidate position bias" in matrix
    assert "diagnostic_only" in matrix or "completed_diagnostic_subset" in matrix


def test_d5f_test_file_does_not_call_ollama_or_external_apis() -> None:
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

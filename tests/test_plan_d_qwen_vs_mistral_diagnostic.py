from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_qwen_vs_mistral_comparison_analysis import (
    AGREEMENT_COLUMNS,
    INSTANCE_COLUMNS,
    MANIFEST_COLUMNS,
    METRIC_COLUMNS,
    MISTRAL_CONDITION,
    MODEL_SUMMARY_COLUMNS,
    PAIRWISE_COLUMNS,
    PHASE,
    QWEN_CONDITION,
    agreement_category,
    agreement_rates,
    build_agreement_profile,
    build_metric_rows,
    build_mistral_manifest,
    build_model_summary,
    build_pairwise_delta,
    candidate_order_signature,
    evaluate_ranked_ids,
    instance_row,
    jaccard,
)
from src.analysis.plan_d_qwen2_strict_prompt_analysis import build_strict_candidate_id_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "src/analysis/plan_d_qwen_vs_mistral_comparison_analysis.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/run_plan_d_qwen_vs_mistral_ml1m_diagnostic.py"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_qwen_vs_mistral_ml1m_diagnostic.md"
REPRO_MATRIX_PATH = PROJECT_ROOT / "report/plan_d_reproducibility_matrix.md"
RAW_OUTPUT_PATH = PROJECT_ROOT / "results/raw_outputs/plan_d_qwen_vs_mistral_ml1m_diagnostic_raw.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen_vs_mistral_ml1m_manifest.csv"
INSTANCE_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen_vs_mistral_ml1m_instance_summary.csv"
MODEL_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen_vs_mistral_ml1m_model_summary.csv"
PAIRWISE_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen_vs_mistral_ml1m_pairwise_delta.csv"
AGREEMENT_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen_vs_mistral_ml1m_agreement_profile.csv"
METRICS_PATH = PROJECT_ROOT / "results/metrics/plan_d_qwen_vs_mistral_ml1m_diagnostic_metrics.csv"


def test_d5i_loads_qwen2_strict_reference_without_rerun_in_code() -> None:
    runner_text = RUNNER_PATH.read_text(encoding="utf-8")

    assert "D5G_RAW_PATH" in runner_text
    assert "QWEN_CONDITION" in runner_text
    assert "model=QWEN_MODEL" not in runner_text
    assert "Qwen2 was not rerun" in runner_text


def test_mistral_strict_prompt_uses_candidate_id_only_constraints() -> None:
    prompt = build_strict_candidate_id_prompt(
        user_id="u1",
        history_preview="history only",
        candidate_ids=["101", "102", "103", "104", "105"],
        top_k=3,
    ).lower()

    assert "return only valid json" in prompt
    assert "select exactly 3 item ids" in prompt
    assert "copied exactly from the allowed candidate id list" in prompt
    assert "do not invent ids" in prompt
    assert "do not output explanations" in prompt


def test_mistral_manifest_preserves_candidate_signature_and_positive_item() -> None:
    frame = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "candidate_ids": '["a","b","c","d","e"]',
                "positive_item_id": "c",
            }
        ]
    )
    manifest = build_mistral_manifest(frame)

    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["positive_item_id"] == "c"
    assert row["candidate_pool_size"] == "5"
    assert row["candidate_order_signature"] == candidate_order_signature(["a", "b", "c", "d", "e"])
    assert row["model"] == "mistral:7b-instruct"


def test_candidate_order_signature_is_deterministic_and_order_sensitive() -> None:
    assert candidate_order_signature(["a", "b", "c"]) == candidate_order_signature(["a", "b", "c"])
    assert candidate_order_signature(["a", "b", "c"]) != candidate_order_signature(["c", "b", "a"])


def test_candidate_validity_hallucination_and_utility_metrics_on_tiny_example() -> None:
    instance = evaluate_ranked_ids(
        ["bad", "b", "b", "c", "d"],
        condition=MISTRAL_CONDITION,
        user_id="u1",
        source_instance_id="src",
        diagnostic_id="diag",
        candidate_ids=["a", "b", "c", "d", "e"],
        positive_item_id="b",
        response_received=True,
        parse_success=True,
        schema_valid=True,
        latency_seconds=0.5,
        parse_status="parsed_with_hallucinations",
        notes="synthetic",
    )

    assert instance.selected_count == 5
    assert instance.valid_selected_count == 4
    assert instance.hallucinated_count == 1
    assert instance.duplicate_count == 1
    assert instance.candidate_validity_rate == 0.8
    assert instance.positive_rank == 2
    assert instance.hit_at_5 == 1.0
    assert math.isclose(instance.ndcg_at_5, 1.0 / math.log2(3))


def test_pairwise_deltas_are_computed_correctly() -> None:
    rows = []
    for condition, model, hit, ndcg, validity, hallucination, latency in (
        (QWEN_CONDITION, "qwen2:7b-instruct", "0.500000", "0.250000", "1.000000", "0.000000", "4.000000"),
        (MISTRAL_CONDITION, "mistral:7b-instruct", "0.250000", "0.100000", "0.800000", "0.200000", "5.000000"),
    ):
        rows.append(
            {
                **{column: "" for column in MODEL_SUMMARY_COLUMNS},
                "condition": condition,
                "model": model,
                "hr_at_5": hit,
                "ndcg_at_5": ndcg,
                "mean_candidate_validity_rate": validity,
                "micro_candidate_validity_rate": validity,
                "hallucination_rate": hallucination,
                "parse_success_rate": "1.000000",
                "schema_valid_rate": "1.000000",
                "exact_top_k_rate": "1.000000",
                "duplicate_rate": "0.000000",
                "mean_latency_seconds": latency,
            }
        )
    agreement = pd.DataFrame(
        [
            {
                **{column: "" for column in AGREEMENT_COLUMNS},
                "agreement_category": "qwen_only_hit",
                "selected_item_jaccard": "0.250000",
                "valid_item_jaccard": "0.250000",
                "qwen_hit": "true",
                "mistral_hit": "false",
            }
        ]
    )
    deltas = build_pairwise_delta(pd.DataFrame(rows), agreement)

    hr = deltas.loc[deltas["metric_name"] == "HR@5_delta"].iloc[0]
    validity = deltas.loc[deltas["metric_name"] == "candidate_validity_delta"].iloc[0]
    assert float(hr["delta"]) == -0.25
    assert float(validity["delta"]) == -0.2


def test_agreement_categories_and_jaccard_overlap() -> None:
    assert agreement_category(True, True) == "both_hit"
    assert agreement_category(True, False) == "qwen_only_hit"
    assert agreement_category(False, True) == "mistral_only_hit"
    assert agreement_category(False, False) == "neither_hit"
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3


def test_agreement_rates_are_computed_correctly() -> None:
    frame = pd.DataFrame(
        [
            {"agreement_category": "both_hit", "qwen_hit": "true", "mistral_hit": "true", "selected_item_jaccard": "0.5", "valid_item_jaccard": "0.5"},
            {"agreement_category": "qwen_only_hit", "qwen_hit": "true", "mistral_hit": "false", "selected_item_jaccard": "0.0", "valid_item_jaccard": "0.0"},
        ]
    )
    rates = agreement_rates(frame)

    assert rates["both_hit_rate"] == 0.5
    assert rates["qwen_only_hit_rate"] == 0.5
    assert rates["positive_selection_agreement_rate"] == 0.5
    assert rates["selected_item_jaccard"] == 0.25


def test_raw_output_path_is_under_results_raw_outputs() -> None:
    raw_path = RAW_OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()

    assert raw_path.startswith("results/raw_outputs/")
    assert raw_path.endswith(".jsonl")


def test_d5i_artifacts_if_present_have_expected_structure() -> None:
    if not MODEL_SUMMARY_PATH.exists():
        pytest.skip("D5I outputs are not generated before the local Mistral diagnostic run")
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    instances = pd.read_csv(INSTANCE_SUMMARY_PATH, dtype=str).fillna("")
    summary = pd.read_csv(MODEL_SUMMARY_PATH, dtype=str).fillna("")
    pairwise = pd.read_csv(PAIRWISE_PATH, dtype=str).fillna("")
    agreement = pd.read_csv(AGREEMENT_PATH, dtype=str).fillna("")
    metrics = pd.read_csv(METRICS_PATH, dtype=str).fillna("")

    assert set(MANIFEST_COLUMNS).issubset(manifest.columns)
    assert set(INSTANCE_COLUMNS).issubset(instances.columns)
    assert set(MODEL_SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(PAIRWISE_COLUMNS).issubset(pairwise.columns)
    assert set(AGREEMENT_COLUMNS).issubset(agreement.columns)
    assert set(METRIC_COLUMNS).issubset(metrics.columns)
    assert len(manifest) == 20
    assert len(instances) == 40
    assert len(agreement) == 20
    assert set(summary["condition"]) == {QWEN_CONDITION, MISTRAL_CONDITION}
    assert set(instances["dataset"]) == {"ml1m"}
    assert set(instances["backend"]) == {"ollama"}
    assert set(instances["top_k"]) == {"5"}
    assert set(instances["candidate_pool_size"]) == {"20"}
    assert {"hit_rate", "ndcg", "hallucination_rate"}.issubset(set(metrics["metric"]))


def test_d5i_report_and_matrix_label_diagnostic_not_exact_if_present() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5I report is not generated before the local diagnostic run")
    report = REPORT_PATH.read_text(encoding="utf-8").lower()
    matrix = REPRO_MATRIX_PATH.read_text(encoding="utf-8").lower()

    assert "diagnostic_subset local open-model comparison" in report
    assert "not exact reproduction" in report
    assert "not full llm reproduction" in report
    assert "qwen2 outputs are reused from d5g and are not rerun" in report
    assert "no openai api call was made" in report
    assert "no anthropic api call was made" in report
    assert "no external api call was made" in report
    assert "no download was performed during d5i" in report
    assert "no model pull was performed during d5i" in report
    assert "no recommender training was performed" in report
    assert "qwen2 vs mistral" in matrix
    assert "diagnostic_only" in matrix or "completed_diagnostic_subset" in matrix


def test_d5i_tests_do_not_call_ollama_or_external_apis() -> None:
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
    assert ANALYSIS_PATH.exists()
    assert RUNNER_PATH.exists()

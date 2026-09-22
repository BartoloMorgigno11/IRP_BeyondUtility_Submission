from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_llm_reranking_analysis import (
    AGREEMENT_COLUMNS,
    CONDITION_SUMMARY_COLUMNS,
    INSTANCE_COLUMNS,
    MANIFEST_COLUMNS,
    METRIC_COLUMNS,
    MISTRAL_CONDITION,
    PAIRWISE_COLUMNS,
    POSITION_SHIFT_COLUMNS,
    QWEN_CONDITION,
    REFERENCE_CONDITION,
    build_agreement_profile,
    build_condition_summary,
    build_pairwise_delta,
    build_position_shift_profile,
    build_reranking_prompt,
    candidate_order_signature,
    evaluate_reranked_ids,
    instance_row,
    jaccard,
    reference_top_k,
    rerank_agreement_category,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "src/analysis/plan_d_llm_reranking_analysis.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/run_plan_d_llm_ml1m_reranking_diagnostic.py"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_llm_ml1m_reranking_diagnostic.md"
REPRO_MATRIX_PATH = PROJECT_ROOT / "report/plan_d_reproducibility_matrix.md"
RAW_OUTPUT_PATH = PROJECT_ROOT / "results/raw_outputs/plan_d_llm_ml1m_reranking_diagnostic_raw.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_reranking_manifest.csv"
INSTANCE_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_reranking_instance_summary.csv"
CONDITION_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_reranking_condition_summary.csv"
PAIRWISE_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_reranking_pairwise_delta.csv"
AGREEMENT_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_reranking_agreement_profile.csv"
POSITION_SHIFT_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_reranking_position_shift_profile.csv"
METRICS_PATH = PROJECT_ROOT / "results/metrics/plan_d_llm_ml1m_reranking_diagnostic_metrics.csv"


def test_reranking_prompt_contains_candidate_id_only_constraints() -> None:
    prompt = build_reranking_prompt(
        user_id="u1",
        history_preview="history only",
        candidate_ids=["101", "102", "103", "104", "105"],
        top_k=3,
    ).lower()

    assert "return only valid json" in prompt
    assert "return exactly 3 item ids" in prompt
    assert "copied exactly from the candidate list" in prompt
    assert "do not invent ids" in prompt
    assert "do not output titles" in prompt
    assert "do not output explanations" in prompt


def test_reranking_prompt_marks_input_as_preranked_baseline_order() -> None:
    prompt = build_reranking_prompt(
        user_id="u1",
        history_preview="history only",
        candidate_ids=["101", "102", "103", "104", "105"],
        top_k=3,
    ).lower()

    assert "re-ranking task" in prompt
    assert "pre-ranked baseline candidate list" in prompt
    assert "input_rank 1" in prompt


def test_reranking_prompt_includes_all_candidate_ids_exactly_once() -> None:
    candidate_ids = ["101", "102", "103", "104", "105"]
    prompt = build_reranking_prompt(
        user_id="u1",
        history_preview="history without ids",
        candidate_ids=candidate_ids,
        top_k=3,
    )

    for item_id in candidate_ids:
        assert prompt.count(item_id) == 1


def test_candidate_order_reference_top5_uses_first_five_candidates() -> None:
    assert reference_top_k(["a", "b", "c", "d", "e", "f"], top_k=5) == ["a", "b", "c", "d", "e"]


def test_candidate_validity_hallucination_and_utility_metrics_on_tiny_example() -> None:
    instance = evaluate_reranked_ids(
        ["bad", "b", "b", "c", "d"],
        condition=QWEN_CONDITION,
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


def test_position_shift_metrics_are_computed_correctly() -> None:
    reference = evaluate_reranked_ids(
        ["a", "b", "c"],
        condition=REFERENCE_CONDITION,
        user_id="u1",
        source_instance_id="src",
        diagnostic_id="ref",
        candidate_ids=["a", "b", "c", "d", "e"],
        positive_item_id="c",
        response_received=True,
        parse_success=True,
        schema_valid=True,
        latency_seconds=0.0,
        parse_status="reference_from_candidate_order",
        notes="ref",
    )
    qwen = evaluate_reranked_ids(
        ["c", "a", "b"],
        condition=QWEN_CONDITION,
        user_id="u1",
        source_instance_id="src",
        diagnostic_id="qwen",
        candidate_ids=["a", "b", "c", "d", "e"],
        positive_item_id="c",
        response_received=True,
        parse_success=True,
        schema_valid=True,
        latency_seconds=0.0,
        parse_status="parsed_valid",
        notes="qwen",
    )
    rows = pd.DataFrame([instance_row(reference), instance_row(qwen)])
    summary = build_condition_summary(rows, runtimes={QWEN_CONDITION: 1.0})
    qwen_row = summary.loc[summary["condition"] == QWEN_CONDITION].iloc[0]

    assert float(qwen_row["mean_selected_input_position"]) == 2.0
    assert math.isclose(float(qwen_row["mean_absolute_position_shift_vs_reference"]), 4 / 3, rel_tol=1e-6)
    assert float(qwen_row["mean_signed_position_shift_vs_reference"]) == 0.0


def test_pairwise_deltas_and_jaccard_overlap_are_computed_correctly() -> None:
    instances = []
    for condition, ranked, diagnostic_id in (
        (REFERENCE_CONDITION, ["a", "b", "c"], "ref"),
        (QWEN_CONDITION, ["c", "a", "b"], "qwen"),
        (MISTRAL_CONDITION, ["a", "d", "e"], "mistral"),
    ):
        instances.append(
            evaluate_reranked_ids(
                ranked,
                condition=condition,
                user_id="u1",
                source_instance_id="src",
                diagnostic_id=diagnostic_id,
                candidate_ids=["a", "b", "c", "d", "e"],
                positive_item_id="c",
                response_received=True,
                parse_success=True,
                schema_valid=True,
                latency_seconds=0.0,
                parse_status="reference_from_candidate_order" if condition == REFERENCE_CONDITION else "parsed_valid",
                notes="synthetic",
            )
        )
    rows = pd.DataFrame([instance_row(instance) for instance in instances])
    summary = build_condition_summary(rows, runtimes={QWEN_CONDITION: 1.0, MISTRAL_CONDITION: 1.0})
    agreement = build_agreement_profile(rows)
    pairwise = build_pairwise_delta(summary, agreement)

    qwen_hr = pairwise.loc[
        (pairwise["comparison"] == "qwen2_rerank_prompt_vs_reference")
        & (pairwise["metric_name"] == "HR@5_delta")
    ].iloc[0]
    overlap = pairwise.loc[
        (pairwise["comparison"] == "qwen2_rerank_prompt_vs_reference")
        & (pairwise["metric_name"] == "selected_item_jaccard")
    ].iloc[0]
    assert float(qwen_hr["delta"]) == 0.0
    assert float(overlap["condition_value"]) == 1.0
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3


def test_agreement_categories_are_computed_correctly() -> None:
    assert rerank_agreement_category(True, True, True) == "all_hit"
    assert rerank_agreement_category(False, True, False) == "qwen_only_hit"
    assert rerank_agreement_category(False, False, True) == "mistral_only_hit"
    assert rerank_agreement_category(True, False, False) == "reference_only_hit"
    assert rerank_agreement_category(False, True, True) == "qwen_mistral_hit_only"
    assert rerank_agreement_category(True, True, False) == "reference_qwen_hit_only"
    assert rerank_agreement_category(True, False, True) == "reference_mistral_hit_only"
    assert rerank_agreement_category(False, False, False) == "none_hit"


def test_candidate_order_signature_is_deterministic() -> None:
    assert candidate_order_signature(["a", "b", "c"]) == candidate_order_signature(["a", "b", "c"])
    assert candidate_order_signature(["a", "b", "c"]) != candidate_order_signature(["c", "b", "a"])


def test_raw_output_path_is_under_results_raw_outputs() -> None:
    raw_path = RAW_OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()

    assert raw_path.startswith("results/raw_outputs/")
    assert raw_path.endswith(".jsonl")


def test_d5j_artifacts_if_present_have_expected_structure() -> None:
    if not CONDITION_SUMMARY_PATH.exists():
        pytest.skip("D5J outputs are not generated before the local reranking diagnostic run")
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    instances = pd.read_csv(INSTANCE_SUMMARY_PATH, dtype=str).fillna("")
    summary = pd.read_csv(CONDITION_SUMMARY_PATH, dtype=str).fillna("")
    pairwise = pd.read_csv(PAIRWISE_PATH, dtype=str).fillna("")
    agreement = pd.read_csv(AGREEMENT_PATH, dtype=str).fillna("")
    shifts = pd.read_csv(POSITION_SHIFT_PATH, dtype=str).fillna("")
    metrics = pd.read_csv(METRICS_PATH, dtype=str).fillna("")

    assert set(MANIFEST_COLUMNS).issubset(manifest.columns)
    assert set(INSTANCE_COLUMNS).issubset(instances.columns)
    assert set(CONDITION_SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(PAIRWISE_COLUMNS).issubset(pairwise.columns)
    assert set(AGREEMENT_COLUMNS).issubset(agreement.columns)
    assert set(POSITION_SHIFT_COLUMNS).issubset(shifts.columns)
    assert set(METRIC_COLUMNS).issubset(metrics.columns)
    assert len(manifest) == 60
    assert len(instances) == 60
    assert len(agreement) == 20
    assert len(shifts) <= 200
    assert set(summary["condition"]) == {REFERENCE_CONDITION, QWEN_CONDITION, MISTRAL_CONDITION}
    assert set(instances["dataset"]) == {"ml1m"}
    assert set(instances["top_k"]) == {"5"}
    assert set(instances["candidate_pool_size"]) == {"20"}
    assert {"hit_rate", "ndcg", "mean_absolute_position_shift_vs_reference"}.issubset(set(metrics["metric"]))


def test_d5j_report_and_matrix_label_diagnostic_not_exact_if_present() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5J report is not generated before the local diagnostic run")
    report = REPORT_PATH.read_text(encoding="utf-8").lower()
    matrix = REPRO_MATRIX_PATH.read_text(encoding="utf-8").lower()

    assert "diagnostic_subset local llm re-ranking diagnostic" in report
    assert "not exact reproduction" in report
    assert "not full llm reproduction" in report
    assert "local ollama only" in report
    assert "no openai api call was made" in report
    assert "no anthropic api call was made" in report
    assert "no external api call was made" in report
    assert "no download was performed during d5j" in report
    assert "no model pull was performed during d5j" in report
    assert "no recommender training was performed" in report
    assert "candidate_order_reference" in report
    assert "re-ranking" in matrix or "reranking" in matrix
    assert "diagnostic_only" in matrix or "completed_diagnostic_subset" in matrix


def test_d5j_tests_do_not_call_ollama_or_external_apis() -> None:
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

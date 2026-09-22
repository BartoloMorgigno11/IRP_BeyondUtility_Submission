from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_history_length_sensitivity_analysis import (
    CONDITION_SUMMARY_COLUMNS,
    INSTANCE_COLUMNS,
    MANIFEST_COLUMNS,
    METRIC_COLUMNS,
    MISTRAL_MODEL,
    PAIRWISE_COLUMNS,
    QWEN_MODEL,
    SHORT_HISTORY,
    MEDIUM_HISTORY,
    LONG_HISTORY,
    STABILITY_COLUMNS,
    HistoryItem,
    build_condition_summary,
    build_history_length_manifest,
    build_history_length_prompt,
    build_pairwise_delta,
    build_stability_profile,
    condition_for,
    evaluate_history_output,
    history_hit_pattern,
    instance_row,
    jaccard,
    parse_candidate_ids,
    truncate_recent_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "src/analysis/plan_d_history_length_sensitivity_analysis.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/run_plan_d_llm_ml1m_history_length_sensitivity.py"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_llm_ml1m_history_length_sensitivity.md"
REPRO_MATRIX_PATH = PROJECT_ROOT / "report/plan_d_reproducibility_matrix.md"
RAW_OUTPUT_PATH = PROJECT_ROOT / "results/raw_outputs/plan_d_llm_ml1m_history_length_sensitivity_raw.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_history_length_manifest.csv"
INSTANCE_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_history_length_instance_summary.csv"
CONDITION_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_history_length_condition_summary.csv"
PAIRWISE_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_history_length_pairwise_delta.csv"
STABILITY_PATH = PROJECT_ROOT / "results/tables/plan_d_llm_ml1m_history_length_stability_profile.csv"
METRICS_PATH = PROJECT_ROOT / "results/metrics/plan_d_llm_ml1m_history_length_sensitivity_metrics.csv"


def test_history_truncation_returns_most_recent_n_items() -> None:
    history = [HistoryItem(str(index), f"title {index}") for index in range(1, 8)]

    assert [item.item_id for item in truncate_recent_history(history, 3)] == ["5", "6", "7"]
    assert [item.item_id for item in truncate_recent_history(history, 10)] == [str(index) for index in range(1, 8)]


def test_short_medium_long_manifest_keep_same_candidate_set() -> None:
    d5e_manifest = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "candidate_ids": '["a","b","c","d","e"]',
                "positive_item_id": "c",
            }
        ]
    )
    histories = {"u1": [HistoryItem(str(index), f"title {index}") for index in range(1, 12)]}
    manifest = build_history_length_manifest(d5e_manifest, histories)

    signatures = set(manifest["candidate_order_signature"])
    assert len(signatures) == 1
    assert set(manifest["history_setting"]) == {SHORT_HISTORY, MEDIUM_HISTORY, LONG_HISTORY}


def test_prompt_includes_correct_history_length_and_candidate_id_only_constraints() -> None:
    prompt = build_history_length_prompt(
        user_id="u1",
        history_items=[HistoryItem("h1", "First Movie"), HistoryItem("h2", "Second Movie")],
        full_history_length=9,
        history_setting=SHORT_HISTORY,
        candidate_ids=["101", "102", "103", "104", "105"],
        top_k=3,
    ).lower()

    assert "recommendation ranking task" in prompt
    assert "history setting: short_history" in prompt
    assert "history length used: 2 of 9" in prompt
    assert "return only valid json" in prompt
    assert "return exactly 3 item ids" in prompt
    assert "copied exactly from the allowed candidate id list" in prompt
    assert "do not invent ids" in prompt
    assert "do not output movie titles" in prompt
    assert "do not output explanations" in prompt


def test_prompt_includes_all_candidate_ids_exactly_once() -> None:
    candidate_ids = ["101", "102", "103", "104", "105"]
    prompt = build_history_length_prompt(
        user_id="u1",
        history_items=[HistoryItem("h1", "A Movie")],
        full_history_length=1,
        history_setting=MEDIUM_HISTORY,
        candidate_ids=candidate_ids,
        top_k=3,
    )

    for item_id in candidate_ids:
        assert prompt.count(item_id) == 1


def test_candidate_validity_hallucination_and_utility_metrics_on_tiny_example() -> None:
    instance = evaluate_history_output(
        ["bad", "b", "b", "c", "d"],
        condition=condition_for(QWEN_MODEL, SHORT_HISTORY),
        history_setting=SHORT_HISTORY,
        user_id="u1",
        source_instance_id="src",
        diagnostic_id="diag",
        model=QWEN_MODEL,
        candidate_ids=["a", "b", "c", "d", "e"],
        positive_item_id="b",
        history_length_used=5,
        full_history_length=12,
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
    rows = _synthetic_rows_for_model(QWEN_MODEL)
    summary = build_condition_summary(rows, runtimes={condition: 1.0 for condition in rows["condition"].unique()})
    pairwise = build_pairwise_delta(summary, rows)

    short_medium = pairwise.loc[
        (pairwise["model"] == QWEN_MODEL)
        & (pairwise["condition_a"] == SHORT_HISTORY)
        & (pairwise["condition_b"] == MEDIUM_HISTORY)
    ].iloc[0]
    assert float(short_medium["HR@5_delta"]) == 1.0
    assert float(short_medium["candidate_validity_delta"]) == 0.0
    assert math.isclose(
        float(short_medium["selected_item_jaccard"]),
        jaccard(["a", "b", "c", "x", "y"], ["a", "b", "e", "x", "y"]),
        abs_tol=1e-6,
    )


def test_history_stability_profile_hit_patterns_are_computed_correctly() -> None:
    rows = _synthetic_rows_for_model(MISTRAL_MODEL)
    profile = build_stability_profile(rows)
    row = profile.iloc[0]

    assert row["hit_pattern"] == "medium_only"
    assert math.isclose(
        float(row["short_medium_jaccard"]),
        jaccard(["a", "b", "c", "x", "y"], ["a", "b", "e", "x", "y"]),
        abs_tol=1e-6,
    )
    assert history_hit_pattern(True, False, True) == "short_long_only"
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3


def test_raw_output_path_is_under_results_raw_outputs() -> None:
    raw_path = RAW_OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()

    assert raw_path.startswith("results/raw_outputs/")
    assert raw_path.endswith(".jsonl")


def test_d5k_artifacts_if_present_have_expected_structure() -> None:
    if not CONDITION_SUMMARY_PATH.exists():
        pytest.skip("D5K outputs are not generated before the local history-length diagnostic run")
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    instances = pd.read_csv(INSTANCE_SUMMARY_PATH, dtype=str).fillna("")
    summary = pd.read_csv(CONDITION_SUMMARY_PATH, dtype=str).fillna("")
    pairwise = pd.read_csv(PAIRWISE_PATH, dtype=str).fillna("")
    stability = pd.read_csv(STABILITY_PATH, dtype=str).fillna("")
    metrics = pd.read_csv(METRICS_PATH, dtype=str).fillna("")

    assert set(MANIFEST_COLUMNS).issubset(manifest.columns)
    assert set(INSTANCE_COLUMNS).issubset(instances.columns)
    assert set(CONDITION_SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(PAIRWISE_COLUMNS).issubset(pairwise.columns)
    assert set(STABILITY_COLUMNS).issubset(stability.columns)
    assert set(METRIC_COLUMNS).issubset(metrics.columns)
    assert len(manifest) == 120
    assert len(instances) == 120
    assert len(summary) == 6
    assert len(pairwise) == 6
    assert len(stability) == 40
    assert set(instances["dataset"]) == {"ml1m"}
    assert set(instances["top_k"]) == {"5"}
    assert set(instances["candidate_pool_size"]) == {"20"}
    assert {"hit_rate", "ndcg", "history_truncation_rate"}.issubset(set(metrics["metric"]))


def test_d5k_report_and_matrix_label_diagnostic_not_exact_if_present() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5K report is not generated before the local diagnostic run")
    report = REPORT_PATH.read_text(encoding="utf-8").lower()
    matrix = REPRO_MATRIX_PATH.read_text(encoding="utf-8").lower()

    assert "diagnostic_subset local llm history length sensitivity diagnostic" in report
    assert "not exact reproduction" in report
    assert "not full llm reproduction" in report
    assert "no openai api call was made" in report
    assert "no anthropic api call was made" in report
    assert "no external api call was made" in report
    assert "no download was performed during d5k" in report
    assert "no model pull was performed during d5k" in report
    assert "no recommender training was performed" in report
    assert "plan c was not modified" in report
    assert "history length sensitivity" in matrix
    assert "diagnostic_only" in matrix


def test_tests_do_not_call_ollama_or_external_apis() -> None:
    test_text = Path(__file__).read_text(encoding="utf-8").lower()

    assert ("subprocess" + ".run") not in test_text
    assert ("urllib" + ".request") not in test_text
    assert ("import " + "openai") not in test_text
    assert ("import " + "anthropic") not in test_text
    assert ("openai" + ".") not in test_text
    assert ("anthropic" + ".") not in test_text


def _synthetic_rows_for_model(model: str) -> pd.DataFrame:
    candidate_ids = ["a", "b", "c", "d", "e"]
    selected_by_setting = {
        SHORT_HISTORY: ["a", "b", "c", "x", "y"],
        MEDIUM_HISTORY: ["a", "b", "e", "x", "y"],
        LONG_HISTORY: ["a", "c", "d", "x", "y"],
    }
    instances = []
    for setting, selected in selected_by_setting.items():
        instances.append(
            evaluate_history_output(
                selected,
                condition=condition_for(model, setting),
                history_setting=setting,
                user_id="u1",
                source_instance_id="src",
                diagnostic_id=f"diag_{setting}",
                model=model,
                candidate_ids=candidate_ids,
                positive_item_id="e",
                history_length_used=5,
                full_history_length=10,
                response_received=True,
                parse_success=True,
                schema_valid=True,
                latency_seconds=0.1,
                parse_status="parsed_valid",
                notes="synthetic",
            )
        )
    return pd.DataFrame([instance_row(instance) for instance in instances])

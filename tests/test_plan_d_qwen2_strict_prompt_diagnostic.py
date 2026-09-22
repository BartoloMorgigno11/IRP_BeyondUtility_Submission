from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_qwen2_strict_prompt_analysis import (
    CONDITION_SUMMARY_COLUMNS,
    INSTANCE_COLUMNS,
    MANIFEST_COLUMNS,
    ORIGINAL_CONDITION,
    ORIGINAL_POST_FILTERED_CONDITION,
    PAIRWISE_COLUMNS,
    PHASE,
    POST_FILTER_PROFILE_COLUMNS,
    PROMPT_STRATEGY,
    STRICT_CONDITION,
    STRICT_POST_FILTERED_CONDITION,
    build_condition_summary,
    build_pairwise_delta,
    build_strict_candidate_id_prompt,
    candidate_order_signature,
    condition_instance_from_selection,
    evaluate_selected_ids,
    post_filter_ranked_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "src/analysis/plan_d_qwen2_strict_prompt_analysis.py"
RUNNER_PATH = PROJECT_ROOT / "src/experiments/run_plan_d_qwen2_ml1m_strict_prompt_diagnostic.py"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_qwen2_ml1m_strict_prompt_diagnostic.md"
REPRO_MATRIX_PATH = PROJECT_ROOT / "report/plan_d_reproducibility_matrix.md"
RAW_OUTPUT_PATH = PROJECT_ROOT / "results/raw_outputs/plan_d_qwen2_ml1m_strict_prompt_diagnostic_raw.jsonl"
METRICS_PATH = PROJECT_ROOT / "results/metrics/plan_d_qwen2_ml1m_strict_prompt_metrics.csv"
MANIFEST_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_strict_prompt_manifest.csv"
INSTANCE_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_strict_prompt_instance_summary.csv"
CONDITION_SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_strict_prompt_condition_summary.csv"
PAIRWISE_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_strict_prompt_pairwise_delta.csv"
POST_FILTER_PROFILE_PATH = PROJECT_ROOT / "results/tables/plan_d_qwen2_ml1m_strict_prompt_post_filter_profile.csv"


def test_strict_prompt_contains_candidate_id_only_constraints() -> None:
    candidate_ids = [f"item_{index}" for index in range(1, 6)]
    prompt = build_strict_candidate_id_prompt(
        user_id="u1",
        history_preview="liked science fiction and animation",
        candidate_ids=candidate_ids,
        top_k=3,
    ).lower()

    assert "return only valid json" in prompt
    assert "select exactly 3 item ids" in prompt
    assert "copied exactly from the allowed candidate id list" in prompt
    assert "do not invent ids" in prompt
    assert "do not output movie titles" in prompt
    assert "do not output explanations" in prompt


def test_strict_prompt_includes_all_candidate_ids_exactly_once() -> None:
    candidate_ids = [f"item_{index}" for index in range(1, 6)]
    prompt = build_strict_candidate_id_prompt(
        user_id="u1",
        history_preview="history without candidate identifiers",
        candidate_ids=candidate_ids,
        top_k=3,
    )

    for item_id in candidate_ids:
        assert prompt.count(item_id) == 1


def test_strict_prompt_does_not_include_non_candidate_selected_id_examples() -> None:
    candidate_ids = [f"candidate_{index}" for index in range(1, 6)]
    prompt = build_strict_candidate_id_prompt(
        user_id="u1",
        history_preview="history without selected ID examples",
        candidate_ids=candidate_ids,
        top_k=3,
    )

    assert "item_id_1" not in prompt
    assert "item_Z" not in prompt
    assert "candidate_999" not in prompt


def test_post_filter_removes_invalid_ids_and_preserves_valid_order_without_backfill() -> None:
    filtered = post_filter_ranked_ids(["bad_1", "a", "b", "bad_2", "c"], {"a", "b", "c", "d", "e"})

    assert filtered == ["a", "b", "c"]
    assert len(filtered) == 3


def test_candidate_validity_and_hallucination_metrics_on_tiny_example() -> None:
    metrics = evaluate_selected_ids(
        ["x", "b", "b", "c", "z"],
        candidate_ids={"a", "b", "c", "d", "e"},
        positive_item_id="b",
        top_k=5,
    )

    assert metrics.selected_count == 5
    assert metrics.valid_selected_count == 3
    assert metrics.hallucinated_count == 2
    assert metrics.duplicate_count == 1
    assert metrics.candidate_validity_rate == 0.6
    assert metrics.positive_rank == 2
    assert metrics.hit_at_5 == 1.0
    assert math.isclose(metrics.ndcg_at_5, 1.0 / math.log2(3))


def test_hr_ndcg_before_and_after_filtering_are_computed_correctly() -> None:
    before = evaluate_selected_ids(
        ["x", "b", "c", "d", "e"],
        candidate_ids={"a", "b", "c", "d", "e"},
        positive_item_id="b",
        top_k=5,
    )
    after = evaluate_selected_ids(
        post_filter_ranked_ids(before.ranked_item_ids, {"a", "b", "c", "d", "e"}),
        candidate_ids={"a", "b", "c", "d", "e"},
        positive_item_id="b",
        top_k=5,
    )

    assert before.positive_rank == 2
    assert after.positive_rank == 1
    assert before.hit_at_5 == after.hit_at_5 == 1.0
    assert math.isclose(after.ndcg_at_5, 1.0)


def test_condition_summary_and_pairwise_deltas_on_synthetic_rows() -> None:
    instances = []
    base_kwargs = {
        "user_id": "u1",
        "source_instance_id": "src_1",
        "candidate_ids": ["a", "b", "c", "d", "e"],
        "positive_item_id": "b",
        "response_received": True,
        "parse_success": True,
        "schema_valid": True,
        "latency_seconds": 0.1,
        "parse_status": "parsed_valid",
        "items_removed_by_filter": 0,
        "filtered_from_condition": "",
        "notes": "synthetic",
    }
    for condition, ranked in (
        (ORIGINAL_CONDITION, ["x", "b", "c", "d", "e"]),
        (ORIGINAL_POST_FILTERED_CONDITION, ["b", "c", "d", "e"]),
        (STRICT_CONDITION, ["b", "a", "c", "d", "e"]),
        (STRICT_POST_FILTERED_CONDITION, ["b", "a", "c", "d", "e"]),
    ):
        instances.append(
            condition_instance_from_selection(
                condition=condition,
                diagnostic_id=f"{PHASE}_{condition}_u1",
                ranked_item_ids=ranked,
                **base_kwargs,
            )
        )
    rows = pd.DataFrame([instance.__dict__ for instance in instances])
    rows = pd.DataFrame(
        [
            {
                **{column: "" for column in INSTANCE_COLUMNS},
                "condition": item.condition,
                "response_received": "true",
                "parse_success": "true",
                "schema_valid": "true",
                "selected_count": str(item.selected_count),
                "valid_selected_count": str(item.valid_selected_count),
                "candidate_validity_rate": f"{item.candidate_validity_rate:.6f}",
                "duplicate_count": str(item.duplicate_count),
                "hallucinated_count": str(item.hallucinated_count),
                "hit_at_5": f"{item.hit_at_5:.6f}",
                "ndcg_at_5": f"{item.ndcg_at_5:.6f}",
                "items_removed_by_filter": str(item.items_removed_by_filter),
            }
            for item in instances
        ]
    )
    summary = build_condition_summary(rows)
    delta = build_pairwise_delta(summary)

    assert set(CONDITION_SUMMARY_COLUMNS).issubset(summary.columns)
    strict_hr = delta.loc[(delta["condition"] == STRICT_CONDITION) & (delta["metric_name"] == "HR@5")].iloc[0]
    assert float(strict_hr["delta"]) == 0.0
    strict_ndcg = delta.loc[(delta["condition"] == STRICT_CONDITION) & (delta["metric_name"] == "NDCG@5")].iloc[0]
    assert float(strict_ndcg["delta"]) > 0


def test_candidate_order_signature_is_order_sensitive() -> None:
    original = ["a", "b", "c"]
    reordered = ["c", "b", "a"]

    assert candidate_order_signature(original) != candidate_order_signature(reordered)
    assert set(original) == set(reordered)


def test_raw_output_path_is_under_results_raw_outputs() -> None:
    raw_path = RAW_OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()

    assert raw_path.startswith("results/raw_outputs/")
    assert raw_path.endswith(".jsonl")


def test_d5g_artifacts_if_present_have_expected_structure() -> None:
    if not CONDITION_SUMMARY_PATH.exists():
        pytest.skip("D5G outputs are not generated before the local diagnostic run")
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    instances = pd.read_csv(INSTANCE_SUMMARY_PATH, dtype=str).fillna("")
    summary = pd.read_csv(CONDITION_SUMMARY_PATH, dtype=str).fillna("")
    pairwise = pd.read_csv(PAIRWISE_PATH, dtype=str).fillna("")
    profile = pd.read_csv(POST_FILTER_PROFILE_PATH, dtype=str).fillna("")
    metrics = pd.read_csv(METRICS_PATH, dtype=str).fillna("")

    assert set(MANIFEST_COLUMNS).issubset(manifest.columns)
    assert set(INSTANCE_COLUMNS).issubset(instances.columns)
    assert set(CONDITION_SUMMARY_COLUMNS).issubset(summary.columns)
    assert set(PAIRWISE_COLUMNS).issubset(pairwise.columns)
    assert set(POST_FILTER_PROFILE_COLUMNS).issubset(profile.columns)
    assert len(manifest) == 60
    assert len(instances) == 80
    assert set(summary["condition"]) == {
        ORIGINAL_CONDITION,
        ORIGINAL_POST_FILTERED_CONDITION,
        STRICT_CONDITION,
        STRICT_POST_FILTERED_CONDITION,
    }
    assert set(instances["dataset"]) == {"ml1m"}
    assert set(instances["model"]) == {"qwen2:7b-instruct"}
    assert set(instances["backend"]) == {"ollama"}
    assert set(instances["top_k"]) == {"5"}
    assert set(instances["candidate_pool_size"]) == {"20"}
    assert {"hit_rate", "ndcg", "hallucination_rate"}.issubset(set(metrics["metric"]))
    for column in ("response_received_rate", "parse_success_rate", "schema_valid_rate", "hallucination_rate"):
        values = pd.to_numeric(summary[column], errors="raise")
        assert ((values >= 0.0) & (values <= 1.0)).all()


def test_d5g_report_and_matrix_label_diagnostic_not_exact_if_present() -> None:
    if not REPORT_PATH.exists():
        pytest.skip("D5G report is not generated before the local diagnostic run")
    report = REPORT_PATH.read_text(encoding="utf-8").lower()
    matrix = REPRO_MATRIX_PATH.read_text(encoding="utf-8").lower()

    assert "diagnostic_subset prompt mitigation diagnostic" in report
    assert "not exact reproduction" in report
    assert "not full llm reproduction" in report
    assert "local qwen2 through ollama only" in report
    assert "no openai api call was made" in report
    assert "no anthropic api call was made" in report
    assert "no external api call was made" in report
    assert "no download was performed" in report
    assert "no model pull was performed" in report
    assert "no recommender training was performed" in report
    assert "strict candidate-id-only prompt" in matrix
    assert "diagnostic_subset" in matrix
    assert "not exact reproduction" in matrix


def test_d5g_test_file_does_not_call_ollama_or_external_apis() -> None:
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

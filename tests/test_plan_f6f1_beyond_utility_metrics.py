from __future__ import annotations

import math
import ast
from pathlib import Path

import pandas as pd

from src.analysis import plan_f6f1_beyond_utility_metrics as f6f1


def _popularity() -> dict[str, f6f1.PopularityData]:
    return {
        "Tiny": f6f1.PopularityData(
            dataset="Tiny",
            train_counts={"a": 1, "b": 2, "c": 2, "d": 3, "p": 5},
            train_unique_user_counts={"a": 1, "b": 2, "c": 4, "d": 8, "p": 2},
            total_train_interactions=13,
            train_users=8,
            long_tail_threshold=2.0,
            source_path="memory",
            source_sha256="memory",
            catalog_items=frozenset({"a", "b", "c", "d", "p"}),
        )
    }


def _row(**overrides):
    base = {
        "source_type": "llm",
        "source_file": "memory",
        "request_id": "r1",
        "source_request_id": "s1",
        "dataset": "Tiny",
        "task": "ranking",
        "model_name": "qwen2:7b-instruct",
        "strategy_id": "paper_base_openp5_parametric_v1",
        "strategy_label": "Base",
        "baseline_family": "",
        "candidate_set_id": "cand",
        "user_id": "u1",
        "positive_item_id": "p",
        "candidate_items": ["a", "b", "c", "d", "p"],
        "candidate_order_hash": f6f1.candidate_order_hash(["a", "b", "c", "d", "p"]),
        "parsed_items": ["a", "b", "c", "d", "p"],
        "top_items": ["a", "b", "c", "d", "p"],
        "valid_list": True,
        "strict_valid": True,
        "invalid_reason": "",
        "positive_present": True,
        "positive_initial_position": 5,
        "positive_rank_one_based": 5,
        "hr_at5_operational": 1.0,
        "ndcg_at5_operational": 1 / math.log2(6),
        "hr_at5_conditional": 1.0,
        "ndcg_at5_conditional": 1 / math.log2(6),
    }
    base.update(overrides)
    return base


def test_import_has_no_network_or_training_imports() -> None:
    source = Path(f6f1.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    forbidden = ["requests", "httpx", "socket", "ollama", "torch", "tensorflow"]
    for token in forbidden:
        assert token not in imports


def test_aplt_threshold_and_tie_handling() -> None:
    metrics = f6f1._row_popularity_metrics(_row(), _popularity())
    assert metrics["aplt_at_5"] == 0.2
    assert metrics["long_tail_threshold"] == 2.0
    assert metrics["missing_popularity_count"] == 0


def test_popularity_uses_train_only_counts() -> None:
    metrics = f6f1._row_popularity_metrics(_row(), _popularity())
    assert metrics["average_popularity_at_5"] == (1 + 2 + 2 + 3 + 5) / 5


def test_novelty_unique_user_formula() -> None:
    metrics = f6f1._row_popularity_metrics(_row(), _popularity())
    expected = sum(math.log2(8 / x) for x in [1, 2, 4, 8, 2]) / 5
    assert metrics["novelty_self_information_at_5"] == expected


def test_serendipity_uses_mostpop_reference() -> None:
    llm = _row(source_type="llm", row_key="k", positive_rank_one_based=3)
    mostpop = _row(
        source_type="baseline",
        row_key="k",
        model_name="mostpop_train_only",
        baseline_family="MostPop",
        strategy_id="",
        strategy_label="",
        positive_rank_one_based=8,
    )
    frame = pd.DataFrame([llm, mostpop])
    enriched = f6f1.attach_serendipity_row_metric(frame)
    llm_value = float(enriched.loc[enriched["source_type"].eq("llm"), "serendipity_at_5"].iloc[0])
    assert llm_value == 1.0


def test_invalid_list_policy_not_defined_for_operational_aplt() -> None:
    valid = _row(row_key="valid")
    invalid = _row(request_id="r2", user_id="u2", row_key="invalid", valid_list=False, strict_valid=False, top_items=[])
    frame = f6f1.enrich_row_metrics(pd.DataFrame([valid, invalid]), _popularity())
    aplt_rows, _ = f6f1.compute_aplt_arp_novelty(frame, _popularity())
    operational = [
        row for row in aplt_rows if row["source_type"] == "llm" and row["population"] == "operational"
    ]
    assert operational
    assert all(row["status"] == "not_defined" for row in operational)


def test_ranking_reranking_excludes_lastfm() -> None:
    ranking = _row(dataset="Amazon Beauty", task="ranking", row_key="rank")
    reranking = _row(dataset="Amazon Beauty", task="reranking", row_key="rerank")
    lastfm = _row(dataset="LastFM", task="ranking", row_key="lastfm")
    rows = f6f1.compute_ranking_vs_reranking(pd.DataFrame([ranking, reranking, lastfm]))
    assert any(row["dataset"] == "LastFM" and row["classification"] == "not_applicable" for row in rows)
    assert any(row["dataset"] == "Amazon Beauty" for row in rows)


def test_canddif_formula() -> None:
    value = f6f1.canddif(0.4, 0.15)
    expected = -math.log(0.6) + math.log(0.85)
    assert value == expected


def test_bootstrap_determinism() -> None:
    left = [0.0, 1.0, 0.0, 1.0]
    right = [1.0, 1.0, 0.0, 0.0]
    assert f6f1.paired_bootstrap_ci(left, right, resamples=100, seed=42) == f6f1.paired_bootstrap_ci(
        left, right, resamples=100, seed=42
    )


def test_candidate_position_bins_and_no_causal_claim() -> None:
    first = _row(positive_initial_position=1, row_key="a")
    second = _row(positive_initial_position=12, row_key="b", user_id="u2")
    rows = f6f1.compute_candidate_position(Path("."), {"inputs": {"plan_d_candidate_position_variant_summary": "missing.csv", "f6e4c_request_level_metrics": "memory"}}, pd.DataFrame([first, second]))
    bin_rows = [row for row in rows if row["analysis_type"] == "observational_position_bin"]
    assert {row["stratum"] for row in bin_rows} >= {"first_half_1_10", "second_half_11_20"}
    assert all("no causal" in row["notes"].lower() or row["canddif_formula_status"] for row in rows)


def test_no_proxy_promoted_to_exact() -> None:
    matrix = f6f1.build_metric_exactness_matrix()
    pop_bias = [row for row in matrix if row["paper_metric"] == "popularity bias"][0]
    assert pop_bias["exactness_classification"] == "blocked_missing_evidence"
    proxy = [row for row in matrix if row["local_metric"] == "novelty_self_information_at_5"][0]
    assert proxy["exactness_classification"] != "local_proxy"


def test_llm_vs_baseline_requires_paired_intersection() -> None:
    llm = _row(source_type="llm", row_key="k")
    baseline = _row(
        source_type="baseline",
        row_key="k",
        model_name="mostpop_train_only",
        baseline_family="MostPop",
        strategy_id="",
        strategy_label="",
    )
    frame = pd.DataFrame([llm, baseline])
    frame = f6f1.enrich_row_metrics(frame, _popularity())
    frame = f6f1.attach_serendipity_row_metric(frame)
    rows = f6f1.compute_llm_vs_baseline(frame, resamples=100, seed=42)
    assert any(row["baseline_family"] == "MostPop" and row["paired_intersection_size"] == "1" for row in rows)


def test_structural_diagnostics_separate_extension_policy() -> None:
    valid = _row(row_key="valid")
    invalid = _row(request_id="r2", user_id="u2", row_key="invalid", valid_list=False, strict_valid=False, top_items=[])
    rows = f6f1.build_structural_diagnostics(pd.DataFrame([valid, invalid]))
    invalid_rate = [
        row
        for row in rows
        if row["scope"] == "dataset_task" and row["metric"] == "invalid_list_rate"
    ][0]
    op_hr = [
        row
        for row in rows
        if row["scope"] == "dataset_task" and row["metric"] == "operational_hr_at_5"
    ][0]
    assert invalid_rate["classification"] == "extension_metric"
    assert invalid_rate["value"] == "0.5"
    assert op_hr["classification"] == "exact_formula_adapted_population"

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from src.extension import h0_history_final_parity_audit as h0
from src.extension.e0_protocol import item_list_hash


TABLE_DIR = Path("results/tables")
MANIFEST_PATH = Path("results/manifests/extension_h0_no_inference_manifest.json")
SCRIPT_PATH = Path("src/extension/h0_history_final_parity_audit.py")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name, keep_default_na=False)


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _json_list(value: object) -> list[str]:
    parsed = json.loads(str(value))
    assert isinstance(parsed, list)
    return [str(item) for item in parsed]


def test_final_cohorts_have_deterministic_200_users_per_dataset() -> None:
    cohorts = _read("extension_h0_final_paired_200_cohorts.csv")
    feasibility = _read("extension_h0_dataset_l20_feasibility.csv")

    assert set(cohorts["dataset"]) == set(h0.DATASETS)
    for dataset, group in cohorts.groupby("dataset", sort=True):
        ordered = group.sort_values("selection_rank", kind="mergesort")
        users = ordered["user_id"].astype(str).tolist()
        assert len(users) == h0.FINAL_USERS_PER_DATASET
        assert ordered["selection_rank"].astype(int).tolist() == list(range(1, h0.FINAL_USERS_PER_DATASET + 1))
        assert ordered["cohort_user_hash"].nunique() == 1
        assert ordered["cohort_user_hash"].iloc[0] == item_list_hash(users)
        expected_hash = feasibility.loc[feasibility["dataset"].eq(dataset), "cohort_user_hash"].iloc[0]
        assert ordered["cohort_user_hash"].iloc[0] == expected_hash


def test_l0_is_explicit_empty_history_with_no_tail_zero_leakage() -> None:
    cohorts = _read("extension_h0_final_paired_200_cohorts.csv")

    assert cohorts["history_l0_count"].astype(int).eq(0).all()
    assert cohorts["history_l0_hash"].eq(item_list_hash([])).all()
    assert cohorts["history_l0_item_ids"].map(_json_list).map(len).eq(0).all()


def test_history_lengths_are_exact_and_preserve_suffix_order() -> None:
    cohorts = _read("extension_h0_final_paired_200_cohorts.csv")

    for row in cohorts.itertuples(index=False):
        l20 = _json_list(row.history_l20_item_ids)
        assert len(l20) == 20
        for length in h0.FINAL_LENGTHS:
            history = _json_list(getattr(row, f"history_l{length}_item_ids"))
            assert len(history) == length
            assert getattr(row, f"history_l{length}_hash") == item_list_hash(history)
            if length:
                assert history == l20[-length:]


def test_candidate_pool_positive_and_title_validity_are_strict() -> None:
    cohorts = _read("extension_h0_final_paired_200_cohorts.csv")

    assert _truthy(cohorts["candidate_titles_valid"]).all()
    assert _truthy(cohorts["fully_paired_l20_valid"]).all()
    for length in h0.FINAL_LENGTHS:
        assert _truthy(cohorts[f"history_l{length}_titles_valid"]).all()

    for row in cohorts.itertuples(index=False):
        candidates = _json_list(row.candidate_item_ids)
        assert len(candidates) == h0.CANDIDATE_COUNT
        assert len(set(candidates)) == h0.CANDIDATE_COUNT
        assert row.positive_item_id in candidates
        assert candidates[int(row.positive_position) - 1] == row.positive_item_id
        assert row.candidate_hash == item_list_hash(candidates)


def test_dataset_feasibility_counts_are_ready_and_unblocked() -> None:
    feasibility = _read("extension_h0_dataset_l20_feasibility.csv")
    expected_strict_l20 = {"beauty": 1105, "sports": 1071, "ml1m": 5954, "lastfm": 1720}

    assert set(feasibility["dataset"]) == set(h0.DATASETS)
    assert feasibility["status"].eq("FEASIBLE_FINAL_PAIRED_200").all()
    assert feasibility["blocker"].eq("").all()
    assert feasibility["selected_users"].astype(int).eq(h0.FINAL_USERS_PER_DATASET).all()
    for row in feasibility.itertuples(index=False):
        assert int(row.strict_paired_l20_users) == expected_strict_l20[row.dataset]
        assert int(row.strict_paired_l20_users) >= h0.FINAL_USERS_PER_DATASET
        assert int(row.strict_l1_users) >= int(row.strict_l20_users)
        assert int(row.strict_l2_users) >= int(row.strict_l20_users)
        assert int(row.strict_l5_users) >= int(row.strict_l20_users)
        assert int(row.strict_l10_users) >= int(row.strict_l20_users)


def test_72_cell_matrix_is_all_to_compute_with_final_contract() -> None:
    matrix = _read("extension_h0_72_cell_history_matrix.csv")

    assert len(matrix) == len(h0.DATASETS) * len(h0.MODELS) * len(h0.FINAL_LENGTHS)
    assert set(matrix["dataset"]) == set(h0.DATASETS)
    assert set(matrix["model_name"]) == set(h0.MODELS)
    assert set(matrix["history_length"].astype(int)) == set(h0.FINAL_LENGTHS)
    assert matrix["status"].eq("TO_COMPUTE").all()
    assert matrix["users"].astype(int).eq(h0.FINAL_USERS_PER_DATASET).all()
    assert matrix["candidate_pool_size"].astype(int).eq(h0.CANDIDATE_COUNT).all()
    assert matrix["requested_output_length"].astype(int).eq(h0.REQUESTED_OUTPUT_LENGTH).all()
    assert matrix["parser_contract"].eq(h0.PARSER_CONTRACT_ID).all()
    assert matrix["metric_k"].astype(int).eq(h0.METRIC_K).all()
    assert matrix["reusable_identical_raw_requests"].astype(int).eq(0).all()


def test_r5c_corrected_is_historical_adapted_evidence_not_reusable() -> None:
    inventory = _read("extension_h0_historical_history_experiment_inventory.csv")
    r5c = inventory.loc[inventory["source"].eq("R5C_CORRECTED")]

    assert len(r5c) == 8
    assert set(r5c["model_name"]) == {"qwen2:7b-instruct", "mistral:7b-instruct"}
    assert r5c["cohort_size"].astype(int).eq(100).all()
    assert r5c["candidate_pool_size"].astype(int).eq(20).all()
    assert r5c["requested_output_length"].astype(int).eq(5).all()
    assert r5c["parser_expected_length"].astype(int).eq(5).all()
    assert r5c["metric_cutoff_k"].astype(int).eq(5).all()
    assert r5c["strict_parser"].eq("r5c_ranked_index_json_top5_v1").all()
    assert r5c["reusable_for_final"].astype(str).str.lower().eq("false").all()
    assert r5c["non_reusable_reason"].str.contains("top5 output/parser", regex=False).all()
    lengths = {(row.dataset, row.history_lengths) for row in r5c.itertuples(index=False)}
    assert ("Beauty", "0;10;20") in lengths
    assert ("Sports", "5;10;20") in lengths
    assert ("ML-1M", "5;10;20") in lengths
    assert ("LastFM", "5;10;20") in lengths


def test_budget_manifest_and_grid_are_no_inference_ready() -> None:
    budget = _read("extension_h0_exact_history_request_budget.csv")
    grid = _read("extension_h0_paper_vs_local_history_grid.csv")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    total = budget.loc[budget["scope"].eq("grand_total")].iloc[0]
    assert int(total["required_raw_requests"]) == h0.EXPECTED_TOTAL_REQUESTS
    assert int(total["reusable_identical_raw_requests"]) == 0
    assert int(total["to_compute_raw_requests"]) == h0.EXPECTED_TOTAL_REQUESTS
    assert int(total["blocked_raw_requests"]) == 0
    assert int(total["derived_metric_extra_requests"]) == 0

    assert manifest["decision"] == "HISTORY_FINAL_PARITY_READY"
    assert manifest["llm_requests_executed"] == h0.EXPECTED_LLM_REQUESTS
    assert manifest["network_calls_executed"] == h0.EXPECTED_NETWORK_CALLS
    assert manifest["downloads_executed"] == h0.EXPECTED_DOWNLOADS
    assert manifest["training_runs_executed"] == h0.EXPECTED_TRAINING_RUNS

    assert len(grid) == len(h0.DATASETS) * len(h0.FINAL_LENGTHS)
    assert grid.loc[grid["history_length"].astype(int).eq(0), "l0_semantics"].eq("explicit_empty_history").all()
    assert set(grid["classification"]).issubset({"OFFICIAL_CODE_SUPPORTED", "LOCAL_EXTENSION_GRID"})


def test_audit_script_has_no_model_network_download_or_training_imports() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_roots = {"anthropic", "ollama", "openai", "requests", "socket", "torch", "tensorflow", "urllib"}

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported.intersection(banned_roots)
    assert "allow_api_calls" not in source
    assert "urlretrieve" not in source

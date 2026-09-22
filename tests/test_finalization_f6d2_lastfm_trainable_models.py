from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.analysis.finalization_f6d2_lastfm_trainable_models import (
    ADAPTED_POOL_PATH,
    AGGREGATE_COLUMNS,
    CANDIDATE_PREDICTION_COLUMNS,
    FULL_CATALOG_COLUMNS,
    MANIFEST_COLUMNS,
    MODEL_ORDER,
    MODEL_PROTOCOL_COLUMNS,
    OUTPUT_TABLES,
    POOL_COLUMNS,
    SEED_METRIC_COLUMNS,
    build_adapted_reranking_pool,
    build_aggregate_metrics,
    build_metrics_from_rankings,
    build_model_protocol,
    build_source_selection,
    candidate_rows_for_run,
    checkpoint_paths,
    file_sha256,
    frame_sha256,
    full_catalog_rows_for_run,
    load_config,
    model_vectors,
    no_absolute_paths,
    normalize_candidates,
    prepare_dataset,
    protected_hash_status,
    score_matrix_from_state,
    set_all_seeds,
    train_or_load_checkpoint,
    train_local_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def config_fixture() -> dict:
    return {
        "phase": "finalization_f6d2",
        "dataset": {},
        "models": {"order": MODEL_ORDER},
        "seed_policy": {"seeds": [42, 43, 44, 45, 46]},
        "training": {
            "device": "cpu",
            "epoch_limit": 1,
            "smoke_epoch_limit": 1,
            "embedding_dimension": 4,
            "hidden_dimension": 4,
            "sequence_length": 3,
            "number_of_layers": 1,
            "dropout": 0.0,
            "learning_rate": 0.02,
            "batch_size": 4,
            "optimizer": "Adam",
            "regularization": 0.0,
            "negative_sampling": "uniform_unobserved_train_item",
        },
        "early_stopping": {"metric": "validation_hr_at_5_full_catalog", "patience": 0, "test_based": False},
        "candidate_evaluation": {
            "cutoff": 5,
            "candidate_count": 4,
            "cohort_name": "synthetic_cohort",
            "candidate_set_name": "synthetic_candidates_4",
        },
        "full_catalog_retrieval": {
            "top_n": 3,
            "exclude_train_history": True,
            "exclude_validation_item": True,
            "exclude_test_positive": False,
        },
        "reranking_pool": {
            "policy": "adapted_multi_model_round_robin_union",
            "candidate_count": 4,
            "model_order": MODEL_ORDER,
        },
    }


def synthetic_inputs() -> dict[str, pd.DataFrame]:
    train = pd.DataFrame(
        [
            {"user_id": "1", "item_id": "1"},
            {"user_id": "1", "item_id": "2"},
            {"user_id": "2", "item_id": "2"},
            {"user_id": "2", "item_id": "3"},
            {"user_id": "3", "item_id": "3"},
            {"user_id": "3", "item_id": "4"},
        ]
    )
    validation = pd.DataFrame(
        [
            {"user_id": "1", "item_id": "3"},
            {"user_id": "2", "item_id": "4"},
            {"user_id": "3", "item_id": "5"},
        ]
    )
    test = pd.DataFrame(
        [
            {"user_id": "1", "item_id": "4"},
            {"user_id": "2", "item_id": "5"},
            {"user_id": "3", "item_id": "6"},
        ]
    )
    catalog = pd.DataFrame(
        {
            "item_id": ["1", "2", "3", "4", "5", "6"],
            "title": ["a", "b", "c", "d", "e", "f"],
        }
    )
    candidates = pd.DataFrame(
        [
            {"candidate_set_id": "c1", "user_id": "1", "ground_truth_item_id": "4", "candidate_item_id": item, "candidate_position": pos, "is_ground_truth": str(item == "4").lower()}
            for pos, item in enumerate(["4", "5", "6", "3"], start=1)
        ]
        + [
            {"candidate_set_id": "c2", "user_id": "2", "ground_truth_item_id": "5", "candidate_item_id": item, "candidate_position": pos, "is_ground_truth": str(item == "5").lower()}
            for pos, item in enumerate(["5", "6", "1", "4"], start=1)
        ]
        + [
            {"candidate_set_id": "c3", "user_id": "3", "ground_truth_item_id": "6", "candidate_item_id": item, "candidate_position": pos, "is_ground_truth": str(item == "6").lower()}
            for pos, item in enumerate(["6", "1", "2", "5"], start=1)
        ]
    )
    return {
        "train": train,
        "validation_split": validation,
        "test": test,
        "catalog": catalog,
        "ranking_candidates": candidates,
        "f5af_atomic": pd.DataFrame(
            [
                {"dataset": "LastFM", "model": "BPRMF", "metric": "hr_at_5", "paper_value_numeric": "0.1"},
                {"dataset": "LastFM", "model": "BPRMF", "metric": "ndcg_at_5", "paper_value_numeric": "0.2"},
            ]
        ),
        "f6d1_missing": pd.DataFrame(
            [
                {"paper_component_id": "PC-LASTFM-BPRMF", "remains_active_gap": "true", "f6d1_status": "active_gap", "resolution_category": "blocked", "completion_criterion": "exact protocol"},
                {"paper_component_id": "PC-LLM-GPT4O", "remains_active_gap": "true", "f6d1_status": "active_gap", "resolution_category": "api", "completion_criterion": "api access"},
            ]
        ),
        "f6d1_utility": pd.DataFrame(
            [
                {"model": "random_seeded", "metric": "HR@5", "value": "0.25"},
                {"model": "random_seeded", "metric": "NDCG@5", "value": "0.15"},
            ]
        ),
    }


def test_config_parsing_and_protocol_schema() -> None:
    config = load_config(PROJECT_ROOT)
    assert config["seed_policy"]["seeds"] == [42, 43, 44, 45, 46]
    protocol = build_model_protocol(config)
    assert list(protocol.columns) == MODEL_PROTOCOL_COLUMNS
    assert protocol["model"].tolist() == MODEL_ORDER
    assert protocol["paper_equivalence"].eq("not_exact").all()
    assert no_absolute_paths(protocol)


def test_seed_setup_is_deterministic_for_python_numpy_and_torch() -> None:
    first = set_all_seeds(123)
    a = torch.rand(4)
    second = set_all_seeds(123)
    b = torch.rand(4)
    assert first["torch_seed"] == second["torch_seed"] == 123
    assert torch.equal(a, b)


def test_candidate_parsing_and_train_validation_test_mapping() -> None:
    prepared = prepare_dataset(synthetic_inputs())
    candidates = normalize_candidates(synthetic_inputs()["ranking_candidates"])
    assert prepared["users"] == ["1", "2", "3"]
    assert prepared["items"] == ["1", "2", "3", "4", "5", "6"]
    assert candidates.groupby("user_id").size().eq(4).all()
    assert candidates.groupby("user_id")["is_ground_truth"].sum().eq(1).all()
    assert prepared["validation_item_by_user"][prepared["user_to_idx"]["1"]] == prepared["item_to_idx"]["3"]
    assert prepared["test_item_by_user"][prepared["user_to_idx"]["1"]] == prepared["item_to_idx"]["4"]


@pytest.mark.parametrize("model", MODEL_ORDER)
def test_trainable_model_fixture_produces_finite_state_and_scores(model: str) -> None:
    config = config_fixture()
    prepared = prepare_dataset(synthetic_inputs())
    result = train_local_state(model, 42, prepared, config)
    assert result["epochs_completed"] == 1
    assert result["history"][0]["train_loss"] > 0
    user_vectors, item_vectors = model_vectors(model, result["state"], prepared, config)
    assert user_vectors.shape == (3, 4)
    assert item_vectors.shape == (6, 4)
    scores = score_matrix_from_state(model, result["state"], prepared, config)
    assert scores.shape == (3, 6)
    assert pd.notna(scores).all()


def test_candidate_only_scoring_topn_tie_breaking_and_schema() -> None:
    config = config_fixture()
    prepared = prepare_dataset(synthetic_inputs())
    scores = pd.DataFrame(0.0, index=range(3), columns=range(6)).to_numpy()
    rows = candidate_rows_for_run("run", "bprmf", 42, scores, prepared, config)
    frame = pd.DataFrame(rows)
    assert list(frame.columns) == CANDIDATE_PREDICTION_COLUMNS
    first_user = frame.loc[frame["user_id"].eq("1")].sort_values("rank")
    assert first_user["candidate_item_id"].tolist() == ["3", "4", "5", "6"]
    assert first_user["is_positive"].eq("true").sum() == 1


def test_full_catalog_excludes_seen_items_but_not_test_positive() -> None:
    config = config_fixture()
    prepared = prepare_dataset(synthetic_inputs())
    scores = pd.DataFrame(1.0, index=range(3), columns=range(6)).to_numpy()
    rows = full_catalog_rows_for_run("run", "bprmf", 42, scores, prepared, config)
    frame = pd.DataFrame(rows)
    assert list(frame.columns) == FULL_CATALOG_COLUMNS
    assert frame["was_seen_in_train"].eq("true").sum() == 0
    assert frame["was_seen_in_validation"].eq("true").sum() == 0
    assert frame.groupby("user_id").size().eq(3).all()
    assert frame["is_positive"].eq("true").sum() >= 1


def test_hr_ndcg_seed_metrics_and_aggregate_schema() -> None:
    config = config_fixture()
    prepared = prepare_dataset(synthetic_inputs())
    rows = []
    for seed in [42, 43, 44, 45, 46]:
        scores = pd.DataFrame(float(seed), index=range(3), columns=range(6)).to_numpy()
        rows.extend(candidate_rows_for_run(f"run-{seed}", "bprmf", seed, scores, prepared, config))
    rankings = pd.DataFrame(rows)
    seed_metrics = build_metrics_from_rankings(rankings, evaluation_scope="candidate_set_20", config=config, total_users=3)
    aggregate = build_aggregate_metrics(seed_metrics, synthetic_inputs())
    assert list(seed_metrics.columns) == SEED_METRIC_COLUMNS
    assert list(aggregate.columns) == AGGREGATE_COLUMNS
    hr = aggregate.loc[aggregate["metric"].eq("HR@5"), "mean"].iloc[0]
    assert float(hr) == 1.0
    assert aggregate.loc[aggregate["metric"].eq("HR@5"), "completed_seed_count"].iloc[0] == 5


def test_best_validation_seed_selection_uses_validation_metric_only() -> None:
    manifest = pd.DataFrame(
        [
            {"execution_id": f"run-{seed}", "model": "bprmf", "seed": seed, "best_validation_metric": value, "checkpoint_file": f"ckpt-{seed}.pt"}
            for seed, value in [(42, "0.2"), (43, "0.5"), (44, "0.5"), (45, "0.1"), (46, "0.4")]
        ],
        columns=["execution_id", "model", "seed", "best_validation_metric", "checkpoint_file"],
    )
    for model in MODEL_ORDER[1:]:
        clone = manifest.loc[manifest["model"].eq("bprmf")].copy()
        clone["model"] = model
        manifest = pd.concat([manifest, clone], ignore_index=True)
    selected = build_source_selection(manifest, config_fixture())
    bpr = selected.loc[(selected["model"].eq("bprmf")) & (selected["selected_for_adapted_pool"].eq("true"))]
    assert bpr["seed"].iloc[0] == 43
    assert bpr["selection_metric"].iloc[0] == "validation_hr_at_5_full_catalog"
    assert "no test metric used" in bpr["selection_reason"].iloc[0].lower()


def test_round_robin_adapted_pool_deduplicates_and_can_absent_positive() -> None:
    config = config_fixture()
    prepared = prepare_dataset(synthetic_inputs())
    rows = []
    for model in MODEL_ORDER:
        for user_id in ["1", "2", "3"]:
            for rank, item in enumerate(["1", "2", "3", "5"], start=1):
                rows.append(
                    {
                        "execution_id": f"{model}-42",
                        "model": model,
                        "seed": 42,
                        "user_id": user_id,
                        "item_id": item,
                        "rank": rank,
                        "score": str(10 - rank),
                        "is_positive": "false",
                        "was_seen_in_train": "false",
                        "was_seen_in_validation": "false",
                        "eligible_for_recommendation": "true",
                        "catalog_name": item,
                        "notes": "",
                    }
                )
    pool = build_adapted_reranking_pool(pd.DataFrame(rows), prepared, config)
    assert list(pool.columns) == POOL_COLUMNS
    assert pool.groupby("user_id").size().eq(4).all()
    assert pool.groupby("user_id")["candidate_item_id"].nunique().eq(4).all()
    assert pool["pool_policy"].eq("adapted_multi_model_round_robin_union").all()


def test_checkpoint_save_load_idempotence(tmp_path: Path) -> None:
    config = config_fixture()
    prepared = prepare_dataset(synthetic_inputs())
    first = train_or_load_checkpoint(tmp_path, "bprmf", 42, prepared, config)
    checkpoint_file, metadata_file = checkpoint_paths(tmp_path, "bprmf", 42)
    first_hash = file_sha256(checkpoint_file)
    second = train_or_load_checkpoint(tmp_path, "bprmf", 42, prepared, config)
    second_hash = file_sha256(checkpoint_file)
    assert checkpoint_file.exists()
    assert metadata_file.exists()
    assert first_hash == second_hash
    assert first["metadata"]["state_sha256"] == second["metadata"]["state_sha256"]


def test_frame_hash_idempotence_and_no_absolute_paths() -> None:
    protocol = build_model_protocol(config_fixture())
    assert frame_sha256(protocol) == frame_sha256(protocol.copy())
    assert no_absolute_paths(protocol)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_real_outputs_schema_when_materialized() -> None:
    validation_path = PROJECT_ROOT / OUTPUT_TABLES["validation"]
    if not validation_path.exists():
        assert True
        return
    validation = pd.read_csv(validation_path)
    seed_metrics = pd.read_csv(PROJECT_ROOT / OUTPUT_TABLES["seed_utility_metrics"])
    aggregate = pd.read_csv(PROJECT_ROOT / OUTPUT_TABLES["aggregate_utility_metrics"])
    assert validation["status"].eq("pass").all()
    assert list(seed_metrics.columns) == SEED_METRIC_COLUMNS
    assert list(aggregate.columns) == AGGREGATE_COLUMNS
    assert (PROJECT_ROOT / ADAPTED_POOL_PATH).exists()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_protected_input_invariance_baseline() -> None:
    assert protected_hash_status(PROJECT_ROOT).endswith("changed=0")

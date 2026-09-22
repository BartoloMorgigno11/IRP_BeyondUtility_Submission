from __future__ import annotations

import pandas as pd

from src.extension.e2a_cell_universe import (
    ACTIVE_DATASETS,
    FIFTH_DATASET,
    GEMMA_MODEL,
    ORIGINAL_DATASETS,
    STATUS_BLOCKED,
    STATUS_NOT_APPLICABLE,
    STATUS_TO_COMPUTE_OFFLINE,
    STATUS_TO_INFER,
    STATUS_TO_TRAIN,
    generate_universe,
    validate_universe,
)


def _universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs = generate_universe()
    return outputs["raw"], outputs["cells"], outputs["parity"]


def test_raw_and_result_ids_are_deterministic() -> None:
    first = generate_universe()
    second = generate_universe()

    assert first["raw"]["raw_experiment_id"].tolist() == second["raw"]["raw_experiment_id"].tolist()
    assert first["cells"]["cell_id"].tolist() == second["cells"]["cell_id"].tolist()


def test_active_dataset_set_is_exactly_original_four_plus_foodcom() -> None:
    raw, cells, _ = _universe()

    assert set(ACTIVE_DATASETS) == {"beauty", "sports", "ml1m", "lastfm", "foodcom"}
    assert set(raw["dataset"]) == set(ACTIVE_DATASETS)
    assert set(cells["dataset"]) == set(ACTIVE_DATASETS)


def test_no_duplicate_raw_experiments_or_result_cells() -> None:
    raw, cells, _ = _universe()

    assert raw["raw_experiment_id"].is_unique
    assert cells["cell_id"].is_unique
    semantic_key = [
        "experiment_family",
        "dataset",
        "task",
        "model_name",
        "prompt_strategy",
        "candidate_set_id",
        "cohort_id",
        "history_length",
        "candidate_order_condition",
        "source_model_or_ranker",
        "profile_condition",
    ]
    assert not raw.duplicated(semantic_key).any()


def test_every_required_blocked_and_not_applicable_cell_is_classified() -> None:
    _, cells, _ = _universe()

    required = cells.loc[cells["required_for_claim"].astype(str).str.startswith("REQUIRED")]
    assert not required.empty
    assert required["status"].ne("").all()

    blocked_or_na = cells.loc[cells["status"].isin([STATUS_BLOCKED, STATUS_NOT_APPLICABLE])]
    assert not blocked_or_na.empty
    assert blocked_or_na["blocking_reason"].fillna("").ne("").all()


def test_foodcom_primary_history_and_canddif_rows_are_materialized() -> None:
    raw, _, parity = _universe()
    food = raw.loc[raw["dataset"].eq(FIFTH_DATASET)]

    primary = food.loc[food["claim_family"].eq("foodcom_primary_parity")]
    assert len(primary) == 6
    assert set(primary["model_name"]) == {"qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b"}
    assert set(primary["prompt_strategy"]) == {"BASE", "LLMRANK"}
    assert primary["cohort_size"].astype(int).eq(1000).all()
    assert primary["request_count"].astype(int).eq(1000).all()
    assert primary["status"].eq(STATUS_TO_INFER).all()

    history = food.loc[food["claim_family"].eq("foodcom_history_parity")]
    assert len(history) == 18
    assert set(history["history_length"].astype(str)) == {"0", "1", "2", "5", "10", "20"}
    assert history["cohort_size"].astype(int).eq(200).all()
    assert history["status"].eq(STATUS_TO_INFER).all()

    canddif = food.loc[food["claim_family"].eq("foodcom_canddif_primary")]
    assert len(canddif) == 3
    assert canddif["candidate_order_condition"].eq("ALTERNATE_SEEDED_ORDER").all()
    assert canddif["status"].eq(STATUS_TO_INFER).all()

    food_summary = parity.loc[parity["parity_family"].str.startswith("Food.com")]
    assert set(food_summary["scenario"]) == {
        "FOODCOM_PRIMARY_1000",
        "FOODCOM_HISTORY_200",
        "FOODCOM_CANDDIF_1000",
        "FOODCOM_PROFILE_CONDITIONAL",
    }


def test_foodcom_traditional_and_reranking_statuses_reflect_training_boundary() -> None:
    raw, _, _ = _universe()
    food = raw.loc[raw["dataset"].eq(FIFTH_DATASET)]

    traditional = food.loc[food["claim_family"].eq("foodcom_traditional_parity")]
    assert len(traditional) == 7
    trainable = traditional.loc[traditional["requires_training"].eq("true")]
    offline = traditional.loc[traditional["requires_training"].eq("false")]
    assert len(trainable) == 4
    assert trainable["status"].eq(STATUS_TO_TRAIN).all()
    assert len(offline) == 3
    assert offline["status"].eq(STATUS_TO_COMPUTE_OFFLINE).all()

    reranking = food.loc[food["claim_family"].eq("foodcom_reranking_parity")]
    assert len(reranking) == 6
    assert reranking["status"].eq(STATUS_BLOCKED).all()
    assert reranking["blocking_reason"].eq("BLOCKED_PENDING_SOURCE_RANKS").all()


def test_gemma_parity_exists_for_all_required_original_dataset_families() -> None:
    raw, _, _ = _universe()
    gemma = raw.loc[raw["model_name"].eq(GEMMA_MODEL)]

    primary = gemma.loc[gemma["claim_family"].eq("gemma_original_four_parity")]
    for dataset in ORIGINAL_DATASETS:
        for task in ("ranking", "reranking"):
            for prompt in ("BASE", "LLMRANK"):
                assert not primary.loc[
                    primary["dataset"].eq(dataset)
                    & primary["task"].eq(task)
                    & primary["prompt_strategy"].eq(prompt)
                ].empty

    history = gemma.loc[gemma["claim_family"].eq("gemma_history_parity")]
    assert set(history["dataset"]) == set(ORIGINAL_DATASETS)
    assert set(history["history_length"].astype(str)) == {"0", "1", "2", "5", "10", "20"}

    canddif = gemma.loc[gemma["claim_family"].eq("gemma_canddif_parity")]
    assert set(canddif["dataset"]) == {"beauty", "sports", "lastfm"}
    assert set(canddif["candidate_order_condition"]) == {"FIRST_ORDER", "RANDOM_ORDER"}


def test_completeness_validator_passes_and_rejects_inactive_dataset_rows() -> None:
    raw, cells, _ = _universe()
    assert validate_universe(raw, cells)["passed"]

    broken = raw.copy()
    first = broken.index[0]
    broken.loc[first, "dataset"] = "inactive_dataset"
    validation = validate_universe(broken, cells)
    assert not validation["passed"]
    assert any("inactive datasets present" in error for error in validation["errors"])

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from src.data.prepare_foodcom_extension import (
    ACQUISITION_BLOCKED_TOKEN,
    SUCCESS_TOKEN,
    prepare_foodcom_extension,
)


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_contracts(root: Path) -> None:
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_csv(
        tables / "extension_e1_metric_contract.csv",
        [
            {
                "metric_id": f"metric_{idx:02d}",
                "metric_name": f"Metric {idx}",
                "family": "utility" if idx < 20 else "local_extension",
                "fifth_dataset_required_data": "canonical data primitives",
            }
            for idx in range(29)
        ],
    )
    _write_csv(
        tables / "extension_e1_four_dataset_coverage.csv",
        [{"metric_or_analysis_id": f"coverage_{idx:02d}"} for idx in range(26)],
    )
    _write_csv(
        tables / "extension_e1_fifth_dataset_requirements.csv",
        [
            {
                "requirement": f"Requirement {idx}",
                "mandatory": "yes" if idx < 20 else "yes_later",
                "requirement_class": "MANDATORY_FOR_PAPER_FRAMEWORK",
            }
            for idx in range(23)
        ],
    )
    _write_csv(
        tables / "extension_e1_inference_requirements.csv",
        [{"analysis_id": f"inference_{idx:02d}"} for idx in range(8)],
    )
    _write_csv(
        tables / "extension_e1_dataset_selection_scorecard_template.csv",
        [{"criterion": f"Criterion {idx}"} for idx in range(25)],
    )
    _write_csv(
        tables / "extension_e2_dataset_selection_decision.csv",
        [
            {
                "decision_token": "EXTENSION_E2_FIFTH_DATASET_SELECTED",
                "primary_dataset_id": "foodcom",
                "backup_dataset_id": "steam",
            }
        ],
    )


def _config(root: Path, *, source_dir: str = "data/raw/foodcom/source") -> dict:
    return {
        "dataset": {
            "name": "foodcom",
            "raw_dir": "data/raw/foodcom",
            "processed_dir": "data/processed/foodcom",
            "sample_dir": "data/samples/foodcom",
            "interactions_path": "data/processed/foodcom/interactions.csv",
            "catalog_path": "data/processed/foodcom/catalog.csv",
            "candidates_path": None,
            "ground_truth_path": "data/processed/foodcom/ground_truth.csv",
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": "data/raw/foodcom",
            "processed_dir": "data/processed/foodcom",
            "sample_dir": "data/samples/foodcom",
            "interactions_path": "data/processed/foodcom/interactions.csv",
            "catalog_path": "data/processed/foodcom/catalog.csv",
            "candidates_path": None,
            "ground_truth_path": "data/processed/foodcom/ground_truth.csv",
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "original_repo": {"root": "original_repo/EvaLLMasRecommender"},
        "experiment": {
            "task": "dataset_integrity_freeze",
            "k_values": [5],
            "seed": 7,
            "sample_size": None,
            "allow_api_calls": False,
        },
        "api": {
            "allow_api_calls": False,
            "provider": None,
            "model": None,
            "api_key_env_var": None,
        },
        "models": {
            "llm_outputs": [],
            "baselines": ["random", "mostpop"],
            "baseline_outputs": [],
        },
        "metrics": {
            "utility": ["hr", "ndcg"],
            "beyond_utility": ["coverage"],
            "matching_protocols": ["exact"],
        },
        "output": {
            "metrics_dir": "results/metrics",
            "tables_dir": "results/tables",
            "plots_dir": "results/plots",
            "raw_outputs_dir": "results/raw_outputs",
            "dataset_audit_path": "results/tables/extension_e3_foodcom_raw_statistics.csv",
            "preparation_summary_path": "results/tables/extension_e3_foodcom_split_statistics.csv",
            "metrics_path": "results/metrics/extension_e3_foodcom_no_metrics.csv",
            "summary_path": "results/tables/extension_e3_foodcom_decision_summary.csv",
            "analysis_path": "report/extension_e3_foodcom_data_integrity.md",
        },
        "extension_e3": {
            "evidence_label": "POST-F6M DATASET-5 EXTENSION EVIDENCE",
            "enforce_git_precondition": False,
            "expected_head": "test",
            "expected_branch": "master",
            "source_dir": source_dir,
            "extracted_dir": "data/raw/foodcom/extracted",
            "determinism_dir": "results/logs/extension_e3_foodcom_determinism",
            "required_archive_count": 1,
            "allowed_archive_extensions": [".zip"],
            "required_raw_files": {
                "interactions": "RAW_interactions.csv",
                "recipes": "RAW_recipes.csv",
            },
            "official_sources": {
                "ucsd_mccauley_page": "https://cseweb.ucsd.edu/~jmcauley/datasets.html#foodcom",
                "kaggle_endpoint": "https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions",
                "berd_record": "https://berd-platform.de/records/dmajj-zp851",
            },
            "authoritative_counts": {},
            "interaction_semantics": {
                "definition": "every raw row is an interaction",
                "rating_threshold": None,
                "zero_ratings_are_interactions": True,
                "classification": "FOODCOM_INTERACTION_SEMANTICS_PARITY",
            },
            "duplicate_policy": {
                "exact_duplicate_rows": "collapse_exact_duplicates",
                "repeated_user_item_pairs": "retain_latest_dated_event",
            },
            "temporal_policy": {
                "date_field": "date",
                "date_granularity": "day",
                "invalid_dates": "block_success",
                "tie_policy": "exclude_users_with_any_tied_interaction_date",
                "target_policy": "last_interaction_test_second_last_validation",
            },
            "preprocessing": {
                "min_user_interactions": 3,
                "min_item_interactions": 2,
                "core_algorithm": "iterative_user_item_k_core_after_duplicate_and_tie_policy",
                "min_sequence_length_after_core": 3,
                "split_policy": "chronological_leave_two_out_train_validation_test",
                "mapping_order": "numeric_raw_id_ascending_zero_based_internal_ids",
            },
            "cohort": {
                "size": 2,
                "seed": 7,
                "algorithm": "seeded_shuffle_over_sorted_eligible_users",
                "min_negative_items_for_future_ranking": 1,
                "cohort_id": "foodcom_test",
            },
            "metadata": {
                "title_field": "name",
                "calibration_feature_source": "tags",
                "calibration_feature_format": "python_literal_list",
                "bm25_text_fields": ["name", "description", "ingredients", "tags"],
                "preserve_duplicate_titles": True,
            },
        },
    }


def _write_config(root: Path, config: dict | None = None) -> Path:
    path = root / "configs" / "foodcom_extension.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config or _config(root), sort_keys=False), encoding="utf-8")
    return path


def _foodcom_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    interactions = []
    plan = {
        "1": ["10", "20", "30", "40"],
        "2": ["10", "20", "30", "50"],
        "3": ["10", "20", "40", "50"],
        "4": ["20", "30", "40", "50"],
    }
    for user_id, items in plan.items():
        for idx, item_id in enumerate(items, start=1):
            interactions.append(
                {
                    "user_id": user_id,
                    "recipe_id": item_id,
                    "date": f"2020-01-0{idx}",
                    "rating": 0 if user_id == "4" and idx == 1 else 5,
                    "review": f"review {user_id}-{item_id}",
                }
            )
    interactions.append(dict(interactions[0]))
    recipes = [
        {
            "name": f"Recipe {item_id}",
            "id": item_id,
            "minutes": 10,
            "contributor_id": 100,
            "submitted": "2019-01-01",
            "tags": "['tag-a', 'tag-b']",
            "nutrition": "[1, 2, 3]",
            "n_steps": 2,
            "steps": "['mix', 'cook']",
            "description": f"Description {item_id}",
            "ingredients": "['salt', 'water']",
            "n_ingredients": 2,
        }
        for item_id in ["10", "20", "30", "40", "50"]
    ]
    return interactions, recipes


def _write_foodcom_archive(root: Path) -> None:
    source = root / "data" / "raw" / "foodcom" / "source"
    source.mkdir(parents=True, exist_ok=True)
    work = root / "tmp_foodcom_source"
    work.mkdir()
    interactions, recipes = _foodcom_rows()
    _write_csv(work / "RAW_interactions.csv", interactions)
    _write_csv(work / "RAW_recipes.csv", recipes)
    with zipfile.ZipFile(source / "food-com-recipes-and-user-interactions.zip", "w") as zf:
        zf.write(work / "RAW_interactions.csv", "RAW_interactions.csv")
        zf.write(work / "RAW_recipes.csv", "RAW_recipes.csv")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prepared_project(tmp_path: Path, *, with_archive: bool) -> Path:
    _write_project_markers(tmp_path)
    _write_contracts(tmp_path)
    config_path = _write_config(tmp_path)
    if with_archive:
        _write_foodcom_archive(tmp_path)
    return config_path


def test_missing_manual_archive_returns_acquisition_blocked(tmp_path: Path) -> None:
    config_path = _prepared_project(tmp_path, with_archive=False)

    result = prepare_foodcom_extension(config_path, project_root=tmp_path)

    assert result.decision_token == ACQUISITION_BLOCKED_TOKEN
    decision = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_decision_summary.csv")
    assert decision[0]["decision_token"] == ACQUISITION_BLOCKED_TOKEN
    metric_rows = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_metric_readiness.csv")
    requirement_rows = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_requirement_realization.csv")
    assert len(metric_rows) == 29
    assert len(requirement_rows) == 23
    assert {row["readiness_status"] for row in metric_rows} == {"UNRESOLVED"}


def test_foodcom_e3_success_builds_deterministic_processed_artifacts(tmp_path: Path) -> None:
    config_path = _prepared_project(tmp_path, with_archive=True)

    result = prepare_foodcom_extension(config_path, project_root=tmp_path)

    assert result.decision_token == SUCCESS_TOKEN
    processed = tmp_path / "data/processed/foodcom"
    for filename in [
        "interactions.csv",
        "ground_truth.csv",
        "catalog.csv",
        "user_mapping.csv",
        "item_mapping.csv",
        "cohort_users.csv",
    ]:
        assert (processed / filename).exists()
    decision = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_decision_summary.csv")
    assert decision[0]["decision_token"] == SUCCESS_TOKEN
    assert decision[0]["processed_tree_sha256"]


def test_foodcom_e3_outputs_have_complete_readiness_and_model_rows(tmp_path: Path) -> None:
    config_path = _prepared_project(tmp_path, with_archive=True)

    prepare_foodcom_extension(config_path, project_root=tmp_path)

    metric_rows = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_metric_readiness.csv")
    requirement_rows = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_requirement_realization.csv")
    model_rows = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_model_readiness.csv")
    protected = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_protected_inputs.csv")
    traces = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_forensic_traces.csv")

    assert len(metric_rows) == 29
    assert len(requirement_rows) == 23
    assert len(model_rows) == 7
    assert {row["readiness_status"] for row in model_rows}.issubset(
        {"READY", "READY_WITH_MAPPING_ADAPTER"}
    )
    assert {row["phase"] for row in protected} == {"before_preprocessing", "after_preprocessing"}
    assert traces
    assert {row["status"] for row in traces} == {"pass"}


def test_foodcom_e3_split_and_mapping_invariants(tmp_path: Path) -> None:
    config_path = _prepared_project(tmp_path, with_archive=True)

    prepare_foodcom_extension(config_path, project_root=tmp_path)

    processed = tmp_path / "data/processed/foodcom"
    interactions = pd.read_csv(processed / "interactions.csv", dtype=str)
    ground_truth = pd.read_csv(processed / "ground_truth.csv", dtype=str)
    user_mapping = pd.read_csv(processed / "user_mapping.csv", dtype=str)
    item_mapping = pd.read_csv(processed / "item_mapping.csv", dtype=str)
    cohort = pd.read_csv(processed / "cohort_users.csv", dtype=str)

    assert set(interactions["split"]) == {"train"}
    assert set(ground_truth["split"]) == {"validation", "test"}
    train_pairs = set(zip(interactions["user_id"], interactions["item_id"]))
    target_pairs = set(zip(ground_truth["user_id"], ground_truth["ground_truth_item_id"]))
    assert not train_pairs.intersection(target_pairs)
    assert user_mapping["raw_user_id"].is_unique
    assert user_mapping["internal_user_id"].astype(int).tolist() == list(range(len(user_mapping)))
    assert item_mapping["raw_recipe_id"].is_unique
    assert item_mapping["internal_item_id"].astype(int).tolist() == list(range(len(item_mapping)))
    assert (cohort["in_operational_cohort"] == "true").sum() == 2


def test_foodcom_e3_determinism_hash_is_stable_across_runs(tmp_path: Path) -> None:
    config_path = _prepared_project(tmp_path, with_archive=True)

    prepare_foodcom_extension(config_path, project_root=tmp_path)
    first = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_decision_summary.csv")[0][
        "processed_tree_sha256"
    ]
    prepare_foodcom_extension(config_path, project_root=tmp_path)
    second = _read_rows(tmp_path / "results/tables/extension_e3_foodcom_decision_summary.csv")[0][
        "processed_tree_sha256"
    ]

    assert first == second

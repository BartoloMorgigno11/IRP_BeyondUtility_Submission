"""Build the Plan E GRU4Rec adapter and sequential preflight audit.

This script audits data contracts only. It does not train GRU4Rec, does not
score project-data candidates, and does not create recommendation or metric
outputs.
"""

from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.plan_e_gru4rec_adapter import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    GRU4REC_PADDING_INDEX,
    GRU4REC_REAL_ITEM_OFFSET,
    PLAN_E2_CANDIDATE_COUNT,
    PLAN_E2_CANDIDATE_SET_IDS,
    audit_index_space,
    audit_plan_e2_gru4rec_candidate_row,
    bool_text,
    build_gru4rec_model_index,
    build_training_histories,
    filter_train_interactions,
    history_bucket,
    json_array,
    load_raw_to_internal_mapping,
    natural_id_sort_key,
    parse_candidate_items,
    read_csv_strings,
)
from src.audit.common_io import write_csv_rows as _write_csv_rows
from src.audit.common_paths import relative_to_root as _common_path_relative_to_root


PROTECTED_HASHES = {
    "src/baselines/plan_d_gru4rec.py": "5E288F1917B8F9A3C037D5C3EEF3BE4C2320DF465B23370D94982F15F63F699D",
    "src/experiments/run_plan_d_gru4rec.py": "42F3EECEB8BE58C94EBB353F5DD02BAE8EC55E9859EE6455470069E8206E40AC",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/neural_environment_inventory.csv": "D3B59338A59A3ED777CA382541594424014F06B895278878884D6E3F0C91FE12",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv": "D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478",
    "data/reconstructed/plan_c/amazon_beauty/datamaps.json": "4EF9E7F9BA6D0F98998EFAC7DFDEA455315635753B9C21FF46269ABC31670423",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv": "E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40",
    "data/reconstructed/plan_c/amazon_sports/datamaps.json": "5A0E99616F2C31DF45E13B82DD9012A7728AFC0507C475876E0E6C78ACFE6A5A",
    "data/reconstructed/plan_c/ml1m/interactions.csv": "1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870",
    "data/reconstructed/plan_c/ml1m/datamaps.json": "D6AC1A6C19E47156955DE87F45EE2255AC69728510E734BE46DBBBECD68BF093",
    "data/reconstructed/plan_c/lastfm/interactions.csv": "C57C4ED08B0F1D45C338E8AFFB0E7E6522786D1D4B9844527F49987257BD7B7B",
    "data/reconstructed/plan_c/lastfm/datamaps.json": "E8DFA8BA693A3DDC0315D1BDCD019CD924F6617CD32C9A02280699D11FDF3DB0",
}

DATASETS = {
    "Amazon Beauty": {
        "folder": "amazon_beauty",
        "task_ids": {
            "ranking": "plan_e_beauty_1000user_ranking_manifest",
            "reranking": "plan_e_beauty_1000user_reranking_manifest",
        },
    },
    "Amazon Sports": {
        "folder": "amazon_sports",
        "task_ids": {
            "ranking": "plan_e_sports_1000user_ranking_manifest",
            "reranking": "plan_e_sports_1000user_reranking_manifest",
        },
    },
    "ML-1M": {
        "folder": "ml1m",
        "task_ids": {
            "ranking": "plan_e_ml1m_1000user_ranking_manifest",
            "reranking": "plan_e_ml1m_1000user_reranking_manifest",
        },
    },
}

LASTFM = {"folder": "lastfm"}

MANIFEST_COLUMNS = [
    "preflight_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "source_path",
    "source_plan",
    "full_cohort_rows",
    "phase2_matched_rows",
    "training_data_path",
    "sequential_data_path",
    "user_mapping_path",
    "item_mapping_path",
    "local_gru4rec_implementation_path",
    "local_gru4rec_runner_path",
    "environment_id",
    "python_executable",
    "python_version",
    "torch_version",
    "cuda_available",
    "cuda_device",
    "history_construction_status",
    "item_index_policy",
    "padding_policy",
    "maximum_sequence_length",
    "minimum_required_history_length",
    "adapter_status",
    "project_data_training_performed",
    "project_data_scoring_performed",
    "recommended_pilot_scope",
    "blocking_reason",
    "notes",
]

SEQUENCE_COLUMNS = [
    "dataset",
    "training_data_path",
    "sequential_data_path",
    "user_id_field",
    "item_id_field",
    "timestamp_field",
    "split_field",
    "chronology_source",
    "timestamp_status",
    "equal_timestamp_tie_rule",
    "train_interaction_count",
    "validation_interaction_count",
    "test_interaction_count",
    "train_user_count",
    "train_item_count",
    "users_with_zero_train_events",
    "users_with_one_train_event",
    "users_with_two_or_more_train_events",
    "users_with_training_pair",
    "minimum_train_history_length",
    "maximum_train_history_length",
    "mean_train_history_length",
    "median_train_history_length",
    "maximum_sequence_length",
    "histories_requiring_truncation",
    "duplicate_user_item_event_count",
    "chronology_ambiguous_user_count",
    "status",
    "blocking_reason",
    "notes",
]

INDEX_COLUMNS = [
    "dataset",
    "entity_type",
    "mapping_path",
    "mapping_direction",
    "raw_id_count",
    "internal_id_count",
    "minimum_source_internal_id",
    "maximum_source_internal_id",
    "source_index_zero_semantics",
    "gru4rec_padding_index",
    "gru4rec_real_item_offset",
    "gru4rec_model_cardinality",
    "duplicate_raw_id_count",
    "duplicate_internal_id_count",
    "contiguous_gru4rec_index_space",
    "train_seen_entity_count",
    "plan_e2_unique_entity_count",
    "plan_e2_mapped_entity_count",
    "plan_e2_train_seen_entity_count",
    "mapping_coverage",
    "train_seen_coverage",
    "status",
    "blocking_reason",
    "notes",
]

ROW_COLUMNS = [
    "candidate_set_id",
    "source_path",
    "dataset",
    "task_type",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "user_mapped",
    "history_available",
    "training_pair_available",
    "raw_train_history_length",
    "effective_history_length",
    "history_truncated",
    "positive_item_mapped",
    "positive_item_seen_in_train",
    "all_candidate_items_mapped",
    "all_candidate_items_seen_in_train",
    "unmapped_candidate_item_count",
    "unmapped_candidate_item_ids",
    "train_unseen_candidate_item_count",
    "train_unseen_candidate_item_ids",
    "padding_index_compatible",
    "chronology_verified",
    "temporal_protocol_unverifiable",
    "sequentially_compatible",
    "eligible_plan_e2_full",
    "eligible_plan_e2_matched",
    "compatibility_status",
    "exclusion_reasons",
    "notes",
]

HISTORY_COLUMNS = [
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "evaluated_rows",
    "rows_history_length_0",
    "rows_history_length_1",
    "rows_history_length_2_to_4",
    "rows_history_length_5_to_9",
    "rows_history_length_10_to_19",
    "rows_history_length_20_or_more",
    "rows_history_truncated",
    "rows_without_training_pair",
    "rows_with_unmapped_candidate",
    "rows_with_train_unseen_candidate",
    "sequentially_compatible_rows",
    "retention_rate",
    "positive_position_distribution_full",
    "positive_position_distribution_compatible",
    "mean_raw_history_length",
    "median_raw_history_length",
    "mean_effective_history_length",
    "median_effective_history_length",
    "notes",
]

POLICY_COLUMNS = [
    "policy_id",
    "policy_name",
    "description",
    "requires_training_history",
    "minimum_history_length",
    "requires_all_candidates_train_seen",
    "preserves_candidate_membership",
    "preserves_phase2_cohort",
    "uses_validation_or_test_history",
    "uses_train_unseen_parameters",
    "introduces_data_leakage",
    "changes_gru4rec_definition",
    "paper_alignment",
    "comparability_impact",
    "methodological_risk",
    "allowed_for_future_execution",
    "recommendation_status",
    "rationale",
    "required_additional_cohort",
    "stop_condition",
    "notes",
]


def main() -> int:
    verify_protected_hashes()
    tables_dir = PROJECT_ROOT / "results" / "tables"
    report_dir = PROJECT_ROOT / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_csv_strings(tables_dir / "ranking_candidate_sets.csv")
    membership = read_csv_strings(tables_dir / "experiment_cohort_membership.csv")
    cohort_summary = read_csv_strings(tables_dir / "experiment_cohort_summary.csv")
    env_inventory = read_csv_strings(tables_dir / "neural_environment_inventory.csv")
    validate_phase2_inputs(candidates, membership)
    env = repo_venv_environment(env_inventory)

    plan_e_candidates = candidates.loc[candidates["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].copy()
    cohort_lookup = build_cohort_lookup(membership)
    summary_lookup = build_summary_lookup(cohort_summary)

    manifest_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    dataset_context: dict[str, dict[str, Any]] = {}

    for dataset, spec in DATASETS.items():
        context = build_dataset_context(dataset, spec, plan_e_candidates, cohort_lookup)
        dataset_context[dataset] = context
        sequence_rows.append(build_sequence_row(context))
        index_rows.extend(build_index_rows(context))
        for row in context["candidate_rows"]:
            key = (str(row["candidate_set_id"]), str(row["user_id"]))
            cohort = cohort_lookup[key]
            audited = audit_plan_e2_gru4rec_candidate_row(
                row,
                user_mapping=context["user_mapping"],
                item_mapping=context["item_mapping"],
                train_histories=context["train_histories"],
                train_seen_items=context["train_seen_items"],
                item_model_index=context["item_model_index"],
                eligible_full=cohort["eligible_plan_e2_paper_like_full"],
                eligible_matched=cohort["eligible_plan_e2_matched_comparison"],
                max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
                chronology_verified=True,
                temporal_protocol_unverifiable=False,
            )
            audited["task_type"] = cohort["task_type"]
            compatibility_rows.append(audited)
        for task_type, candidate_set_id in spec["task_ids"].items():
            manifest_rows.append(
                build_manifest_row(
                    dataset=dataset,
                    task_type=task_type,
                    candidate_set_id=candidate_set_id,
                    context=context,
                    summary_lookup=summary_lookup,
                    env=env,
                )
            )

    lastfm_context = build_lastfm_context()
    sequence_rows.append(build_lastfm_sequence_row(lastfm_context))
    index_rows.extend(build_lastfm_index_rows(lastfm_context))
    manifest_rows.append(build_lastfm_manifest_row(env, lastfm_context))

    compatibility_rows = sort_compatibility_rows(compatibility_rows)
    history_rows = build_history_rows(compatibility_rows, plan_e_candidates)
    policy_rows = build_policy_rows()

    write_csv(tables_dir / "plan_e_gru4rec_preflight_manifest.csv", MANIFEST_COLUMNS, manifest_rows)
    write_csv(tables_dir / "plan_e_gru4rec_sequence_audit.csv", SEQUENCE_COLUMNS, sequence_rows)
    write_csv(tables_dir / "plan_e_gru4rec_index_audit.csv", INDEX_COLUMNS, index_rows)
    write_csv(
        tables_dir / "plan_e_gru4rec_row_compatibility.csv",
        ROW_COLUMNS,
        serialize_bool_rows(compatibility_rows),
    )
    write_csv(tables_dir / "plan_e_gru4rec_history_impact.csv", HISTORY_COLUMNS, history_rows)
    write_csv(tables_dir / "plan_e_gru4rec_policy_options.csv", POLICY_COLUMNS, policy_rows)

    validate_outputs(tables_dir)
    write_report(
        report_dir / "plan_e_gru4rec_adapter_preflight.md",
        manifest_rows=manifest_rows,
        sequence_rows=sequence_rows,
        index_rows=index_rows,
        compatibility_rows=compatibility_rows,
        history_rows=history_rows,
        policy_rows=policy_rows,
        env=env,
    )

    print(f"preflight_manifest_rows={len(manifest_rows)}")
    print(f"sequence_audit_rows={len(sequence_rows)}")
    print(f"index_audit_rows={len(index_rows)}")
    print(f"row_compatibility_rows={len(compatibility_rows)}")
    print(f"history_impact_rows={len(history_rows)}")
    print(f"policy_option_rows={len(policy_rows)}")
    print("project_data_training_performed=false")
    print("project_data_scoring_performed=false")
    return 0


def verify_protected_hashes() -> None:
    for rel_path, expected in PROTECTED_HASHES.items():
        path = PROJECT_ROOT / rel_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if actual != expected:
            raise RuntimeError(f"protected input hash mismatch for {rel_path}: {actual} != {expected}")


def validate_phase2_inputs(candidates: pd.DataFrame, membership: pd.DataFrame) -> None:
    if len(candidates) != 76121:
        raise RuntimeError(f"expected 76121 canonical candidate rows, found {len(candidates)}")
    required = {
        "candidate_set_id",
        "source_path",
        "source_plan",
        "dataset",
        "user_id",
        "positive_item_id",
        "candidate_items",
        "candidate_count",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise RuntimeError(f"ranking_candidate_sets.csv missing columns: {sorted(missing)}")
    plan_e = candidates.loc[candidates["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].copy()
    if len(plan_e) != 6000:
        raise RuntimeError(f"expected 6000 Plan E2 rows, found {len(plan_e)}")
    if set(plan_e["candidate_set_id"]) != PLAN_E2_CANDIDATE_SET_IDS:
        raise RuntimeError("Plan E2 candidate-set IDs do not match the expected six IDs")
    for row in plan_e.itertuples(index=False):
        items = parse_candidate_items(getattr(row, "candidate_items"))
        if len(items) != PLAN_E2_CANDIDATE_COUNT:
            raise RuntimeError("Plan E2 row with non-20 candidate count")
        positive = str(getattr(row, "positive_item_id"))
        if items.count(positive) != 1:
            raise RuntimeError("Plan E2 row does not contain exactly one positive item")
    full = membership["eligible_plan_e2_paper_like_full"].astype(str).str.lower().eq("true")
    matched = membership["eligible_plan_e2_matched_comparison"].astype(str).str.lower().eq("true")
    if int(full.sum()) != 6000:
        raise RuntimeError(f"expected 6000 full-cohort rows, found {int(full.sum())}")
    if int(matched.sum()) != 5814:
        raise RuntimeError(f"expected 5814 matched-comparison rows, found {int(matched.sum())}")


def repo_venv_environment(env_inventory: pd.DataFrame) -> dict[str, str]:
    row = env_inventory.loc[env_inventory["environment_id"].astype(str) == "repo_venv_cuda"]
    if row.empty:
        raise RuntimeError("repo_venv_cuda not found in neural environment inventory")
    first = row.iloc[0].to_dict()
    return {str(key): str(value) for key, value in first.items()}


def build_cohort_lookup(membership: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    plan_e = membership.loc[membership["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].copy()
    for row in plan_e.to_dict("records"):
        key = (str(row["candidate_set_id"]), str(row["user_id"]))
        lookup[key] = {
            "task_type": str(row["task_type"]),
            "eligible_plan_e2_paper_like_full": str(row["eligible_plan_e2_paper_like_full"]).lower() == "true",
            "eligible_plan_e2_matched_comparison": str(row["eligible_plan_e2_matched_comparison"]).lower()
            == "true",
        }
    return lookup


def build_summary_lookup(summary: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary.to_dict("records"):
        lookup[(str(row["cohort_id"]), str(row["candidate_set_id"]))] = row
    return lookup


def build_dataset_context(
    dataset: str,
    spec: dict[str, Any],
    plan_e_candidates: pd.DataFrame,
    cohort_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    folder = spec["folder"]
    base = PROJECT_ROOT / "data" / "reconstructed" / "plan_c" / folder
    interactions_path = base / "interactions.csv"
    datamap_path = base / "datamaps.json"
    sequential_path = base / "sequential_data.txt"
    interactions = read_csv_strings(interactions_path)
    train = filter_train_interactions(interactions)
    train_histories, chronology, ordered_train = build_training_histories(interactions)
    train_seen_users = set(train_histories)
    train_seen_items = {str(item) for sequence in train_histories.values() for item in sequence}
    user_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="user")
    item_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="item")
    rows = (
        plan_e_candidates.loc[plan_e_candidates["dataset"].astype(str) == dataset]
        .copy()
        .sort_values(["candidate_set_id", "user_id"], key=lambda s: s.map(natural_id_sort_key), kind="mergesort")
        .to_dict("records")
    )
    for row in rows:
        key = (str(row["candidate_set_id"]), str(row["user_id"]))
        if key not in cohort_lookup:
            raise RuntimeError(f"missing cohort row for {key}")
        row["task_type"] = cohort_lookup[key]["task_type"]

    plan_e_users = {str(row["user_id"]) for row in rows}
    plan_e_items: set[str] = set()
    for row in rows:
        plan_e_items.update(parse_candidate_items(row["candidate_items"]))
        plan_e_items.add(str(row["positive_item_id"]))
    item_model_index = build_gru4rec_model_index(train_seen_items)
    return {
        "dataset": dataset,
        "folder": folder,
        "training_data_path": rel(interactions_path),
        "sequential_data_path": rel(sequential_path) if sequential_path.exists() else "",
        "datamap_path": rel(datamap_path),
        "interactions": interactions,
        "train": train,
        "ordered_train": ordered_train,
        "chronology": chronology,
        "train_histories": train_histories,
        "train_seen_users": train_seen_users,
        "train_seen_items": train_seen_items,
        "plan_e_users": plan_e_users,
        "plan_e_items": plan_e_items,
        "candidate_rows": rows,
        "user_mapping": user_mapping,
        "item_mapping": item_mapping,
        "item_model_index": item_model_index,
    }


def build_lastfm_context() -> dict[str, Any]:
    base = PROJECT_ROOT / "data" / "reconstructed" / "plan_c" / "lastfm"
    interactions_path = base / "interactions.csv"
    datamap_path = base / "datamaps.json"
    interactions = read_csv_strings(interactions_path)
    train = filter_train_interactions(interactions)
    user_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="user")
    item_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="item")
    return {
        "dataset": "LastFM",
        "folder": "lastfm",
        "training_data_path": rel(interactions_path),
        "sequential_data_path": "",
        "datamap_path": rel(datamap_path),
        "interactions": interactions,
        "train": train,
        "train_histories": {},
        "train_seen_users": set(),
        "train_seen_items": set(),
        "plan_e_users": set(),
        "plan_e_items": set(),
        "user_mapping": user_mapping,
        "item_mapping": item_mapping,
        "item_model_index": {},
    }


def build_sequence_row(context: dict[str, Any]) -> dict[str, Any]:
    interactions = context["interactions"]
    train = context["train"]
    histories = context["train_histories"]
    lengths = [len(history) for history in histories.values()]
    plan_e_user_lengths = [len(histories.get(user_id, ())) for user_id in sorted(context["plan_e_users"])]
    chronology = context["chronology"]
    duplicate_user_item_event_count = len(train) - len(train.drop_duplicates(["user_id", "item_id"]))
    status = "warning" if chronology.chronology_ambiguous_user_count else "passed"
    split = interactions["split"].astype(str).str.lower() if "split" in interactions.columns else pd.Series([], dtype=str)
    return {
        "dataset": context["dataset"],
        "training_data_path": context["training_data_path"],
        "sequential_data_path": context["sequential_data_path"],
        "user_id_field": "user_id",
        "item_id_field": "item_id",
        "timestamp_field": "timestamp",
        "split_field": "split",
        "chronology_source": chronology.chronology_source,
        "timestamp_status": chronology.timestamp_status,
        "equal_timestamp_tie_rule": chronology.equal_timestamp_tie_rule,
        "train_interaction_count": len(train),
        "validation_interaction_count": int(split.eq("validation").sum()),
        "test_interaction_count": int(split.eq("test").sum()),
        "train_user_count": len(histories),
        "train_item_count": len(context["train_seen_items"]),
        "users_with_zero_train_events": sum(length == 0 for length in plan_e_user_lengths),
        "users_with_one_train_event": sum(length == 1 for length in plan_e_user_lengths),
        "users_with_two_or_more_train_events": sum(length >= 2 for length in plan_e_user_lengths),
        "users_with_training_pair": sum(length >= 2 for length in plan_e_user_lengths),
        "minimum_train_history_length": min(lengths) if lengths else 0,
        "maximum_train_history_length": max(lengths) if lengths else 0,
        "mean_train_history_length": f"{(statistics.fmean(lengths) if lengths else 0.0):.12g}",
        "median_train_history_length": f"{(statistics.median(lengths) if lengths else 0.0):.12g}",
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "histories_requiring_truncation": sum(length > DEFAULT_MAX_SEQUENCE_LENGTH for length in plan_e_user_lengths),
        "duplicate_user_item_event_count": duplicate_user_item_event_count,
        "chronology_ambiguous_user_count": chronology.chronology_ambiguous_user_count,
        "status": status,
        "blocking_reason": "",
        "notes": (
            "equal timestamps are sorted with the documented repository-row-order tie key; "
            "zero/one/two-user counts are over unique Plan E2 users"
        ),
    }


def build_lastfm_sequence_row(context: dict[str, Any]) -> dict[str, Any]:
    interactions = context["interactions"]
    split = interactions["split"].astype(str).str.lower() if "split" in interactions.columns else pd.Series([], dtype=str)
    return {
        "dataset": "LastFM",
        "training_data_path": context["training_data_path"],
        "sequential_data_path": "",
        "user_id_field": "user_id",
        "item_id_field": "item_id",
        "timestamp_field": "timestamp",
        "split_field": "split",
        "chronology_source": "not_applicable_without_train_split_or_plan_e2_rows",
        "timestamp_status": "blocked_no_training_split",
        "equal_timestamp_tie_rule": "not_applicable",
        "train_interaction_count": 0,
        "validation_interaction_count": int(split.eq("validation").sum()),
        "test_interaction_count": int(split.eq("test").sum()),
        "train_user_count": 0,
        "train_item_count": 0,
        "users_with_zero_train_events": 0,
        "users_with_one_train_event": 0,
        "users_with_two_or_more_train_events": 0,
        "users_with_training_pair": 0,
        "minimum_train_history_length": 0,
        "maximum_train_history_length": 0,
        "mean_train_history_length": "0",
        "median_train_history_length": "0",
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "histories_requiring_truncation": 0,
        "duplicate_user_item_event_count": 0,
        "chronology_ambiguous_user_count": 0,
        "status": "blocked",
        "blocking_reason": "LastFM interactions are split=observed and no Plan E2 candidate rows exist.",
        "notes": "LastFM remains explicit not-applicable/blocked for this Plan E2 GRU4Rec preflight.",
    }


def build_index_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entity_type, mapping, train_seen, plan_e_ids in [
        ("user", context["user_mapping"], context["train_seen_users"], context["plan_e_users"]),
        ("item", context["item_mapping"], context["train_seen_items"], context["plan_e_items"]),
    ]:
        audit = audit_index_space(mapping)
        mapped_count = sum(1 for raw_id in plan_e_ids if raw_id in mapping.raw_to_internal)
        train_seen_count = sum(1 for raw_id in plan_e_ids if raw_id in train_seen)
        plan_e_count = len(plan_e_ids)
        mapping_coverage = mapped_count / plan_e_count if plan_e_count else 0.0
        train_seen_coverage = train_seen_count / plan_e_count if plan_e_count else 0.0
        if mapping_coverage < 1.0:
            status = "blocked"
            blocking = "at least one Plan E2 raw ID lacks an explicit datamap entry"
        elif train_seen_coverage < 1.0 and entity_type == "item":
            status = "warning"
            blocking = ""
        else:
            status = "passed"
            blocking = ""
        rows.append(
            {
                "dataset": context["dataset"],
                "entity_type": entity_type,
                "mapping_path": context["datamap_path"],
                "mapping_direction": "raw_to_internal via user2id/item2id",
                "raw_id_count": mapping.raw_id_count,
                "internal_id_count": mapping.internal_id_count,
                "minimum_source_internal_id": audit.minimum_internal_id,
                "maximum_source_internal_id": audit.maximum_internal_id,
                "source_index_zero_semantics": audit.index_zero_semantics,
                "gru4rec_padding_index": GRU4REC_PADDING_INDEX if entity_type == "item" else "not_applicable",
                "gru4rec_real_item_offset": GRU4REC_REAL_ITEM_OFFSET if entity_type == "item" else "not_applicable",
                "gru4rec_model_cardinality": (len(train_seen) + 1) if entity_type == "item" else len(train_seen),
                "duplicate_raw_id_count": 0,
                "duplicate_internal_id_count": 0,
                "contiguous_gru4rec_index_space": bool_text(bool(train_seen) or entity_type == "user"),
                "train_seen_entity_count": len(train_seen),
                "plan_e2_unique_entity_count": plan_e_count,
                "plan_e2_mapped_entity_count": mapped_count,
                "plan_e2_train_seen_entity_count": train_seen_count,
                "mapping_coverage": f"{mapping_coverage:.12g}",
                "train_seen_coverage": f"{train_seen_coverage:.12g}",
                "status": status,
                "blocking_reason": blocking,
                "notes": (
                    "item model cardinality includes padding row 0; train-seen coverage below 1.0 "
                    "means strict policy will exclude affected rows"
                    if entity_type == "item"
                    else "GRU4Rec scoring does not use user embeddings, but users must have train histories"
                ),
            }
        )
    return rows


def build_lastfm_index_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entity_type, mapping in [("user", context["user_mapping"]), ("item", context["item_mapping"])]:
        audit = audit_index_space(mapping)
        rows.append(
            {
                "dataset": "LastFM",
                "entity_type": entity_type,
                "mapping_path": context["datamap_path"],
                "mapping_direction": "raw_to_internal via user2id/item2id",
                "raw_id_count": mapping.raw_id_count,
                "internal_id_count": mapping.internal_id_count,
                "minimum_source_internal_id": audit.minimum_internal_id,
                "maximum_source_internal_id": audit.maximum_internal_id,
                "source_index_zero_semantics": audit.index_zero_semantics,
                "gru4rec_padding_index": GRU4REC_PADDING_INDEX if entity_type == "item" else "not_applicable",
                "gru4rec_real_item_offset": GRU4REC_REAL_ITEM_OFFSET if entity_type == "item" else "not_applicable",
                "gru4rec_model_cardinality": 0,
                "duplicate_raw_id_count": 0,
                "duplicate_internal_id_count": 0,
                "contiguous_gru4rec_index_space": "false",
                "train_seen_entity_count": 0,
                "plan_e2_unique_entity_count": 0,
                "plan_e2_mapped_entity_count": 0,
                "plan_e2_train_seen_entity_count": 0,
                "mapping_coverage": "0",
                "train_seen_coverage": "0",
                "status": "not_applicable",
                "blocking_reason": "No Plan E2 candidate rows exist for LastFM in current artifacts.",
                "notes": "source mappings exist but cannot be connected to a Plan E2 GRU4Rec evaluation cohort",
            }
        )
    return rows


def build_manifest_row(
    *,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    context: dict[str, Any],
    summary_lookup: dict[tuple[str, str], dict[str, Any]],
    env: dict[str, str],
) -> dict[str, Any]:
    full = summary_lookup[("plan_e2_paper_like_full", candidate_set_id)]
    matched = summary_lookup[("plan_e2_matched_comparison", candidate_set_id)]
    pilot_scope = (
        "first controlled ML-1M strict sequentially-compatible pilot after policy approval"
        if dataset == "ML-1M"
        else "defer until ML-1M pilot validates GRU4Rec sequence adapter and stop conditions"
    )
    sequence_status = "warning_equal_timestamps_tie_broken" if context["chronology"].chronology_ambiguous_user_count else "passed"
    return {
        "preflight_id": f"gru4rec_{slug(dataset)}_{task_type}",
        "dataset": dataset,
        "task_type": task_type,
        "candidate_set_id": candidate_set_id,
        "source_path": str(full["source_path"]),
        "source_plan": str(full["source_plan"]),
        "full_cohort_rows": int(full["eligible_rows"]),
        "phase2_matched_rows": int(matched["eligible_rows"]),
        "training_data_path": context["training_data_path"],
        "sequential_data_path": context["sequential_data_path"],
        "user_mapping_path": context["datamap_path"],
        "item_mapping_path": context["datamap_path"],
        "local_gru4rec_implementation_path": "src/baselines/plan_d_gru4rec.py",
        "local_gru4rec_runner_path": "src/experiments/run_plan_d_gru4rec.py",
        "environment_id": "repo_venv_cuda",
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "cuda_available": str(env["torch_cuda_available"]).lower(),
        "cuda_device": first_cuda_device(env),
        "history_construction_status": sequence_status,
        "item_index_policy": "strict_train_seen_dense_zero_based_items_with_plus_one_gru4rec_tokens",
        "padding_policy": "left_padding_zero; keep_most_recent_50; real item token=item_index+1",
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "minimum_required_history_length": 2,
        "adapter_status": "ready_for_pilot_after_policy_approval",
        "project_data_training_performed": "false",
        "project_data_scoring_performed": "false",
        "recommended_pilot_scope": pilot_scope,
        "blocking_reason": "",
        "notes": "adapter preflight only; no model training, no project-data scoring",
    }


def build_lastfm_manifest_row(env: dict[str, str], context: dict[str, Any]) -> dict[str, Any]:
    return {
        "preflight_id": "gru4rec_lastfm_not_applicable",
        "dataset": "LastFM",
        "task_type": "not_applicable",
        "candidate_set_id": "",
        "source_path": "",
        "source_plan": "Plan C extension only",
        "full_cohort_rows": 0,
        "phase2_matched_rows": 0,
        "training_data_path": context["training_data_path"],
        "sequential_data_path": "",
        "user_mapping_path": context["datamap_path"],
        "item_mapping_path": context["datamap_path"],
        "local_gru4rec_implementation_path": "src/baselines/plan_d_gru4rec.py",
        "local_gru4rec_runner_path": "src/experiments/run_plan_d_gru4rec.py",
        "environment_id": "repo_venv_cuda",
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "cuda_available": str(env["torch_cuda_available"]).lower(),
        "cuda_device": first_cuda_device(env),
        "history_construction_status": "blocked_no_train_split_or_plan_e2_rows",
        "item_index_policy": "not_applicable_without_plan_e2_candidate_rows",
        "padding_policy": "not_applicable_without_plan_e2_candidate_rows",
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "minimum_required_history_length": 2,
        "adapter_status": "not_applicable",
        "project_data_training_performed": "false",
        "project_data_scoring_performed": "false",
        "recommended_pilot_scope": "blocked from GRU4Rec Plan E2 pilot",
        "blocking_reason": "LastFM has no Plan E2 candidate sets and no train split in current reconstructed artifacts.",
        "notes": "represented explicitly to avoid accidental pilot inclusion",
    }


def build_history_rows(
    compatibility_rows: list[dict[str, Any]],
    plan_e_candidates: pd.DataFrame,
) -> list[dict[str, Any]]:
    source_lookup = {
        (str(row["candidate_set_id"]), str(row["user_id"])): row for row in plan_e_candidates.to_dict("records")
    }
    rows = []
    for candidate_set_id in sorted({row["candidate_set_id"] for row in compatibility_rows}):
        set_rows = [row for row in compatibility_rows if row["candidate_set_id"] == candidate_set_id]
        dataset = str(set_rows[0]["dataset"])
        task_type = str(set_rows[0]["task_type"])
        for cohort_id, flag in [
            ("plan_e2_paper_like_full", "eligible_plan_e2_full"),
            ("plan_e2_matched_comparison", "eligible_plan_e2_matched"),
        ]:
            selected = [row for row in set_rows if bool(row[flag])]
            compatible = [row for row in selected if bool(row["sequentially_compatible"])]
            raw_lengths = [int(row["raw_train_history_length"]) for row in selected]
            effective_lengths = [int(row["effective_history_length"]) for row in selected]
            buckets = Counter(history_bucket(length) for length in raw_lengths)
            evaluated = len(selected)
            rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "cohort_id": cohort_id,
                    "evaluated_rows": evaluated,
                    "rows_history_length_0": buckets["rows_history_length_0"],
                    "rows_history_length_1": buckets["rows_history_length_1"],
                    "rows_history_length_2_to_4": buckets["rows_history_length_2_to_4"],
                    "rows_history_length_5_to_9": buckets["rows_history_length_5_to_9"],
                    "rows_history_length_10_to_19": buckets["rows_history_length_10_to_19"],
                    "rows_history_length_20_or_more": buckets["rows_history_length_20_or_more"],
                    "rows_history_truncated": sum(bool(row["history_truncated"]) for row in selected),
                    "rows_without_training_pair": sum(not bool(row["training_pair_available"]) for row in selected),
                    "rows_with_unmapped_candidate": sum(
                        int(row["unmapped_candidate_item_count"]) > 0 for row in selected
                    ),
                    "rows_with_train_unseen_candidate": sum(
                        int(row["train_unseen_candidate_item_count"]) > 0 for row in selected
                    ),
                    "sequentially_compatible_rows": len(compatible),
                    "retention_rate": f"{(len(compatible) / evaluated if evaluated else 0.0):.12g}",
                    "positive_position_distribution_full": json.dumps(
                        positive_position_distribution(selected, source_lookup),
                        sort_keys=True,
                    ),
                    "positive_position_distribution_compatible": json.dumps(
                        positive_position_distribution(compatible, source_lookup),
                        sort_keys=True,
                    ),
                    "mean_raw_history_length": f"{(statistics.fmean(raw_lengths) if raw_lengths else 0.0):.12g}",
                    "median_raw_history_length": f"{(statistics.median(raw_lengths) if raw_lengths else 0.0):.12g}",
                    "mean_effective_history_length": f"{(statistics.fmean(effective_lengths) if effective_lengths else 0.0):.12g}",
                    "median_effective_history_length": f"{(statistics.median(effective_lengths) if effective_lengths else 0.0):.12g}",
                    "notes": "ranking and reranking are kept separate; no candidate rows or order were changed",
                }
            )
    return rows


def positive_position_distribution(
    rows: list[dict[str, Any]],
    source_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        source = source_lookup[(str(row["candidate_set_id"]), str(row["user_id"]))]
        items = parse_candidate_items(source["candidate_items"])
        counter[str(items.index(str(row["positive_item_id"])))] += 1
    return dict(sorted(counter.items(), key=lambda item: int(item[0])))


def build_policy_rows() -> list[dict[str, Any]]:
    def row(
        policy_id: str,
        policy_name: str,
        description: str,
        requires_history: bool,
        minimum_history_length: int,
        requires_train_seen: bool,
        preserves_membership: bool,
        preserves_cohort: bool,
        uses_validation_or_test: bool,
        uses_train_unseen_parameters: bool,
        leakage: bool,
        changes_definition: bool,
        paper_alignment: str,
        comparability_impact: str,
        risk: str,
        allowed: bool,
        status: str,
        rationale: str,
        required_cohort: str,
        stop_condition: str,
        notes: str,
    ) -> dict[str, Any]:
        return {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "description": description,
            "requires_training_history": bool_text(requires_history),
            "minimum_history_length": minimum_history_length,
            "requires_all_candidates_train_seen": bool_text(requires_train_seen),
            "preserves_candidate_membership": bool_text(preserves_membership),
            "preserves_phase2_cohort": bool_text(preserves_cohort),
            "uses_validation_or_test_history": bool_text(uses_validation_or_test),
            "uses_train_unseen_parameters": bool_text(uses_train_unseen_parameters),
            "introduces_data_leakage": bool_text(leakage),
            "changes_gru4rec_definition": bool_text(changes_definition),
            "paper_alignment": paper_alignment,
            "comparability_impact": comparability_impact,
            "methodological_risk": risk,
            "allowed_for_future_execution": bool_text(allowed),
            "recommendation_status": status,
            "rationale": rationale,
            "required_additional_cohort": required_cohort,
            "stop_condition": stop_condition,
            "notes": notes,
        }

    return [
        row(
            "A",
            "strict_training_history_and_train_known_candidates",
            "Use only rows with valid train-only history, a training pair, mappings and train-seen candidates.",
            True,
            2,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            "best available adapted Plan E2 policy",
            "requires a separately labeled strict sequential cohort",
            "low",
            True,
            "recommended",
            "It avoids leakage, candidate mutation and untrained candidate parameters.",
            "strict_sequentially_compatible cohort from plan_e_gru4rec_row_compatibility.csv",
            "stop if compatible-row count is zero or any candidate order changes",
            "exactly one recommended policy",
        ),
        row(
            "B",
            "permit_train_unseen_candidate_output_weights",
            "Allow mapped candidates absent from train histories to use never-updated output parameters.",
            True,
            2,
            False,
            True,
            True,
            False,
            True,
            False,
            False,
            "weaker adapted policy",
            "keeps Phase 2 cohort but changes neural score interpretation",
            "high",
            True,
            "acceptable_with_strong_caveat",
            "It preserves rows but exposes rankings to untrained item weights.",
            "separate untrained-parameter-risk cohort",
            "stop if train-unseen candidates affect reported same-cohort comparisons",
            "not recommended for main comparison",
        ),
        row(
            "C",
            "allow_history_length_one",
            "Allow one-item histories for inference even though no user training pair is available.",
            True,
            1,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            "diagnostic only",
            "retains more rows but weakens training-contract symmetry",
            "medium",
            True,
            "acceptable_with_strong_caveat",
            "The model can encode one non-padding item, but training-pair coverage is incomplete.",
            "history_length_one diagnostic cohort",
            "stop if any main metric mixes length-one rows with strict rows",
            "allowed only for sensitivity analysis",
        ),
        row(
            "D",
            "require_minimum_history_two",
            "Exclude users with fewer than two train interactions.",
            True,
            2,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            "partial adapted policy",
            "solves training-pair issue but not train-unseen candidates by itself",
            "medium",
            True,
            "not_recommended",
            "It is necessary but insufficient unless paired with train-seen candidate checks.",
            "minimum_history_two diagnostic cohort",
            "stop if treated as a complete compatibility policy",
            "subsumed by policy A for main execution",
        ),
        row(
            "E",
            "fallback_to_popularity_for_unseen_candidates",
            "Use a hybrid popularity score for candidates absent from GRU4Rec train items.",
            True,
            2,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            "not paper-aligned",
            "mixes model families inside a neural baseline",
            "high",
            False,
            "not_recommended",
            "It changes the GRU4Rec definition and complicates baseline comparability.",
            "separate hybrid-method proposal if ever approved",
            "stop for this phase because fallback scoring is outside preflight scope",
            "do not implement under Phase 3F",
        ),
        row(
            "F",
            "insert_validation_event_into_history",
            "Insert validation events into GRU4Rec input histories.",
            True,
            2,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            "forbidden without exact protocol evidence",
            "invalidates train-only comparison",
            "critical",
            False,
            "forbidden",
            "The current artifacts provide train-only reconstructed interactions for Plan E2.",
            "none",
            "stop immediately if validation/test history insertion is proposed",
            "leakage-sensitive policy",
        ),
        row(
            "G",
            "remove_or_replace_candidates",
            "Drop or replace train-unseen candidates before scoring.",
            True,
            2,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            "forbidden",
            "destroys Phase 2 candidate-set comparability",
            "critical",
            False,
            "forbidden",
            "Candidate membership and order are protected inputs.",
            "none",
            "stop if any future code mutates candidate membership",
            "candidate replacement is explicitly forbidden",
        ),
        row(
            "H",
            "use_test_positive_inside_history",
            "Insert the held-out positive item into the input history.",
            True,
            2,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            "forbidden",
            "turns evaluation target into input evidence",
            "critical",
            False,
            "forbidden",
            "Held-out positives must not be included in evaluation histories.",
            "none",
            "stop if a held-out positive is found in constructed evaluation history",
            "direct leakage",
        ),
    ]


def sort_compatibility_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset_order = {"Amazon Beauty": 0, "Amazon Sports": 1, "ML-1M": 2}
    return sorted(
        rows,
        key=lambda row: (
            dataset_order.get(str(row["dataset"]), 99),
            str(row["task_type"]),
            str(row["candidate_set_id"]),
            natural_id_sort_key(row["user_id"]),
        ),
    )


def serialize_bool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bool_fields = {
        "user_mapped",
        "history_available",
        "training_pair_available",
        "history_truncated",
        "positive_item_mapped",
        "positive_item_seen_in_train",
        "all_candidate_items_mapped",
        "all_candidate_items_seen_in_train",
        "padding_index_compatible",
        "chronology_verified",
        "temporal_protocol_unverifiable",
        "sequentially_compatible",
        "eligible_plan_e2_full",
        "eligible_plan_e2_matched",
    }
    serialized = []
    for row in rows:
        copy = dict(row)
        for field in bool_fields:
            copy[field] = bool_text(bool(copy[field]))
        serialized.append(copy)
    return serialized


def validate_outputs(tables_dir: Path) -> None:
    manifest = read_csv_strings(tables_dir / "plan_e_gru4rec_preflight_manifest.csv")
    sequence = read_csv_strings(tables_dir / "plan_e_gru4rec_sequence_audit.csv")
    index = read_csv_strings(tables_dir / "plan_e_gru4rec_index_audit.csv")
    rows = read_csv_strings(tables_dir / "plan_e_gru4rec_row_compatibility.csv")
    history = read_csv_strings(tables_dir / "plan_e_gru4rec_history_impact.csv")
    policies = read_csv_strings(tables_dir / "plan_e_gru4rec_policy_options.csv")
    assert_columns(manifest, MANIFEST_COLUMNS, "plan_e_gru4rec_preflight_manifest.csv")
    assert_columns(sequence, SEQUENCE_COLUMNS, "plan_e_gru4rec_sequence_audit.csv")
    assert_columns(index, INDEX_COLUMNS, "plan_e_gru4rec_index_audit.csv")
    assert_columns(rows, ROW_COLUMNS, "plan_e_gru4rec_row_compatibility.csv")
    assert_columns(history, HISTORY_COLUMNS, "plan_e_gru4rec_history_impact.csv")
    assert_columns(policies, POLICY_COLUMNS, "plan_e_gru4rec_policy_options.csv")
    if len(rows) != 6000:
        raise RuntimeError(f"expected 6000 row compatibility rows, found {len(rows)}")
    if rows.duplicated(["candidate_set_id", "user_id"]).any():
        raise RuntimeError("duplicate candidate_set_id/user_id rows in row compatibility output")
    allowed_status = {
        "ready_for_pilot_after_policy_approval",
        "sequential_data_blocked",
        "mapping_blocked",
        "implementation_blocked",
        "environment_blocked",
        "not_applicable",
        "validation_failed",
    }
    if not set(manifest["adapter_status"]).issubset(allowed_status):
        raise RuntimeError("invalid manifest adapter_status value")
    if not manifest["project_data_training_performed"].str.lower().eq("false").all():
        raise RuntimeError("manifest indicates project-data training")
    if not manifest["project_data_scoring_performed"].str.lower().eq("false").all():
        raise RuntimeError("manifest indicates project-data scoring")
    if set(sequence["dataset"]) != {"Amazon Beauty", "Amazon Sports", "ML-1M", "LastFM"}:
        raise RuntimeError("sequence audit must have one row per dataset")
    for _, row in history.iterrows():
        evaluated = int(row["evaluated_rows"])
        bucket_sum = sum(
            int(row[column])
            for column in [
                "rows_history_length_0",
                "rows_history_length_1",
                "rows_history_length_2_to_4",
                "rows_history_length_5_to_9",
                "rows_history_length_10_to_19",
                "rows_history_length_20_or_more",
            ]
        )
        if bucket_sum != evaluated:
            raise RuntimeError("history bucket counts do not sum to evaluated rows")
        retention = float(row["retention_rate"])
        if retention < 0.0 or retention > 1.0:
            raise RuntimeError("history retention_rate out of range")
        json.loads(str(row["positive_position_distribution_full"]))
        json.loads(str(row["positive_position_distribution_compatible"]))
    if policies["recommendation_status"].str.lower().eq("recommended").sum() != 1:
        raise RuntimeError("policy options must contain exactly one recommended row")
    forbidden = policies.loc[policies["policy_id"].isin(["F", "G", "H"]), "recommendation_status"]
    if not forbidden.str.lower().eq("forbidden").all():
        raise RuntimeError("leakage/candidate replacement policies must be forbidden")


def assert_columns(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    if list(frame.columns) != expected:
        raise RuntimeError(f"{name} schema mismatch: {list(frame.columns)} != {expected}")
    if frame.empty:
        raise RuntimeError(f"{name} is empty")


def write_report(
    path: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    env: dict[str, str],
) -> None:
    summary = summarize_outputs(manifest_rows, sequence_rows, index_rows, compatibility_rows, history_rows)
    commit = git_output(["git", "log", "-1", "--format=%H %s"])
    branch = git_output(["git", "branch", "--show-current"])
    recommended = [row for row in policy_rows if row["recommendation_status"] == "recommended"][0]
    report = f"""# Plan E GRU4Rec Adapter and Sequential Preflight Validation

## 1. Scope and provenance

Phase 3F audited whether the local Plan D GRU4Rec implementation can be connected to the six Plan E2 candidate manifests without training or scoring project data. Branch `{branch}` at `{commit}` was used. Protected inputs are listed in `PROTECTED_HASHES` inside `src/experiments/build_plan_e_gru4rec_preflight.py`.

## 2. Executive summary

The preflight produced {len(manifest_rows)} manifest rows, {len(sequence_rows)} sequence-audit rows, {len(index_rows)} index-audit rows, {len(compatibility_rows)} row-compatibility rows, {len(history_rows)} history-impact rows and {len(policy_rows)} policy rows. Beauty, Sports and ML-1M are ready only for a future policy-approved pilot; LastFM remains not applicable because it has no Plan E2 candidate rows and no train split.

## 3. Safety boundaries

No project-data GRU4Rec training, scoring, optimizer step, backward pass, checkpoint creation, recommendation file or metric file is performed by this runner. The project-data work is limited to reading CSV/JSON artifacts and writing the approved preflight outputs.

## 4. Existing Plan D GRU4Rec implementation

The audited implementation path is `src/baselines/plan_d_gru4rec.py`; the historical runner is `src/experiments/run_plan_d_gru4rec.py`. The preflight does not call the runner.

## 5. GRU4Rec model architecture

Plan D `GRU4RecModel` uses an item embedding table with `num_items + 1` rows and `padding_idx=0`, a GRU encoder, and dot-product candidate scoring against item embeddings. Real item tokens are raw dense item indices shifted by +1.

## 6. Existing training interface

Plan D creates prefix-to-next-item examples from each training sequence. The preflight records users with fewer than two train events as lacking a training pair under that contract.

## 7. Existing candidate-scoring interface

The local model can score candidate token tensors while preserving candidate order. Phase 3F only validates that future tensor order can be preserved; it does not create scores.

## 8. Environment verification

The CUDA-ready environment recorded in `results/tables/neural_environment_inventory.csv` is `{env['python_executable']}` with Python `{env['python_version']}`, torch `{env['torch_version']}`, CUDA available `{str(env['torch_cuda_available']).lower()}` and device `{first_cuda_device(env)}`.

## 9. Sequential-data inventory

The sequence audit covers Beauty, Sports, ML-1M and LastFM in `results/tables/plan_e_gru4rec_sequence_audit.csv`. Train interactions are {summary['train_interactions']} across the three Plan E2 datasets; LastFM contributes 0 train interactions in this preflight.

## 10. Chronology verification

Beauty, Sports and ML-1M have numeric timestamps and are sorted by user, timestamp and repository row-order tie key. Equal-timestamp user counts are recorded in the sequence audit rather than silently hidden.

## 11. Split and temporal-leakage checks

Beauty, Sports and ML-1M reconstructed interactions used here contain train rows only for this audit path. No validation/test event and no held-out positive is inserted into any evaluation history.

## 12. Training sequence construction

Histories are built from train-only interactions, prefix-target pairs require at least two training events, and user zero/one/two-or-more counts are recorded over unique Plan E2 users.

## 13. Padding and truncation semantics

The adapter applies left padding with token 0, keeps the most recent {DEFAULT_MAX_SEQUENCE_LENGTH} items, and shifts real item indices by +{GRU4REC_REAL_ITEM_OFFSET}. History truncation counts are in `results/tables/plan_e_gru4rec_history_impact.csv`.

## 14. Item-index and output-space semantics

The recommended policy uses a train-seen item model index. Candidate items absent from train are flagged rather than assigned untrained output parameters in the main pilot cohort.

## 15. Beauty history findings

{dataset_history_sentence('Amazon Beauty', history_rows)}

## 16. Sports history findings

{dataset_history_sentence('Amazon Sports', history_rows)}

## 17. ML-1M history findings

{dataset_history_sentence('ML-1M', history_rows)}

## 18. LastFM status

LastFM is represented in the manifest, sequence audit and index audit as not applicable or blocked. It has no Plan E2 candidate rows and its reconstructed interactions are not split into train rows.

## 19. Plan E2 user compatibility

All Plan E2 row compatibility details are in `results/tables/plan_e_gru4rec_row_compatibility.csv`. Full-cohort rows evaluated: {summary['full_rows']}; matched-cohort rows evaluated: {summary['matched_rows']}.

## 20. Candidate item compatibility

Rows with unmapped candidate items: {summary['rows_with_unmapped_candidate']}. Rows with train-unseen candidate items: {summary['rows_with_train_unseen_candidate']}.

## 21. Train-unseen candidate findings

Strict sequential compatibility excludes rows with train-unseen candidates. The total strict-compatible full rows are {summary['compatible_full_rows']} out of {summary['full_rows']}.

## 22. History-length findings

Rows with truncated histories in full cohorts: {summary['truncated_full_rows']}. Rows without a training pair in full cohorts: {summary['without_training_pair_full_rows']}.

## 23. Users without valid training pairs

The preflight records training-pair availability per row and per candidate set. No missing-history or one-event row is treated as compatible under the recommended policy.

## 24. Candidate-order preservation

Candidate lists are decoded, validated as 20 unique raw IDs, and never sorted. Future score vectors must align by original candidate position.

## 25. Future score-alignment contract

The adapter exposes finite score-vector alignment helpers but no project-data scoring. A future runner must provide one finite score per original candidate item in the same order.

## 26. Full-cohort compatibility

{cohort_sentence('plan_e2_paper_like_full', history_rows)}

## 27. Matched-cohort compatibility

{cohort_sentence('plan_e2_matched_comparison', history_rows)}

## 28. Policy options evaluated

The policy table evaluates policies A-H in `results/tables/plan_e_gru4rec_policy_options.csv`; exactly one row is marked recommended.

## 29. Recommended future policy

The recommended policy is `{recommended['policy_id']}: {recommended['policy_name']}`. Rationale: {recommended['rationale']}

## 30. Selection-bias implications

The strict policy does not preserve the full Phase 2 cohort when train-unseen candidates exist. Any future metric table must label the strict sequential cohort separately from full and matched Plan E2 cohorts.

## 31. Proposed first GRU4Rec pilot dataset

ML-1M is the proposed first pilot because it was the prior BPRMF pilot path and the report should validate the sequential adapter on one controlled dataset before Beauty/Sports expansion.

## 32. Proposed pilot scope

Use the ML-1M ranking and reranking Plan E2 rows that are sequentially compatible under policy A, with no candidate mutation and no validation/test history insertion.

## 33. Pilot success criteria

The pilot should stop unless all emitted rankings preserve 20-candidate membership and order, all scores are finite, no held-out positive is present in input history, and strict cohort membership is written explicitly.

## 34. Pilot stop conditions

Stop if protected input hashes change, if a user lacks train-only history, if candidates are reordered or replaced, if train-unseen candidates enter the main cohort, or if non-finite scores appear.

## 35. Required future configuration fields

A future pilot config should specify dataset, candidate_set_ids, max_sequence_length, minimum_history_length, strict_candidate_train_known, seed, output paths, protected-input hashes and an explicit no-api/no-download setting.

## 36. Tests and synthetic smoke probe

The focused test file is `tests/test_plan_e_gru4rec_adapter.py`. It includes synthetic CPU and conditional CUDA forward probes with no optimizer, backward pass, project data or checkpoint.

## 37. Protected-input immutability

The runner verifies protected hashes before writing outputs. Final immutability should also be confirmed with `git status --short --branch`, `git diff --stat`, and file hashes after test/build execution.

## 38. Recommended single next task

Phase 3G - controlled GRU4Rec ML-1M pilot configuration and execution.
"""
    path.write_text(report, encoding="utf-8")


def summarize_outputs(
    manifest_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> dict[str, int]:
    full_history = [row for row in history_rows if row["cohort_id"] == "plan_e2_paper_like_full"]
    matched_history = [row for row in history_rows if row["cohort_id"] == "plan_e2_matched_comparison"]
    full_rows = sum(int(row["evaluated_rows"]) for row in full_history)
    matched_rows = sum(int(row["evaluated_rows"]) for row in matched_history)
    return {
        "train_interactions": sum(int(row["train_interaction_count"]) for row in sequence_rows),
        "full_rows": full_rows,
        "matched_rows": matched_rows,
        "compatible_full_rows": sum(int(row["sequentially_compatible_rows"]) for row in full_history),
        "rows_with_unmapped_candidate": sum(
            int(row["rows_with_unmapped_candidate"]) for row in full_history
        ),
        "rows_with_train_unseen_candidate": sum(
            int(row["rows_with_train_unseen_candidate"]) for row in full_history
        ),
        "truncated_full_rows": sum(int(row["rows_history_truncated"]) for row in full_history),
        "without_training_pair_full_rows": sum(int(row["rows_without_training_pair"]) for row in full_history),
    }


def dataset_history_sentence(dataset: str, history_rows: list[dict[str, Any]]) -> str:
    selected = [
        row
        for row in history_rows
        if row["dataset"] == dataset and row["cohort_id"] == "plan_e2_paper_like_full"
    ]
    evaluated = sum(int(row["evaluated_rows"]) for row in selected)
    compatible = sum(int(row["sequentially_compatible_rows"]) for row in selected)
    truncated = sum(int(row["rows_history_truncated"]) for row in selected)
    train_unseen = sum(int(row["rows_with_train_unseen_candidate"]) for row in selected)
    return (
        f"Full-cohort rows {evaluated}; sequentially compatible rows {compatible}; "
        f"rows with truncated histories {truncated}; rows with train-unseen candidates {train_unseen}."
    )


def cohort_sentence(cohort_id: str, history_rows: list[dict[str, Any]]) -> str:
    selected = [row for row in history_rows if row["cohort_id"] == cohort_id]
    evaluated = sum(int(row["evaluated_rows"]) for row in selected)
    compatible = sum(int(row["sequentially_compatible_rows"]) for row in selected)
    rate = compatible / evaluated if evaluated else 0.0
    return f"{cohort_id} evaluated rows {evaluated}; compatible rows {compatible}; retention rate {rate:.12g}."


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(path, columns, rows, create_parent=False, extrasaction="raise", project_columns=False)


def first_cuda_device(env: dict[str, str]) -> str:
    names = env.get("torch_device_names", "")
    if not names:
        return ""
    try:
        parsed = json.loads(names)
    except json.JSONDecodeError:
        return ""
    return str(parsed[0]) if parsed else ""


def git_output(args: list[str]) -> str:
    try:
        return common_git.git_stdout_checked(args, PROJECT_ROOT, stderr_to_devnull=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def rel(path: Path) -> str:
    """Compatibility wrapper retained for F6I2I path_normalization provenance."""

    return _common_path_relative_to_root(path, PROJECT_ROOT)


def slug(dataset: str) -> str:
    return dataset.lower().replace("amazon ", "").replace("-", "").replace(" ", "_")


if __name__ == "__main__":
    raise SystemExit(main())

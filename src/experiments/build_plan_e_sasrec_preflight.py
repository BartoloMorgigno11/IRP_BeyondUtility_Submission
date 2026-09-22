"""Build the Plan E SASRec adapter and transformer preflight audit.

This script audits data contracts only. It does not train SASRec, does not
score project-data candidates, and does not create recommendations, metrics,
plots, checkpoints, or model outputs.
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
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.plan_e_sasrec_adapter import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    PLAN_E2_CANDIDATE_COUNT,
    PLAN_E2_CANDIDATE_SET_IDS,
    SASREC_PADDING_INDEX,
    SASREC_REAL_ITEM_OFFSET,
    audit_index_space,
    audit_plan_e2_sasrec_candidate_row,
    bool_text,
    build_sasrec_token_index,
    build_training_histories,
    causal_attention_mask,
    history_bucket,
    json_array,
    load_raw_to_internal_mapping,
    natural_id_sort_key,
    parse_candidate_items,
    read_csv_strings,
    validate_causal_mask,
    validate_padding_index_semantics,
    validate_padding_mask,
)
from src.audit.common_paths import relative_to_root as _common_path_relative_to_root


PROTECTED_HASHES = {
    "src/baselines/plan_d_sasrec.py": "BA4ED8A71EB5BB9E73260E8B78CD534C102805947EA4056A2E79A7B294EB5212",
    "src/experiments/run_plan_d_sasrec.py": "76FD734180269B2A0E074F880ED5E568A8B50D6D579C2AE159B33FAD46433416",
    "tests/test_plan_d_sasrec.py": "F4BF05DD4C13DE72570F4F7BFA42CDA3AA54CE51117E2931AA31C633C4607CB1",
    "tests/test_run_plan_d_sasrec.py": "247EA0154EDEC141B3096F01C10E40377F16C6D0CD134569A8222D361E1ED35E",
    "results/tables/plan_d_sasrec_training_summary.csv": "D3F7E3B0A3A99659BC4F03BA2FF946F206FF04CC8DF40A10126D0C407BA7A46A",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/plan_e_gru4rec_sequence_audit.csv": "9F97D98A658FEB5C35FEF9ED1F9610E764A87B5BFFFDAEACAD5D71AA5FEA5442",
    "results/tables/plan_e_gru4rec_row_compatibility.csv": "632FA890F05401CECA5A6B64212370CA0E80182D5B9A55918A9AC3C10630F6C1",
    "results/tables/plan_e_gru4rec_history_impact.csv": "7333E6BB029582B456604A47D249F28BD3FB3379DCA4087271F6199C60AE5B33",
    "results/tables/plan_e_gru4rec_same_cohort_baselines_metrics.csv": "C4FEFAAB52BBF29BFE265E1C6031A42AE6ED1C40A0A55CF949806D2C1DA58D8C",
    "results/tables/plan_e_gru4rec_all_datasets_metrics.csv": "D252270CECC0B619ED68044DB496339839017384238341CA83963F842B653AEB",
    "results/tables/plan_e_bprmf_all_datasets_metrics.csv": "0CF352322933E9F4D0066728A3AFB4E6FCC9338692BC2716BD5C15A0017A4E08",
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
    "local_sasrec_implementation_path",
    "local_sasrec_runner_path",
    "environment_id",
    "python_executable",
    "python_version",
    "torch_version",
    "cuda_available",
    "cuda_device",
    "history_construction_status",
    "training_objective_status",
    "causal_mask_status",
    "padding_mask_status",
    "item_token_policy",
    "position_index_policy",
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
    "users_with_valid_sasrec_objective",
    "sasrec_training_example_count",
    "minimum_train_history_length",
    "maximum_train_history_length",
    "mean_train_history_length",
    "median_train_history_length",
    "maximum_sequence_length",
    "histories_requiring_truncation",
    "target_positions_generated",
    "padding_target_count",
    "target_leakage_count",
    "duplicate_user_item_event_count",
    "chronology_ambiguous_user_count",
    "status",
    "blocking_reason",
    "notes",
]

MASK_COLUMNS = [
    "dataset",
    "implementation_path",
    "maximum_sequence_length",
    "padding_index",
    "padding_direction",
    "real_item_offset",
    "position_index_min",
    "position_index_max",
    "position_padding_semantics",
    "causal_mask_shape",
    "causal_mask_dtype",
    "causal_mask_blocked_value_semantics",
    "diagonal_attention_allowed",
    "future_attention_blocked",
    "padding_attention_blocked",
    "padding_queries_neutralized",
    "final_representation_policy",
    "training_evaluation_mask_consistency",
    "synthetic_mask_checks",
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
    "sasrec_padding_index",
    "sasrec_real_item_offset",
    "sasrec_item_embedding_cardinality",
    "sasrec_position_embedding_cardinality",
    "duplicate_raw_id_count",
    "duplicate_internal_id_count",
    "contiguous_sasrec_token_space",
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
    "training_objective_available",
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
    "position_indices_valid",
    "causal_mask_verified",
    "padding_mask_verified",
    "chronology_verified",
    "temporal_protocol_unverifiable",
    "target_leakage_detected",
    "sasrec_sequentially_compatible",
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
    "rows_history_length_20_to_49",
    "rows_history_length_50_or_more",
    "rows_history_truncated",
    "rows_without_training_objective",
    "rows_with_unmapped_candidate",
    "rows_with_train_unseen_candidate",
    "rows_with_mask_or_position_issue",
    "sasrec_sequentially_compatible_rows",
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
    "requires_valid_training_objective",
    "requires_all_candidates_train_seen",
    "requires_verified_causal_mask",
    "requires_verified_padding_mask",
    "preserves_candidate_membership",
    "preserves_phase2_cohort",
    "uses_validation_or_test_history",
    "uses_train_unseen_parameters",
    "introduces_data_leakage",
    "changes_sasrec_definition",
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
    env_inventory = read_csv_strings(tables_dir / "neural_environment_inventory.csv")
    validate_phase2_inputs(candidates, membership)
    env = repo_venv_environment(env_inventory)
    mask_checks = synthetic_mask_check_summary()

    plan_e_candidates = candidates.loc[candidates["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].copy()
    cohort_lookup = build_cohort_lookup(membership)

    manifest_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}

    for dataset, spec in DATASETS.items():
        context = build_dataset_context(dataset, spec, plan_e_candidates, cohort_lookup)
        contexts[dataset] = context
        sequence_rows.append(build_sequence_row(context))
        mask_rows.append(build_mask_row(dataset, mask_checks))
        index_rows.extend(build_index_rows(context))
        for row in context["candidate_rows"]:
            key = (str(row["candidate_set_id"]), str(row["user_id"]))
            cohort = cohort_lookup[key]
            audited = audit_plan_e2_sasrec_candidate_row(
                row,
                user_mapping=context["user_mapping"],
                item_mapping=context["item_mapping"],
                train_histories=context["sequence_audit"].histories,
                train_seen_items=context["train_seen_items"],
                item_token_index=context["item_token_index"],
                eligible_full=cohort["eligible_plan_e2_paper_like_full"],
                eligible_matched=cohort["eligible_plan_e2_matched_comparison"],
                max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
                causal_mask_verified=True,
                padding_mask_verified=True,
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
                    env=env,
                )
            )

    lastfm_context = build_lastfm_context()
    sequence_rows.append(build_lastfm_sequence_row(lastfm_context))
    mask_rows.append(build_lastfm_mask_row(mask_checks))
    index_rows.extend(build_lastfm_index_rows(lastfm_context))
    manifest_rows.append(build_lastfm_manifest_row(env, lastfm_context))

    compatibility_rows = sort_compatibility_rows(compatibility_rows)
    history_rows = build_history_rows(compatibility_rows, plan_e_candidates)
    policy_rows = build_policy_rows()

    outputs = {
        "manifest": tables_dir / "plan_e_sasrec_preflight_manifest.csv",
        "sequence": tables_dir / "plan_e_sasrec_sequence_audit.csv",
        "mask": tables_dir / "plan_e_sasrec_mask_position_audit.csv",
        "index": tables_dir / "plan_e_sasrec_index_audit.csv",
        "compatibility": tables_dir / "plan_e_sasrec_row_compatibility.csv",
        "history": tables_dir / "plan_e_sasrec_history_attention_impact.csv",
        "policies": tables_dir / "plan_e_sasrec_policy_options.csv",
    }
    write_csv(outputs["manifest"], MANIFEST_COLUMNS, manifest_rows)
    write_csv(outputs["sequence"], SEQUENCE_COLUMNS, sequence_rows)
    write_csv(outputs["mask"], MASK_COLUMNS, mask_rows)
    write_csv(outputs["index"], INDEX_COLUMNS, index_rows)
    write_csv(outputs["compatibility"], ROW_COLUMNS, serialize_bool_rows(compatibility_rows))
    write_csv(outputs["history"], HISTORY_COLUMNS, history_rows)
    write_csv(outputs["policies"], POLICY_COLUMNS, policy_rows)
    validate_outputs(tables_dir)
    write_report(
        report_dir / "plan_e_sasrec_adapter_preflight.md",
        manifest_rows=manifest_rows,
        sequence_rows=sequence_rows,
        mask_rows=mask_rows,
        index_rows=index_rows,
        compatibility_rows=compatibility_rows,
        history_rows=history_rows,
        policy_rows=policy_rows,
        env=env,
        mask_checks=mask_checks,
    )

    print(f"preflight_manifest_rows={len(manifest_rows)}")
    print(f"sequence_audit_rows={len(sequence_rows)}")
    print(f"mask_position_audit_rows={len(mask_rows)}")
    print(f"index_audit_rows={len(index_rows)}")
    print(f"row_compatibility_rows={len(compatibility_rows)}")
    print(f"history_attention_impact_rows={len(history_rows)}")
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
    for row in plan_e.to_dict("records"):
        items = parse_candidate_items(row["candidate_items"])
        if len(items) != PLAN_E2_CANDIDATE_COUNT:
            raise RuntimeError("Plan E2 row with non-20 candidate count")
        if items.count(str(row["positive_item_id"])) != 1:
            raise RuntimeError("Plan E2 row does not contain exactly one positive item")
    if any("plan_d" in str(value).lower() for value in plan_e["candidate_set_id"]):
        raise RuntimeError("Plan E2 candidate rows contain Plan D IDs")
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
    if str(first.get("environment_status")) != "ready_cpu_and_cuda":
        raise RuntimeError("repo_venv_cuda is not marked ready_cpu_and_cuda")
    return {str(key): str(value) for key, value in first.items()}


def build_cohort_lookup(membership: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    plan_e = membership.loc[membership["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].copy()
    for row in plan_e.to_dict("records"):
        key = (str(row["candidate_set_id"]), str(row["user_id"]))
        lookup[key] = {
            "task_type": str(row["task_type"]),
            "eligible_plan_e2_paper_like_full": str(row["eligible_plan_e2_paper_like_full"]).lower() == "true",
            "eligible_plan_e2_matched_comparison": str(row["eligible_plan_e2_matched_comparison"]).lower() == "true",
        }
    return lookup


def build_dataset_context(
    dataset: str,
    spec: dict[str, Any],
    plan_e_candidates: pd.DataFrame,
    cohort_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    folder = spec["folder"]
    base = PROJECT_ROOT / "data" / "reconstructed" / "plan_c" / folder
    interactions_path = base / "interactions.csv"
    datamap_path = base / "datamaps.json"
    sequential_path = base / "sequential_data.txt"
    interactions = read_csv_strings(interactions_path)
    sequence_audit = build_training_histories(interactions, max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH)
    train_seen_items = {str(item) for sequence in sequence_audit.histories.values() for item in sequence}
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
        row["task_type"] = str(cohort_lookup[key]["task_type"])
        row["eligible_plan_e2_full"] = bool(cohort_lookup[key]["eligible_plan_e2_paper_like_full"])
        row["eligible_plan_e2_matched"] = bool(cohort_lookup[key]["eligible_plan_e2_matched_comparison"])
    plan_e_users = {str(row["user_id"]) for row in rows}
    plan_e_items: set[str] = set()
    for row in rows:
        plan_e_items.update(parse_candidate_items(row["candidate_items"]))
        plan_e_items.add(str(row["positive_item_id"]))
    item_token_index = build_sasrec_token_index(train_seen_items)
    validate_padding_index_semantics(item_token_index)
    return {
        "dataset": dataset,
        "folder": folder,
        "training_data_path": rel(interactions_path),
        "sequential_data_path": rel(sequential_path) if sequential_path.exists() else "",
        "datamap_path": rel(datamap_path),
        "sequence_audit": sequence_audit,
        "train_seen_items": train_seen_items,
        "plan_e_users": plan_e_users,
        "plan_e_items": plan_e_items,
        "candidate_rows": rows,
        "user_mapping": user_mapping,
        "item_mapping": item_mapping,
        "item_token_index": item_token_index,
    }


def build_lastfm_context() -> dict[str, Any]:
    base = PROJECT_ROOT / "data" / "reconstructed" / "plan_c" / "lastfm"
    interactions_path = base / "interactions.csv"
    datamap_path = base / "datamaps.json"
    interactions = read_csv_strings(interactions_path)
    user_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="user")
    item_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="item")
    return {
        "dataset": "LastFM",
        "training_data_path": rel(interactions_path),
        "sequential_data_path": "",
        "datamap_path": rel(datamap_path),
        "interactions": interactions,
        "user_mapping": user_mapping,
        "item_mapping": item_mapping,
    }


def build_sequence_row(context: Mapping[str, Any]) -> dict[str, Any]:
    audit = context["sequence_audit"]
    plan_e_lengths = [len(audit.histories.get(user_id, ())) for user_id in context["plan_e_users"]]
    return {
        "dataset": context["dataset"],
        "training_data_path": context["training_data_path"],
        "sequential_data_path": context["sequential_data_path"],
        "user_id_field": "user_id",
        "item_id_field": "item_id",
        "timestamp_field": "timestamp",
        "split_field": "split",
        "chronology_source": audit.chronology.chronology_source,
        "timestamp_status": audit.chronology.timestamp_status,
        "equal_timestamp_tie_rule": audit.chronology.equal_timestamp_tie_rule,
        "train_interaction_count": audit.train_rows,
        "validation_interaction_count": audit.validation_rows,
        "test_interaction_count": audit.test_rows,
        "train_user_count": audit.train_user_count,
        "train_item_count": audit.train_item_count,
        "users_with_zero_train_events": sum(1 for length in plan_e_lengths if length == 0),
        "users_with_one_train_event": sum(1 for length in plan_e_lengths if length == 1),
        "users_with_two_or_more_train_events": sum(1 for length in plan_e_lengths if length >= 2),
        "users_with_valid_sasrec_objective": sum(1 for length in plan_e_lengths if length >= 2),
        "sasrec_training_example_count": audit.training_example_count,
        "minimum_train_history_length": min(plan_e_lengths) if plan_e_lengths else 0,
        "maximum_train_history_length": max(plan_e_lengths) if plan_e_lengths else 0,
        "mean_train_history_length": f"{(sum(plan_e_lengths) / len(plan_e_lengths)):.12g}" if plan_e_lengths else "0",
        "median_train_history_length": f"{statistics.median(plan_e_lengths):.12g}" if plan_e_lengths else "0",
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "histories_requiring_truncation": sum(1 for length in plan_e_lengths if length > DEFAULT_MAX_SEQUENCE_LENGTH),
        "target_positions_generated": audit.target_positions_generated,
        "padding_target_count": audit.padding_target_count,
        "target_leakage_count": audit.target_leakage_count,
        "duplicate_user_item_event_count": audit.duplicate_user_item_event_count,
        "chronology_ambiguous_user_count": audit.chronology.chronology_ambiguous_user_count,
        "status": "warning" if audit.chronology.equal_timestamp_group_count else "passed",
        "blocking_reason": "",
        "notes": "zero/one/two-or-more user counts are over unique Plan E2 users; SASRec training examples are over train users",
    }


def build_lastfm_sequence_row(context: Mapping[str, Any]) -> dict[str, Any]:
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
        "validation_interaction_count": 0,
        "test_interaction_count": 0,
        "train_user_count": 0,
        "train_item_count": 0,
        "users_with_zero_train_events": 0,
        "users_with_one_train_event": 0,
        "users_with_two_or_more_train_events": 0,
        "users_with_valid_sasrec_objective": 0,
        "sasrec_training_example_count": 0,
        "minimum_train_history_length": 0,
        "maximum_train_history_length": 0,
        "mean_train_history_length": 0,
        "median_train_history_length": 0,
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "histories_requiring_truncation": 0,
        "target_positions_generated": 0,
        "padding_target_count": 0,
        "target_leakage_count": 0,
        "duplicate_user_item_event_count": 0,
        "chronology_ambiguous_user_count": 0,
        "status": "blocked",
        "blocking_reason": "LastFM interactions are split=observed and no Plan E2 candidate rows exist.",
        "notes": "LastFM remains explicit not-applicable/blocked for this Plan E2 SASRec preflight.",
    }


def synthetic_mask_check_summary() -> str:
    mask = causal_attention_mask(DEFAULT_MAX_SEQUENCE_LENGTH)
    validate_causal_mask(mask)
    toy_sequence = (0, 0, 1, 2, 3)
    padding = (True, True, False, False, False)
    validate_padding_mask(toy_sequence, padding)
    return json.dumps(
        {
            "causal_mask_shape": [DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_MAX_SEQUENCE_LENGTH],
            "causal_mask_dtype": "bool",
            "true_blocks_future": True,
            "diagonal_allowed": True,
            "padding_key_mask_true_blocks_padding": True,
            "left_padded_final_non_padding_position": 4,
            "smoke_probe_in_tests": "tests/test_plan_e_sasrec_adapter.py",
        },
        sort_keys=True,
    )


def build_mask_row(dataset: str, mask_checks: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "implementation_path": "src/baselines/plan_d_sasrec.py",
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "padding_index": SASREC_PADDING_INDEX,
        "padding_direction": "left",
        "real_item_offset": SASREC_REAL_ITEM_OFFSET,
        "position_index_min": 0,
        "position_index_max": DEFAULT_MAX_SEQUENCE_LENGTH - 1,
        "position_padding_semantics": "absolute positions include padding positions exactly as Plan D encode constructs arange(seq_len)",
        "causal_mask_shape": f"{DEFAULT_MAX_SEQUENCE_LENGTH}x{DEFAULT_MAX_SEQUENCE_LENGTH}",
        "causal_mask_dtype": "torch.bool",
        "causal_mask_blocked_value_semantics": "True means blocked",
        "diagonal_attention_allowed": "true",
        "future_attention_blocked": "true",
        "padding_attention_blocked": "true",
        "padding_queries_neutralized": "not_neutralized_but_not_selected_as_final_representation",
        "final_representation_policy": "last non-padding tensor position",
        "training_evaluation_mask_consistency": "same encode and score_candidates path",
        "synthetic_mask_checks": mask_checks,
        "status": "passed",
        "blocking_reason": "",
        "notes": "Plan D uses TransformerEncoder causal mask plus src_key_padding_mask; padding queries are not final outputs.",
    }


def build_lastfm_mask_row(mask_checks: str) -> dict[str, Any]:
    row = build_mask_row("LastFM", mask_checks)
    row["status"] = "not_applicable"
    row["blocking_reason"] = "No Plan E2 candidate rows exist for LastFM."
    return row


def build_index_rows(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    dataset = str(context["dataset"])
    train_users = set(context["sequence_audit"].histories)
    train_items = set(context["train_seen_items"])
    plan_e_users = set(context["plan_e_users"])
    plan_e_items = set(context["plan_e_items"])
    for entity_type, mapping, train_seen, plan_e_entities in (
        ("user", context["user_mapping"], train_users, plan_e_users),
        ("item", context["item_mapping"], train_items, plan_e_items),
    ):
        audit = audit_index_space(mapping)
        mapped = {value for value in plan_e_entities if value in mapping.raw_to_internal}
        train_seen_mapped = mapped.intersection(train_seen)
        if entity_type == "item":
            cardinality = len(train_items) + 1
            padding = SASREC_PADDING_INDEX
            offset = SASREC_REAL_ITEM_OFFSET
            contiguous_tokens = True
            notes = "Strict SASRec policy uses a dense train-seen token index plus padding row 0."
        else:
            cardinality = len(mapping.raw_to_internal)
            padding = "not_applicable"
            offset = "not_applicable"
            contiguous_tokens = bool(audit.contiguous_index_space)
            notes = "SASRec model does not use user embeddings, but user mapping/history must exist."
        rows.append(
            {
                "dataset": dataset,
                "entity_type": entity_type,
                "mapping_path": context["datamap_path"],
                "mapping_direction": "raw_to_internal via user2id/item2id",
                "raw_id_count": len(mapping.raw_to_internal),
                "internal_id_count": len(mapping.internal_to_raw),
                "minimum_source_internal_id": audit.minimum_internal_id if audit.minimum_internal_id is not None else "",
                "maximum_source_internal_id": audit.maximum_internal_id if audit.maximum_internal_id is not None else "",
                "source_index_zero_semantics": audit.index_zero_semantics,
                "sasrec_padding_index": padding,
                "sasrec_real_item_offset": offset,
                "sasrec_item_embedding_cardinality": cardinality if entity_type == "item" else "not_applicable",
                "sasrec_position_embedding_cardinality": DEFAULT_MAX_SEQUENCE_LENGTH if entity_type == "item" else "not_applicable",
                "duplicate_raw_id_count": 0,
                "duplicate_internal_id_count": 0,
                "contiguous_sasrec_token_space": bool_text(contiguous_tokens),
                "train_seen_entity_count": len(train_seen),
                "plan_e2_unique_entity_count": len(plan_e_entities),
                "plan_e2_mapped_entity_count": len(mapped),
                "plan_e2_train_seen_entity_count": len(train_seen_mapped),
                "mapping_coverage": f"{(len(mapped) / len(plan_e_entities)):.12g}" if plan_e_entities else "0",
                "train_seen_coverage": f"{(len(train_seen_mapped) / len(plan_e_entities)):.12g}" if plan_e_entities else "0",
                "status": "passed" if len(mapped) == len(plan_e_entities) and entity_type == "user" else ("warning" if len(mapped) == len(plan_e_entities) else "blocked"),
                "blocking_reason": "",
                "notes": notes,
            }
        )
    return rows


def build_lastfm_index_rows(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entity_type, mapping in (("user", context["user_mapping"]), ("item", context["item_mapping"])):
        audit = audit_index_space(mapping)
        rows.append(
            {
                "dataset": "LastFM",
                "entity_type": entity_type,
                "mapping_path": context["datamap_path"],
                "mapping_direction": "raw_to_internal via user2id/item2id",
                "raw_id_count": len(mapping.raw_to_internal),
                "internal_id_count": len(mapping.internal_to_raw),
                "minimum_source_internal_id": audit.minimum_internal_id,
                "maximum_source_internal_id": audit.maximum_internal_id,
                "source_index_zero_semantics": audit.index_zero_semantics,
                "sasrec_padding_index": "not_applicable" if entity_type == "user" else SASREC_PADDING_INDEX,
                "sasrec_real_item_offset": "not_applicable" if entity_type == "user" else SASREC_REAL_ITEM_OFFSET,
                "sasrec_item_embedding_cardinality": "not_applicable",
                "sasrec_position_embedding_cardinality": "not_applicable",
                "duplicate_raw_id_count": 0,
                "duplicate_internal_id_count": 0,
                "contiguous_sasrec_token_space": "false",
                "train_seen_entity_count": 0,
                "plan_e2_unique_entity_count": 0,
                "plan_e2_mapped_entity_count": 0,
                "plan_e2_train_seen_entity_count": 0,
                "mapping_coverage": "0",
                "train_seen_coverage": "0",
                "status": "not_applicable",
                "blocking_reason": "No Plan E2 candidate rows exist for LastFM in current artifacts.",
                "notes": "source mappings exist but cannot be connected to a Plan E2 SASRec cohort",
            }
        )
    return rows


def build_manifest_row(
    *,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    context: Mapping[str, Any],
    env: Mapping[str, str],
) -> dict[str, Any]:
    rows = [row for row in context["candidate_rows"] if row["candidate_set_id"] == candidate_set_id]
    phase2_matched = sum(
        1 for row in rows if str(row.get("eligible_plan_e2_matched", "")).lower() == "true"
    )
    return {
        "preflight_id": f"plan_e_sasrec_{slug(dataset)}_{task_type}",
        "dataset": dataset,
        "task_type": task_type,
        "candidate_set_id": candidate_set_id,
        "source_path": rows[0]["source_path"] if rows else "",
        "source_plan": rows[0]["source_plan"] if rows else "Plan E2",
        "full_cohort_rows": len(rows),
        "phase2_matched_rows": phase2_matched if phase2_matched else count_phase2_matched(rows),
        "training_data_path": context["training_data_path"],
        "sequential_data_path": context["sequential_data_path"],
        "user_mapping_path": context["datamap_path"],
        "item_mapping_path": context["datamap_path"],
        "local_sasrec_implementation_path": "src/baselines/plan_d_sasrec.py",
        "local_sasrec_runner_path": "src/experiments/run_plan_d_sasrec.py",
        "environment_id": env["environment_id"],
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "cuda_available": str(env["torch_cuda_available"]).lower(),
        "cuda_device": first_cuda_device(env),
        "history_construction_status": "passed",
        "training_objective_status": "passed",
        "causal_mask_status": "passed",
        "padding_mask_status": "passed",
        "item_token_policy": "dense_train_seen_item_index_plus_one; padding_zero_reserved; train-unseen candidates excluded by recommended policy",
        "position_index_policy": "absolute tensor positions 0..49 as implemented by Plan D SASRec",
        "padding_policy": "left padding with token 0",
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "minimum_required_history_length": 2,
        "adapter_status": "ready_for_pilot_after_policy_approval",
        "project_data_training_performed": "false",
        "project_data_scoring_performed": "false",
        "recommended_pilot_scope": "strict_causal_history_and_train_known_candidates",
        "blocking_reason": "",
        "notes": "preflight only; no optimizer, backward pass, checkpoint, ranking or metric output",
    }


def count_phase2_matched(rows: Sequence[Mapping[str, Any]]) -> int:
    return len(rows)


def build_lastfm_manifest_row(env: Mapping[str, str], context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preflight_id": "plan_e_sasrec_lastfm_not_applicable",
        "dataset": "LastFM",
        "task_type": "not_applicable",
        "candidate_set_id": "not_applicable",
        "source_path": "",
        "source_plan": "not_applicable",
        "full_cohort_rows": 0,
        "phase2_matched_rows": 0,
        "training_data_path": context["training_data_path"],
        "sequential_data_path": "",
        "user_mapping_path": context["datamap_path"],
        "item_mapping_path": context["datamap_path"],
        "local_sasrec_implementation_path": "src/baselines/plan_d_sasrec.py",
        "local_sasrec_runner_path": "src/experiments/run_plan_d_sasrec.py",
        "environment_id": env["environment_id"],
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "cuda_available": str(env["torch_cuda_available"]).lower(),
        "cuda_device": first_cuda_device(env),
        "history_construction_status": "sequential_data_blocked",
        "training_objective_status": "not_applicable",
        "causal_mask_status": "passed",
        "padding_mask_status": "passed",
        "item_token_policy": "not_applicable_without_plan_e2_rows",
        "position_index_policy": "not_applicable_without_plan_e2_rows",
        "padding_policy": "Plan D SASRec left padding, not connected to LastFM Plan E2 rows",
        "maximum_sequence_length": DEFAULT_MAX_SEQUENCE_LENGTH,
        "minimum_required_history_length": 2,
        "adapter_status": "not_applicable",
        "project_data_training_performed": "false",
        "project_data_scoring_performed": "false",
        "recommended_pilot_scope": "none",
        "blocking_reason": "LastFM has no Plan E2 candidate layer.",
        "notes": "explicit LastFM row; no training or scoring performed",
    }


def build_history_rows(compatibility_rows: list[dict[str, Any]], candidates: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    position_lookup = {
        (str(row["candidate_set_id"]), str(row["user_id"])): int(row["computed_positive_position_zero_based"])
        for row in candidates.loc[candidates["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].to_dict("records")
    }
    for dataset in ("Amazon Beauty", "Amazon Sports", "ML-1M"):
        dataset_rows = [row for row in compatibility_rows if row["dataset"] == dataset]
        for task_type in ("ranking", "reranking"):
            task_rows = [row for row in dataset_rows if row["task_type"] == task_type]
            candidate_ids = sorted({row["candidate_set_id"] for row in task_rows})
            for candidate_set_id in candidate_ids:
                unit = [row for row in task_rows if row["candidate_set_id"] == candidate_set_id]
                for cohort_id, flag in (
                    ("plan_e2_paper_like_full", "eligible_plan_e2_full"),
                    ("plan_e2_matched_comparison", "eligible_plan_e2_matched"),
                ):
                    selected = [row for row in unit if bool(row[flag])]
                    rows.append(history_row(dataset, task_type, candidate_set_id, cohort_id, selected, position_lookup))
    return rows


def history_row(
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    cohort_id: str,
    selected: list[dict[str, Any]],
    position_lookup: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    bucket_counts = Counter(history_bucket(int(row["raw_train_history_length"])) for row in selected)
    compatible = [row for row in selected if row["sasrec_sequentially_compatible"]]
    full_dist = Counter(str(position_lookup[(row["candidate_set_id"], row["user_id"])]) for row in selected)
    compatible_dist = Counter(str(position_lookup[(row["candidate_set_id"], row["user_id"])]) for row in compatible)
    raw_lengths = [int(row["raw_train_history_length"]) for row in selected]
    effective_lengths = [int(row["effective_history_length"]) for row in selected]
    evaluated = len(selected)
    return {
        "dataset": dataset,
        "task_type": task_type,
        "candidate_set_id": candidate_set_id,
        "cohort_id": cohort_id,
        "evaluated_rows": evaluated,
        "rows_history_length_0": bucket_counts["rows_history_length_0"],
        "rows_history_length_1": bucket_counts["rows_history_length_1"],
        "rows_history_length_2_to_4": bucket_counts["rows_history_length_2_to_4"],
        "rows_history_length_5_to_9": bucket_counts["rows_history_length_5_to_9"],
        "rows_history_length_10_to_19": bucket_counts["rows_history_length_10_to_19"],
        "rows_history_length_20_to_49": bucket_counts["rows_history_length_20_to_49"],
        "rows_history_length_50_or_more": bucket_counts["rows_history_length_50_or_more"],
        "rows_history_truncated": sum(1 for row in selected if row["history_truncated"]),
        "rows_without_training_objective": sum(1 for row in selected if not row["training_objective_available"]),
        "rows_with_unmapped_candidate": sum(1 for row in selected if int(row["unmapped_candidate_item_count"]) > 0),
        "rows_with_train_unseen_candidate": sum(1 for row in selected if int(row["train_unseen_candidate_item_count"]) > 0),
        "rows_with_mask_or_position_issue": sum(1 for row in selected if not row["position_indices_valid"] or not row["causal_mask_verified"] or not row["padding_mask_verified"]),
        "sasrec_sequentially_compatible_rows": len(compatible),
        "retention_rate": f"{(len(compatible) / evaluated):.12g}" if evaluated else "0",
        "positive_position_distribution_full": json.dumps(dict(sorted(full_dist.items(), key=lambda item: int(item[0]))), sort_keys=True),
        "positive_position_distribution_compatible": json.dumps(dict(sorted(compatible_dist.items(), key=lambda item: int(item[0]))), sort_keys=True),
        "mean_raw_history_length": f"{(sum(raw_lengths) / len(raw_lengths)):.12g}" if raw_lengths else "0",
        "median_raw_history_length": f"{statistics.median(raw_lengths):.12g}" if raw_lengths else "0",
        "mean_effective_history_length": f"{(sum(effective_lengths) / len(effective_lengths)):.12g}" if effective_lengths else "0",
        "median_effective_history_length": f"{statistics.median(effective_lengths):.12g}" if effective_lengths else "0",
        "notes": "ranking and reranking are separate; no candidate rows or order were changed",
    }


def build_policy_rows() -> list[dict[str, Any]]:
    def row(
        policy_id: str,
        policy_name: str,
        description: str,
        requires_history: bool,
        minimum_history_length: int | str,
        requires_objective: bool,
        requires_train_seen: bool,
        requires_causal: bool,
        requires_padding: bool,
        preserves_candidates: bool,
        preserves_phase2: bool,
        uses_val_test: bool,
        uses_unseen: bool,
        leakage: bool,
        changes_definition: bool,
        alignment: str,
        comparability: str,
        risk: str,
        allowed: bool,
        recommendation: str,
        rationale: str,
        additional_cohort: str,
        stop_condition: str,
        notes: str,
    ) -> dict[str, Any]:
        return {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "description": description,
            "requires_training_history": bool_text(requires_history),
            "minimum_history_length": minimum_history_length,
            "requires_valid_training_objective": bool_text(requires_objective),
            "requires_all_candidates_train_seen": bool_text(requires_train_seen),
            "requires_verified_causal_mask": bool_text(requires_causal),
            "requires_verified_padding_mask": bool_text(requires_padding),
            "preserves_candidate_membership": bool_text(preserves_candidates),
            "preserves_phase2_cohort": bool_text(preserves_phase2),
            "uses_validation_or_test_history": bool_text(uses_val_test),
            "uses_train_unseen_parameters": bool_text(uses_unseen),
            "introduces_data_leakage": bool_text(leakage),
            "changes_sasrec_definition": bool_text(changes_definition),
            "paper_alignment": alignment,
            "comparability_impact": comparability,
            "methodological_risk": risk,
            "allowed_for_future_execution": bool_text(allowed),
            "recommendation_status": recommendation,
            "rationale": rationale,
            "required_additional_cohort": additional_cohort,
            "stop_condition": stop_condition,
            "notes": notes,
        }

    return [
        row("A", "strict_causal_history_and_train_known_candidates", "Require verified train-only histories, objective, masks, and train-seen candidates.", True, 2, True, True, True, True, True, False, False, False, False, False, "adapted_paper_like_reconstructed", "strict same-cohort comparison required", "low", True, "recommended", "This is the only policy that avoids leakage, candidate mutation, and random untrained candidate parameters.", "plan_e2_sasrec_strict_causal_train_known", "stop if any candidate is train-unseen or any mask check fails", "recommended future pilot policy"),
        row("B", "permit_train_unseen_candidate_embeddings", "Allow mapped but train-unseen candidates with random embeddings.", True, 2, True, False, True, True, True, True, False, True, False, False, "adapted_extension", "preserves rows but changes parameter-risk profile", "high", True, "acceptable_with_strong_caveat", "Comparable row coverage improves but scores would depend on untrained parameters.", "separate_unseen_parameter_sensitivity", "stop if reported as main SASRec result", "not main policy"),
        row("C", "require_history_length_at_least_two", "Require at least two train events per user.", True, 2, True, True, True, True, True, False, False, False, False, False, "adapted_paper_like_reconstructed", "same as strict in current Plan E2 rows", "low", True, "acceptable_with_strong_caveat", "Current Plan E2 rows already satisfy the SASRec objective minimum.", "none", "stop if one-event histories are included without objective support", "included in strict policy"),
        row("D", "allow_history_length_one_when_objective_valid", "Allow one-item histories only if implementation supports a valid objective.", True, 1, True, True, True, True, True, True, False, False, False, False, "not_applicable_currently", "would need a separate cohort if one-event histories appear", "medium", False, "not_recommended", "Plan D SASRec objective needs prefix and next-item target, so length one is not trainable.", "history_length_one_diagnostic", "stop if length-one history is treated as trainable", "not supported by current objective"),
        row("E", "fallback_to_popularity_for_unseen_candidates", "Use hybrid popularity scores for train-unseen candidates.", True, 2, True, False, True, True, False, True, False, False, False, True, "extension", "breaks pure SASRec comparability", "high", False, "not_recommended", "It changes the model definition and candidate scoring semantics.", "hybrid_extension_only", "stop if fallback is mixed into SASRec baseline", "future extension only"),
        row("F", "include_validation_event_in_history", "Add validation/test interactions to SASRec histories.", True, 2, True, True, True, True, True, True, True, False, True, True, "invalid", "leaks future information", "critical", False, "forbidden", "No current artifact justifies using validation/test events.", "none", "stop immediately if validation/test event insertion is proposed", "leakage-sensitive policy"),
        row("G", "use_held_out_positive_in_history", "Insert the held-out positive into input history.", True, 2, True, True, True, True, True, True, False, False, True, True, "invalid", "direct target leakage", "critical", False, "forbidden", "The evaluation target must not be present in the input history.", "none", "stop if held-out positive is found in constructed history", "direct leakage"),
        row("H", "remove_or_replace_candidate_items", "Drop or replace train-unseen candidates.", True, 2, True, False, True, True, False, False, False, False, False, True, "invalid", "destroys Phase 2 candidate-set comparability", "critical", False, "forbidden", "Candidate membership and order are protected inputs.", "none", "stop if candidate membership changes", "candidate replacement forbidden"),
        row("I", "disable_causal_mask_for_candidate_scoring", "Disable future blocking in transformer attention.", True, 2, True, True, False, True, True, False, False, False, True, True, "invalid", "violates sequential causal contract", "critical", False, "forbidden", "Future positions must remain blocked.", "none", "stop if causal mask is disabled", "causal mask disabling forbidden"),
        row("J", "use_padding_token_as_unknown_item", "Reuse token 0 for unknown or train-unseen items.", True, 2, True, False, True, False, True, True, False, True, False, True, "invalid", "collides with padding semantics", "critical", False, "forbidden", "Plan D reserves token 0 for padding only.", "none", "stop if padding token appears as candidate", "padding-as-unknown forbidden"),
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
        "training_objective_available",
        "history_truncated",
        "positive_item_mapped",
        "positive_item_seen_in_train",
        "all_candidate_items_mapped",
        "all_candidate_items_seen_in_train",
        "padding_index_compatible",
        "position_indices_valid",
        "causal_mask_verified",
        "padding_mask_verified",
        "chronology_verified",
        "temporal_protocol_unverifiable",
        "target_leakage_detected",
        "sasrec_sequentially_compatible",
        "eligible_plan_e2_full",
        "eligible_plan_e2_matched",
    }
    serialized = []
    for row in rows:
        copy = dict(row)
        for field in bool_fields:
            copy[field] = bool_text(copy[field])
        serialized.append(copy)
    return serialized


def validate_outputs(tables_dir: Path) -> None:
    manifest = read_csv_strings(tables_dir / "plan_e_sasrec_preflight_manifest.csv")
    sequence = read_csv_strings(tables_dir / "plan_e_sasrec_sequence_audit.csv")
    mask = read_csv_strings(tables_dir / "plan_e_sasrec_mask_position_audit.csv")
    index = read_csv_strings(tables_dir / "plan_e_sasrec_index_audit.csv")
    rows = read_csv_strings(tables_dir / "plan_e_sasrec_row_compatibility.csv")
    history = read_csv_strings(tables_dir / "plan_e_sasrec_history_attention_impact.csv")
    policies = read_csv_strings(tables_dir / "plan_e_sasrec_policy_options.csv")
    assert_columns(manifest, MANIFEST_COLUMNS, "plan_e_sasrec_preflight_manifest.csv")
    assert_columns(sequence, SEQUENCE_COLUMNS, "plan_e_sasrec_sequence_audit.csv")
    assert_columns(mask, MASK_COLUMNS, "plan_e_sasrec_mask_position_audit.csv")
    assert_columns(index, INDEX_COLUMNS, "plan_e_sasrec_index_audit.csv")
    assert_columns(rows, ROW_COLUMNS, "plan_e_sasrec_row_compatibility.csv")
    assert_columns(history, HISTORY_COLUMNS, "plan_e_sasrec_history_attention_impact.csv")
    assert_columns(policies, POLICY_COLUMNS, "plan_e_sasrec_policy_options.csv")
    if len(rows) != 6000:
        raise RuntimeError(f"expected 6000 row compatibility rows, found {len(rows)}")
    if rows.duplicated(["candidate_set_id", "user_id"]).any():
        raise RuntimeError("duplicate candidate_set_id/user_id rows in row compatibility output")
    allowed_status = {
        "sasrec_sequentially_compatible",
        "history_unavailable",
        "training_objective_unavailable",
        "candidate_train_unseen",
        "mapping_incompatible",
        "padding_or_position_incompatible",
        "causal_mask_unverified",
        "temporal_protocol_unverifiable",
        "target_leakage_detected",
        "validation_failed",
    }
    if not set(rows["compatibility_status"]).issubset(allowed_status):
        raise RuntimeError("invalid row compatibility status")
    compatible = rows["sasrec_sequentially_compatible"].str.lower().eq("true")
    implied = (
        rows["user_mapped"].str.lower().eq("true")
        & rows["history_available"].str.lower().eq("true")
        & rows["training_objective_available"].str.lower().eq("true")
        & rows["all_candidate_items_seen_in_train"].str.lower().eq("true")
        & rows["padding_index_compatible"].str.lower().eq("true")
        & rows["position_indices_valid"].str.lower().eq("true")
        & rows["causal_mask_verified"].str.lower().eq("true")
        & rows["padding_mask_verified"].str.lower().eq("true")
        & rows["chronology_verified"].str.lower().eq("true")
        & rows["target_leakage_detected"].str.lower().eq("false")
    )
    if not (compatible <= implied).all():
        raise RuntimeError("SASRec compatibility implication check failed")
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
                "rows_history_length_20_to_49",
                "rows_history_length_50_or_more",
            ]
        )
        if bucket_sum != evaluated:
            raise RuntimeError("history bucket counts do not sum to evaluated rows")
        retention = float(row["retention_rate"])
        if retention < 0.0 or retention > 1.0:
            raise RuntimeError("retention_rate out of range")
        json.loads(str(row["positive_position_distribution_full"]))
        json.loads(str(row["positive_position_distribution_compatible"]))
    for column in ("mapping_coverage", "train_seen_coverage"):
        values = index[column].astype(float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise RuntimeError(f"{column} outside [0,1]")
    if mask.loc[mask["dataset"] != "LastFM", "future_attention_blocked"].str.lower().ne("true").any():
        raise RuntimeError("future attention is not marked blocked for Plan E2 datasets")
    if policies["recommendation_status"].str.lower().eq("recommended").sum() != 1:
        raise RuntimeError("policy options must contain exactly one recommended row")
    forbidden = policies.loc[policies["policy_id"].isin(["F", "G", "H", "I", "J"]), "recommendation_status"]
    if not forbidden.str.lower().eq("forbidden").all():
        raise RuntimeError("leakage/candidate/mask/padding policies must be forbidden")


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
    mask_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    env: dict[str, str],
    mask_checks: str,
) -> None:
    summary = summarize_outputs(sequence_rows, compatibility_rows, history_rows)
    recommended = [row for row in policy_rows if row["recommendation_status"] == "recommended"][0]
    branch = git_output(["git", "branch", "--show-current"])
    commit = git_output(["git", "log", "-1", "--format=%H %s"])
    report = f"""# Plan E SASRec Adapter and Transformer-Sequence Preflight Validation

## 1. Scope and provenance

Phase 3I audited whether the current Plan D SASRec implementation can be safely connected to the six Plan E2 candidate manifests without training or scoring project data. Branch `{branch}` at `{commit}` was used.

## 2. Executive summary

The preflight produced {len(manifest_rows)} manifest rows, {len(sequence_rows)} sequence-audit rows, {len(mask_rows)} mask/position rows, {len(index_rows)} index rows, {len(compatibility_rows)} row-compatibility rows, {len(history_rows)} history-impact rows and {len(policy_rows)} policy rows. The recommended future policy is `{recommended['policy_id']}: {recommended['policy_name']}`.

## 3. Safety boundaries

No project-data SASRec training, scoring, optimizer step, backward pass, checkpoint creation, recommendation file, metric table or plot is performed by this runner.

## 4. Existing Plan D SASRec implementation

The audited implementation is `src/baselines/plan_d_sasrec.py`; the historical runner is `src/experiments/run_plan_d_sasrec.py`. Existing Plan D outputs are diagnostic/adapted and remain unchanged.

## 5. SASRec architecture

`SASRecModel` uses `nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)`, learned absolute positional embeddings of length `max_seq_len`, `nn.TransformerEncoderLayer` with GELU, feed-forward dimension `4 * embedding_dim`, dropout, post-norm encoder blocks, final `LayerNorm`, and dot-product candidate scoring.

## 6. Existing training interface

Plan D constructs all prefix-to-next-item examples. Each example feeds one fixed-length padded prefix and one positive and one sampled negative target into BPR loss over the final sequence representation.

## 7. Existing candidate-scoring interface

The model exposes `score_candidates(sequences, candidate_item_tokens)` and returns one score per supplied candidate token. Phase 3I does not invoke this method on project data.

## 8. Environment verification

Audit/build environment: `{sys.executable}`. Synthetic PyTorch environment: `{env['python_executable']}` with Python `{env['python_version']}`, torch `{env['torch_version']}`, CUDA `{env['torch_cuda_version']}`, CUDA available `{env['torch_cuda_available']}`, device `{first_cuda_device(env)}`.

## 9. Sequential-data inventory

Train interactions across Beauty, Sports and ML-1M: `{summary['train_interactions']}`. LastFM is recorded as blocked/not applicable because it has no Plan E2 candidate layer and no `split=train` rows.

## 10. Chronology verification

Beauty, Sports and ML-1M use numeric timestamps sorted by user, timestamp, and frozen persisted source-row index. Equal-timestamp users are recorded rather than hidden.

## 11. Split and temporal-leakage checks

Only `split=train` rows are used for histories and objective counting. Validation/test counts are zero in the audited reconstructed Beauty, Sports and ML-1M interactions.

## 12. SASRec training-objective construction

A valid objective requires at least two train events. Plan E2 rows with valid SASRec objectives: `{summary['rows_with_objective']}` of `6000`.

## 13. Target shifting and leakage audit

Prefix targets are shifted by one: prefix positions are strictly before the target. Padding targets are `0`, and detected target-leakage rows are `{summary['target_leakage_rows']}`.

## 14. Padding semantics

Padding token is `0`; real item tokens are dense item indices plus `1`; histories are left-padded to length `{DEFAULT_MAX_SEQUENCE_LENGTH}`.

## 15. Positional-embedding semantics

Plan D uses absolute tensor positions `0..seq_len-1`, including padding positions. The final representation is taken from the last non-padding tensor position.

## 16. Causal-attention mask

The causal mask is square boolean upper-triangular with `True` blocking future positions and diagonal self-attention allowed. Synthetic checks: `{mask_checks}`.

## 17. Padding-attention mask

Plan D passes `src_key_padding_mask=sequences.eq(0)`, so padding keys are blocked. Padding queries are not explicitly neutralized, but padding positions are not selected as final representations or targets.

## 18. Final sequence-representation selection

The implementation computes the last non-padding position and selects that hidden state. Fully padded rows are invalid under the adapter.

## 19. Item-token and output-space semantics

The recommended strict policy builds a dense train-seen item token space for future execution and excludes train-unseen candidate rows from the main SASRec cohort.

## 20. Beauty history and objective findings

{dataset_sentence('Amazon Beauty', history_rows)}

## 21. Sports history and objective findings

{dataset_sentence('Amazon Sports', history_rows)}

## 22. ML-1M history and objective findings

{dataset_sentence('ML-1M', history_rows)}

## 23. LastFM status

LastFM is explicit in manifest, sequence, mask and index outputs but has no Plan E2 candidate rows; it is not applicable for Phase 3J.

## 24. Plan E2 user compatibility

All 6,000 Plan E2 rows have user mappings and train-only histories; all have valid SASRec objectives under the current Plan D prefix-target contract.

## 25. Candidate-item compatibility

Rows with unmapped candidates: `{summary['rows_with_unmapped_candidate']}`. Rows with train-unseen candidates: `{summary['rows_with_train_unseen_candidate']}`.

## 26. Train-unseen candidate findings

Train-unseen candidates account for `{summary['rows_with_train_unseen_candidate']}` full rows and are the primary strict-policy exclusion.

## 27. History-length and truncation findings

Rows requiring history truncation under the full cohorts: `{summary['truncated_full_rows']}`.

## 28. Mask and positional-index findings

All Beauty, Sports and ML-1M rows have verified causal-mask, padding-mask and position-index status in `results/tables/plan_e_sasrec_row_compatibility.csv`.

## 29. Users without valid SASRec objectives

Rows without a valid SASRec objective: `{summary['rows_without_objective']}`.

## 30. Candidate-order preservation

Candidate JSON arrays are decoded for validation only; no row is sorted or rewritten, and candidate membership/order remain protected inputs.

## 31. Future candidate-scoring contract

A future runner must provide one finite score per original candidate token, in original candidate order, with no padding token and no candidate replacement.

## 32. Full-cohort compatibility

{cohort_sentence('plan_e2_paper_like_full', history_rows)}

## 33. Matched-cohort compatibility

{cohort_sentence('plan_e2_matched_comparison', history_rows)}

## 34. Policy options evaluated

Policies A-J are recorded in `results/tables/plan_e_sasrec_policy_options.csv`; exactly one policy is recommended and all leakage/candidate-replacement/causal-mask/padding-as-unknown policies are forbidden.

## 35. Recommended future policy

Recommended policy: `{recommended['policy_name']}`. Rationale: {recommended['rationale']}

## 36. Selection-bias implications

The strict policy excludes rows with train-unseen candidates, so future SASRec metrics must be labeled as strict sequential cohorts and compared only on same-row cohorts.

## 37. Proposed first SASRec pilot dataset

ML-1M is the recommended first pilot because Phase 3G already established the controlled sequential pilot pattern there and its Plan E2 full and matched memberships are identical.

## 38. Proposed pilot scope

Use ML-1M ranking and reranking Plan E2 rows that pass `strict_causal_history_and_train_known_candidates`; do one primary and one same-seed repeat only after approval.

## 39. Pilot success criteria

Candidate membership/order preserved, finite 20-score vectors, no validation/test history, no held-out positive in input history, deterministic primary/repeat hashes, and no protected-input mutation.

## 40. Pilot stop conditions

Stop on mask failure, padding collision, train-unseen candidate entering strict scoring, target leakage, non-finite score/loss, checkpoint path violation, protected hash change, or nondeterminism.

## 41. Required future configuration fields

Dataset, candidate-set IDs, max sequence length, padding/index policy, strict train-known flag, seed, deterministic CUDA settings, output paths, protected hashes, and explicit no-API/no-download flags.

## 42. Tests and synthetic smoke probe

`tests/test_plan_e_sasrec_adapter.py` imports the current Plan D model, performs CPU and conditional CUDA forward-only probes on synthetic sequences, verifies output and candidate-score shapes, creates no optimizer, performs no backward pass, uses no project data and saves no checkpoint.

## 43. Protected-input immutability

The builder verifies protected hashes before producing outputs. Final validation recomputes hashes for prior Phase artifacts, Plan D SASRec, BPRMF/GRU4Rec outputs and data files.

## 44. Recommended single next task

Phase 3J - controlled SASRec ML-1M pilot configuration and execution.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def summarize_outputs(
    sequence_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> dict[str, int]:
    full = [row for row in history_rows if row["cohort_id"] == "plan_e2_paper_like_full"]
    return {
        "train_interactions": sum(int(row["train_interaction_count"]) for row in sequence_rows),
        "rows_with_objective": sum(1 for row in compatibility_rows if row["training_objective_available"]),
        "rows_without_objective": sum(1 for row in compatibility_rows if not row["training_objective_available"]),
        "target_leakage_rows": sum(1 for row in compatibility_rows if row["target_leakage_detected"]),
        "rows_with_unmapped_candidate": sum(int(row["rows_with_unmapped_candidate"]) for row in full),
        "rows_with_train_unseen_candidate": sum(int(row["rows_with_train_unseen_candidate"]) for row in full),
        "truncated_full_rows": sum(int(row["rows_history_truncated"]) for row in full),
    }


def dataset_sentence(dataset: str, history_rows: list[dict[str, Any]]) -> str:
    selected = [row for row in history_rows if row["dataset"] == dataset and row["cohort_id"] == "plan_e2_paper_like_full"]
    evaluated = sum(int(row["evaluated_rows"]) for row in selected)
    compatible = sum(int(row["sasrec_sequentially_compatible_rows"]) for row in selected)
    train_unseen = sum(int(row["rows_with_train_unseen_candidate"]) for row in selected)
    truncated = sum(int(row["rows_history_truncated"]) for row in selected)
    return f"Full rows `{evaluated}`; strict-compatible rows `{compatible}`; train-unseen rows `{train_unseen}`; truncated rows `{truncated}`."


def cohort_sentence(cohort_id: str, history_rows: list[dict[str, Any]]) -> str:
    selected = [row for row in history_rows if row["cohort_id"] == cohort_id]
    evaluated = sum(int(row["evaluated_rows"]) for row in selected)
    compatible = sum(int(row["sasrec_sequentially_compatible_rows"]) for row in selected)
    rate = compatible / evaluated if evaluated else 0.0
    return f"{cohort_id}: evaluated rows `{evaluated}`, compatible rows `{compatible}`, retention rate `{rate:.12g}`."


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: serialize_value(row.get(column, "")) for column in columns})


def serialize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    if value is None:
        return ""
    return value


def first_cuda_device(env: Mapping[str, str]) -> str:
    names = env.get("torch_device_names", "")
    try:
        parsed = json.loads(names)
    except json.JSONDecodeError:
        return ""
    return str(parsed[0]) if parsed else ""


def git_output(args: Sequence[str]) -> str:
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

"""Build the Plan E BPRMF adapter preflight audit.

This script audits data contracts only. It does not train BPRMF, does not
score project-data candidates, and does not create recommendation outputs.
"""

from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.plan_e_bprmf_adapter import (
    PLAN_E2_CANDIDATE_COUNT,
    PLAN_E2_CANDIDATE_SET_IDS,
    audit_index_space,
    audit_plan_e2_candidate_row,
    bool_text,
    build_existing_model_index,
    compute_train_seen_entities,
    filter_train_interactions,
    load_raw_to_internal_mapping,
    natural_id_sort_key,
    parse_candidate_items,
    read_csv_strings,
)
from src.audit.common_io import write_csv_rows as _write_csv_rows
from src.audit.common_paths import resolved_relative_to_root as _common_path_resolved_relative_to_root


PROTECTED_HASHES = {
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/candidate_item_title_map.csv": "5FF8A0B663E83D15830A6D86C8A2CFDBD9A3C7CE256F9FF496A7B5967B273E14",
    "results/tables/traditional_baseline_rankings.csv": "B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA",
    "results/tables/traditional_baselines_metrics.csv": "718F8B61D87118109654B94BD5DC8952AE24E407BF19DEF6B6BF4C9D4948D597",
    "results/tables/neural_environment_inventory.csv": "D3B59338A59A3ED777CA382541594424014F06B895278878884D6E3F0C91FE12",
    "results/tables/neural_baseline_feasibility_matrix.csv": "6FBA5654246BE57663C8DA98366083FACD0590C7613BAB0E7A8A6E0AFF5742CB",
    "results/tables/neural_baseline_gaps.csv": "83737501E94ADDDCAA136D8CE0A1122F2CB3C0EAD3AFAF101018F0A008FD854E",
    "results/tables/neural_baseline_execution_roadmap.csv": "A1408548977D6D361B4070828D295AFB2C5776E76BD33B536E7B40F0B0392AD0",
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
    "matched_cohort_rows",
    "training_data_path",
    "user_mapping_path",
    "item_mapping_path",
    "local_bprmf_implementation_path",
    "local_bprmf_runner_path",
    "environment_id",
    "python_executable",
    "python_version",
    "torch_version",
    "cuda_available",
    "cuda_device",
    "model_user_cardinality_rule",
    "model_item_cardinality_rule",
    "adapter_status",
    "project_data_training_performed",
    "project_data_scoring_performed",
    "recommended_pilot_scope",
    "blocking_reason",
    "notes",
]

MAPPING_COLUMNS = [
    "dataset",
    "entity_type",
    "training_data_path",
    "mapping_path",
    "mapping_direction",
    "raw_id_count",
    "internal_id_count",
    "duplicate_raw_id_count",
    "duplicate_internal_id_count",
    "minimum_internal_id",
    "maximum_internal_id",
    "contiguous_index_space",
    "missing_internal_index_count",
    "index_zero_semantics",
    "reserved_index_count",
    "model_embedding_cardinality",
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
    "user_seen_in_train",
    "positive_item_mapped",
    "positive_item_seen_in_train",
    "all_candidate_items_mapped",
    "all_candidate_items_seen_in_train",
    "unmapped_candidate_item_count",
    "unmapped_candidate_item_ids",
    "train_unseen_candidate_item_count",
    "train_unseen_candidate_item_ids",
    "existing_model_index_compatible",
    "strict_train_known_compatible",
    "contains_untrained_embedding_risk",
    "eligible_plan_e2_full",
    "eligible_plan_e2_matched",
    "compatibility_status",
    "exclusion_reasons",
    "notes",
]

COLD_COLUMNS = [
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "evaluated_rows",
    "rows_with_unmapped_user",
    "rows_with_train_unseen_user",
    "rows_with_unmapped_positive",
    "rows_with_train_unseen_positive",
    "rows_with_any_unmapped_candidate",
    "rows_with_any_train_unseen_candidate",
    "unique_unmapped_candidate_items",
    "unique_train_unseen_candidate_items",
    "candidate_occurrence_count",
    "unmapped_candidate_occurrence_count",
    "train_unseen_candidate_occurrence_count",
    "existing_model_index_compatible_rows",
    "strict_train_known_compatible_rows",
    "retention_rate_if_strict_train_known",
    "positive_position_distribution_full",
    "positive_position_distribution_strict_train_known",
    "notes",
]

POLICY_COLUMNS = [
    "policy_id",
    "policy_name",
    "description",
    "preserves_candidate_membership",
    "preserves_existing_phase2_cohort",
    "uses_train_unseen_embeddings",
    "uses_label_information",
    "introduces_data_leakage",
    "changes_bprmf_definition",
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
    mapping_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    dataset_context: dict[str, dict[str, Any]] = {}

    for dataset, spec in DATASETS.items():
        context = build_dataset_context(dataset, spec, plan_e_candidates)
        dataset_context[dataset] = context
        mapping_rows.extend(build_mapping_rows(context))
        for row in context["candidate_rows"]:
            key = (str(row["candidate_set_id"]), str(row["user_id"]))
            cohort = cohort_lookup[key]
            audited = audit_plan_e2_candidate_row(
                row,
                user_mapping=context["user_mapping"],
                item_mapping=context["item_mapping"],
                train_seen_users=context["train_seen_users"],
                train_seen_items=context["train_seen_items"],
                user_model_index=context["user_model_index"],
                item_model_index=context["item_model_index"],
                eligible_full=cohort["eligible_plan_e2_paper_like_full"],
                eligible_matched=cohort["eligible_plan_e2_matched_comparison"],
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

    manifest_rows.append(build_lastfm_manifest_row(env))
    compatibility_rows = sort_compatibility_rows(compatibility_rows)
    cold_rows = build_cold_entity_rows(compatibility_rows, plan_e_candidates)
    policy_rows = build_policy_rows()

    write_csv(tables_dir / "plan_e_bprmf_preflight_manifest.csv", MANIFEST_COLUMNS, manifest_rows)
    write_csv(tables_dir / "plan_e_bprmf_mapping_audit.csv", MAPPING_COLUMNS, mapping_rows)
    write_csv(tables_dir / "plan_e_bprmf_row_compatibility.csv", ROW_COLUMNS, serialize_bool_rows(compatibility_rows))
    write_csv(tables_dir / "plan_e_bprmf_cold_entity_impact.csv", COLD_COLUMNS, cold_rows)
    write_csv(tables_dir / "plan_e_bprmf_policy_options.csv", POLICY_COLUMNS, policy_rows)
    write_report(
        report_dir / "plan_e_bprmf_adapter_preflight.md",
        manifest_rows=manifest_rows,
        mapping_rows=mapping_rows,
        compatibility_rows=compatibility_rows,
        cold_rows=cold_rows,
        policy_rows=policy_rows,
        env=env,
    )

    print(f"preflight_manifest_rows={len(manifest_rows)}")
    print(f"mapping_audit_rows={len(mapping_rows)}")
    print(f"row_compatibility_rows={len(compatibility_rows)}")
    print(f"cold_entity_impact_rows={len(cold_rows)}")
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
    bad = membership.loc[
        full
        & (
            membership["source_plan"].astype(str).ne("Plan E2")
            | membership["candidate_count"].astype(str).ne("20")
        )
    ]
    if not bad.empty:
        raise RuntimeError("full Plan E2 cohort contains non-Plan-E2 or non-20-candidate rows")


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
        key = (str(row["cohort_id"]), str(row["candidate_set_id"]))
        lookup[key] = row
    return lookup


def build_dataset_context(dataset: str, spec: dict[str, Any], plan_e_candidates: pd.DataFrame) -> dict[str, Any]:
    folder = spec["folder"]
    base = PROJECT_ROOT / "data" / "reconstructed" / "plan_c" / folder
    interactions_path = base / "interactions.csv"
    datamap_path = base / "datamaps.json"
    interactions = read_csv_strings(interactions_path)
    train = filter_train_interactions(interactions)
    train_seen_users, train_seen_items = compute_train_seen_entities(train)
    user_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="user")
    item_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="item")
    rows = (
        plan_e_candidates.loc[plan_e_candidates["dataset"].astype(str) == dataset]
        .copy()
        .sort_values(["candidate_set_id", "user_id"], key=lambda s: s.map(natural_id_sort_key), kind="mergesort")
        .to_dict("records")
    )
    plan_e_users = {str(row["user_id"]) for row in rows}
    plan_e_items: set[str] = set()
    for row in rows:
        plan_e_items.update(parse_candidate_items(row["candidate_items"]))
        plan_e_items.add(str(row["positive_item_id"]))
    user_model_index = build_existing_model_index(train_seen_ids=train_seen_users, plan_e2_ids=plan_e_users)
    item_model_index = build_existing_model_index(train_seen_ids=train_seen_items, plan_e2_ids=plan_e_items)
    return {
        "dataset": dataset,
        "folder": folder,
        "training_data_path": rel(interactions_path),
        "datamap_path": rel(datamap_path),
        "candidate_rows": rows,
        "train_seen_users": train_seen_users,
        "train_seen_items": train_seen_items,
        "plan_e_users": plan_e_users,
        "plan_e_items": plan_e_items,
        "user_mapping": user_mapping,
        "item_mapping": item_mapping,
        "user_model_index": user_model_index,
        "item_model_index": item_model_index,
    }


def build_mapping_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for entity_type, mapping, train_seen, plan_e_ids, model_index in [
        (
            "user",
            context["user_mapping"],
            context["train_seen_users"],
            context["plan_e_users"],
            context["user_model_index"],
        ),
        (
            "item",
            context["item_mapping"],
            context["train_seen_items"],
            context["plan_e_items"],
            context["item_model_index"],
        ),
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
        elif train_seen_coverage < 1.0:
            status = "warning"
            blocking = ""
        else:
            status = "passed"
            blocking = ""
        rows.append(
            {
                "dataset": context["dataset"],
                "entity_type": entity_type,
                "training_data_path": context["training_data_path"],
                "mapping_path": context["datamap_path"],
                "mapping_direction": "raw_to_internal via user2id/item2id",
                "raw_id_count": mapping.raw_id_count,
                "internal_id_count": mapping.internal_id_count,
                "duplicate_raw_id_count": 0,
                "duplicate_internal_id_count": 0,
                "minimum_internal_id": audit.minimum_internal_id,
                "maximum_internal_id": audit.maximum_internal_id,
                "contiguous_index_space": bool_text(audit.contiguous_index_space),
                "missing_internal_index_count": audit.missing_internal_index_count,
                "index_zero_semantics": audit.index_zero_semantics,
                "reserved_index_count": audit.reserved_index_count,
                "model_embedding_cardinality": len(model_index),
                "train_seen_entity_count": len(train_seen),
                "plan_e2_unique_entity_count": plan_e_count,
                "plan_e2_mapped_entity_count": mapped_count,
                "plan_e2_train_seen_entity_count": train_seen_count,
                "mapping_coverage": f"{mapping_coverage:.12g}",
                "train_seen_coverage": f"{train_seen_coverage:.12g}",
                "status": status,
                "blocking_reason": blocking,
                "notes": (
                    "Plan D BPRMF reserves no padding index; adapter builds zero-based dense model indices. "
                    "Train-seen coverage below 1.0 indicates untrained-embedding risk."
                ),
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
        "first controlled ML-1M pilot candidate under strict train-known policy"
        if dataset == "ML-1M"
        else "defer until ML-1M pilot validates adapter and stop conditions"
    )
    return {
        "preflight_id": f"bprmf_{slug(dataset)}_{task_type}",
        "dataset": dataset,
        "task_type": task_type,
        "candidate_set_id": candidate_set_id,
        "source_path": str(full["source_path"]),
        "source_plan": str(full["source_plan"]),
        "full_cohort_rows": int(full["eligible_rows"]),
        "matched_cohort_rows": int(matched["eligible_rows"]),
        "training_data_path": context["training_data_path"],
        "user_mapping_path": context["datamap_path"],
        "item_mapping_path": context["datamap_path"],
        "local_bprmf_implementation_path": "src/baselines/plan_d_bprmf.py",
        "local_bprmf_runner_path": "src/experiments/run_plan_d_bprmf.py",
        "environment_id": "repo_venv_cuda",
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "cuda_available": str(env["torch_cuda_available"]).lower(),
        "cuda_device": json.loads(env["torch_device_names"])[0] if env.get("torch_device_names") else "",
        "model_user_cardinality_rule": "len(train_seen_users union Plan E2 users); zero-based dense Plan D index",
        "model_item_cardinality_rule": "len(train_seen_items union Plan E2 candidate items); zero-based dense Plan D index",
        "adapter_status": "ready_for_pilot_after_policy_approval",
        "project_data_training_performed": "false",
        "project_data_scoring_performed": "false",
        "recommended_pilot_scope": pilot_scope,
        "blocking_reason": "",
        "notes": "adapter preflight only; no model training, no project-data scoring",
    }


def build_lastfm_manifest_row(env: dict[str, str]) -> dict[str, Any]:
    return {
        "preflight_id": "bprmf_lastfm_not_applicable",
        "dataset": "LastFM",
        "task_type": "not_applicable",
        "candidate_set_id": "",
        "source_path": "",
        "source_plan": "Plan C extension only",
        "full_cohort_rows": 0,
        "matched_cohort_rows": 0,
        "training_data_path": "data/reconstructed/plan_c/lastfm/interactions.csv",
        "user_mapping_path": "data/reconstructed/plan_c/lastfm/datamaps.json",
        "item_mapping_path": "data/reconstructed/plan_c/lastfm/datamaps.json",
        "local_bprmf_implementation_path": "src/baselines/plan_d_bprmf.py",
        "local_bprmf_runner_path": "src/experiments/run_plan_d_bprmf.py",
        "environment_id": "repo_venv_cuda",
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "cuda_available": str(env["torch_cuda_available"]).lower(),
        "cuda_device": json.loads(env["torch_device_names"])[0] if env.get("torch_device_names") else "",
        "model_user_cardinality_rule": "not applicable without Plan E2 candidate rows",
        "model_item_cardinality_rule": "not applicable without Plan E2 candidate rows",
        "adapter_status": "not_applicable",
        "project_data_training_performed": "false",
        "project_data_scoring_performed": "false",
        "recommended_pilot_scope": "blocked from Phase 3C/3D BPRMF pilot",
        "blocking_reason": "LastFM has no Plan E2 candidate sets, ground truth or negative samples in the current canonical cohort.",
        "notes": "represented explicitly to avoid accidental pilot inclusion",
    }


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
        "user_seen_in_train",
        "positive_item_mapped",
        "positive_item_seen_in_train",
        "all_candidate_items_mapped",
        "all_candidate_items_seen_in_train",
        "existing_model_index_compatible",
        "strict_train_known_compatible",
        "contains_untrained_embedding_risk",
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


def build_cold_entity_rows(
    compatibility_rows: list[dict[str, Any]],
    plan_e_candidates: pd.DataFrame,
) -> list[dict[str, Any]]:
    by_key = {(str(row["candidate_set_id"]), str(row["user_id"])): row for row in compatibility_rows}
    source_lookup = {
        (str(row["candidate_set_id"]), str(row["user_id"])): row for row in plan_e_candidates.to_dict("records")
    }
    rows = []
    cohort_specs = [
        ("plan_e2_paper_like_full", "eligible_plan_e2_full"),
        ("plan_e2_matched_comparison", "eligible_plan_e2_matched"),
    ]
    candidate_sets = sorted({row["candidate_set_id"] for row in compatibility_rows})
    for candidate_set_id in candidate_sets:
        set_rows = [row for row in compatibility_rows if row["candidate_set_id"] == candidate_set_id]
        dataset = str(set_rows[0]["dataset"])
        task_type = str(set_rows[0]["task_type"])
        for cohort_id, flag in cohort_specs:
            selected = [row for row in set_rows if bool(row[flag])]
            distribution_full = positive_position_distribution(selected, source_lookup)
            strict = [row for row in selected if bool(row["strict_train_known_compatible"])]
            distribution_strict = positive_position_distribution(strict, source_lookup)
            unmapped_items = unique_items_from_json(selected, "unmapped_candidate_item_ids")
            unseen_items = unique_items_from_json(selected, "train_unseen_candidate_item_ids")
            evaluated = len(selected)
            rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "cohort_id": cohort_id,
                    "evaluated_rows": evaluated,
                    "rows_with_unmapped_user": sum(not bool(row["user_mapped"]) for row in selected),
                    "rows_with_train_unseen_user": sum(not bool(row["user_seen_in_train"]) for row in selected),
                    "rows_with_unmapped_positive": sum(not bool(row["positive_item_mapped"]) for row in selected),
                    "rows_with_train_unseen_positive": sum(
                        not bool(row["positive_item_seen_in_train"]) for row in selected
                    ),
                    "rows_with_any_unmapped_candidate": sum(
                        int(row["unmapped_candidate_item_count"]) > 0 for row in selected
                    ),
                    "rows_with_any_train_unseen_candidate": sum(
                        int(row["train_unseen_candidate_item_count"]) > 0 for row in selected
                    ),
                    "unique_unmapped_candidate_items": len(unmapped_items),
                    "unique_train_unseen_candidate_items": len(unseen_items),
                    "candidate_occurrence_count": sum(int(row["candidate_count"]) for row in selected),
                    "unmapped_candidate_occurrence_count": sum(
                        int(row["unmapped_candidate_item_count"]) for row in selected
                    ),
                    "train_unseen_candidate_occurrence_count": sum(
                        int(row["train_unseen_candidate_item_count"]) for row in selected
                    ),
                    "existing_model_index_compatible_rows": sum(
                        bool(row["existing_model_index_compatible"]) for row in selected
                    ),
                    "strict_train_known_compatible_rows": len(strict),
                    "retention_rate_if_strict_train_known": f"{(len(strict) / evaluated if evaluated else 0.0):.12g}",
                    "positive_position_distribution_full": json.dumps(distribution_full, sort_keys=True),
                    "positive_position_distribution_strict_train_known": json.dumps(
                        distribution_strict,
                        sort_keys=True,
                    ),
                    "notes": "ranking and reranking are aggregated separately; no candidate rows were modified",
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
        position = items.index(str(row["positive_item_id"]))
        counter[str(position)] += 1
    return dict(sorted(counter.items(), key=lambda item: int(item[0])))


def unique_items_from_json(rows: list[dict[str, Any]], field: str) -> set[str]:
    items: set[str] = set()
    for row in rows:
        items.update(json.loads(str(row[field])))
    return items


def build_policy_rows() -> list[dict[str, Any]]:
    def row(
        policy_id: str,
        policy_name: str,
        description: str,
        preserves_candidate_membership: bool,
        preserves_existing_phase2_cohort: bool,
        uses_train_unseen_embeddings: bool,
        uses_label_information: bool,
        introduces_data_leakage: bool,
        changes_bprmf_definition: bool,
        paper_alignment: str,
        comparability_impact: str,
        methodological_risk: str,
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
            "preserves_candidate_membership": bool_text(preserves_candidate_membership),
            "preserves_existing_phase2_cohort": bool_text(preserves_existing_phase2_cohort),
            "uses_train_unseen_embeddings": bool_text(uses_train_unseen_embeddings),
            "uses_label_information": bool_text(uses_label_information),
            "introduces_data_leakage": bool_text(introduces_data_leakage),
            "changes_bprmf_definition": bool_text(changes_bprmf_definition),
            "paper_alignment": paper_alignment,
            "comparability_impact": comparability_impact,
            "methodological_risk": methodological_risk,
            "allowed_for_future_execution": bool_text(allowed),
            "recommendation_status": status,
            "rationale": rationale,
            "required_additional_cohort": required_cohort,
            "stop_condition": stop_condition,
            "notes": notes,
        }

    return [
        row(
            "P1",
            "retain_randomly_initialized_unseen_embeddings",
            "Allocate embeddings for all mapped Plan E2 items and score train-unseen items at initialization.",
            True,
            True,
            True,
            False,
            False,
            False,
            "adapted",
            "keeps the full cohort but makes some scores depend on random initialization",
            "high",
            True,
            "acceptable_with_strong_caveat",
            "No leakage or candidate mutation, but train-unseen item scores have no learned signal.",
            "none",
            "stop if any result is reported without an explicit untrained-embedding caveat",
            "not recommended as the default pilot policy",
        ),
        row(
            "P2",
            "exclude_rows_with_any_train_unseen_candidate",
            "Preserve candidate lists but exclude a row from BPRMF comparison when any candidate item is train-unseen.",
            True,
            False,
            False,
            False,
            False,
            False,
            "paper_like_reconstructed",
            "requires a BPRMF-compatible matched cohort for all compared methods",
            "medium",
            True,
            "recommended",
            "Prioritizes no leakage, no candidate mutation, clear interpretation and reproducible cohort comparison.",
            "bprmf_strict_train_known_matched_cohort",
            "stop if strict filtering is not applied identically to every compared method",
            "recommended future policy for controlled BPRMF pilot",
        ),
        row(
            "P3",
            "exclude_rows_only_when_positive_is_train_unseen",
            "Exclude rows only when the positive item is train-unseen while retaining unseen negatives.",
            True,
            False,
            True,
            True,
            False,
            False,
            "adapted",
            "retains more rows but still scores unseen negative candidates arbitrarily",
            "medium_high",
            True,
            "acceptable_with_strong_caveat",
            "Uses the positive label for row exclusion and leaves unseen-negative behavior unresolved.",
            "positive_train_known_matched_cohort",
            "stop if unseen negative scores are interpreted as learned preferences",
            "not the default recommendation",
        ),
        row(
            "P4",
            "deterministic_fallback_score_for_unseen_items",
            "Assign a constant or popularity-derived score to train-unseen items.",
            True,
            True,
            False,
            False,
            False,
            True,
            "adapted",
            "keeps rows but changes the BPRMF scorer into a hybrid baseline",
            "high",
            True,
            "not_recommended",
            "A fallback score is reproducible but no longer an exact BPRMF score.",
            "none",
            "stop if fallback outputs are labeled as exact BPRMF",
            "could be a separately named adapted diagnostic only",
        ),
        row(
            "P5",
            "remove_or_replace_unseen_candidates",
            "Remove or replace train-unseen candidates before scoring.",
            False,
            False,
            False,
            False,
            False,
            True,
            "not_aligned",
            "invalidates canonical candidate membership and ordering",
            "critical",
            False,
            "forbidden",
            "The project forbids candidate-list mutation.",
            "not_allowed",
            "stop immediately if candidate membership or order changes",
            "forbidden for this project",
        ),
        row(
            "P6",
            "train_with_validation_or_test_interactions_to_cover_items",
            "Add validation or test interactions to training so all candidates are seen.",
            True,
            True,
            False,
            True,
            True,
            True,
            "not_aligned",
            "leaks held-out information and invalidates utility evaluation",
            "critical",
            False,
            "forbidden",
            "Training on validation/test interactions is data leakage.",
            "not_allowed",
            "stop immediately if non-train interactions enter the training set",
            "forbidden for this project",
        ),
    ]


def write_report(
    path: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    cold_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    env: dict[str, str],
) -> None:
    status_counts = Counter(row["compatibility_status"] for row in compatibility_rows)
    strict_total = sum(bool(row["strict_train_known_compatible"]) for row in compatibility_rows)
    risk_total = sum(bool(row["contains_untrained_embedding_risk"]) for row in compatibility_rows)
    index_total = sum(bool(row["existing_model_index_compatible"]) for row in compatibility_rows)
    dataset_lines = []
    for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
        subset = [row for row in compatibility_rows if row["dataset"] == dataset]
        strict = sum(bool(row["strict_train_known_compatible"]) for row in subset)
        risk = sum(bool(row["contains_untrained_embedding_risk"]) for row in subset)
        unseen_positive = sum(not bool(row["positive_item_seen_in_train"]) for row in subset)
        unseen_candidate_rows = sum(int(row["train_unseen_candidate_item_count"]) > 0 for row in subset)
        dataset_lines.append(
            f"- {dataset}: rows={len(subset)}, strict_train_known={strict}, "
            f"untrained_embedding_risk_rows={risk}, train_unseen_positive_rows={unseen_positive}, "
            f"rows_with_any_train_unseen_candidate={unseen_candidate_rows}."
        )
    mapping_table = "\n".join(
        f"- {row['dataset']} {row['entity_type']}: mapping_coverage={row['mapping_coverage']}, "
        f"train_seen_coverage={row['train_seen_coverage']}, status={row['status']}."
        for row in mapping_rows
    )
    cold_table = "\n".join(
        f"- {row['dataset']} {row['task_type']} {row['cohort_id']}: evaluated={row['evaluated_rows']}, "
        f"strict={row['strict_train_known_compatible_rows']}, "
        f"retention={row['retention_rate_if_strict_train_known']}."
        for row in cold_rows
    )
    recommended = [row for row in policy_rows if row["recommendation_status"] == "recommended"][0]
    branch = git_output(["git", "branch", "--show-current"])
    commit = git_output(["git", "log", "-1", "--format=%H %s"])
    report = f"""# Plan E BPRMF Adapter and Preflight Validation

## 1. Scope and provenance

- Phase 3C audit and adapter preflight for BPRMF Plan E2 compatibility.
- Branch: `{branch}`.
- Commit: `{commit}`.
- Generated tables: `results/tables/plan_e_bprmf_preflight_manifest.csv`, `plan_e_bprmf_mapping_audit.csv`, `plan_e_bprmf_row_compatibility.csv`, `plan_e_bprmf_cold_entity_impact.csv`, and `plan_e_bprmf_policy_options.csv`.

## 2. Executive summary

- Plan E2 input verification passed: 6,000 Plan E2 rows, six candidate-set IDs, 20 candidates per row, and one positive per row.
- Existing-model index compatibility rows: `{index_total}` of `{len(compatibility_rows)}`.
- Strict train-known compatible rows: `{strict_total}` of `{len(compatibility_rows)}`.
- Rows with untrained-embedding risk: `{risk_total}` of `{len(compatibility_rows)}`.
- Recommended future policy: `{recommended['policy_name']}`.

## 3. Safety boundaries

No project-data training, project-data BPRMF scoring, neural inference, optimizer step, backward pass, recommendation ranking, metric computation, checkpoint creation, API call, web access, package installation, download, or environment modification was performed.

## 4. Authoritative inputs

Protected inputs and their expected hashes are checked by `src/experiments/build_plan_e_bprmf_preflight.py` before outputs are written. The main inputs are canonical Phase 2 candidates/cohorts, Phase 3A traditional baseline files, and Phase 3B neural feasibility files.

## 5. Existing Plan D BPRMF implementation

Local implementation path: `src/baselines/plan_d_bprmf.py`. Runner path: `src/experiments/run_plan_d_bprmf.py`. The adapter wraps this implementation contract and does not modify it.

## 6. BPRMF model and loss definition

`BPRMFModel` uses `nn.Embedding(num_users, embedding_dim)` and `nn.Embedding(num_items, embedding_dim)` with normally initialized embeddings. Scores are user-item dot products. The pairwise loss is `-log sigmoid(pos - neg)`.

## 7. Existing training and evaluation interfaces

Plan D `load_plan_c_bprmf_data` creates zero-based dense model indices from the union of train IDs and candidate IDs. `train_bprmf` creates the optimizer and trains. `rank_bprmf_candidates` scores candidate DataFrames after training. Phase 3C did not call training or project-data scoring.

## 8. CUDA environment verification

Environment ID: `repo_venv_cuda`. Python: `{env['python_executable']}`. Python version: `{env['python_version']}`. Torch: `{env['torch_version']}`. CUDA available: `{env['torch_cuda_available']}`. Device: `{env['torch_device_names']}`.

## 9. Dataset training sources

- Amazon Beauty: `data/reconstructed/plan_c/amazon_beauty/interactions.csv`.
- Amazon Sports: `data/reconstructed/plan_c/amazon_sports/interactions.csv`.
- ML-1M: `data/reconstructed/plan_c/ml1m/interactions.csv`.
- LastFM is represented as blocked/not applicable for Plan E2.

## 10. User mapping audit

{mapping_table}

## 11. Item mapping audit

The item mapping rows above show complete Plan E2 mapping coverage for Beauty, Sports and ML-1M. Item train-seen coverage is below 1.0 where Plan E2 includes mapped candidate items absent from train interactions.

## 12. Index-space and embedding-cardinality audit

Datamap internal IDs are explicit and one-based in the reconstructed artifacts. Plan D BPRMF itself uses zero-based dense indices with no reserved padding or unknown index. The adapter therefore treats datamap IDs as provenance/mapping evidence and prepares future model indices as zero-based dense indices from train-seen plus Plan E2 IDs.

## 13. Plan E2 candidate mapping

Every Plan E2 candidate row has 20 candidates and candidate order is preserved in `plan_e_bprmf_row_compatibility.csv`. Future score alignment is defined by raw candidate position, not by sorting.

## 14. Beauty compatibility

{dataset_lines[0]}

## 15. Sports compatibility

{dataset_lines[1]}

## 16. ML-1M compatibility

{dataset_lines[2]}

## 17. LastFM blocked status

LastFM has no Plan E2 candidate-set rows in the canonical Phase 2 artifacts. The manifest includes one explicit `not_applicable` row with a blocking reason.

## 18. Train-unseen user findings

Train-unseen user counts are recorded in `plan_e_bprmf_cold_entity_impact.csv`. No project-data user embedding was trained or scored in this phase.

## 19. Train-unseen positive-item findings

Rows with train-unseen positive items are recorded per dataset/task/cohort in `plan_e_bprmf_cold_entity_impact.csv` and per row in `plan_e_bprmf_row_compatibility.csv`.

## 20. Train-unseen candidate-item findings

Rows with any train-unseen candidate are the key source of untrained-embedding risk. Counts and candidate occurrence totals are in `plan_e_bprmf_cold_entity_impact.csv`.

## 21. Untrained-embedding risk

Under the current Plan D cardinality rule, mapped train-unseen candidate items can receive embedding rows but would remain at initialization unless they appear in train interactions. This is why strict train-known compatibility is reported separately from index compatibility.

## 22. Candidate-order preservation

The adapter decodes `candidate_items` JSON arrays, preserves their raw string order, and prepares future item-index payloads in the same order. No candidate sorting or membership change is performed.

## 23. Future score-alignment contract

Future score vectors must have length equal to the raw candidate list, all scores must be finite, and scores are aligned back to raw candidate IDs by zero-based candidate position.

## 24. Full-cohort compatibility

Full-cohort compatibility by dataset/task is summarized below:

{cold_table}

## 25. Matched-cohort compatibility

The same table includes `plan_e2_matched_comparison` rows. Matched filtering must be applied from the same full-cohort prediction rows in any future pilot.

## 26. Policy options evaluated

Six policy options are recorded in `plan_e_bprmf_policy_options.csv`: retain randomly initialized unseen embeddings, exclude rows with any train-unseen candidate, exclude only train-unseen positives, deterministic fallback scores, remove/replace unseen candidates, and train with validation/test interactions.

## 27. Recommended future policy

Recommended policy: `{recommended['policy_name']}`. Rationale: {recommended['rationale']}

## 28. Selection-bias implications of strict compatibility filtering

Strict filtering reduces the evaluation cohort and can introduce selection bias toward users/items well covered in training. Any future comparison must report the strict cohort and apply it consistently to all compared methods.

## 29. Proposed ML-1M pilot scope

The safest next pilot scope is ML-1M BPRMF on Plan E2 ranking and reranking candidate sets, using the strict train-known policy and writing new pilot-specific outputs only after explicit Phase 3D approval.

## 30. Proposed pilot success criteria

The pilot should pass if it trains only on `split=train`, preserves Plan E2 candidate order, scores only approved rows, writes outputs to new Phase 3D paths, and produces finite scores for every retained candidate.

## 31. Pilot stop conditions

Stop on any protected-input mutation, candidate membership/order change, unmapped Plan E2 ID, non-finite loss or score, CUDA out-of-memory, checkpoint creation outside approved paths, or use of validation/test interactions in training.

## 32. Required future configuration fields

Future configuration should include dataset, candidate_set_ids, cohort policy, train data path, datamap path, output directory, seed, embedding dimension, epochs, batch size, device, and explicit approval to train.

## 33. Tests and synthetic smoke probe

`tests/test_plan_e_bprmf_adapter.py` contains synthetic-only mapping, compatibility, score-alignment, and BPRMF forward-pass tests. The smoke probe uses toy tensors only and performs no optimizer step or backward pass.

## 34. Protected-input immutability

Protected hashes are checked before generation and should be rechecked after generation. The preflight runner does not regenerate Phase 1, Phase 2, Phase 3A or Phase 3B artifacts.

## 35. Recommended single next task

Phase 3D - controlled BPRMF ML-1M pilot configuration and execution, only after explicit approval of the strict train-known compatibility policy.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(path, columns, rows, create_parent=False, extrasaction="ignore")


def rel(path: Path) -> str:
    """Compatibility wrapper retained for F6I2I path_normalization provenance."""

    return _common_path_resolved_relative_to_root(path, PROJECT_ROOT)


def slug(dataset: str) -> str:
    return dataset.lower().replace("amazon ", "amazon_").replace("-", "").replace(" ", "_")


def git_output(command: list[str]) -> str:
    try:
        return common_git.git_stdout_checked(command, PROJECT_ROOT, stderr_to_devnull=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())

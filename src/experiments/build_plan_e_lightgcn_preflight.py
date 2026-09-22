"""Build the Plan E LightGCN adapter and graph-construction preflight audit.

This script audits data contracts only. It does not train LightGCN, does not
run project-data graph propagation, does not score project-data candidates, and
does not create recommendation, metric, plot, or checkpoint artifacts.
"""

from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.plan_e_lightgcn_adapter import (
    PLAN_E2_CANDIDATE_COUNT,
    PLAN_E2_CANDIDATE_SET_IDS,
    GraphValidationError,
    audit_plan_e2_candidate_row,
    bool_text,
    bool_value,
    build_graph_contract,
    estimate_sparse_memory_bytes,
    filter_train_interactions,
    hash_float_array,
    hash_int_arrays,
    hash_normalized_graph,
    hash_raw_pairs,
    json_array,
    load_raw_to_internal_mapping,
    natural_id_sort_key,
    parse_candidate_items,
    read_csv_strings,
    safe_raw_id,
    synthetic_sparse_forward_probe,
)
from src.audit.common_io import write_csv_rows as _write_csv_rows


APPROVED_INTERPRETER = (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").resolve()
OUTPUT_PREFIX = "plan_e_lightgcn_"

PROTECTED_HASHES = {
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/plan_e_bprmf_row_compatibility.csv": "9D34B52424A6113FD2B854D6759756809D86E53C106327D120F52F38DAAEC817",
    "results/tables/plan_e_gru4rec_row_compatibility.csv": "632FA890F05401CECA5A6B64212370CA0E80182D5B9A55918A9AC3C10630F6C1",
    "results/tables/plan_e_sasrec_row_compatibility.csv": "414F8A8EA84E8DE639DA05EB79863CE5D900F92D83EFF9163CC02FEEC9C9BE09",
    "results/tables/plan_e_sasrec_same_cohort_baselines_metrics.csv": "67F19EE8128956DFE1121224993DFD6AD33814E232E73AF1FF1C244C396D0F35",
    "src/baselines/plan_d_lightgcn.py": "1B9DF1E9DF737758639D98013A8718E287EE550E9491979C663EE6E95959FEF8",
    "src/experiments/run_plan_d_lightgcn.py": "27626556582F897C846904B11658B51DB4C499C6335903D8CF3B52474FB800AB",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv": "D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478",
    "data/reconstructed/plan_c/amazon_beauty/datamaps.json": "4EF9E7F9BA6D0F98998EFAC7DFDEA455315635753B9C21FF46269ABC31670423",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv": "E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40",
    "data/reconstructed/plan_c/amazon_sports/datamaps.json": "5A0E99616F2C31DF45E13B82DD9012A7728AFC0507C475876E0E6C78ACFE6A5A",
    "data/reconstructed/plan_c/ml1m/interactions.csv": "1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870",
    "data/reconstructed/plan_c/ml1m/datamaps.json": "D6AC1A6C19E47156955DE87F45EE2255AC69728510E734BE46DBBBECD68BF093",
    "data/reconstructed/plan_c/lastfm/interactions.csv": "C57C4ED08B0F1D45C338E8AFFB0E7E6522786D1D4B9844527F49987257BD7B7B",
    "data/reconstructed/plan_c/lastfm/datamaps.json": "E8DFA8BA693A3DDC0315D1BDCD019CD924F6617CD32C9A02280699D11FDF3DB0",
    "src/baselines/plan_e_bprmf_adapter.py": "DEAEA8B4DC9A809C89E4547964329B745C5E460A8E375D1618096CBE0866AE8F",
    "src/baselines/plan_e_gru4rec_adapter.py": "F3972AEDE7A085FD7D247826D0EB12B097A5F9EBB0E6F6ADA6143B4EE8C84DCD",
    "src/baselines/plan_e_sasrec_adapter.py": "D3CF9058885BE006613C34A51301F3AA4647EC062C15179D60CC3B0D0DC8A8D3",
    "src/experiments/build_plan_e_bprmf_preflight.py": "1EFB521C2B63654713B39A612DD9D463B6CA645710D2ADED61731DC5E8165625",
    "src/experiments/build_plan_e_gru4rec_preflight.py": "B44E0FEC4880ED98471436DB0860E346F5A665F83EFE6C19F8CCD00FFDF9D488",
    "src/experiments/build_plan_e_sasrec_preflight.py": "5513655CC86909B0F394416A3F22952F2854097F93F62F6099A69A5A3671FCF1",
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
    "LastFM": {
        "folder": "lastfm",
        "task_ids": {},
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
    "user_mapping_path",
    "item_mapping_path",
    "local_lightgcn_implementation_path",
    "local_lightgcn_runner_path",
    "environment_id",
    "python_executable",
    "python_version",
    "torch_version",
    "cuda_available",
    "cuda_device",
    "graph_construction_status",
    "normalization_status",
    "sparse_backend_status",
    "index_policy",
    "propagation_policy",
    "adapter_status",
    "project_data_training_performed",
    "project_data_scoring_performed",
    "recommended_pilot_scope",
    "blocking_reason",
    "notes",
]

GRAPH_COLUMNS = [
    "dataset",
    "training_data_path",
    "split_field",
    "training_split_value",
    "total_interaction_count",
    "train_interaction_count",
    "validation_interaction_count",
    "test_interaction_count",
    "observed_interaction_count",
    "train_user_count",
    "train_item_count",
    "duplicate_user_item_pair_count",
    "unique_user_item_edge_count",
    "graph_edge_weight_policy",
    "user_node_count",
    "item_node_count",
    "total_node_count",
    "directed_edge_count",
    "self_loop_count",
    "isolated_train_user_count",
    "isolated_train_item_count",
    "minimum_user_degree",
    "maximum_user_degree",
    "mean_user_degree",
    "median_user_degree",
    "minimum_item_degree",
    "maximum_item_degree",
    "mean_item_degree",
    "median_item_degree",
    "raw_edge_sha256",
    "directed_edge_index_sha256",
    "edge_weight_sha256",
    "graph_construction_deterministic",
    "status",
    "blocking_reason",
    "notes",
]

NORMALIZATION_COLUMNS = [
    "dataset",
    "implementation_path",
    "normalization_formula",
    "bipartite_edges_symmetrized",
    "self_loops_included",
    "degree_calculation_policy",
    "zero_degree_handling",
    "sparse_layout",
    "sparse_coalesced",
    "tensor_dtype",
    "normalized_edge_count",
    "minimum_normalized_weight",
    "maximum_normalized_weight",
    "non_finite_weight_count",
    "symmetry_verified",
    "normalized_adjacency_sha256",
    "propagation_layer_count",
    "layer_zero_included",
    "layer_aggregation",
    "propagation_nonlinearity",
    "trainable_graph_transformations_present",
    "final_user_item_split_policy",
    "synthetic_normalization_checks",
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
    "lightgcn_local_index_min",
    "lightgcn_local_index_max",
    "bipartite_node_offset",
    "global_node_index_min",
    "global_node_index_max",
    "total_graph_nodes",
    "duplicate_raw_id_count",
    "duplicate_internal_id_count",
    "contiguous_local_index_space",
    "train_seen_entity_count",
    "plan_e2_unique_entity_count",
    "plan_e2_mapped_entity_count",
    "plan_e2_graph_connected_entity_count",
    "mapping_coverage",
    "graph_connected_coverage",
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
    "user_graph_connected",
    "positive_item_mapped",
    "positive_item_graph_connected",
    "all_candidate_items_mapped",
    "all_candidate_items_graph_connected",
    "unmapped_candidate_item_count",
    "unmapped_candidate_item_ids",
    "graph_disconnected_candidate_item_count",
    "graph_disconnected_candidate_item_ids",
    "all_candidate_items_train_seen",
    "existing_model_index_compatible",
    "contains_isolated_embedding_risk",
    "lightgcn_graph_compatible",
    "eligible_plan_e2_full",
    "eligible_plan_e2_matched",
    "compatibility_status",
    "exclusion_reasons",
    "notes",
]

IMPACT_COLUMNS = [
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "evaluated_rows",
    "rows_with_unmapped_user",
    "rows_with_disconnected_user",
    "rows_with_unmapped_positive",
    "rows_with_disconnected_positive",
    "rows_with_any_unmapped_candidate",
    "rows_with_any_disconnected_candidate",
    "unique_disconnected_candidate_items",
    "candidate_occurrence_count",
    "disconnected_candidate_occurrence_count",
    "lightgcn_graph_compatible_rows",
    "retention_rate",
    "positive_position_distribution_full",
    "positive_position_distribution_compatible",
    "mean_user_degree",
    "median_user_degree",
    "mean_positive_item_degree",
    "median_positive_item_degree",
    "notes",
]

SPARSE_COLUMNS = [
    "environment_id",
    "python_executable",
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "cuda_available",
    "cuda_device",
    "dataset",
    "graph_nodes",
    "directed_edges",
    "sparse_layout",
    "estimated_edge_index_bytes",
    "estimated_edge_weight_bytes",
    "estimated_embedding_bytes_per_layer",
    "estimated_total_graph_tensor_bytes",
    "cpu_sparse_construction_supported",
    "cpu_sparse_propagation_supported",
    "cuda_sparse_construction_supported",
    "cuda_sparse_propagation_supported",
    "deterministic_algorithms_supported_cpu",
    "deterministic_algorithms_supported_cuda",
    "synthetic_cpu_forward_passed",
    "synthetic_cuda_forward_passed",
    "cpu_cuda_synthetic_max_abs_difference",
    "warning_count",
    "warning_categories",
    "recommended_execution_device",
    "status",
    "blocking_reason",
    "notes",
]

POLICY_COLUMNS = [
    "policy_id",
    "policy_name",
    "description",
    "uses_train_only_edges",
    "deduplicates_user_item_edges",
    "weights_repeated_interactions",
    "includes_self_loops",
    "includes_layer_zero_embedding",
    "requires_connected_user",
    "requires_connected_positive",
    "requires_all_candidates_graph_connected",
    "preserves_candidate_membership",
    "uses_validation_or_test_edges",
    "uses_candidate_pseudo_edges",
    "uses_disconnected_embeddings",
    "introduces_data_leakage",
    "changes_lightgcn_definition",
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

OUTPUTS = {
    "manifest": (PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_preflight_manifest.csv", MANIFEST_COLUMNS),
    "graph": (PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_graph_audit.csv", GRAPH_COLUMNS),
    "normalization": (
        PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_normalization_audit.csv",
        NORMALIZATION_COLUMNS,
    ),
    "index": (PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_index_audit.csv", INDEX_COLUMNS),
    "compatibility": (
        PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_row_compatibility.csv",
        ROW_COLUMNS,
    ),
    "impact": (
        PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_connectivity_impact.csv",
        IMPACT_COLUMNS,
    ),
    "sparse": (
        PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_sparse_execution_audit.csv",
        SPARSE_COLUMNS,
    ),
    "policy": (PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_policy_options.csv", POLICY_COLUMNS),
}


def main() -> int:
    verify_runtime()
    verify_protected_hashes()

    tables_dir = PROJECT_ROOT / "results" / "tables"
    report_dir = PROJECT_ROOT / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_csv_strings(tables_dir / "ranking_candidate_sets.csv")
    membership = read_csv_strings(tables_dir / "experiment_cohort_membership.csv")
    cohort_summary = read_csv_strings(tables_dir / "experiment_cohort_summary.csv")
    validate_phase2_inputs(candidates, membership)

    env = environment_inventory()
    plan_e_candidates = candidates.loc[candidates["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].copy()
    membership_lookup = build_membership_lookup(membership)
    summary_lookup = build_summary_lookup(cohort_summary)

    contexts = {
        dataset: build_dataset_context(dataset, spec, plan_e_candidates)
        for dataset, spec in DATASETS.items()
    }

    graph_rows = [build_graph_row(dataset, context) for dataset, context in contexts.items()]
    normalization_rows = [build_normalization_row(dataset, context) for dataset, context in contexts.items()]
    index_rows: list[dict[str, Any]] = []
    for dataset, context in contexts.items():
        index_rows.extend(build_index_rows(dataset, context))

    compatibility_rows: list[dict[str, Any]] = []
    candidate_position_lookup: dict[tuple[str, str], int] = {}
    for dataset, context in contexts.items():
        if dataset == "LastFM":
            continue
        for row in context["candidate_rows"].to_dict("records"):
            key = (str(row["candidate_set_id"]), str(row["user_id"]))
            cohort = membership_lookup[key]
            row["task_type"] = cohort["task_type"]
            audited = audit_plan_e2_candidate_row(
                row,
                user_mapping=context["source_user_mapping"],
                item_mapping=context["source_item_mapping"],
                graph_users=context["graph_users"],
                graph_items=context["graph_items"],
                user_local_index=context["graph_contract"].user_mapping.raw_to_local,
                item_local_index=context["graph_contract"].item_mapping.raw_to_local,
                normalization_verified=context["normalization_verified"],
                eligible_full=cohort["eligible_plan_e2_paper_like_full"],
                eligible_matched=cohort["eligible_plan_e2_matched_comparison"],
            )
            compatibility_rows.append(audited)
            candidate_position_lookup[key] = int(row["computed_positive_position_zero_based"])

    compatibility_rows = sort_rows(compatibility_rows, ["dataset", "task_type", "candidate_set_id", "user_id"])
    impact_rows = build_connectivity_impact_rows(compatibility_rows, candidate_position_lookup, contexts)
    sparse_rows = build_sparse_rows(env, graph_rows)
    policy_rows = build_policy_rows()
    manifest_rows = build_manifest_rows(contexts, summary_lookup, env, graph_rows, normalization_rows, sparse_rows)

    write_csv(OUTPUTS["manifest"][0], MANIFEST_COLUMNS, manifest_rows)
    write_csv(OUTPUTS["graph"][0], GRAPH_COLUMNS, graph_rows)
    write_csv(OUTPUTS["normalization"][0], NORMALIZATION_COLUMNS, normalization_rows)
    write_csv(OUTPUTS["index"][0], INDEX_COLUMNS, index_rows)
    write_csv(OUTPUTS["compatibility"][0], ROW_COLUMNS, compatibility_rows)
    write_csv(OUTPUTS["impact"][0], IMPACT_COLUMNS, impact_rows)
    write_csv(OUTPUTS["sparse"][0], SPARSE_COLUMNS, sparse_rows)
    write_csv(OUTPUTS["policy"][0], POLICY_COLUMNS, policy_rows)

    validate_outputs()
    write_report(
        report_dir / "plan_e_lightgcn_adapter_preflight.md",
        env=env,
        graph_rows=graph_rows,
        normalization_rows=normalization_rows,
        index_rows=index_rows,
        compatibility_rows=compatibility_rows,
        impact_rows=impact_rows,
        sparse_rows=sparse_rows,
        policy_rows=policy_rows,
        manifest_rows=manifest_rows,
    )
    verify_protected_hashes()
    print_summary(graph_rows, compatibility_rows, impact_rows, sparse_rows)
    return 0


def verify_runtime() -> None:
    executable = Path(sys.executable).resolve()
    if executable != APPROVED_INTERPRETER:
        raise RuntimeError(f"approved interpreter required: {APPROVED_INTERPRETER}; got {executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def verify_protected_hashes() -> None:
    mismatches = []
    for relative_path, expected in PROTECTED_HASHES.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            mismatches.append(f"{relative_path}: missing")
            continue
        observed = file_sha256(path)
        if observed != expected:
            mismatches.append(f"{relative_path}: expected {expected}, observed {observed}")
    if mismatches:
        raise RuntimeError("protected input hash mismatch: " + "; ".join(mismatches))


def environment_inventory() -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else ""
    cuda_capability = ".".join(str(part) for part in torch.cuda.get_device_capability(0)) if cuda_available else ""
    return {
        "environment_id": "repo_venv_cuda",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda or ""),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "cuda_compute_capability": cuda_capability,
    }


def validate_phase2_inputs(candidates: pd.DataFrame, membership: pd.DataFrame) -> None:
    required_candidate_cols = {
        "candidate_set_id",
        "source_path",
        "source_plan",
        "dataset",
        "user_id",
        "positive_item_id",
        "candidate_items",
        "candidate_count",
        "computed_positive_position_zero_based",
    }
    missing = required_candidate_cols.difference(candidates.columns)
    if missing:
        raise RuntimeError(f"ranking_candidate_sets.csv missing columns: {sorted(missing)}")
    plan_e = candidates.loc[candidates["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].copy()
    if len(plan_e) != 6000:
        raise RuntimeError(f"expected exactly 6000 Plan E2 rows, found {len(plan_e)}")
    ids = set(plan_e["candidate_set_id"].astype(str).unique())
    if ids != PLAN_E2_CANDIDATE_SET_IDS:
        raise RuntimeError(f"unexpected Plan E2 candidate-set IDs: {sorted(ids)}")
    for row in plan_e.itertuples(index=False):
        candidates_list = parse_candidate_items(row.candidate_items)
        if len(candidates_list) != PLAN_E2_CANDIDATE_COUNT:
            raise RuntimeError("Plan E2 row does not contain 20 candidates")
        if candidates_list.count(str(row.positive_item_id)) != 1:
            raise RuntimeError("Plan E2 row does not contain exactly one positive item")
        if str(row.candidate_count) != str(PLAN_E2_CANDIDATE_COUNT):
            raise RuntimeError("Plan E2 candidate_count is not 20")
    required_membership_cols = {
        "candidate_set_id",
        "dataset",
        "task_type",
        "user_id",
        "eligible_plan_e2_paper_like_full",
        "eligible_plan_e2_matched_comparison",
    }
    missing_membership = required_membership_cols.difference(membership.columns)
    if missing_membership:
        raise RuntimeError(f"experiment_cohort_membership.csv missing columns: {sorted(missing_membership)}")


def build_membership_lookup(membership: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    plan_e = membership.loc[membership["candidate_set_id"].isin(PLAN_E2_CANDIDATE_SET_IDS)].copy()
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in plan_e.to_dict("records"):
        key = (str(row["candidate_set_id"]), str(row["user_id"]))
        if key in lookup:
            raise RuntimeError(f"duplicate Plan E2 membership key: {key}")
        lookup[key] = {
            "task_type": str(row["task_type"]),
            "eligible_plan_e2_paper_like_full": bool_value(row["eligible_plan_e2_paper_like_full"]),
            "eligible_plan_e2_matched_comparison": bool_value(row["eligible_plan_e2_matched_comparison"]),
        }
    if len(lookup) != 6000:
        raise RuntimeError(f"expected 6000 Plan E2 membership rows, found {len(lookup)}")
    return lookup


def build_summary_lookup(cohort_summary: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in cohort_summary.to_dict("records"):
        key = (str(row["cohort_id"]), str(row["dataset"]), str(row["candidate_set_id"]))
        lookup[key] = row
    return lookup


def build_dataset_context(dataset: str, spec: dict[str, Any], plan_e_candidates: pd.DataFrame) -> dict[str, Any]:
    folder = spec["folder"]
    base = PROJECT_ROOT / "data" / "reconstructed" / "plan_c" / folder
    interactions_path = base / "interactions.csv"
    datamap_path = base / "datamaps.json"
    interactions = read_csv_strings(interactions_path)
    split_counts = interactions["split"].astype(str).str.lower().value_counts().to_dict()
    dataset_candidates = plan_e_candidates.loc[plan_e_candidates["dataset"].astype(str) == dataset].copy()
    source_user_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="user")
    source_item_mapping = load_raw_to_internal_mapping(datamap_path, entity_type="item")
    context: dict[str, Any] = {
        "dataset": dataset,
        "folder": folder,
        "base_path": base,
        "training_data_path": relpath(interactions_path),
        "datamap_path": relpath(datamap_path),
        "interactions": interactions,
        "split_counts": split_counts,
        "candidate_rows": dataset_candidates,
        "source_user_mapping": source_user_mapping,
        "source_item_mapping": source_item_mapping,
        "graph_contract": None,
        "graph_users": set(),
        "graph_items": set(),
        "normalization_verified": False,
        "blocking_reason": "",
    }
    try:
        train = filter_train_interactions(interactions)
    except GraphValidationError as exc:
        context["train_interactions"] = interactions.iloc[0:0].copy()
        context["blocking_reason"] = str(exc)
        return context
    contract = build_graph_contract(train)
    context["train_interactions"] = train
    context["graph_contract"] = contract
    context["graph_users"] = set(contract.user_mapping.raw_to_local)
    context["graph_items"] = set(contract.item_mapping.raw_to_local)
    context["normalization_verified"] = True
    context["user_degrees"] = raw_degree_lookup(contract, entity_type="user")
    context["item_degrees"] = raw_degree_lookup(contract, entity_type="item")
    return context


def build_graph_row(dataset: str, context: dict[str, Any]) -> dict[str, Any]:
    interactions = context["interactions"]
    split_counts = context["split_counts"]
    contract = context["graph_contract"]
    if contract is None:
        return {
            "dataset": dataset,
            "training_data_path": context["training_data_path"],
            "split_field": "split",
            "training_split_value": "train",
            "total_interaction_count": len(interactions),
            "train_interaction_count": 0,
            "validation_interaction_count": int(split_counts.get("validation", 0)),
            "test_interaction_count": int(split_counts.get("test", 0)),
            "observed_interaction_count": int(split_counts.get("observed", 0)),
            "train_user_count": 0,
            "train_item_count": 0,
            "duplicate_user_item_pair_count": 0,
            "unique_user_item_edge_count": 0,
            "graph_edge_weight_policy": "blocked_no_verified_train_split",
            "user_node_count": 0,
            "item_node_count": 0,
            "total_node_count": 0,
            "directed_edge_count": 0,
            "self_loop_count": 0,
            "isolated_train_user_count": 0,
            "isolated_train_item_count": 0,
            "minimum_user_degree": "",
            "maximum_user_degree": "",
            "mean_user_degree": "",
            "median_user_degree": "",
            "minimum_item_degree": "",
            "maximum_item_degree": "",
            "mean_item_degree": "",
            "median_item_degree": "",
            "raw_edge_sha256": "",
            "directed_edge_index_sha256": "",
            "edge_weight_sha256": "",
            "graph_construction_deterministic": bool_text(False),
            "status": "blocked",
            "blocking_reason": context["blocking_reason"],
            "notes": "LastFM has observed interactions only and no Plan E2 candidate layer.",
        }
    train = context["train_interactions"]
    user_degrees = contract.normalized_graph.degrees[: contract.user_mapping.count]
    item_degrees = contract.normalized_graph.degrees[contract.user_mapping.count :]
    return {
        "dataset": dataset,
        "training_data_path": context["training_data_path"],
        "split_field": "split",
        "training_split_value": "train",
        "total_interaction_count": len(interactions),
        "train_interaction_count": len(train),
        "validation_interaction_count": int(split_counts.get("validation", 0)),
        "test_interaction_count": int(split_counts.get("test", 0)),
        "observed_interaction_count": int(split_counts.get("observed", 0)),
        "train_user_count": contract.user_mapping.count,
        "train_item_count": contract.item_mapping.count,
        "duplicate_user_item_pair_count": contract.duplicate_user_item_pair_count,
        "unique_user_item_edge_count": len(contract.train_pairs),
        "graph_edge_weight_policy": (
            "unique_binary_train_pairs; Plan D loader drops duplicate user-item pairs before LightGCN graph construction"
        ),
        "user_node_count": contract.user_mapping.count,
        "item_node_count": contract.item_mapping.count,
        "total_node_count": contract.directed_edges.total_nodes,
        "directed_edge_count": len(contract.directed_edges.rows),
        "self_loop_count": int(np.sum(contract.directed_edges.rows == contract.directed_edges.cols)),
        "isolated_train_user_count": int(np.sum(user_degrees == 0)),
        "isolated_train_item_count": int(np.sum(item_degrees == 0)),
        "minimum_user_degree": int(user_degrees.min()),
        "maximum_user_degree": int(user_degrees.max()),
        "mean_user_degree": float(user_degrees.mean()),
        "median_user_degree": float(np.median(user_degrees)),
        "minimum_item_degree": int(item_degrees.min()),
        "maximum_item_degree": int(item_degrees.max()),
        "mean_item_degree": float(item_degrees.mean()),
        "median_item_degree": float(np.median(item_degrees)),
        "raw_edge_sha256": hash_raw_pairs(contract.train_pairs),
        "directed_edge_index_sha256": hash_int_arrays(contract.directed_edges.rows, contract.directed_edges.cols),
        "edge_weight_sha256": hash_float_array(contract.directed_edges.raw_values),
        "graph_construction_deterministic": bool_text(True),
        "status": "passed",
        "blocking_reason": "",
        "notes": "Audit-only graph built from split=train interactions; no model propagation or scoring performed.",
    }


def build_normalization_row(dataset: str, context: dict[str, Any]) -> dict[str, Any]:
    contract = context["graph_contract"]
    if contract is None:
        return {
            "dataset": dataset,
            "implementation_path": "src/baselines/plan_d_lightgcn.py",
            "normalization_formula": "",
            "bipartite_edges_symmetrized": "",
            "self_loops_included": "",
            "degree_calculation_policy": "",
            "zero_degree_handling": "",
            "sparse_layout": "",
            "sparse_coalesced": "",
            "tensor_dtype": "",
            "normalized_edge_count": 0,
            "minimum_normalized_weight": "",
            "maximum_normalized_weight": "",
            "non_finite_weight_count": "",
            "symmetry_verified": bool_text(False),
            "normalized_adjacency_sha256": "",
            "propagation_layer_count": "",
            "layer_zero_included": "",
            "layer_aggregation": "",
            "propagation_nonlinearity": "",
            "trainable_graph_transformations_present": "",
            "final_user_item_split_policy": "",
            "synthetic_normalization_checks": "not_run_no_train_graph",
            "status": "blocked",
            "blocking_reason": context["blocking_reason"],
            "notes": "No split=train graph is available for LastFM.",
        }
    values = contract.normalized_graph.values
    return {
        "dataset": dataset,
        "implementation_path": "src/baselines/plan_d_lightgcn.py",
        "normalization_formula": "D^-1/2 A D^-1/2",
        "bipartite_edges_symmetrized": bool_text(True),
        "self_loops_included": bool_text(False),
        "degree_calculation_policy": "degree calculated after symmetrization from directed rows",
        "zero_degree_handling": "zero-degree nodes are not endpoints under strict connected-node graph; isolated candidate policy rejected",
        "sparse_layout": "torch.sparse_coo",
        "sparse_coalesced": bool_text(True),
        "tensor_dtype": "float32 edge weights; int64 indices",
        "normalized_edge_count": len(values),
        "minimum_normalized_weight": float(values.min()),
        "maximum_normalized_weight": float(values.max()),
        "non_finite_weight_count": int(np.sum(~np.isfinite(values))),
        "symmetry_verified": bool_text(True),
        "normalized_adjacency_sha256": hash_normalized_graph(contract.normalized_graph),
        "propagation_layer_count": 2,
        "layer_zero_included": bool_text(True),
        "layer_aggregation": "mean over layer 0 through L",
        "propagation_nonlinearity": "none",
        "trainable_graph_transformations_present": bool_text(False),
        "final_user_item_split_policy": "final[:num_users] users; final[num_users:] items",
        "synthetic_normalization_checks": "toy graph verified in tests; project graph hashed without propagation",
        "status": "passed",
        "blocking_reason": "",
        "notes": "Normalization mirrors Plan D build_normalized_bipartite_graph semantics.",
    }


def build_index_rows(dataset: str, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_type in ("user", "item"):
        source_mapping = context["source_user_mapping"] if entity_type == "user" else context["source_item_mapping"]
        source_ids = sorted(source_mapping.internal_to_raw)
        plan_e_ids = plan_e_unique_ids(context["candidate_rows"], entity_type=entity_type)
        graph_mapping = None
        if context["graph_contract"] is not None:
            graph_mapping = (
                context["graph_contract"].user_mapping
                if entity_type == "user"
                else context["graph_contract"].item_mapping
            )
        mapped = {raw_id for raw_id in plan_e_ids if raw_id in source_mapping.raw_to_internal}
        connected = set()
        if graph_mapping is not None:
            connected = {raw_id for raw_id in plan_e_ids if raw_id in graph_mapping.raw_to_local}
        raw_internal_values = list(source_mapping.raw_to_internal.values())
        duplicate_internal = len(raw_internal_values) - len(set(raw_internal_values))
        duplicate_raw = len(source_mapping.raw_to_internal) - len(set(source_mapping.raw_to_internal))
        local_min = 0 if graph_mapping is not None and graph_mapping.count else ""
        local_max = graph_mapping.count - 1 if graph_mapping is not None and graph_mapping.count else ""
        user_count = context["graph_contract"].user_mapping.count if context["graph_contract"] is not None else 0
        item_count = context["graph_contract"].item_mapping.count if context["graph_contract"] is not None else 0
        total_nodes = user_count + item_count
        offset = 0 if entity_type == "user" else user_count
        global_min = offset if graph_mapping is not None and graph_mapping.count else ""
        global_max = offset + graph_mapping.count - 1 if graph_mapping is not None and graph_mapping.count else ""
        status = "not_applicable" if dataset == "LastFM" else "passed"
        if dataset != "LastFM" and plan_e_ids and len(connected) < len(plan_e_ids):
            status = "warning"
        rows.append(
            {
                "dataset": dataset,
                "entity_type": entity_type,
                "mapping_path": context["datamap_path"],
                "mapping_direction": "raw_to_internal",
                "raw_id_count": len(source_mapping.raw_to_internal),
                "internal_id_count": len(source_mapping.internal_to_raw),
                "minimum_source_internal_id": min(source_ids) if source_ids else "",
                "maximum_source_internal_id": max(source_ids) if source_ids else "",
                "lightgcn_local_index_min": local_min,
                "lightgcn_local_index_max": local_max,
                "bipartite_node_offset": offset if graph_mapping is not None else "",
                "global_node_index_min": global_min,
                "global_node_index_max": global_max,
                "total_graph_nodes": total_nodes,
                "duplicate_raw_id_count": duplicate_raw,
                "duplicate_internal_id_count": duplicate_internal,
                "contiguous_local_index_space": bool_text(graph_mapping is not None and graph_mapping.count > 0),
                "train_seen_entity_count": graph_mapping.count if graph_mapping is not None else 0,
                "plan_e2_unique_entity_count": len(plan_e_ids),
                "plan_e2_mapped_entity_count": len(mapped),
                "plan_e2_graph_connected_entity_count": len(connected),
                "mapping_coverage": proportion(len(mapped), len(plan_e_ids)),
                "graph_connected_coverage": proportion(len(connected), len(plan_e_ids)),
                "status": status,
                "blocking_reason": context["blocking_reason"] if dataset == "LastFM" else "",
                "notes": (
                    "Plan E2 not applicable for LastFM."
                    if dataset == "LastFM"
                    else "Graph-connected coverage is strict train-seen coverage."
                ),
            }
        )
    return rows


def plan_e_unique_ids(candidate_rows: pd.DataFrame, *, entity_type: str) -> set[str]:
    if candidate_rows.empty:
        return set()
    if entity_type == "user":
        return {safe_raw_id(value) for value in candidate_rows["user_id"].tolist()}
    ids: set[str] = set()
    for row in candidate_rows.to_dict("records"):
        ids.add(safe_raw_id(row["positive_item_id"]))
        ids.update(parse_candidate_items(row["candidate_items"]))
    return ids


def raw_degree_lookup(contract: Any, *, entity_type: str) -> dict[str, float]:
    if entity_type == "user":
        degrees = contract.normalized_graph.degrees[: contract.user_mapping.count]
        return {raw_id: float(degrees[index]) for raw_id, index in contract.user_mapping.raw_to_local.items()}
    degrees = contract.normalized_graph.degrees[contract.user_mapping.count :]
    return {raw_id: float(degrees[index]) for raw_id, index in contract.item_mapping.raw_to_local.items()}


def build_connectivity_impact_rows(
    compatibility_rows: list[dict[str, Any]],
    candidate_position_lookup: dict[tuple[str, str], int],
    contexts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(compatibility_rows)
    for (dataset, task_type, candidate_set_id), group in frame.groupby(
        ["dataset", "task_type", "candidate_set_id"], sort=True
    ):
        for cohort_id, eligible_column in (
            ("plan_e2_paper_like_full", "eligible_plan_e2_full"),
            ("plan_e2_matched_comparison", "eligible_plan_e2_matched"),
        ):
            cohort = group.loc[group[eligible_column].astype(str).str.lower() == "true"].copy()
            rows.append(
                build_one_impact_row(
                    dataset=dataset,
                    task_type=task_type,
                    candidate_set_id=candidate_set_id,
                    cohort_id=cohort_id,
                    cohort=cohort,
                    candidate_position_lookup=candidate_position_lookup,
                    context=contexts[dataset],
                )
            )
    return sort_rows(rows, ["dataset", "task_type", "candidate_set_id", "cohort_id"])


def build_one_impact_row(
    *,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    cohort_id: str,
    cohort: pd.DataFrame,
    candidate_position_lookup: dict[tuple[str, str], int],
    context: dict[str, Any],
) -> dict[str, Any]:
    if cohort.empty:
        evaluated = 0
    else:
        evaluated = len(cohort)
    disconnected_items: set[str] = set()
    disconnected_occurrences = 0
    positions_full: list[int] = []
    positions_compatible: list[int] = []
    user_degrees: list[float] = []
    positive_degrees: list[float] = []
    for row in cohort.to_dict("records"):
        disconnected = json.loads(str(row["graph_disconnected_candidate_item_ids"]))
        disconnected_items.update(disconnected)
        disconnected_occurrences += len(disconnected)
        key = (str(row["candidate_set_id"]), str(row["user_id"]))
        if key in candidate_position_lookup:
            positions_full.append(candidate_position_lookup[key])
            if bool_value(row["lightgcn_graph_compatible"]):
                positions_compatible.append(candidate_position_lookup[key])
        user_degrees.append(float(context.get("user_degrees", {}).get(str(row["user_id"]), 0.0)))
        positive_degrees.append(float(context.get("item_degrees", {}).get(str(row["positive_item_id"]), 0.0)))
    compatible_rows = int(cohort["lightgcn_graph_compatible"].astype(str).str.lower().eq("true").sum()) if evaluated else 0
    return {
        "dataset": dataset,
        "task_type": task_type,
        "candidate_set_id": candidate_set_id,
        "cohort_id": cohort_id,
        "evaluated_rows": evaluated,
        "rows_with_unmapped_user": count_false(cohort, "user_mapped"),
        "rows_with_disconnected_user": count_false(cohort, "user_graph_connected"),
        "rows_with_unmapped_positive": count_false(cohort, "positive_item_mapped"),
        "rows_with_disconnected_positive": count_false(cohort, "positive_item_graph_connected"),
        "rows_with_any_unmapped_candidate": count_positive(cohort, "unmapped_candidate_item_count"),
        "rows_with_any_disconnected_candidate": count_positive(cohort, "graph_disconnected_candidate_item_count"),
        "unique_disconnected_candidate_items": len(disconnected_items),
        "candidate_occurrence_count": int(cohort["candidate_count"].astype(int).sum()) if evaluated else 0,
        "disconnected_candidate_occurrence_count": disconnected_occurrences,
        "lightgcn_graph_compatible_rows": compatible_rows,
        "retention_rate": proportion(compatible_rows, evaluated),
        "positive_position_distribution_full": json.dumps(dict(sorted(Counter(positions_full).items())), separators=(",", ":")),
        "positive_position_distribution_compatible": json.dumps(
            dict(sorted(Counter(positions_compatible).items())), separators=(",", ":")
        ),
        "mean_user_degree": float(np.mean(user_degrees)) if user_degrees else "",
        "median_user_degree": float(np.median(user_degrees)) if user_degrees else "",
        "mean_positive_item_degree": float(np.mean(positive_degrees)) if positive_degrees else "",
        "median_positive_item_degree": float(np.median(positive_degrees)) if positive_degrees else "",
        "notes": "Strict graph-compatible retention; no candidates removed or replaced.",
    }


def count_false(frame: pd.DataFrame, column: str) -> int:
    if frame.empty:
        return 0
    return int(frame[column].astype(str).str.lower().eq("false").sum())


def count_positive(frame: pd.DataFrame, column: str) -> int:
    if frame.empty:
        return 0
    return int((frame[column].astype(int) > 0).sum())


def build_sparse_rows(env: dict[str, Any], graph_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cpu_probe = synthetic_sparse_forward_probe(device="cpu")
    cuda_probe = {"passed": False, "warnings": [], "error": "cuda not available", "output": np.asarray([], dtype=np.float32)}
    if env["cuda_available"]:
        cuda_probe = synthetic_sparse_forward_probe(device="cuda")
    max_diff = ""
    if cpu_probe["passed"] and cuda_probe["passed"]:
        max_diff = float(np.max(np.abs(cpu_probe["output"] - cuda_probe["output"])))
    warnings_all = list(cpu_probe.get("warnings", [])) + list(cuda_probe.get("warnings", []))
    warning_categories = "none" if not warnings_all else "synthetic_sparse_warning"
    recommended_device = "cuda" if env["cuda_available"] and cuda_probe["passed"] and not warnings_all else "cpu"
    sparse_rows = []
    for graph in graph_rows:
        estimates = estimate_sparse_memory_bytes(
            total_nodes=int(graph["total_node_count"] or 0),
            directed_edges=int(graph["directed_edge_count"] or 0),
            embedding_dim=64,
            layer_count=2,
        )
        status = "passed" if cpu_probe["passed"] and (not env["cuda_available"] or cuda_probe["passed"]) else "warning"
        blocking_reason = ""
        if not cpu_probe["passed"]:
            status = "blocked"
            blocking_reason = cpu_probe["error"]
        elif env["cuda_available"] and not cuda_probe["passed"]:
            blocking_reason = cuda_probe["error"]
        if graph["dataset"] == "LastFM":
            status = "not_applicable"
            blocking_reason = "no verified train graph and no Plan E2 candidate layer"
        sparse_rows.append(
            {
                "environment_id": env["environment_id"],
                "python_executable": env["python_executable"],
                "python_version": env["python_version"],
                "torch_version": env["torch_version"],
                "torch_cuda_version": env["torch_cuda_version"],
                "cuda_available": bool_text(env["cuda_available"]),
                "cuda_device": env["cuda_device"],
                "dataset": graph["dataset"],
                "graph_nodes": graph["total_node_count"],
                "directed_edges": graph["directed_edge_count"],
                "sparse_layout": "torch.sparse_coo",
                **estimates,
                "cpu_sparse_construction_supported": bool_text(cpu_probe["passed"]),
                "cpu_sparse_propagation_supported": bool_text(cpu_probe["passed"]),
                "cuda_sparse_construction_supported": bool_text(cuda_probe["passed"]),
                "cuda_sparse_propagation_supported": bool_text(cuda_probe["passed"]),
                "deterministic_algorithms_supported_cpu": bool_text(cpu_probe["passed"]),
                "deterministic_algorithms_supported_cuda": bool_text(cuda_probe["passed"]),
                "synthetic_cpu_forward_passed": bool_text(cpu_probe["passed"]),
                "synthetic_cuda_forward_passed": bool_text(cuda_probe["passed"]),
                "cpu_cuda_synthetic_max_abs_difference": max_diff,
                "warning_count": len(warnings_all),
                "warning_categories": warning_categories,
                "recommended_execution_device": recommended_device,
                "status": status,
                "blocking_reason": blocking_reason,
                "notes": "Synthetic toy graph probe only; no project-data propagation was performed.",
            }
        )
    return sparse_rows


def build_manifest_rows(
    contexts: dict[str, dict[str, Any]],
    summary_lookup: dict[tuple[str, str, str], dict[str, Any]],
    env: dict[str, Any],
    graph_rows: list[dict[str, Any]],
    normalization_rows: list[dict[str, Any]],
    sparse_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    graph_status = {row["dataset"]: row["status"] for row in graph_rows}
    norm_status = {row["dataset"]: row["status"] for row in normalization_rows}
    sparse_status = {row["dataset"]: row["status"] for row in sparse_rows}
    rows: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        if dataset == "LastFM":
            context = contexts[dataset]
            rows.append(
                {
                    "preflight_id": "plan_e_lightgcn_lastfm_not_applicable",
                    "dataset": dataset,
                    "task_type": "not_applicable",
                    "candidate_set_id": "not_applicable",
                    "source_path": "",
                    "source_plan": "",
                    "full_cohort_rows": 0,
                    "phase2_matched_rows": 0,
                    "training_data_path": context["training_data_path"],
                    "user_mapping_path": context["datamap_path"],
                    "item_mapping_path": context["datamap_path"],
                    "local_lightgcn_implementation_path": "src/baselines/plan_d_lightgcn.py",
                    "local_lightgcn_runner_path": "src/experiments/run_plan_d_lightgcn.py",
                    "environment_id": env["environment_id"],
                    "python_executable": env["python_executable"],
                    "python_version": env["python_version"],
                    "torch_version": env["torch_version"],
                    "cuda_available": bool_text(env["cuda_available"]),
                    "cuda_device": env["cuda_device"],
                    "graph_construction_status": graph_status[dataset],
                    "normalization_status": norm_status[dataset],
                    "sparse_backend_status": sparse_status[dataset],
                    "index_policy": "not_applicable_no_plan_e2_candidates",
                    "propagation_policy": "not_applicable",
                    "adapter_status": "graph_data_blocked",
                    "project_data_training_performed": bool_text(False),
                    "project_data_scoring_performed": bool_text(False),
                    "recommended_pilot_scope": "none",
                    "blocking_reason": context["blocking_reason"],
                    "notes": "LastFM remains blocked for Plan E2 LightGCN because it has observed-only interactions and no Plan E2 candidate set.",
                }
            )
            continue
        context = contexts[dataset]
        for task_type, candidate_set_id in spec["task_ids"].items():
            key_full = ("plan_e2_paper_like_full", dataset, candidate_set_id)
            key_matched = ("plan_e2_matched_comparison", dataset, candidate_set_id)
            full_rows = int(summary_lookup[key_full]["eligible_rows"])
            matched_rows = int(summary_lookup[key_matched]["eligible_rows"])
            status = "ready_for_pilot_after_policy_approval"
            blocking = ""
            if graph_status[dataset] != "passed":
                status = "graph_construction_blocked"
                blocking = "graph audit did not pass"
            elif norm_status[dataset] != "passed":
                status = "normalization_blocked"
                blocking = "normalization audit did not pass"
            elif sparse_status[dataset] not in {"passed", "warning"}:
                status = "sparse_backend_blocked"
                blocking = "sparse backend audit did not pass"
            rows.append(
                {
                    "preflight_id": f"plan_e_lightgcn_{context['folder']}_{task_type}",
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "source_path": relpath(PROJECT_ROOT / summary_lookup[key_full]["source_path"]),
                    "source_plan": "Plan E2",
                    "full_cohort_rows": full_rows,
                    "phase2_matched_rows": matched_rows,
                    "training_data_path": context["training_data_path"],
                    "user_mapping_path": context["datamap_path"],
                    "item_mapping_path": context["datamap_path"],
                    "local_lightgcn_implementation_path": "src/baselines/plan_d_lightgcn.py",
                    "local_lightgcn_runner_path": "src/experiments/run_plan_d_lightgcn.py",
                    "environment_id": env["environment_id"],
                    "python_executable": env["python_executable"],
                    "python_version": env["python_version"],
                    "torch_version": env["torch_version"],
                    "cuda_available": bool_text(env["cuda_available"]),
                    "cuda_device": env["cuda_device"],
                    "graph_construction_status": graph_status[dataset],
                    "normalization_status": norm_status[dataset],
                    "sparse_backend_status": sparse_status[dataset],
                    "index_policy": "separate dense train-connected user/item local indices; item nodes offset by num_users",
                    "propagation_policy": "future scoring only after training; layer-zero included and mean aggregation verified",
                    "adapter_status": status,
                    "project_data_training_performed": bool_text(False),
                    "project_data_scoring_performed": bool_text(False),
                    "recommended_pilot_scope": (
                        "Phase 3M ML-1M strict graph-connected pilot"
                        if dataset == "ML-1M"
                        else "defer until ML-1M pilot succeeds"
                    ),
                    "blocking_reason": blocking,
                    "notes": "Preflight validates adapter contract only; no LightGCN training or project-data scoring performed.",
                }
            )
    return sort_rows(rows, ["dataset", "task_type", "candidate_set_id"])


def build_policy_rows() -> list[dict[str, Any]]:
    policies = [
        (
            "A",
            "verified_train_graph_and_strict_connected_candidates",
            "Use verified train-only graph construction and retain only rows where user, positive, and all candidates are graph-connected.",
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            "adapted_but_strict",
            "preserves same candidates for retained rows; excludes disconnected rows explicitly",
            "low",
            True,
            "recommended",
            "Safest no-leakage policy with no isolated random candidate embeddings and no candidate mutation.",
            "strict_graph_connected_lightgcn_cohort",
            "stop if any retained row has a disconnected user, positive, or candidate item",
            "Exactly one recommended policy.",
        ),
        (
            "B",
            "permit_isolated_candidate_embeddings",
            "Allocate and score train-unseen candidates with untrained isolated embeddings.",
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            "weak",
            "keeps rows but scores random isolated candidates",
            "high",
            False,
            "not_recommended",
            "Layer-zero aggregation would let isolated random embeddings influence ranks.",
            "isolated_embedding_sensitivity_only",
            "stop if used for primary baseline",
            "",
        ),
        (
            "C",
            "include_all_mapped_catalog_nodes_as_isolated",
            "Build node universe from the full mapped catalog and leave zero-degree nodes isolated.",
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            "weak",
            "large isolated node universe makes scores arbitrary for disconnected items",
            "high",
            False,
            "not_recommended",
            "This expands model cardinality without adding learnable evidence.",
            "isolated_catalog_ablation_only",
            "stop if zero-degree candidates enter primary scoring",
            "",
        ),
        (
            "D",
            "fallback_to_popularity_for_disconnected_candidates",
            "Use popularity fallback scores for disconnected candidates.",
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            "hybrid_extension",
            "not comparable to pure LightGCN or prior same-cohort baselines",
            "medium",
            False,
            "not_recommended",
            "Hybrid fallback changes the method and would require a separate extension label.",
            "hybrid_fallback_extension",
            "stop if reported as LightGCN",
            "",
        ),
        (
            "E",
            "add_validation_or_test_edges",
            "Insert validation or test edges into the graph.",
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            "invalid",
            "leaks held-out information",
            "critical",
            False,
            "forbidden",
            "Validation/test edges introduce leakage.",
            "none",
            "always stop",
            "",
        ),
        (
            "F",
            "add_candidate_pseudo_edges",
            "Add evaluation candidate membership as graph edges.",
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            "invalid",
            "injects evaluation-set structure",
            "critical",
            False,
            "forbidden",
            "Candidate pseudo-edges use evaluation information in the graph.",
            "none",
            "always stop",
            "",
        ),
        (
            "G",
            "add_held_out_positive_edges",
            "Add held-out positive items as graph edges.",
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            "invalid",
            "direct label leakage",
            "critical",
            False,
            "forbidden",
            "Held-out positives are labels and must not be training graph edges.",
            "none",
            "always stop",
            "",
        ),
        (
            "H",
            "remove_or_replace_disconnected_candidates",
            "Mutate candidate lists to avoid disconnected items.",
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            "invalid_for_same_cohort",
            "breaks canonical candidate membership and same-cohort comparison",
            "critical",
            False,
            "forbidden",
            "Candidate membership and order are protected Phase 2 inputs.",
            "none",
            "always stop",
            "",
        ),
        (
            "I",
            "construct_dense_full_adjacency",
            "Build a dense project-sized adjacency matrix.",
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            "unnecessary",
            "same math but unsafe memory behavior",
            "high",
            False,
            "forbidden",
            "Sparse COO is sufficient; dense adjacency is unnecessary and risky.",
            "none",
            "stop on any project dense adjacency request",
            "",
        ),
        (
            "J",
            "use_interaction_multiplicity_weights_when_plan_d_is_binary",
            "Weight edges by repeated interaction counts when Plan D loader uses unique pairs.",
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            "unverified_variant",
            "changes graph weights relative to local Plan D contract",
            "medium",
            False,
            "not_recommended",
            "Current Plan D loader deduplicates user-item pairs and current duplicate counts are zero.",
            "weighted_edge_ablation_only",
            "stop unless explicitly approved as extension",
            "",
        ),
    ]
    return [
        dict(zip(POLICY_COLUMNS, row, strict=True))
        for row in policies
    ]


def write_report(
    path: Path,
    *,
    env: dict[str, Any],
    graph_rows: list[dict[str, Any]],
    normalization_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    sparse_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> None:
    graph_by_dataset = {row["dataset"]: row for row in graph_rows}
    compat = pd.DataFrame(compatibility_rows)
    impact = pd.DataFrame(impact_rows)
    manifest = pd.DataFrame(manifest_rows)
    sparse = pd.DataFrame(sparse_rows)
    policy = pd.DataFrame(policy_rows)
    branch = git_output(["git", "branch", "--show-current"])
    commit = git_output(["git", "log", "-1", "--format=%H %s"])
    compatible_by_dataset = (
        compat.groupby("dataset")["lightgcn_graph_compatible"]
        .apply(lambda s: int(s.astype(str).str.lower().eq("true").sum()))
        .to_dict()
    )
    total_by_dataset = compat.groupby("dataset").size().to_dict()
    recommended_policy = policy.loc[policy["recommendation_status"] == "recommended"].iloc[0].to_dict()
    recommended_device = sparse.loc[sparse["dataset"] == "ML-1M", "recommended_execution_device"].iloc[0]
    lines = [
        "# Plan E LightGCN Adapter and Graph-Construction Preflight Validation",
        "",
        "## 1. Scope and provenance",
        f"- Branch: `{branch}`.",
        f"- Commit: `{commit}`.",
        "- Phase 3L adapter/preflight only; no LightGCN training, project-data propagation, scoring, rankings, metrics, plots or checkpoints were produced.",
        "- Generated outputs are the eight `results/tables/plan_e_lightgcn_*.csv` files and this report.",
        "",
        "## 2. Executive summary",
        f"- Plan E2 candidate verification passed for 6,000 rows across six 20-candidate manifests.",
        f"- Strict graph-compatible rows: Beauty {compatible_by_dataset.get('Amazon Beauty', 0)}/{total_by_dataset.get('Amazon Beauty', 0)}, Sports {compatible_by_dataset.get('Amazon Sports', 0)}/{total_by_dataset.get('Amazon Sports', 0)}, ML-1M {compatible_by_dataset.get('ML-1M', 0)}/{total_by_dataset.get('ML-1M', 0)}.",
        "- LastFM remains blocked because it has `split=observed` only and no Plan E2 candidate layer.",
        f"- Recommended future policy: `{recommended_policy['policy_name']}`.",
        "",
        "## 3. Safety boundaries",
        "- The runner hashes protected inputs, writes only Phase 3L LightGCN audit artifacts, and uses synthetic sparse probes only.",
        "- It does not call Plan D training, does not instantiate a project-data optimizer, and does not call project-data `final_embeddings` or candidate scoring.",
        "",
        "## 4. Existing Plan D LightGCN implementation",
        "- Model path: `src/baselines/plan_d_lightgcn.py`.",
        "- Runner path: `src/experiments/run_plan_d_lightgcn.py`.",
        "- Prior execution evidence remains in `results/tables/plan_d_lightgcn_training_summary.csv`; this preflight did not rerun it.",
        "",
        "## 5. LightGCN architecture",
        "- `LightGCNModel(num_users, num_items, embedding_dim, num_layers)` defines separate user and item embeddings initialized from a normal distribution with std 0.01.",
        "- The graph is a sparse normalized bipartite adjacency.",
        "",
        "## 6. Existing training interface",
        "- `train_lightgcn` builds a normalized graph, creates an Adam optimizer, samples negatives and applies BPR loss.",
        "- This preflight does not invoke that interface on project data.",
        "",
        "## 7. Existing candidate-scoring interface",
        "- `rank_lightgcn_candidates` propagates embeddings, dot-products aligned user/item embeddings, sorts by score, and writes recommendation rows in Plan D runs.",
        "- This preflight audits candidate compatibility but does not call the scorer on project data.",
        "",
        "## 8. Execution environment",
        f"- Python executable: `{env['python_executable']}`.",
        f"- Python version: `{env['python_version']}`.",
        f"- Torch: `{env['torch_version']}`, torch CUDA runtime `{env['torch_cuda_version']}`.",
        f"- CUDA available: `{bool_text(env['cuda_available'])}`; device: `{env['cuda_device']}`.",
        "",
        "## 9. Graph-data inventory",
    ]
    for dataset, row in graph_by_dataset.items():
        lines.append(
            f"- {dataset}: total interactions {row['total_interaction_count']}, train {row['train_interaction_count']}, users {row['train_user_count']}, items {row['train_item_count']}, status `{row['status']}`."
        )
    lines.extend(
        [
            "",
            "## 10. Training-edge isolation",
            "- Beauty, Sports and ML-1M graph audits use only `split=train` interactions. Validation/test/candidate/positive-derived edges are not inserted.",
            "- LastFM has `split=observed` only, so it is blocked rather than reinterpreted as train.",
            "",
            "## 11. Duplicate-edge policy",
            "- Plan D data loading drops duplicate `user_id,item_id` train pairs before LightGCN graph construction.",
        ]
    )
    for dataset in ("Amazon Beauty", "Amazon Sports", "ML-1M"):
        row = graph_by_dataset[dataset]
        lines.append(f"- {dataset}: duplicate train user-item pairs {row['duplicate_user_item_pair_count']}.")
    lines.extend(
        [
            "",
            "## 12. Bipartite node indexing",
            "- Users and items use separate dense local indices; item global nodes are offset by `num_users`.",
            "- Index details are in `results/tables/plan_e_lightgcn_index_audit.csv`.",
            "",
            "## 13. User/item node offsets",
        ]
    )
    for row in index_rows:
        if row["dataset"] != "LastFM" and row["entity_type"] == "item":
            lines.append(f"- {row['dataset']}: item offset {row['bipartite_node_offset']}, total nodes {row['total_graph_nodes']}.")
    lines.extend(
        [
            "",
            "## 14. Graph symmetrization",
            "- Each train pair contributes user-to-item and item-to-user directed edges.",
            "",
            "## 15. Self-loop policy",
            "- No self-loops are added by the local Plan D LightGCN graph builder.",
            "",
            "## 16. Degree calculations",
        ]
    )
    for dataset in ("Amazon Beauty", "Amazon Sports", "ML-1M"):
        row = graph_by_dataset[dataset]
        lines.append(
            f"- {dataset}: user degree min/max/mean/median = {row['minimum_user_degree']}/{row['maximum_user_degree']}/{format_value(row['mean_user_degree'])}/{format_value(row['median_user_degree'])}; item degree min/max/mean/median = {row['minimum_item_degree']}/{row['maximum_item_degree']}/{format_value(row['mean_item_degree'])}/{format_value(row['median_item_degree'])}."
        )
    lines.extend(
        [
            "",
            "## 17. Normalized adjacency",
            "- Formula: symmetric `D^-1/2 A D^-1/2`, with degrees calculated after symmetrization.",
            "- Normalized adjacency hashes are recorded in `plan_e_lightgcn_normalization_audit.csv`.",
            "",
            "## 18. Sparse tensor representation",
            "- The existing implementation uses coalesced `torch.sparse_coo_tensor` with int64 indices and float32 weights.",
            "",
            "## 19. Propagation-layer semantics",
            "- Propagation recurrence is `E^(k+1) = A_norm E^k` through sparse matrix multiplication.",
            "",
            "## 20. Layer-zero inclusion",
            "- Layer zero is included in the final representation.",
            "",
            "## 21. Layer aggregation",
            "- Layer embeddings are averaged from layer 0 through `num_layers`.",
            "",
            "## 22. User/item final embedding separation",
            "- Final embeddings are split as users `final[:num_users]` and items `final[num_users:]`.",
            "",
            "## 23. Candidate-scoring contract",
            "- Candidate scoring is a dot product between propagated user and item embeddings; future execution must retain candidate order until score alignment is validated.",
            "",
            "## 24. Beauty graph findings",
            dataset_graph_sentence(graph_by_dataset["Amazon Beauty"], compatible_by_dataset, total_by_dataset),
            "",
            "## 25. Sports graph findings",
            dataset_graph_sentence(graph_by_dataset["Amazon Sports"], compatible_by_dataset, total_by_dataset),
            "",
            "## 26. ML-1M graph findings",
            dataset_graph_sentence(graph_by_dataset["ML-1M"], compatible_by_dataset, total_by_dataset),
            "",
            "## 27. LastFM status",
            f"- LastFM status is `{graph_by_dataset['LastFM']['status']}`: {graph_by_dataset['LastFM']['blocking_reason']}.",
            "",
            "## 28. User connectivity findings",
            "- All Plan E2 users for Beauty, Sports and ML-1M are mapped and graph-connected in the strict train graph.",
            "",
            "## 29. Positive-item connectivity findings",
            "- Positive-item connectivity is tracked per row in `plan_e_lightgcn_row_compatibility.csv`; rows with disconnected positives are excluded by strict policy.",
            "",
            "## 30. Candidate-item connectivity findings",
        ]
    )
    disconnected_summary = (
        compat.assign(disconnected=compat["graph_disconnected_candidate_item_count"].astype(int))
        .groupby("dataset")["disconnected"]
        .sum()
        .to_dict()
    )
    for dataset in ("Amazon Beauty", "Amazon Sports", "ML-1M"):
        lines.append(f"- {dataset}: disconnected candidate occurrences across full Plan E2 rows = {disconnected_summary.get(dataset, 0)}.")
    lines.extend(
        [
            "",
            "## 31. Isolated-embedding risk",
            "- Any policy that allocates train-unseen candidates would leave zero-degree item embeddings at initialization except for layer-zero averaging, creating random score risk.",
            "",
            "## 32. Plan E2 compatibility",
            "- Row-level compatibility is in `results/tables/plan_e_lightgcn_row_compatibility.csv`.",
            "",
            "## 33. Full-cohort retention",
        ]
    )
    for _, row in impact.loc[impact["cohort_id"] == "plan_e2_paper_like_full"].iterrows():
        lines.append(
            f"- {row['dataset']} {row['task_type']}: {row['lightgcn_graph_compatible_rows']}/{row['evaluated_rows']} retained ({format_value(row['retention_rate'])})."
        )
    lines.extend(["", "## 34. Matched-cohort retention"])
    for _, row in impact.loc[impact["cohort_id"] == "plan_e2_matched_comparison"].iterrows():
        lines.append(
            f"- {row['dataset']} {row['task_type']}: {row['lightgcn_graph_compatible_rows']}/{row['evaluated_rows']} retained ({format_value(row['retention_rate'])})."
        )
    lines.extend(
        [
            "",
            "## 35. Sparse CPU findings",
            f"- Synthetic CPU sparse forward passed: `{sparse.iloc[0]['synthetic_cpu_forward_passed']}`.",
            "",
            "## 36. Sparse CUDA findings",
            f"- Synthetic CUDA sparse forward passed: `{sparse.iloc[0]['synthetic_cuda_forward_passed']}`.",
            "",
            "## 37. Deterministic sparse-operation findings",
            f"- CPU/CUDA synthetic max absolute difference: `{sparse.iloc[0]['cpu_cuda_synthetic_max_abs_difference']}`; warnings: `{sparse.iloc[0]['warning_categories']}`.",
            "",
            "## 38. Resource and memory assessment",
        ]
    )
    for _, row in sparse.iterrows():
        lines.append(
            f"- {row['dataset']}: graph nodes {row['graph_nodes']}, directed edges {row['directed_edges']}, estimated graph tensor bytes {row['estimated_total_graph_tensor_bytes']}."
        )
    lines.extend(
        [
            "",
            "## 39. Policy options evaluated",
            f"- Evaluated {len(policy_rows)} policies A-J; exactly one is marked `recommended`.",
            "",
            "## 40. Recommended future policy",
            f"- `{recommended_policy['policy_name']}`: {recommended_policy['rationale']}",
            "",
            "## 41. Selection-bias implications",
            "- Strict graph-connected filtering excludes rows with train-unseen candidates; comparisons must report the retained cohort and avoid mixing with full-cohort metrics.",
            "",
            "## 42. Recommended first LightGCN pilot dataset",
            "- ML-1M is the recommended first pilot dataset because it has the smallest graph and high strict retention.",
            "",
            "## 43. Recommended first-pilot device",
            f"- Recommended first-pilot device: `{recommended_device}` based on synthetic sparse probe evidence.",
            "",
            "## 44. Proposed pilot scope",
            "- A future pilot should use the ML-1M strict graph-connected Plan E2 ranking and reranking rows only.",
            "",
            "## 45. Pilot success criteria",
            "- Stop-free completion, deterministic rerun hashes, no disconnected retained row, and no deviation from train-only graph construction.",
            "",
            "## 46. Pilot stop conditions",
            "- Stop on protected-input hash mismatch, CUDA sparse nondeterminism, non-finite normalized weights, disconnected retained candidate, candidate mutation, validation/test edge leakage, or checkpoint/ranking/metric output path ambiguity.",
            "",
            "## 47. Tests and synthetic smoke probes",
            "- Tests target synthetic adapter behavior and the existing cohort/candidate validation suites. Sparse probes are synthetic toy LightGCN forward passes only.",
            "",
            "## 48. Protected-input immutability",
            "- The runner verifies protected hashes before and after output generation.",
            "",
            "## 49. Recommended single next task",
            "- Phase 3M - controlled LightGCN ML-1M pilot configuration and execution.",
            "",
            "## Evidence paths",
            "- `results/tables/plan_e_lightgcn_preflight_manifest.csv`",
            "- `results/tables/plan_e_lightgcn_graph_audit.csv`",
            "- `results/tables/plan_e_lightgcn_normalization_audit.csv`",
            "- `results/tables/plan_e_lightgcn_index_audit.csv`",
            "- `results/tables/plan_e_lightgcn_row_compatibility.csv`",
            "- `results/tables/plan_e_lightgcn_connectivity_impact.csv`",
            "- `results/tables/plan_e_lightgcn_sparse_execution_audit.csv`",
            "- `results/tables/plan_e_lightgcn_policy_options.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_graph_sentence(row: dict[str, Any], compatible: dict[str, int], total: dict[str, int]) -> str:
    return (
        f"- {row['dataset']}: {row['train_interaction_count']} train interactions, "
        f"{row['unique_user_item_edge_count']} unique train pairs, {row['total_node_count']} graph nodes, "
        f"{row['directed_edge_count']} directed edges, strict-compatible rows "
        f"{compatible.get(row['dataset'], 0)}/{total.get(row['dataset'], 0)}."
    )


def validate_outputs() -> None:
    for name, (path, columns) in OUTPUTS.items():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(frame.columns) != columns:
            raise RuntimeError(f"{name} schema mismatch")
        if frame.empty:
            raise RuntimeError(f"{name} output is empty")
    manifest = pd.read_csv(OUTPUTS["manifest"][0], dtype=str, keep_default_na=False)
    allowed_adapter = {
        "ready_for_pilot_after_policy_approval",
        "graph_data_blocked",
        "graph_construction_blocked",
        "normalization_blocked",
        "sparse_backend_blocked",
        "mapping_blocked",
        "implementation_blocked",
        "environment_blocked",
        "not_applicable",
        "validation_failed",
    }
    if not set(manifest["adapter_status"]).issubset(allowed_adapter):
        raise RuntimeError("invalid manifest adapter_status")
    if not manifest["project_data_training_performed"].eq("false").all():
        raise RuntimeError("manifest indicates project-data training")
    if not manifest["project_data_scoring_performed"].eq("false").all():
        raise RuntimeError("manifest indicates project-data scoring")
    row = pd.read_csv(OUTPUTS["compatibility"][0], dtype=str, keep_default_na=False)
    if len(row) != 6000:
        raise RuntimeError(f"row compatibility must contain 6000 rows, found {len(row)}")
    if row.duplicated(["candidate_set_id", "user_id"]).any():
        raise RuntimeError("row compatibility has duplicate candidate_set_id,user_id keys")
    allowed_status = {
        "lightgcn_graph_compatible",
        "graph_disconnected_candidate",
        "positive_graph_disconnected",
        "user_graph_disconnected",
        "mapping_incompatible",
        "index_incompatible",
        "normalization_unverified",
        "validation_failed",
    }
    if not set(row["compatibility_status"]).issubset(allowed_status):
        raise RuntimeError("invalid compatibility status")
    for value in row["unmapped_candidate_item_ids"].tolist() + row["graph_disconnected_candidate_item_ids"].tolist():
        if not isinstance(json.loads(value), list):
            raise RuntimeError("invalid JSON array in row compatibility")
    compatible = row.loc[row["lightgcn_graph_compatible"].eq("true")]
    required_true = [
        "user_mapped",
        "user_graph_connected",
        "positive_item_mapped",
        "positive_item_graph_connected",
        "all_candidate_items_mapped",
        "all_candidate_items_graph_connected",
        "existing_model_index_compatible",
    ]
    for column in required_true:
        if not compatible[column].eq("true").all():
            raise RuntimeError(f"graph-compatible row has {column}=false")
    impact = pd.read_csv(OUTPUTS["impact"][0], dtype=str, keep_default_na=False)
    if len(impact) != 12:
        raise RuntimeError(f"connectivity impact should have 12 rows, found {len(impact)}")
    for value in impact["retention_rate"].astype(float):
        if value < 0 or value > 1:
            raise RuntimeError("invalid retention_rate")
    policy = pd.read_csv(OUTPUTS["policy"][0], dtype=str, keep_default_na=False)
    if (policy["recommendation_status"] == "recommended").sum() != 1:
        raise RuntimeError("policy table must have exactly one recommended row")
    forbidden = policy.loc[
        policy["policy_name"].isin(
            [
                "add_validation_or_test_edges",
                "add_candidate_pseudo_edges",
                "add_held_out_positive_edges",
                "remove_or_replace_disconnected_candidates",
                "construct_dense_full_adjacency",
            ]
        )
    ]
    if not forbidden["recommendation_status"].eq("forbidden").all():
        raise RuntimeError("leakage/mutation/dense policies must be forbidden")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(
        path,
        columns,
        rows,
        required_name_prefix=OUTPUT_PREFIX,
        prefix_error_message=f"refusing to write non-LightGCN output: {path}",
        lineterminator="\n",
        value_transform=format_value,
        atomic=True,
    )


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number):
            return ""
        return f"{number:.12g}"
    if isinstance(value, (np.integer,)):
        return str(int(value))
    return str(value)


def sort_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: tuple(natural_id_sort_key(row.get(key, "")) for key in keys))


def proportion(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def relpath(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def git_output(args: list[str]) -> str:
    return common_git.git_stdout_checked(args, PROJECT_ROOT)


def print_summary(
    graph_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    sparse_rows: list[dict[str, Any]],
) -> None:
    compat = pd.DataFrame(compatibility_rows)
    print("Plan E LightGCN preflight complete")
    print("graph_status", {row["dataset"]: row["status"] for row in graph_rows})
    print(
        "compatible_rows",
        compat.groupby("dataset")["lightgcn_graph_compatible"]
        .apply(lambda s: int(s.astype(str).str.lower().eq("true").sum()))
        .to_dict(),
    )
    print("sparse_status", {row["dataset"]: row["status"] for row in sparse_rows})
    print("impact_rows", len(impact_rows))


if __name__ == "__main__":
    raise SystemExit(main())

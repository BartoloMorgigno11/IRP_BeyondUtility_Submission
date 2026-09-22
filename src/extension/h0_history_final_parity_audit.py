"""Offline H0 final history-length parity audit.

This audit materializes the final four-dataset history-length readiness state
without inference. It reads frozen E0/E1/R5C/Plan D/official artifacts, freezes
the final E0 paired cohorts, and classifies the 72 final raw-history cells.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.extension.e0_candidates import attach_titles, normalize_candidate_manifest
from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    HISTORY_LENGTHS,
    METRIC_K,
    PARSER_CONTRACT_ID,
    REQUESTED_OUTPUT_LENGTH,
    item_list_hash,
    sha256_json,
    stable_int_seed,
    stable_json,
)
from src.utils.config import find_project_root


PHASE_ID = "extension_h0_history_final_parity_audit"
DATASETS = ("beauty", "sports", "ml1m", "lastfm")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
FINAL_LENGTHS = (0, 1, 2, 5, 10, 20)
FINAL_USERS_PER_DATASET = 200
EXPECTED_TOTAL_REQUESTS = 14_400
EXPECTED_LLM_REQUESTS = 0
EXPECTED_NETWORK_CALLS = 0
EXPECTED_DOWNLOADS = 0
EXPECTED_TRAINING_RUNS = 0

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")

E0_HISTORY_MANIFEST = MANIFEST_DIR / "extension_e0_history_manifest.csv"
E0_HISTORY_SUMMARY = TABLE_DIR / "extension_e0_history_dry_run_summary.csv"
E1_PILOT_MANIFEST = MANIFEST_DIR / "extension_e1_pilot_manifest.csv"
E1_RESPONSE_MANIFEST = MANIFEST_DIR / "extension_e1_response_manifest.csv"
E1_HISTORY_VALIDATION = TABLE_DIR / "extension_e1_history_validation.csv"
R5C_MANIFEST = TABLE_DIR / "reproduction_r5c_history_length_execution_manifest.csv"
R5C_CORRECTED_MANIFEST = TABLE_DIR / "reproduction_r5c_history_length_corrected_execution_manifest.csv"
PLAN_D_MANIFEST = TABLE_DIR / "plan_d_llm_ml1m_history_length_manifest.csv"
R1_OFFICIAL_RESULTS = TABLE_DIR / "reproduction_r1_history_length_official_results.csv"
R4_RECONCILIATION = TABLE_DIR / "reproduction_r4_history_length_reconciliation.csv"

OUTPUT_HISTORICAL_INVENTORY = TABLE_DIR / "extension_h0_historical_history_experiment_inventory.csv"
OUTPUT_PROTOCOL_IDENTITY = TABLE_DIR / "extension_h0_final_history_protocol_identity_matrix.csv"
OUTPUT_DATASET_FEASIBILITY = TABLE_DIR / "extension_h0_dataset_l20_feasibility.csv"
OUTPUT_COHORTS = TABLE_DIR / "extension_h0_final_paired_200_cohorts.csv"
OUTPUT_CELL_MATRIX = TABLE_DIR / "extension_h0_72_cell_history_matrix.csv"
OUTPUT_CLASSIFICATION = TABLE_DIR / "extension_h0_reusable_vs_to_compute.csv"
OUTPUT_REQUEST_BUDGET = TABLE_DIR / "extension_h0_exact_history_request_budget.csv"
OUTPUT_GRID_ALIGNMENT = TABLE_DIR / "extension_h0_paper_vs_local_history_grid.csv"
OUTPUT_MANIFEST = MANIFEST_DIR / "extension_h0_no_inference_manifest.json"
OUTPUT_REPORT = REPORT_DIR / "extension_h0_history_final_parity_audit.md"

CATALOG_PATHS = {
    "beauty": Path("data/reconstructed/plan_c/amazon_beauty/catalog.csv"),
    "sports": Path("data/reconstructed/plan_c/amazon_sports/catalog.csv"),
    "ml1m": Path("data/processed/ml1m/catalog.csv"),
    "lastfm": Path("data/processed/finalization_f6c_lastfm_catalog.csv"),
}
INTERACTIONS_PATHS = {
    "beauty": Path("data/reconstructed/plan_c/amazon_beauty/interactions.csv"),
    "sports": Path("data/reconstructed/plan_c/amazon_sports/interactions.csv"),
    "ml1m": Path("data/processed/ml1m/interactions.csv"),
    "lastfm": Path("data/processed/finalization_f6c_lastfm_interactions.csv"),
}
CANDIDATE_PATHS = {
    "beauty": Path("data/reconstructed/plan_c/amazon_beauty/candidates.csv"),
    "sports": Path("data/reconstructed/plan_c/amazon_sports/candidates.csv"),
    "ml1m": Path("data/reconstructed/plan_c/ml1m/candidates.csv"),
    "lastfm": Path("data/processed/finalization_f6c_lastfm_ranking_candidates.csv"),
}
SELECTION_SEED = 20260824


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline H0 history-length parity audit.")
    parser.add_argument("--output-root", default=".")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_h0(root)
    write_outputs(root, outputs)
    summary = outputs["summary"]
    print(f"final_cells={summary['final_cells']}")
    print(f"reusable_cells={summary['reusable_cells']}")
    print(f"to_compute_cells={summary['to_compute_cells']}")
    print(f"blocked_cells={summary['blocked_cells']}")
    print(f"to_compute_requests={summary['to_compute_requests']}")
    print(f"decision={summary['decision']}")
    print(f"llm_requests={EXPECTED_LLM_REQUESTS}")
    return 0


def generate_h0(root: Path) -> dict[str, pd.DataFrame | dict[str, Any]]:
    e0_manifest = read_csv(root / E0_HISTORY_MANIFEST)
    e0_summary = read_csv(root / E0_HISTORY_SUMMARY)
    catalog_frames = {dataset: read_catalog_csv(root / path) for dataset, path in CATALOG_PATHS.items()}
    catalogs = {dataset: title_map(frame) for dataset, frame in catalog_frames.items()}

    full_pool = build_full_feasibility_pool(root, catalogs, catalog_frames)
    feasibility, cohorts = build_dataset_feasibility_and_cohorts(e0_manifest, e0_summary, full_pool)
    historical = build_historical_inventory(root, e0_manifest, e0_summary)
    protocol_identity = build_protocol_identity(historical)
    cell_matrix = build_cell_matrix(feasibility, historical)
    classification = build_reusable_classification(cell_matrix)
    budget = build_request_budget(cell_matrix)
    grid_alignment = build_grid_alignment(root)
    summary = build_summary(feasibility, cell_matrix, budget)
    manifest = build_manifest(root, summary)
    return {
        "historical_inventory": historical,
        "protocol_identity": protocol_identity,
        "dataset_feasibility": feasibility,
        "cohorts": cohorts,
        "full_pool": full_pool,
        "cell_matrix": cell_matrix,
        "classification": classification,
        "request_budget": budget,
        "grid_alignment": grid_alignment,
        "manifest": manifest,
        "summary": summary,
    }


def write_outputs(root: Path, outputs: Mapping[str, pd.DataFrame | dict[str, Any]]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "historical_inventory": root / OUTPUT_HISTORICAL_INVENTORY,
        "protocol_identity": root / OUTPUT_PROTOCOL_IDENTITY,
        "dataset_feasibility": root / OUTPUT_DATASET_FEASIBILITY,
        "cohorts": root / OUTPUT_COHORTS,
        "cell_matrix": root / OUTPUT_CELL_MATRIX,
        "classification": root / OUTPUT_CLASSIFICATION,
        "request_budget": root / OUTPUT_REQUEST_BUDGET,
        "grid_alignment": root / OUTPUT_GRID_ALIGNMENT,
    }
    for key, path in table_paths.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(path, index=False)

    manifest = dict(outputs["manifest"])  # type: ignore[arg-type]
    manifest["artifact_hashes"] = {path.relative_to(root).as_posix(): sha256_file(path) for path in table_paths.values()}
    (root / OUTPUT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / OUTPUT_REPORT).write_text(render_report(outputs), encoding="utf-8")


def build_full_feasibility_pool(
    root: Path,
    catalogs: Mapping[str, Mapping[str, str]],
    catalog_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build one row per structurally valid long-history candidate user."""

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        catalog = catalogs[dataset]
        catalog_frame = catalog_frames[dataset]
        train = train_interactions(read_interactions_csv(root / INTERACTIONS_PATHS[dataset]))
        train_by_user = {
            str(user_id): group.sort_values("__history_order", kind="mergesort")["item_id"].astype(str).tolist()
            for user_id, group in train.groupby("user_id", sort=False)
        }
        train_eligible_users = {user_id for user_id, history in train_by_user.items() if len(history) >= max(FINAL_LENGTHS)}
        raw_candidates = read_candidate_csv(root / CANDIDATE_PATHS[dataset])
        raw_candidates.columns = [str(column).strip().lower() for column in raw_candidates.columns]
        raw_candidates = raw_candidates.loc[raw_candidates["user_id"].astype(str).isin(train_eligible_users)].copy()
        candidates = normalize_candidate_manifest(raw_candidates, dataset=dataset, seed=SELECTION_SEED)
        candidates = attach_titles(candidates, catalog_frame)
        candidate_groups = normalized_candidate_groups(candidates)
        for user_id in sorted(set(train_by_user).intersection(candidate_groups)):
            history_all = train_by_user[user_id]
            if len(history_all) < max(FINAL_LENGTHS):
                continue
            candidate = candidate_groups[user_id]
            if not candidate["structurally_valid"]:
                continue
            candidate_items = candidate["candidate_item_ids"]
            candidate_titles_ok = bool(candidate["candidate_titles_valid"])
            histories: dict[int, list[str]] = {}
            history_title_ok_by_l: dict[int, bool] = {}
            for length in FINAL_LENGTHS:
                history_ids = history_all[-length:] if length else []
                histories[length] = history_ids
                history_title_ok_by_l[length] = all(title_is_valid(catalog.get(str(item_id), ""), str(item_id)) for item_id in history_ids)
            rows.append(
                {
                    "dataset": dataset,
                    "user_id": str(user_id),
                    "candidate_item_ids": stable_json(candidate_items),
                    "candidate_hash": item_list_hash(candidate_items),
                    "unordered_candidate_set_hash": sha256_json(sorted(candidate_items)),
                    "positive_item_id": str(candidate["positive_item_id"]),
                    "positive_position": int(candidate["positive_position"]),
                    "candidate_titles_valid": candidate_titles_ok,
                    **{f"history_l{length}_item_ids": stable_json(histories[length]) for length in FINAL_LENGTHS},
                    **{f"history_l{length}_hash": item_list_hash(histories[length]) for length in FINAL_LENGTHS},
                    **{f"history_l{length}_count": len(histories[length]) for length in FINAL_LENGTHS},
                    **{f"history_l{length}_titles_valid": history_title_ok_by_l[length] for length in FINAL_LENGTHS},
                    "fully_paired_l20_valid": candidate_titles_ok and history_title_ok_by_l[20] and len(histories[20]) == 20,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "user_id"], kind="mergesort").reset_index(drop=True)


def build_dataset_feasibility_and_cohorts(
    e0_manifest: pd.DataFrame,
    e0_summary: pd.DataFrame,
    full_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feasibility_rows: list[dict[str, Any]] = []
    cohort_rows: list[pd.DataFrame] = []
    summary_by_dataset = e0_summary.set_index("dataset").to_dict("index")
    for dataset in DATASETS:
        e0_data = e0_manifest.loc[e0_manifest["dataset"].eq(dataset)].copy()
        pool = full_pool.loc[full_pool["dataset"].eq(dataset)].copy()
        strict_by_length = {
            length: int(
                (
                    pool["candidate_titles_valid"].astype(bool)
                    & pool[f"history_l{length}_titles_valid"].astype(bool)
                    & pool[f"history_l{length}_count"].astype(int).eq(length)
                ).sum()
            )
            for length in FINAL_LENGTHS
        }
        strict_l20 = pool.loc[pool["fully_paired_l20_valid"].astype(bool)].copy()
        selected_ids = deterministic_history_users(strict_l20["user_id"].astype(str).tolist(), FINAL_USERS_PER_DATASET, dataset=dataset)
        selected = strict_l20.loc[strict_l20["user_id"].astype(str).isin(set(selected_ids))].copy()
        selected["selection_rank"] = selected["user_id"].astype(str).map({user_id: idx for idx, user_id in enumerate(selected_ids, start=1)})
        selected = selected.sort_values("selection_rank", kind="mergesort").reset_index(drop=True)
        selected["cohort_user_hash"] = item_list_hash(selected_ids)
        selected["cohort_candidate_hash"] = item_list_hash(
            [item_id for raw in selected["candidate_item_ids"] for item_id in json.loads(str(raw))]
        )
        cohort_rows.append(selected)
        summary = summary_by_dataset[dataset]
        feasibility_rows.append(
            {
                "dataset": dataset,
                "eligible_users_ge20": int(summary["eligible_users_ge20"]),
                "structurally_valid_long_history_users": int(len(pool)),
                "selected_users": int(len(selected)),
                "e0_manifest_rows": int(len(e0_data)),
                "e0_manifest_selected_users": int(e0_data["user_id"].nunique()),
                "models": int(e0_data["model_name"].nunique()),
                "history_lengths": ";".join(str(length) for length in FINAL_LENGTHS),
                "candidate_pool_20_users": int(len(pool)),
                "candidate_titles_valid_users": int(pool["candidate_titles_valid"].astype(bool).sum()),
                "strict_l1_users": strict_by_length[1],
                "strict_l2_users": strict_by_length[2],
                "strict_l5_users": strict_by_length[5],
                "strict_l10_users": strict_by_length[10],
                "strict_l20_users": strict_by_length[20],
                "strict_paired_l20_users": int(strict_l20["user_id"].nunique()),
                "l0_empty_history_users": int(pool["history_l0_count"].astype(int).eq(0).sum()),
                "same_candidates_all_l_users": int(len(pool)),
                "same_positive_all_l_users": int(len(pool)),
                "selection_seed": SELECTION_SEED,
                "selection_rule": "sort_by_stable_int_seed(seed,'history_cohort',dataset,user_id),user_id over strict L20 pool",
                "status": "FEASIBLE_FINAL_PAIRED_200" if len(selected) >= FINAL_USERS_PER_DATASET else "BLOCKED",
                "blocker": "" if len(selected) >= FINAL_USERS_PER_DATASET else "fewer_than_200_strict_l20_users",
                "cohort_user_hash": str(selected["cohort_user_hash"].iloc[0]) if not selected.empty else "",
                "cohort_candidate_hash": str(selected["cohort_candidate_hash"].iloc[0]) if not selected.empty else "",
            }
        )
    cohorts = pd.concat(cohort_rows, ignore_index=True) if cohort_rows else pd.DataFrame()
    return pd.DataFrame(feasibility_rows), cohorts


def deterministic_history_users(users: Sequence[str], count: int, *, dataset: str) -> list[str]:
    ordered = sorted((str(user) for user in users), key=lambda user: (stable_int_seed(SELECTION_SEED, "history_cohort", dataset, user), user))
    return ordered[: min(count, len(ordered))]


def normalized_candidate_groups(candidates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    frame = candidates.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame["user_id"] = frame["user_id"].astype(str)
    frame["candidate_item_id"] = frame["candidate_item_id"].astype(str)
    frame["candidate_position"] = pd.to_numeric(frame["candidate_position"], errors="coerce").fillna(-1).astype(int)
    frame["is_ground_truth_bool"] = truthy(frame["is_ground_truth"])
    groups: dict[str, dict[str, Any]] = {}
    for user_id, group in frame.groupby("user_id", sort=False):
        ordered = group.sort_values("candidate_position", kind="mergesort")
        item_ids = ordered["candidate_item_id"].astype(str).tolist()
        titles = ordered["candidate_item_title"].astype(str).tolist()
        positions = ordered["candidate_position"].astype(int).tolist()
        positives = ordered.loc[ordered["is_ground_truth_bool"], "candidate_item_id"].astype(str).tolist()
        candidate_titles_valid = all(title_is_valid(title, item_id) for title, item_id in zip(titles, item_ids))
        structurally_valid = (
            len(item_ids) == CANDIDATE_COUNT
            and len(set(item_ids)) == CANDIDATE_COUNT
            and positions == list(range(1, CANDIDATE_COUNT + 1))
            and len(positives) == 1
        )
        groups[str(user_id)] = {
            "candidate_item_ids": item_ids,
            "positive_item_id": positives[0] if positives else "",
            "positive_position": positions[item_ids.index(positives[0])] if positives and positives[0] in item_ids else 0,
            "candidate_titles_valid": candidate_titles_valid,
            "structurally_valid": structurally_valid,
        }
    return groups


def train_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    frame = interactions.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    split = frame["split"].astype(str).str.lower() if "split" in frame.columns else pd.Series("train", index=frame.index)
    frame = frame.loc[split.eq("train")].copy()
    frame["__input_order"] = range(len(frame))
    if "timestamp" in frame.columns:
        frame["__timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.sort_values(["user_id", "__timestamp", "__input_order"], kind="mergesort")
    else:
        frame = frame.sort_values(["user_id", "__input_order"], kind="mergesort")
    frame["__history_order"] = frame.groupby("user_id", sort=False).cumcount() + 1
    return frame


def truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def build_historical_inventory(root: Path, e0_manifest: pd.DataFrame, e0_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(inventory_e0(e0_manifest, e0_summary))
    rows.extend(inventory_e1(root))
    rows.extend(inventory_r5c(root / R5C_MANIFEST, "R5C_ORIGINAL", "HISTORICAL_ADAPTED_EVIDENCE_NOT_FINAL_REUSABLE"))
    rows.extend(inventory_r5c(root / R5C_CORRECTED_MANIFEST, "R5C_CORRECTED", "HISTORICAL_ADAPTED_EVIDENCE_NOT_FINAL_REUSABLE"))
    rows.extend(inventory_plan_d(root / PLAN_D_MANIFEST))
    rows.extend(inventory_official(root / R1_OFFICIAL_RESULTS))
    rows.extend(inventory_paper_official_grid(root / R4_RECONCILIATION))
    return pd.DataFrame(rows)


def inventory_e0(e0_manifest: pd.DataFrame, e0_summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in e0_summary.itertuples(index=False):
        dataset = str(row.dataset)
        data = e0_manifest.loc[e0_manifest["dataset"].eq(dataset)]
        rows.append(
            inventory_row(
                source="E0_FINAL_HISTORY_MANIFEST",
                dataset=dataset,
                model_name=";".join(sorted(data["model_name"].astype(str).unique())),
                cohort_size=int(row.selected_users),
                user_ids_hash=item_list_hash(data.drop_duplicates("user_id")["user_id"].astype(str).tolist()),
                history_lengths=sorted_ints(data["history_length"].unique()),
                candidate_pool_size=unique_or_mixed(data["candidate_count"]),
                requested_output_length=unique_or_mixed(data["requested_output_length"]),
                parser_expected_length=10,
                metric_k=unique_or_mixed(data["metric_k"]),
                prompt_family=unique_or_mixed(data["prompt_strategy"]),
                strict_parser=unique_or_mixed(data["parser_contract"]),
                candidate_identity="same_candidate_hash_across_L_per_user",
                positive_identity="same_positive_across_L_per_user",
                evidence_class="FINAL_PROTOCOL_MANIFEST_ONLY_NO_RAW_OUTPUTS",
                reusable_for_final=False,
                non_reusable_reason="manifest/planned only; no raw LLM outputs executed",
            )
        )
    return rows


def inventory_e1(root: Path) -> list[dict[str, Any]]:
    path = root / E1_PILOT_MANIFEST
    if not path.exists():
        return []
    pilot = read_csv(path)
    responses = read_csv(root / E1_RESPONSE_MANIFEST) if (root / E1_RESPONSE_MANIFEST).exists() else pd.DataFrame()
    rows = []
    for (dataset, model), group in pilot.groupby(["dataset", "model_name"], sort=True):
        rows.append(
            inventory_row(
                source="E1_CONTROLLED_PILOT",
                dataset=str(dataset),
                model_name=str(model),
                cohort_size=int(group["user_id"].nunique()),
                user_ids_hash=item_list_hash(group["user_id"].astype(str).unique().tolist()),
                history_lengths=sorted_ints(group["history_length"].unique()),
                candidate_pool_size=unique_or_mixed(group["candidate_count"]),
                requested_output_length=unique_or_mixed(group["requested_output_length"]),
                parser_expected_length=10,
                metric_k=unique_or_mixed(group["metric_k"]),
                prompt_family=unique_or_mixed(group["prompt_strategy"]),
                strict_parser=unique_or_mixed(group["parser_contract"]),
                candidate_identity="pilot_only",
                positive_identity="pilot_only",
                evidence_class=f"PILOT_EXECUTED_RESPONSES_{len(responses)}_ROWS",
                reusable_for_final=False,
                non_reusable_reason="pilot scope and fifth-dataset context; not full four-dataset final history matrix",
            )
        )
    return rows


def inventory_r5c(path: Path, source: str, evidence_class: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_csv(path)
    rows = []
    for (dataset, model), group in data.groupby(["dataset", "model_name"], sort=True):
        rows.append(
            inventory_row(
                source=source,
                dataset=str(dataset),
                model_name=str(model),
                cohort_size=int(group.groupby("history_length")["user_id"].nunique().min()),
                user_ids_hash=item_list_hash(sorted(group["user_id"].astype(str).unique().tolist())),
                history_lengths=sorted_ints(group["history_length"].unique()),
                candidate_pool_size=unique_or_mixed(group["candidate_count"]),
                requested_output_length=5,
                parser_expected_length=5,
                metric_k=5,
                prompt_family=unique_or_mixed(group["prompt_template_id"]),
                strict_parser=unique_or_mixed(group["parser_contract"]),
                candidate_identity="frozen_R5A_candidate_hashes",
                positive_identity="same_positive_across_R5C_group",
                evidence_class=evidence_class,
                reusable_for_final=False,
                non_reusable_reason="top5 output/parser, 100 users/cell, Qwen+Mistral only, partial length grid",
            )
        )
    return rows


def inventory_plan_d(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_csv(path)
    rows = []
    for (dataset, model), group in data.groupby(["dataset", "model"], sort=True):
        rows.append(
            inventory_row(
                source="PLAN_D_ML1M_HISTORY_DIAGNOSTIC",
                dataset=str(dataset),
                model_name=str(model),
                cohort_size=int(group.groupby("history_length_used")["user_id"].nunique().min()),
                user_ids_hash=item_list_hash(sorted(group["user_id"].astype(str).unique().tolist())),
                history_lengths=sorted_ints(group["history_length_used"].unique()),
                candidate_pool_size=unique_or_mixed(group["candidate_pool_size"]),
                requested_output_length="unknown_from_manifest",
                parser_expected_length="candidate_id_diagnostic",
                metric_k=5,
                prompt_family=unique_or_mixed(group["prompt_strategy"]),
                strict_parser="strict_candidate_id_only_history_length",
                candidate_identity="diagnostic_subset_candidate_order_signature",
                positive_identity="diagnostic_subset_positive_item_id",
                evidence_class="DIAGNOSTIC_SUBSET_NOT_FINAL_REUSABLE",
                reusable_for_final=False,
                non_reusable_reason="ML-1M 20-user diagnostic subset and candidate-ID contract",
            )
        )
    return rows


def inventory_official(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_csv(path)
    rows = []
    for (dataset, model), group in data.groupby(["dataset", "model"], sort=True):
        rows.append(
            inventory_row(
                source="ORIGINAL_REPO_ARCHIVED_HISTORY_RESULTS",
                dataset=str(dataset),
                model_name=str(model),
                cohort_size="metrics_archive_only",
                user_ids_hash="UNAVAILABLE",
                history_lengths=sorted_ints(group["history_length"].unique()),
                candidate_pool_size=20 if group["condition_evidence"].astype(str).str.contains("candidate20", regex=False).any() else "unknown",
                requested_output_length="metrics_archive_only",
                parser_expected_length="metrics_archive_only",
                metric_k=5,
                prompt_family="history_only_archived_results",
                strict_parser="unknown_from_metrics_table",
                candidate_identity="official_archive_not_local_final_cohort",
                positive_identity="official_archive_not_local_final_cohort",
                evidence_class="OFFICIAL_ARCHIVED_METRICS_NOT_FINAL_REUSABLE",
                reusable_for_final=False,
                non_reusable_reason="different models and no raw final top10 local outputs for Qwen/Mistral/Gemma",
            )
        )
    return rows


def inventory_paper_official_grid(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_csv(path)
    rows = []
    for row in data.itertuples(index=False):
        rows.append(
            inventory_row(
                source="PAPER_OFFICIAL_GRID_RECONCILIATION",
                dataset=str(row.dataset),
                model_name="not_model_specific",
                cohort_size="not_model_specific",
                user_ids_hash="UNAVAILABLE",
                history_lengths=str(row.official_grid),
                candidate_pool_size="not_model_specific",
                requested_output_length="not_model_specific",
                parser_expected_length="not_model_specific",
                metric_k=5,
                prompt_family="paper/official history sensitivity protocol",
                strict_parser="not_model_specific",
                candidate_identity="not_model_specific",
                positive_identity="not_model_specific",
                evidence_class=str(row.grid_classification),
                reusable_for_final=False,
                non_reusable_reason="grid/protocol evidence only; not raw final outputs",
            )
        )
    return rows


def inventory_row(**values: Any) -> dict[str, Any]:
    return {
        "source": values["source"],
        "dataset": values["dataset"],
        "model_name": values["model_name"],
        "cohort_size": values["cohort_size"],
        "user_ids_hash": values["user_ids_hash"],
        "history_lengths": ";".join(str(v) for v in values["history_lengths"]) if isinstance(values["history_lengths"], list) else values["history_lengths"],
        "candidate_pool_size": values["candidate_pool_size"],
        "requested_output_length": values["requested_output_length"],
        "parser_expected_length": values["parser_expected_length"],
        "metric_cutoff_k": values["metric_k"],
        "prompt_family": values["prompt_family"],
        "strict_parser": values["strict_parser"],
        "candidate_identity": values["candidate_identity"],
        "positive_identity": values["positive_identity"],
        "evidence_class": values["evidence_class"],
        "reusable_for_final": bool(values["reusable_for_final"]),
        "non_reusable_reason": values["non_reusable_reason"],
    }


def build_protocol_identity(historical: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("datasets", "beauty;sports;ml1m;lastfm", "R5C has same four datasets; E1 is pilot-only evidence", "FINAL_ONLY_FOR_E0_MANIFEST"),
        ("models", ";".join(MODELS), "R5C has Qwen/Mistral only; Plan D has Qwen/Mistral; E0 has all three", "FINAL_ONLY_FOR_E0_MANIFEST"),
        ("prompt", "BASE only", "R5C corrected prompt family differs; Plan D candidate-id prompt differs", "FINAL_ONLY_FOR_E0_MANIFEST"),
        ("history_lengths", "0;1;2;5;10;20", "R5C partial grid; Plan D 5/10/20; official grids vary by dataset", "FINAL_ONLY_FOR_E0_MANIFEST"),
        ("paired_users", "200 per dataset", "R5C 100/cell; Plan D 20; E1 pilot only", "FINAL_ONLY_FOR_E0_MANIFEST"),
        ("candidate_pool", "20", "R5C/Plan D/E0 all use 20 where applicable", "IDENTICAL_WHERE_PRESENT"),
        ("positive_count", "1", "E0/R5C group validation; E1 pilot validation", "IDENTICAL_WHERE_PRESENT"),
        ("negative_count", "19", "E0 final contract", "FINAL_ONLY_FOR_E0_MANIFEST"),
        ("requested_output", "10 positions", "R5C corrected requests exactly 5", "FINAL_ONLY_FOR_E0_MANIFEST"),
        ("parser", PARSER_CONTRACT_ID, "R5C parser is r5c_ranked_index_json_top5_v1", "FINAL_ONLY_FOR_E0_MANIFEST"),
        ("metric_k", str(METRIC_K), "R5C/official use K=5 where applicable", "IDENTICAL_WHERE_PRESENT"),
        ("l0_semantics", "explicit empty history []", "E0 code uses empty frame for length 0; no tail(0) leak", "IDENTICAL_FINAL_CONTRACT"),
        ("raw_output_reuse", "none", f"{int(historical['reusable_for_final'].sum())} reusable historical rows", "NO_REUSABLE_RAW_OUTPUTS"),
    ]
    return pd.DataFrame(rows, columns=["dimension", "final_contract", "historical_observation", "status"])


def build_cell_matrix(feasibility: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    feasible = set(feasibility.loc[feasibility["status"].eq("FEASIBLE_FINAL_PAIRED_200"), "dataset"].astype(str))
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for length in FINAL_LENGTHS:
                status = "TO_COMPUTE" if dataset in feasible else "BLOCKED"
                historical_note = historical_note_for(dataset, model, length, historical)
                rows.append(
                    {
                        "dataset": dataset,
                        "model_name": model,
                        "history_length": length,
                        "status": status,
                        "users": FINAL_USERS_PER_DATASET if dataset in feasible else 0,
                        "required_raw_requests": FINAL_USERS_PER_DATASET if dataset in feasible else 0,
                        "reusable_identical_raw_requests": 0,
                        "to_compute_raw_requests": FINAL_USERS_PER_DATASET if status == "TO_COMPUTE" else 0,
                        "blocked_raw_requests": FINAL_USERS_PER_DATASET if status == "BLOCKED" else 0,
                        "candidate_pool_size": CANDIDATE_COUNT,
                        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                        "parser_contract": PARSER_CONTRACT_ID,
                        "metric_k": METRIC_K,
                        "prompt_strategy": "BASE",
                        "historical_evidence_note": historical_note,
                    }
                )
    return pd.DataFrame(rows)


def historical_note_for(dataset: str, model: str, length: int, historical: pd.DataFrame) -> str:
    matches = historical.loc[
        historical["dataset"].astype(str).str.lower().eq(dataset.lower())
        & historical["model_name"].astype(str).eq(model)
        & historical["history_lengths"].astype(str).str.split(";").map(lambda values: str(length) in values)
    ]
    if matches.empty:
        if model == "gemma2:9b":
            return "no historical Gemma final-history output found"
        return "no protocol-identical historical raw output found"
    return "; ".join(sorted(matches["non_reusable_reason"].astype(str).unique()))


def build_reusable_classification(cell_matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        cell_matrix.groupby(["status", "historical_evidence_note"], sort=True)
        .agg(
            cells=("status", "size"),
            raw_requests=("required_raw_requests", "sum"),
            reusable_identical_raw_requests=("reusable_identical_raw_requests", "sum"),
            to_compute_raw_requests=("to_compute_raw_requests", "sum"),
            blocked_raw_requests=("blocked_raw_requests", "sum"),
        )
        .reset_index()
    )


def build_request_budget(cell_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, model), group in cell_matrix.groupby(["dataset", "model_name"], sort=True):
        rows.append(budget_row("dataset_model", dataset, model, group))
    for dataset, group in cell_matrix.groupby("dataset", sort=True):
        rows.append(budget_row("dataset_total", dataset, "ALL_MODELS", group))
    for model, group in cell_matrix.groupby("model_name", sort=True):
        rows.append(budget_row("model_total", "ALL_DATASETS", model, group))
    rows.append(budget_row("grand_total", "ALL_DATASETS", "ALL_MODELS", cell_matrix))
    return pd.DataFrame(rows)


def budget_row(scope: str, dataset: str, model: str, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "scope": scope,
        "dataset": dataset,
        "model_name": model,
        "users_per_dataset_model": FINAL_USERS_PER_DATASET if scope == "dataset_model" else "",
        "history_conditions": len(FINAL_LENGTHS) if scope == "dataset_model" else "",
        "required_raw_requests": int(group["required_raw_requests"].sum()),
        "reusable_identical_raw_requests": int(group["reusable_identical_raw_requests"].sum()),
        "to_compute_raw_requests": int(group["to_compute_raw_requests"].sum()),
        "blocked_raw_requests": int(group["blocked_raw_requests"].sum()),
        "derived_metric_extra_requests": 0,
    }


def build_grid_alignment(root: Path) -> pd.DataFrame:
    reconciliation = read_csv(root / R4_RECONCILIATION) if (root / R4_RECONCILIATION).exists() else pd.DataFrame()
    official_by_dataset = {
        str(row.dataset).lower(): set(str(row.official_grid).replace(" ", "").split(";"))
        for row in reconciliation.itertuples(index=False)
    } if not reconciliation.empty else {}
    rows = []
    for dataset in DATASETS:
        official = official_by_dataset.get({"beauty": "beauty", "sports": "sports", "ml1m": "ml-1m", "lastfm": "lastfm"}[dataset], set())
        for length in FINAL_LENGTHS:
            if str(length) in official:
                classification = "OFFICIAL_CODE_SUPPORTED"
            else:
                classification = "LOCAL_EXTENSION_GRID"
            rows.append(
                {
                    "dataset": dataset,
                    "history_length": length,
                    "classification": classification,
                    "paper_alignment_note": "paper history protocol exists, but exact paper length grid is not proven from local artifacts",
                    "official_code_supported": str(length) in official,
                    "local_final_grid": True,
                    "l0_semantics": "explicit_empty_history" if length == 0 else "",
                }
            )
    return pd.DataFrame(rows)


def build_summary(feasibility: pd.DataFrame, cell_matrix: pd.DataFrame, budget: pd.DataFrame) -> dict[str, Any]:
    total = budget.loc[budget["scope"].eq("grand_total")].iloc[0]
    blocked_cells = int(cell_matrix["status"].eq("BLOCKED").sum())
    return {
        "phase_id": PHASE_ID,
        "decision": "HISTORY_FINAL_PARITY_READY" if blocked_cells == 0 else "HISTORY_FINAL_PARITY_BLOCKED",
        "feasible_datasets": int(feasibility["status"].eq("FEASIBLE_FINAL_PAIRED_200").sum()),
        "final_cells": int(len(cell_matrix)),
        "reusable_cells": int(cell_matrix["status"].eq("EXISTING_REUSABLE_IDENTICAL").sum()),
        "to_compute_cells": int(cell_matrix["status"].eq("TO_COMPUTE").sum()),
        "blocked_cells": blocked_cells,
        "required_requests": int(total["required_raw_requests"]),
        "reusable_requests": int(total["reusable_identical_raw_requests"]),
        "to_compute_requests": int(total["to_compute_raw_requests"]),
        "blocked_requests": int(total["blocked_raw_requests"]),
        "llm_requests_executed": EXPECTED_LLM_REQUESTS,
        "network_calls_executed": EXPECTED_NETWORK_CALLS,
        "downloads_executed": EXPECTED_DOWNLOADS,
        "training_runs_executed": EXPECTED_TRAINING_RUNS,
    }


def build_manifest(root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(summary),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_head(root),
        "source_artifacts": {
            "extension_e0_history_manifest": str(E0_HISTORY_MANIFEST),
            "extension_e0_history_summary": str(E0_HISTORY_SUMMARY),
            "r5c_corrected_manifest": str(R5C_CORRECTED_MANIFEST),
            "plan_d_manifest": str(PLAN_D_MANIFEST),
            "official_history_results": str(R1_OFFICIAL_RESULTS),
        },
    }


def render_report(outputs: Mapping[str, pd.DataFrame | dict[str, Any]]) -> str:
    summary = outputs["summary"]  # type: ignore[assignment]
    feasibility = outputs["dataset_feasibility"]  # type: ignore[assignment]
    cell_matrix = outputs["cell_matrix"]  # type: ignore[assignment]
    budget = outputs["request_budget"]  # type: ignore[assignment]
    grid = outputs["grid_alignment"]  # type: ignore[assignment]
    return f"""# H0 Final History-Length Parity Audit

Phase token: `{summary['decision']}`

No LLM inference, recommender training, downloads, network calls, fifth-dataset
selection, or commits were performed.

## Final Contract

Datasets: Beauty, Sports, ML-1M, LastFM.
Models: Qwen2 7B, Mistral 7B, Gemma2 9B.
Lengths: 0, 1, 2, 5, 10, 20.
Prompt: BASE only. Candidate pool: 20. Requested output: 10 unique positions.
Parser: `{PARSER_CONTRACT_ID}`. Metric K: {METRIC_K}.

## Dataset Feasibility

{feasibility.to_markdown(index=False) if isinstance(feasibility, pd.DataFrame) else ''}

## Cell Matrix

{cell_matrix.groupby('status').size().rename('cells').reset_index().to_markdown(index=False) if isinstance(cell_matrix, pd.DataFrame) else ''}

## Request Budget

{budget.loc[budget['scope'].eq('grand_total')].to_markdown(index=False) if isinstance(budget, pd.DataFrame) else ''}

## Paper vs Local Grid

{grid.to_markdown(index=False) if isinstance(grid, pd.DataFrame) else ''}
"""


def title_map(catalog: pd.DataFrame) -> dict[str, str]:
    frame = catalog.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "item_id" not in frame.columns:
        raise ValueError("catalog missing item_id")
    if "title" not in frame.columns:
        raise ValueError("catalog missing title")
    frame["item_id"] = frame["item_id"].astype(str)
    frame["title"] = frame["title"].astype(str)
    return dict(zip(frame["item_id"], frame["title"]))


def title_is_valid(title: Any, item_id: str) -> bool:
    value = str(title or "").strip()
    return bool(value) and value.lower() not in {"nan", "none", "null"} and value != str(item_id)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def read_catalog_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False, usecols=lambda column: str(column).strip().lower() in {"item_id", "title"})


def read_interactions_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        keep_default_na=False,
        usecols=lambda column: str(column).strip().lower() in {"user_id", "item_id", "split", "timestamp"},
    )


def read_candidate_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        keep_default_na=False,
        usecols=lambda column: str(column).strip().lower()
        in {"user_id", "candidate_item_id", "candidate_position", "is_ground_truth", "candidate_item_title"},
    )


def sorted_ints(values: Iterable[Any]) -> list[int]:
    return sorted(int(value) for value in values)


def unique_or_mixed(values: Iterable[Any]) -> str | int:
    unique = sorted({str(value) for value in values})
    if len(unique) == 1:
        value = unique[0]
        return int(value) if value.isdigit() else value
    return "MIXED:" + ";".join(unique)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

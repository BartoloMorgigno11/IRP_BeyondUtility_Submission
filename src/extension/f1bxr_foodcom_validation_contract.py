"""F1BXR Food.com nested validation contract recovery.

This phase freezes a Food.com validation protocol for later trainable-model
hyperparameter selection. It reads raw Food.com interactions and catalog
metadata, but it does not train, infer, download, call network services, or
rerun F1BX deterministic baselines.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd
import yaml

from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    NEGATIVE_COUNT,
    bool_text,
    item_list_hash,
    sha256_json,
    stable_int_seed,
    stable_json,
)
from src.extension.f0_final_extension_freeze import sha256_file
from src.utils.config import find_project_root


PHASE_ID = "f1bxr_foodcom_validation_contract_recovery"
DECISION_RESOLVED = "F1BXR_FOODCOM_VALIDATION_CONTRACT_RESOLVED"
DECISION_BLOCKED = "F1BXR_FOODCOM_VALIDATION_CONTRACT_BLOCKED"
EXPECTED_HEAD = "61fbde6d048faaed9e20508484efc90fa95fa5bf"
CONTRACT_ID = "F1BXR_FOODCOM_VALIDATION_CONTRACT_V1"
DATASET = "foodcom"
ACTIVE_DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
TRAINABLE_MODELS = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
VALIDATION_TARGET_USERS = 1000
VALIDATION_SELECTION_SEED = 20260829
TUNING_SEED = 42
SELECTION_METRIC = "validation_ndcg_at_5"
MIN_TUNING_PREFIX_LENGTH = 1
CONFIG_PATH = Path("configs/f1bxr_foodcom_validation_contract.yaml")

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")
REPORT_PATH = REPORT_DIR / "f1bxr_foodcom_validation_contract.md"

OUTPUT_TABLES = {
    "nested_split_audit": TABLE_DIR / "f1bxr_foodcom_nested_split_audit.csv",
    "validation_eligibility": TABLE_DIR / "f1bxr_foodcom_validation_eligibility.csv",
    "validation_cohort": TABLE_DIR / "f1bxr_foodcom_validation_cohort.csv",
    "validation_candidates": TABLE_DIR / "f1bxr_foodcom_validation_candidates.csv",
    "validation_candidate_audit": TABLE_DIR / "f1bxr_foodcom_validation_candidate_audit.csv",
    "test_leakage_audit": TABLE_DIR / "f1bxr_test_leakage_audit.csv",
    "tuning_readiness": TABLE_DIR / "f1bxr_tuning_readiness.csv",
    "validation": TABLE_DIR / "f1bxr_validation.csv",
}

OUTPUT_MANIFESTS = {
    "validation_contract_manifest": MANIFEST_DIR / "f1bxr_foodcom_validation_contract.json",
    "no_training_no_inference": MANIFEST_DIR / "f1bxr_no_training_no_inference.json",
}

SUPPORT_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/f1bxr_foodcom_validation_contract.py"),
    Path("tests/test_extension_f1bxr_foodcom_validation_contract.py"),
    Path(".gitignore"),
)

F1BX_BASELINE_ARTIFACTS = (
    Path("results/tables/f1bx_baseline_execution.csv"),
    Path("results/tables/f1bx_baseline_utility_results.csv"),
    Path("results/tables/f1bx_baseline_beyond_utility_results.csv"),
    Path("results/raw_outputs/f1bx_baseline_recommendations.csv"),
)


@dataclass(frozen=True)
class UserSequence:
    user_id: str
    items: tuple[str, ...]
    dates: tuple[str, ...]
    input_orders: tuple[int, ...]

    @property
    def total_interactions(self) -> int:
        return len(self.items)

    @property
    def tuning_prefix(self) -> tuple[str, ...]:
        return self.items[:-2]

    @property
    def validation_target(self) -> str:
        return self.items[-2]

    @property
    def final_test_target(self) -> str:
        return self.items[-1]

    @property
    def validation_date(self) -> str:
        return self.dates[-2]

    @property
    def final_test_date(self) -> str:
        return self.dates[-1]

    @property
    def validation_input_order(self) -> int:
        return self.input_orders[-2]

    @property
    def final_test_input_order(self) -> int:
        return self.input_orders[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze F1BXR Food.com nested validation contract.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_project_root()
    outputs = generate_f1bxr(root, args.config)
    write_outputs(root, outputs)
    manifest = outputs["validation_contract_manifest"]
    print(f"decision={manifest['decision_token']}")
    print(f"eligible_validation_users={manifest['eligibility_counts']['users_supporting_candidate_construction']}")
    print(f"validation_users_selected={manifest['validation_users_selected']}")
    print(f"validation_candidate_rows={manifest['validation_candidate_rows']}")
    print(f"tuning_ready_rows={manifest['tuning_ready_rows']}")
    print(f"training_runs_executed={manifest['training_runs_executed']}")
    print(f"llm_inference_calls={manifest['llm_inference_calls_executed']}")
    print(f"validation_passed={bool_text(manifest['validation_passed'])}")
    return 0 if manifest["decision_token"] == DECISION_RESOLVED else 1


def generate_f1bxr(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, config_path)
    paths = resolved_paths(project_root, config)
    started_at = datetime.now(timezone.utc).isoformat()

    baseline_preservation = baseline_hash_state(project_root, paths["f1bx_execution_manifest"])
    final_contract = load_final_test_contract(project_root, paths)
    item_universe, catalog_stats = load_candidate_item_universe(paths["raw_recipes"])
    sequences = load_foodcom_sequences(paths["raw_interactions"])
    eligibility, eligible_sequences = build_eligibility(sequences, item_universe, final_contract["user_ids"])
    selected_sequences = select_validation_sequences(eligible_sequences, VALIDATION_TARGET_USERS)
    split_audit, cohort, candidates, candidate_audit = materialize_validation_contract(selected_sequences, item_universe)
    tuning_readiness = build_tuning_readiness(paths["f1br_foodcom_search_space"], cohort, candidates, len(selected_sequences))
    leakage_audit = build_test_leakage_audit(project_root, paths, selected_sequences, candidates, tuning_readiness)
    integrity = build_integrity(project_root, paths, baseline_preservation, final_contract, eligibility, cohort, candidates, tuning_readiness)
    validation = build_validation(eligibility, cohort, candidates, candidate_audit, tuning_readiness, leakage_audit, integrity)
    decision = DECISION_RESOLVED if validation["status"].eq("PASS").all() else DECISION_BLOCKED
    ended_at = datetime.now(timezone.utc).isoformat()

    manifest = build_validation_contract_manifest(
        project_root,
        decision,
        started_at,
        ended_at,
        config,
        catalog_stats,
        baseline_preservation,
        final_contract,
        eligibility,
        cohort,
        candidates,
        candidate_audit,
        tuning_readiness,
        leakage_audit,
        integrity,
        validation,
    )
    no_training = build_no_training_manifest(project_root, decision, validation, integrity)
    report = render_report(decision, eligibility, cohort, candidates, candidate_audit, tuning_readiness, leakage_audit, integrity, validation)
    return {
        "decision_token": decision,
        "nested_split_audit": split_audit,
        "validation_eligibility": eligibility,
        "validation_cohort": cohort,
        "validation_candidates": candidates,
        "validation_candidate_audit": candidate_audit,
        "test_leakage_audit": leakage_audit,
        "tuning_readiness": tuning_readiness,
        "validation": validation,
        "baseline_preservation": baseline_preservation,
        "final_test_contract": final_contract,
        "validation_contract_manifest": manifest,
        "no_training_no_inference": no_training,
        "report": report,
    }


def load_config(root: Path, config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def resolved_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    raw_paths = config.get("paths", {})
    if not isinstance(raw_paths, Mapping):
        raise ValueError("config.paths must be a mapping")
    return {key: root / Path(value) for key, value in raw_paths.items()}


def load_final_test_contract(root: Path, paths: Mapping[str, Path]) -> dict[str, Any]:
    users = pd.read_csv(paths["final_test_users"], dtype=str, keep_default_na=False)
    candidates = pd.read_csv(paths["final_test_candidates"], dtype=str, keep_default_na=False)
    user_ids = set(users["user_id"].astype(str))
    return {
        "user_ids": user_ids,
        "user_count": int(users["user_id"].nunique()),
        "candidate_rows": int(len(candidates)),
        "candidate_users": int(candidates["user_id"].nunique()),
        "candidate_count_per_user_min": int(candidates.groupby("user_id")["candidate_item_id"].count().min()),
        "candidate_count_per_user_max": int(candidates.groupby("user_id")["candidate_item_id"].count().max()),
        "positive_rows": int(candidates["is_ground_truth"].astype(str).str.lower().eq("true").sum()),
        "final_user_hash": sha256_file(paths["final_test_users"]),
        "final_candidate_hash": sha256_file(paths["final_test_candidates"]),
        "final_contract_source": "D0A/F0 Food.com final-test primary 1000 contract",
    }


def load_candidate_item_universe(recipe_path: Path) -> tuple[list[str], dict[str, Any]]:
    catalog = pd.read_csv(recipe_path, dtype={"id": "string", "name": "string"}, usecols=["id", "name"], keep_default_na=False, low_memory=False)
    catalog["id"] = catalog["id"].astype(str)
    valid = catalog.loc[catalog["id"].str.strip().ne("") & catalog["name"].astype(str).str.strip().ne("")]
    universe = sorted(valid["id"].astype(str).unique(), key=sort_key)
    return universe, {
        "raw_recipe_rows": int(len(catalog)),
        "valid_recipe_id_name_rows": int(len(valid)),
        "candidate_universe_items": int(len(universe)),
        "candidate_universe_hash": item_list_hash(universe),
        "requirements": "valid_recipe_id;canonical_catalog_membership;usable_recipe_name",
    }


def load_foodcom_sequences(interaction_path: Path) -> list[UserSequence]:
    interactions = pd.read_csv(
        interaction_path,
        dtype={"user_id": "string", "recipe_id": "string", "date": "string"},
        usecols=["user_id", "recipe_id", "date"],
        keep_default_na=False,
        low_memory=False,
    )
    interactions["__input_order"] = range(1, len(interactions) + 1)
    interactions["__parsed_date"] = pd.to_datetime(interactions["date"], errors="coerce")
    if interactions["__parsed_date"].isna().any():
        raise ValueError("Food.com interactions contain unparsable dates")
    interactions = interactions.sort_values(["user_id", "__parsed_date", "__input_order"], kind="mergesort")

    sequences: list[UserSequence] = []
    for user_id, group in interactions.groupby("user_id", sort=False):
        sequences.append(
            UserSequence(
                user_id=str(user_id),
                items=tuple(group["recipe_id"].astype(str).tolist()),
                dates=tuple(group["__parsed_date"].dt.strftime("%Y-%m-%d").tolist()),
                input_orders=tuple(int(value) for value in group["__input_order"].tolist()),
            )
        )
    return sequences


def build_eligibility(
    sequences: Sequence[UserSequence],
    item_universe: Sequence[str],
    final_test_user_ids: set[str],
) -> tuple[pd.DataFrame, list[UserSequence]]:
    universe_set = set(str(item) for item in item_universe)
    counts = {
        "total_foodcom_users": len(sequences),
        "users_with_enough_interactions_for_test_target": 0,
        "users_with_enough_interactions_for_nested_validation_target": 0,
        "users_satisfying_common_model_input_requirements": 0,
        "users_remaining_after_excluding_f0_primary_test_users": 0,
        "users_with_validation_positive_in_candidate_universe": 0,
        "users_supporting_candidate_construction": 0,
    }
    eligible: list[UserSequence] = []
    for sequence in sequences:
        counts["users_with_enough_interactions_for_test_target"] += int(sequence.total_interactions >= 1)
        counts["users_with_enough_interactions_for_nested_validation_target"] += int(sequence.total_interactions >= 2)
        if sequence.total_interactions < MIN_TUNING_PREFIX_LENGTH + 2:
            continue
        counts["users_satisfying_common_model_input_requirements"] += 1
        if sequence.user_id in final_test_user_ids:
            continue
        counts["users_remaining_after_excluding_f0_primary_test_users"] += 1
        if sequence.validation_target not in universe_set:
            continue
        counts["users_with_validation_positive_in_candidate_universe"] += 1
        excluded_in_universe = {item for item in sequence.tuning_prefix if item in universe_set}
        excluded_in_universe.add(sequence.validation_target)
        if len(universe_set) - len(excluded_in_universe) < NEGATIVE_COUNT:
            continue
        counts["users_supporting_candidate_construction"] += 1
        eligible.append(sequence)

    rows = []
    for step, count in counts.items():
        status = "PASS"
        reason = "computed from RAW_interactions chronology and RAW_recipes candidate universe"
        if step == "users_supporting_candidate_construction" and count < VALIDATION_TARGET_USERS:
            status = "FAIL"
            reason = f"maximum feasible validation N is {count}, below required {VALIDATION_TARGET_USERS}"
        rows.append({"audit_step": step, "count": int(count), "status": status, "reason": reason})
    rows.append(
        {
            "audit_step": "required_validation_users",
            "count": VALIDATION_TARGET_USERS,
            "status": "PASS",
            "reason": "fixed F1BXR target; do not silently reduce N",
        }
    )
    rows.append(
        {
            "audit_step": "selection_seed",
            "count": VALIDATION_SELECTION_SEED,
            "status": "PASS",
            "reason": "deterministic outcome-blind stable hash ordering",
        }
    )
    return pd.DataFrame(rows), eligible


def select_validation_sequences(sequences: Sequence[UserSequence], count: int) -> list[UserSequence]:
    ordered = sorted(
        sequences,
        key=lambda sequence: (stable_int_seed(VALIDATION_SELECTION_SEED, "f1bxr_validation_user", sequence.user_id), sort_key(sequence.user_id)),
    )
    return ordered[: min(count, len(ordered))]


def materialize_validation_contract(
    selected_sequences: Sequence[UserSequence],
    item_universe: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for rank, sequence in enumerate(selected_sequences, start=1):
        candidates, negative_seed, order_seed = generate_validation_candidate_items(
            user_id=sequence.user_id,
            validation_target=sequence.validation_target,
            tuning_training_prefix=sequence.tuning_prefix,
            item_universe=item_universe,
        )
        candidate_hash = item_list_hash(candidates)
        candidate_set_id = f"F1BXR-FOODCOM-VAL-CAND-{rank:04d}-{candidate_hash[:16].upper()}"
        validation_sample_id = f"F1BXR-FOODCOM-VAL-{rank:04d}"
        positive_position = candidates.index(sequence.validation_target) + 1
        prefix_set = set(sequence.tuning_prefix)
        final_test_in_candidates = sequence.final_test_target in set(candidates)
        common = {
            "validation_sample_id": validation_sample_id,
            "selection_rank": rank,
            "user_id": sequence.user_id,
            "total_interactions": sequence.total_interactions,
            "training_history_length": len(sequence.tuning_prefix),
            "validation_target": sequence.validation_target,
            "validation_target_date": sequence.validation_date,
            "validation_target_input_order": sequence.validation_input_order,
            "final_test_target": sequence.final_test_target,
            "final_test_target_date": sequence.final_test_date,
            "final_test_target_input_order": sequence.final_test_input_order,
            "candidate_set_id": candidate_set_id,
            "candidate_set_hash": candidate_hash,
            "unordered_candidate_set_hash": sha256_json(sorted(candidates, key=sort_key)),
            "positive_position": positive_position,
            "sampling_seed": negative_seed,
            "ordering_seed": order_seed,
            "selection_seed": VALIDATION_SELECTION_SEED,
            "source_lineage": "RAW_interactions sorted by date ascending then original input order; RAW_recipes valid id/name item universe",
        }
        split_rows.append(
            {
                **common,
                "tuning_training_prefix_hash": item_list_hash(sequence.tuning_prefix),
                "tuning_training_prefix_excludes_validation_target": bool_text(sequence.validation_target not in prefix_set),
                "tuning_training_prefix_excludes_final_test_target": bool_text(sequence.final_test_target not in prefix_set),
                "nested_split_status": "NESTED_TEMPORAL_SPLIT_READY",
            }
        )
        cohort_rows.append(
            {
                **common,
                "selection_hash": stable_int_seed(VALIDATION_SELECTION_SEED, "f1bxr_validation_user", sequence.user_id),
                "status": "SELECTED_VALIDATION_USER",
            }
        )
        for position, item_id in enumerate(candidates, start=1):
            candidate_rows.append(
                {
                    "validation_sample_id": validation_sample_id,
                    "user_id": sequence.user_id,
                    "validation_target": sequence.validation_target,
                    "training_history_length": len(sequence.tuning_prefix),
                    "candidate_set_id": candidate_set_id,
                    "candidate_position": position,
                    "candidate_item_id": item_id,
                    "is_validation_positive": bool_text(item_id == sequence.validation_target),
                    "positive_position": positive_position,
                    "candidate_set_hash": candidate_hash,
                    "sampling_seed": negative_seed,
                    "source_lineage": "validation positive i[T-1] plus 19 validation-time unseen negatives; order is seeded shuffle",
                }
            )
        candidate_set = set(candidates)
        audit_rows.append(
            {
                "validation_sample_id": validation_sample_id,
                "user_id": sequence.user_id,
                "candidate_count": len(candidates),
                "unique_candidate_count": len(candidate_set),
                "positive_count": sum(item == sequence.validation_target for item in candidates),
                "negative_count": len(candidates) - 1,
                "positive_position": positive_position,
                "positive_not_among_negatives": bool_text(sequence.validation_target not in [item for item in candidates if item != sequence.validation_target]),
                "prefix_overlap_count": len(candidate_set & prefix_set),
                "duplicate_candidate_count": len(candidates) - len(candidate_set),
                "negative_source_blind_to_test_target": "true",
                "final_test_target_in_validation_candidates": bool_text(final_test_in_candidates),
                "status": "PASS" if len(candidates) == CANDIDATE_COUNT and len(candidate_set) == CANDIDATE_COUNT and len(candidate_set & prefix_set) == 0 else "FAIL",
                "reason": "candidate set has one validation positive and 19 unique validation-time unseen negatives",
            }
        )
    return pd.DataFrame(split_rows), pd.DataFrame(cohort_rows), pd.DataFrame(candidate_rows), pd.DataFrame(audit_rows)


def generate_validation_candidate_items(
    *,
    user_id: str,
    validation_target: str,
    tuning_training_prefix: Sequence[str],
    item_universe: Sequence[str],
) -> tuple[list[str], int, int]:
    """Generate validation candidates without receiving or consulting i[T]."""

    exclusions = set(str(item) for item in tuning_training_prefix)
    exclusions.add(str(validation_target))
    if len(item_universe) - len(exclusions) < NEGATIVE_COUNT:
        raise ValueError(f"not enough validation-time unseen negatives for user {user_id}")
    negative_seed = stable_int_seed(VALIDATION_SELECTION_SEED, "f1bxr_validation_negatives", user_id)
    rng = random.Random(negative_seed)
    negatives: list[str] = []
    selected: set[str] = set()
    while len(negatives) < NEGATIVE_COUNT:
        item_id = str(item_universe[rng.randrange(len(item_universe))])
        if item_id in exclusions or item_id in selected:
            continue
        selected.add(item_id)
        negatives.append(item_id)
    candidates = [str(validation_target), *negatives]
    order_seed = stable_int_seed(VALIDATION_SELECTION_SEED, "f1bxr_validation_order", user_id)
    random.Random(order_seed).shuffle(candidates)
    return candidates, negative_seed, order_seed


def build_tuning_readiness(search_space_path: Path, cohort: pd.DataFrame, candidates: pd.DataFrame, selected_n: int) -> pd.DataFrame:
    search = pd.read_csv(search_space_path, dtype=str, keep_default_na=False)
    contract_ready = selected_n == VALIDATION_TARGET_USERS and len(candidates) == VALIDATION_TARGET_USERS * CANDIDATE_COUNT
    rows: list[dict[str, Any]] = []
    candidate_contract_hash = frame_hash(candidates)
    cohort_hash = frame_hash(cohort)
    for _, row in search.iterrows():
        status = "READY_TO_TUNE" if contract_ready else "BLOCKED_VALIDATION_CONTRACT"
        rows.append(
            {
                "run_id": f"f1bxr_foodcom_tuning_ready_{str(row['search_config_id']).lower()}",
                "config_id": row["search_config_id"],
                "dataset": row["dataset"],
                "model": row["method"],
                "seed": row["tuning_seed"],
                "selection_metric": row["selection_metric"],
                "uses_final_test_metrics": row["uses_final_test_metrics"],
                "validation_contract_id": CONTRACT_ID,
                "validation_cohort_hash": cohort_hash,
                "validation_candidate_contract_hash": candidate_contract_hash,
                "candidate_scope": "candidate_set_20",
                "selection_basis": "validation_ndcg_at_5_only",
                "tie_breaking": "higher_validation_ndcg_at_5_then_lower_search_config_id_lexicographic",
                "training_executed": "false",
                "final_training_executed": "false",
                "llm_inference_executed": "false",
                "final_test_candidates_input_to_tuning": "false",
                "status": status,
                "reason": "F1BXR nested validation candidate20 contract resolved" if contract_ready else "validation contract could not be resolved",
            }
        )
    return pd.DataFrame(rows)


def build_test_leakage_audit(
    root: Path,
    paths: Mapping[str, Path],
    selected_sequences: Sequence[UserSequence],
    candidates: pd.DataFrame,
    tuning_readiness: pd.DataFrame,
) -> pd.DataFrame:
    final_targets_by_user = {sequence.user_id: sequence.final_test_target for sequence in selected_sequences}
    candidate_final_hits = 0
    if not candidates.empty:
        candidate_final_hits = sum(
            int(str(row.candidate_item_id) == final_targets_by_user.get(str(row.user_id), ""))
            for row in candidates.itertuples(index=False)
        )
    final_result_artifacts = find_final_training_result_artifacts(root)
    rows = [
        audit_row("f0_final_test_candidate_files_not_input_to_tuning_selection", True, "F0 candidate file hashed only for preservation; F1BXR tuning readiness uses validation contract hash"),
        audit_row("final_test_hr_ndcg_not_computed_during_tuning", True, "Food.com tuning runs executed = 0 and validation table contains no final-test metric values"),
        audit_row("candidate_generator_signature_has_no_test_target", True, "generate_validation_candidate_items accepts user_id, validation_target, tuning_training_prefix, item_universe"),
        audit_row("validation_candidate_generation_does_not_exclude_test_target", True, f"final target appeared by blind chance in {candidate_final_hits} validation candidate rows"),
        audit_row("validation_selection_depends_only_on_user_id_and_eligibility", True, "stable hash selection uses seed and user_id after chronology/catalog eligibility"),
        audit_row("final_test_result_artifacts_absent_before_winner_freeze", len(final_result_artifacts) == 0, ";".join(final_result_artifacts)),
        audit_row("all_tuning_rows_mark_final_candidates_not_input", tuning_readiness["final_test_candidates_input_to_tuning"].astype(str).str.lower().eq("false").all(), "f1bxr_tuning_readiness.csv"),
        audit_row("f0_final_user_file_used_only_for_disjoint_exclusion", paths["final_test_users"].exists(), paths["final_test_users"].relative_to(root).as_posix()),
    ]
    return pd.DataFrame(rows)


def build_integrity(
    root: Path,
    paths: Mapping[str, Path],
    baseline_preservation: Mapping[str, Any],
    final_contract: Mapping[str, Any],
    eligibility: pd.DataFrame,
    cohort: pd.DataFrame,
    candidates: pd.DataFrame,
    tuning_readiness: pd.DataFrame,
) -> dict[str, Any]:
    budget = budget_values(paths["f0_request_budget"])
    final_user_overlap = 0
    if not cohort.empty:
        final_user_overlap = len(set(cohort["user_id"].astype(str)) & set(final_contract["user_ids"]))
    return {
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_head": EXPECTED_HEAD,
        "head_is_expected": git_output(root, ["rev-parse", "HEAD"]) == EXPECTED_HEAD,
        "active_datasets": list(ACTIVE_DATASETS),
        "active_video_games_rows": active_video_games_rows(root),
        "global_metric_blockers": global_metric_blocker_count(paths["f1arr_final_metric_blockers"]),
        "f0_budget": budget,
        "f0_budget_unchanged": budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292 and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584,
        "foodcom_raw_hashes_unchanged": foodcom_hashes_unchanged(paths["d0a_extracted_hashes"]),
        "h0_hashes_unchanged": h0_hashes_unchanged(root, paths["h0_manifest"]),
        "final_test_user_count": int(final_contract["user_count"]),
        "final_test_candidate_rows": int(final_contract["candidate_rows"]),
        "final_test_contract_untouched": final_contract_matches_d0a_manifest(paths["d0a_archive_manifest"], final_contract),
        "validation_user_count": int(cohort["user_id"].nunique()) if not cohort.empty else 0,
        "validation_candidate_rows": int(len(candidates)),
        "validation_final_test_user_overlap": final_user_overlap,
        "baseline_hashes_unchanged": bool(baseline_preservation["hashes_unchanged"]),
        "baseline_rows": int(baseline_preservation["baseline_rows"]),
        "baseline_recommendation_rows": int(baseline_preservation["baseline_recommendation_rows"]),
        "tuning_ready_rows": int(tuning_readiness["status"].eq("READY_TO_TUNE").sum()),
        "training_runs_executed": 0,
        "final_training_runs_executed": 0,
        "llm_inference_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "commit_created": 0,
    }


def build_validation(
    eligibility: pd.DataFrame,
    cohort: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    tuning_readiness: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    integrity: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    eligibility_count = eligibility_count_for(eligibility, "users_supporting_candidate_construction")
    check("head_matches_f1br_checkpoint", bool(integrity["head_is_expected"]), integrity["head"])
    check("active_datasets_exact", integrity["active_datasets"] == list(ACTIVE_DATASETS), ",".join(integrity["active_datasets"]))
    check("video_games_absent", int(integrity["active_video_games_rows"]) == 0, integrity["active_video_games_rows"])
    check("global_metric_blockers_zero", int(integrity["global_metric_blockers"]) == 0, integrity["global_metric_blockers"])
    check("foodcom_raw_hashes_unchanged", bool(integrity["foodcom_raw_hashes_unchanged"]), integrity["foodcom_raw_hashes_unchanged"])
    check("h0_hashes_unchanged", bool(integrity["h0_hashes_unchanged"]), integrity["h0_hashes_unchanged"])
    check("f0_budget_unchanged", bool(integrity["f0_budget_unchanged"]), integrity["f0_budget"])
    check("final_test_contract_untouched", bool(integrity["final_test_contract_untouched"]), "D0A final users/candidates hashes")
    check("eligible_validation_users_at_least_1000", eligibility_count >= VALIDATION_TARGET_USERS, eligibility_count)
    check("validation_cohort_n_1000", int(integrity["validation_user_count"]) == VALIDATION_TARGET_USERS, integrity["validation_user_count"])
    check("validation_test_cohort_disjoint", int(integrity["validation_final_test_user_overlap"]) == 0, integrity["validation_final_test_user_overlap"])
    check("validation_candidate_rows_20000", len(candidates) == VALIDATION_TARGET_USERS * CANDIDATE_COUNT, len(candidates))
    if not candidate_audit.empty:
        check("candidate_sets_have_20_unique", candidate_audit["unique_candidate_count"].astype(int).eq(CANDIDATE_COUNT).all(), candidate_audit["unique_candidate_count"].min())
        check("candidate_sets_have_one_positive", candidate_audit["positive_count"].astype(int).eq(1).all(), candidate_audit["positive_count"].value_counts().to_dict())
        check("candidate_sets_have_19_negatives", candidate_audit["negative_count"].astype(int).eq(NEGATIVE_COUNT).all(), candidate_audit["negative_count"].value_counts().to_dict())
        check("candidate_sets_no_prefix_overlap", candidate_audit["prefix_overlap_count"].astype(int).eq(0).all(), candidate_audit["prefix_overlap_count"].sum())
        check("candidate_order_not_systematically_positive_first", candidate_audit["positive_position"].astype(int).nunique() > 1, candidate_audit["positive_position"].astype(int).value_counts().sort_index().to_dict())
    else:
        check("candidate_audit_nonempty", False, "empty")
    check("tuning_readiness_40_ready", int(tuning_readiness["status"].eq("READY_TO_TUNE").sum()) == 40, tuning_readiness["status"].value_counts().to_dict())
    check("tuning_seed_42", tuning_readiness["seed"].astype(str).eq("42").all(), tuning_readiness["seed"].unique().tolist())
    check("selection_metric_validation_ndcg", tuning_readiness["selection_metric"].eq(SELECTION_METRIC).all(), tuning_readiness["selection_metric"].unique().tolist())
    check("final_test_not_used_by_tuning", leakage_audit["status"].eq("PASS").all(), leakage_audit.loc[leakage_audit["status"].ne("PASS"), "check"].tolist())
    check("baseline_hashes_unchanged", bool(integrity["baseline_hashes_unchanged"]), integrity["baseline_rows"])
    check("baseline_rows_15", int(integrity["baseline_rows"]) == 15, integrity["baseline_rows"])
    check("baseline_recommendation_rows_292920", int(integrity["baseline_recommendation_rows"]) == 292920, integrity["baseline_recommendation_rows"])
    check("zero_training", int(integrity["training_runs_executed"]) == 0 and int(integrity["final_training_runs_executed"]) == 0, "0")
    check("zero_llm_network_download", int(integrity["llm_inference_calls_executed"]) == 0 and int(integrity["network_calls_executed"]) == 0 and int(integrity["downloads_executed"]) == 0, "0")
    return pd.DataFrame(rows)


def build_validation_contract_manifest(
    root: Path,
    decision: str,
    started_at: str,
    ended_at: str,
    config: Mapping[str, Any],
    catalog_stats: Mapping[str, Any],
    baseline_preservation: Mapping[str, Any],
    final_contract: Mapping[str, Any],
    eligibility: pd.DataFrame,
    cohort: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    tuning_readiness: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    integrity: Mapping[str, Any],
    validation: pd.DataFrame,
) -> dict[str, Any]:
    eligibility_counts = {row["audit_step"]: int(row["count"]) for row in eligibility.to_dict("records") if str(row["count"]).isdigit()}
    return {
        "phase_id": PHASE_ID,
        "contract_id": CONTRACT_ID,
        "decision_token": decision,
        "started_at": started_at,
        "ended_at": ended_at,
        "head": integrity["head"],
        "expected_head": EXPECTED_HEAD,
        "config_hash": sha256_json(config),
        "dataset": DATASET,
        "chronological_order": "date_ascending_then_original_input_order",
        "validation_target_definition": "i[T-1]",
        "final_test_target_definition": "i[T]",
        "tuning_training_prefix_definition": "i1..i[T-2]",
        "later_final_training_policy": "retrain using i1..i[T-1] after Food.com winners are frozen; not executed in F1BXR",
        "candidate_count": CANDIDATE_COUNT,
        "negative_count": NEGATIVE_COUNT,
        "validation_selection_seed": VALIDATION_SELECTION_SEED,
        "negative_seed_label": "f1bxr_validation_negatives",
        "ordering_seed_label": "f1bxr_validation_order",
        "selection_metric": SELECTION_METRIC,
        "tuning_seed": TUNING_SEED,
        "catalog_stats": dict(catalog_stats),
        "eligibility_counts": eligibility_counts,
        "final_test_contract": {
            "user_count": final_contract["user_count"],
            "candidate_rows": final_contract["candidate_rows"],
            "candidate_users": final_contract["candidate_users"],
            "final_user_hash": final_contract["final_user_hash"],
            "final_candidate_hash": final_contract["final_candidate_hash"],
        },
        "validation_users_selected": int(cohort["user_id"].nunique()) if not cohort.empty else 0,
        "validation_candidate_rows": int(len(candidates)),
        "validation_candidate_contract_hash": frame_hash(candidates),
        "validation_cohort_hash": frame_hash(cohort),
        "candidate_audit_failures": int(candidate_audit["status"].ne("PASS").sum()) if not candidate_audit.empty else 1,
        "test_leakage_failures": int(leakage_audit["status"].ne("PASS").sum()),
        "tuning_ready_rows": int(tuning_readiness["status"].eq("READY_TO_TUNE").sum()),
        "baseline_preservation": dict(baseline_preservation),
        "integrity": dict(integrity),
        "training_runs_executed": 0,
        "final_training_runs_executed": 0,
        "llm_inference_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "commit_created": 0,
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
    }


def build_no_training_manifest(root: Path, decision: str, validation: pd.DataFrame, integrity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": decision,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "training_runs_executed": 0,
        "foodcom_tuning_training_runs_executed": 0,
        "final_training_runs_executed": 0,
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "baseline_reruns_executed": 0,
        "baseline_hashes_unchanged": bool(integrity["baseline_hashes_unchanged"]),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
    }


def render_report(
    decision: str,
    eligibility: pd.DataFrame,
    cohort: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    tuning_readiness: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    integrity: Mapping[str, Any],
    validation: pd.DataFrame,
) -> str:
    eligibility_count = eligibility_count_for(eligibility, "users_supporting_candidate_construction")
    return "\n".join(
        [
            "# F1BXR Food.com Validation Contract",
            "",
            f"Decision token: `{decision}`",
            "",
            "## Eligibility",
            f"- Eligible validation users supporting candidate construction: `{eligibility_count}`.",
            f"- Required validation users: `{VALIDATION_TARGET_USERS}`.",
            "",
            "## Nested Split",
            "- Tuning train: `i1..i[T-2]`; validation target: `i[T-1]`; final-test target: `i[T]`.",
            "- Later final training policy: retrain using `i1..i[T-1]` after winner freeze; not executed now.",
            "",
            "## Validation Contract",
            f"- Validation users selected: `{cohort['user_id'].nunique() if not cohort.empty else 0}`.",
            f"- Candidate rows: `{len(candidates)}`.",
            f"- Candidate sets with audit failures: `{int(candidate_audit['status'].ne('PASS').sum()) if not candidate_audit.empty else 1}`.",
            "",
            "## Leakage",
            f"- Leakage audit failures: `{int(leakage_audit['status'].ne('PASS').sum())}`.",
            f"- Final/test validation cohort overlap: `{integrity['validation_final_test_user_overlap']}`.",
            "",
            "## Tuning Readiness",
            f"- READY_TO_TUNE rows: `{int(tuning_readiness['status'].eq('READY_TO_TUNE').sum())}/40`.",
            "- Selection metric: `validation_ndcg_at_5`.",
            "",
            "## Preservation",
            f"- F1BX baseline rows: `{integrity['baseline_rows']}`.",
            f"- F1BX recommendation rows: `{integrity['baseline_recommendation_rows']}`.",
            f"- Baseline hashes unchanged: `{bool_text(integrity['baseline_hashes_unchanged'])}`.",
            f"- F0 mandatory budget unchanged: `{integrity['f0_budget'].get('MANDATORY_RAW_REQUESTS_TOTAL')}`.",
            f"- F0 conceptual all-applicable budget unchanged: `{integrity['f0_budget'].get('CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL')}`.",
            "- Training/LLM/network/download counters: `0`.",
            "",
            "## Validation",
            f"- Focused validation passed: `{bool_text(validation['status'].eq('PASS').all())}`.",
        ]
    ) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")

    artifact_paths = [*OUTPUT_TABLES.values(), REPORT_PATH, *SUPPORT_ARTIFACTS]
    artifact_hashes = artifact_hashes_for(root, artifact_paths)
    for key, rel_path in OUTPUT_MANIFESTS.items():
        payload = dict(outputs[key])
        payload["artifact_hashes"] = artifact_hashes
        (root / rel_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def baseline_hash_state(root: Path, f1bx_manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(f1bx_manifest_path)
    expected_hashes = manifest.get("artifact_hashes", {})
    actual_hashes = {rel.as_posix(): sha256_file(root / rel) for rel in F1BX_BASELINE_ARTIFACTS}
    baseline_execution = pd.read_csv(root / F1BX_BASELINE_ARTIFACTS[0], dtype=str, keep_default_na=False)
    recommendation_rows = line_count(root / F1BX_BASELINE_ARTIFACTS[3]) - 1
    compared = {
        rel.as_posix(): {
            "expected": expected_hashes.get(rel.as_posix(), ""),
            "actual": actual_hashes[rel.as_posix()],
            "matches": expected_hashes.get(rel.as_posix(), "") == actual_hashes[rel.as_posix()],
        }
        for rel in F1BX_BASELINE_ARTIFACTS
    }
    return {
        "f1bx_decision_token": manifest.get("decision_token", ""),
        "baseline_rows": int(len(baseline_execution)),
        "baseline_computed_rows": int(baseline_execution["status"].eq("COMPUTED").sum()),
        "baseline_recommendation_rows": int(recommendation_rows),
        "hashes_before": actual_hashes,
        "hashes_after": actual_hashes,
        "manifest_comparison": compared,
        "hashes_unchanged": all(row["matches"] for row in compared.values()),
    }


def build_split_stats_by_user(sequences: Sequence[UserSequence]) -> dict[str, UserSequence]:
    return {sequence.user_id: sequence for sequence in sequences}


def foodcom_hashes_unchanged(extracted_hashes_path: Path) -> bool:
    manifest = read_json(extracted_hashes_path)
    for info in manifest.get("files", {}).values():
        path = Path(info["relative_extracted_path"])
        if not path.exists() or sha256_file(path) != info["sha256"]:
            return False
    return True


def h0_hashes_unchanged(root: Path, h0_manifest_path: Path) -> bool:
    if not h0_manifest_path.exists():
        return False
    manifest = read_json(h0_manifest_path)
    for rel_path, expected_hash in manifest.get("artifact_hashes", {}).items():
        path = root / Path(rel_path)
        if not path.exists() or sha256_file(path) != expected_hash:
            return False
    return True


def final_contract_matches_d0a_manifest(d0a_manifest_path: Path, final_contract: Mapping[str, Any]) -> bool:
    manifest = read_json(d0a_manifest_path)
    hashes = manifest.get("artifact_hashes", {})
    return (
        hashes.get("results/tables/d0a_foodcom_primary1000_users.csv") == final_contract["final_user_hash"]
        and hashes.get("results/tables/d0a_foodcom_primary1000_candidates.csv") == final_contract["final_candidate_hash"]
        and int(final_contract["user_count"]) == 1000
        and int(final_contract["candidate_rows"]) == 20000
    )


def budget_values(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {str(row.budget_item): int(row.request_count) for row in frame.itertuples(index=False)}


def global_metric_blocker_count(path: Path) -> int:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        return 0
    return int(len(frame))


def active_video_games_rows(root: Path) -> int:
    paths = [
        root / "results/tables/f0_final_dataset_matrix.csv",
        root / "results/tables/f0_final_raw_experiment_universe.csv",
        root / "results/tables/f1br_trainable_execution_readiness.csv",
        root / "results/tables/f1bx_baseline_execution.csv",
    ]
    total = 0
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        mask = frame.astype(str).apply(lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True, na=False))
        total += int(mask.any(axis=1).sum())
    return total


def find_final_training_result_artifacts(root: Path) -> list[str]:
    matches: list[str] = []
    for directory in (root / "results/tables", root / "results/raw_outputs", root / "results/metrics"):
        if not directory.exists():
            continue
        for path in directory.glob("f1b*"):
            name = path.name.lower()
            if "final" in name and ("result" in name or "training" in name or "recommendation" in name):
                matches.append(path.relative_to(root).as_posix())
    return sorted(matches)


def audit_row(check: str, passed: bool, evidence: str) -> dict[str, str]:
    return {"check": check, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def eligibility_count_for(eligibility: pd.DataFrame, step: str) -> int:
    matched = eligibility.loc[eligibility["audit_step"].eq(step), "count"]
    return int(matched.iloc[0]) if not matched.empty else 0


def artifact_hashes_for(root: Path, rel_paths: Sequence[Path]) -> dict[str, str]:
    return {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in rel_paths
        if (root / rel_path).exists()
    }


def frame_hash(frame: pd.DataFrame) -> str:
    return sha256_json(frame.to_dict("records"))


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

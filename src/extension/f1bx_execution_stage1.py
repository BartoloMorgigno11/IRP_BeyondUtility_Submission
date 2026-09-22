"""F1BX deterministic baselines and Food.com tuning gate.

The phase executes only deterministic candidate-constrained baselines. It
materializes Food.com tuning rows as terminal blocked rows when the frozen
validation candidate contract is unavailable, preventing final-test leakage or
implicit protocol invention.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import numpy as np
import pandas as pd
import yaml

from src.extension.e0_protocol import sha256_json, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1a_offline_metric_closure import (
    average_recommendation_popularity,
    calibration_at_k,
    clean,
    exposure_counts,
    format_float,
    gini_index,
    ild_jaccard_at_k,
    jain_index,
    long_tail_rate_at_k,
    ndcg_at_k_from_rank,
    oic_at_k,
    reciprocal_rank,
    self_information_at_k,
    shannon_category_entropy_at_k,
)
from src.extension.f1arr_popreo_dpd_recovery import (
    dpd_at_k_from_ranks,
    popreo_at_k_from_hits,
)
from src.utils.config import find_project_root


PHASE_ID = "f1bx_traditional_execution_stage1"
DECISION_VALIDATED = "F1BX_TUNING_AND_BASELINES_VALIDATED"
DECISION_BLOCKED = "F1BX_TUNING_AND_BASELINES_BLOCKED"
EXPECTED_HEAD = "61fbde6d048faaed9e20508484efc90fa95fa5bf"
ACTIVE_DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
BASELINE_METHODS = ("Random", "MostPop", "BM25-like/content")
TRAINABLE_MODELS = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
TOP_K = 5

UTILITY_METRICS = ("hr_at_5", "ndcg_at_5", "mrr_diagnostic")
BEYOND_METRICS = (
    "self_information_at_5",
    "serendipity_at_5",
    "arp_at_5",
    "aplt_at_5",
    "popreo_at_5",
    "item_coverage_at_5",
    "catalog_coverage_at_5",
    "oic_at_5",
    "shannon_category_entropy_at_5",
    "gini_at_5",
    "dpd_at_5",
    "jain_index_at_5",
    "ild_at_5",
    "calibration_at_5",
)

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
RAW_OUTPUT_DIR = Path("results/raw_outputs")
REPORT_DIR = Path("report")
CONFIG_PATH = Path("configs/f1bx_execution_stage1.yaml")
REPORT_PATH = REPORT_DIR / "f1bx_baselines_and_foodcom_tuning.md"
RECOMMENDATIONS_PATH = RAW_OUTPUT_DIR / "f1bx_baseline_recommendations.csv"

OUTPUT_TABLES = {
    "baseline_execution": TABLE_DIR / "f1bx_baseline_execution.csv",
    "baseline_utility_results": TABLE_DIR / "f1bx_baseline_utility_results.csv",
    "baseline_beyond_utility_results": TABLE_DIR / "f1bx_baseline_beyond_utility_results.csv",
    "foodcom_tuning_runs": TABLE_DIR / "f1bx_foodcom_tuning_runs.csv",
    "foodcom_validation_results": TABLE_DIR / "f1bx_foodcom_validation_results.csv",
    "foodcom_selected_configs": TABLE_DIR / "f1bx_foodcom_selected_configs.csv",
    "runtime_provenance": TABLE_DIR / "f1bx_runtime_provenance.csv",
    "validation": TABLE_DIR / "f1bx_validation.csv",
}

OUTPUT_MANIFESTS = {
    "execution_manifest": MANIFEST_DIR / "f1bx_execution_manifest.json",
    "foodcom_selection_freeze": MANIFEST_DIR / "f1bx_foodcom_selection_freeze.json",
    "no_llm_inference": MANIFEST_DIR / "f1bx_no_llm_inference.json",
}

SUPPORT_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/f1bx_execution_stage1.py"),
    Path("tests/test_extension_f1bx_execution_stage1.py"),
    Path(".gitignore"),
)

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    cohort_id: str
    evaluation_users: int
    candidate_source: Path
    cohort_filter_source: Path
    interaction_source: Path
    catalog_source: Path
    feature_source: str
    bm25_text_source: str


@dataclass(frozen=True)
class DatasetMaterial:
    spec: DatasetSpec
    candidates: pd.DataFrame
    train_counts: Counter[str]
    total_train_interactions: int
    history_items_by_user: dict[str, list[str]]
    history_lengths_by_user: dict[str, int]
    catalog_items: list[str]
    bm25_terms_by_item: dict[str, Counter[str]]
    bm25_doc_lengths: dict[str, int]
    bm25_idf: dict[str, float]
    bm25_average_doc_length: float
    item_features: dict[str, list[str]]
    candidate_set_hash: str
    target_leakage_count: int
    catalog_missing_candidate_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1BX deterministic baseline stage and Food.com tuning gate.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_f1bx(root, Path(args.config))
    if not args.validate_only:
        write_outputs(root, outputs)
    validation = outputs["validation"]
    manifest = outputs["execution_manifest"]
    print(f"decision={manifest['decision_token']}")
    print(f"baseline_rows={len(outputs['baseline_execution'])}")
    print(f"baseline_computed={int(outputs['baseline_execution']['status'].eq('COMPUTED').sum())}")
    print(f"foodcom_tuning_planned={len(outputs['foodcom_tuning_runs'])}")
    print(f"foodcom_tuning_executed={manifest['foodcom_tuning_executed']}")
    print(f"foodcom_winners_selected={manifest['foodcom_winners_selected']}")
    print(f"final_training_runs_executed={manifest['final_training_runs_executed']}")
    print(f"llm_inference_calls={manifest['llm_inference_calls_executed']}")
    print(f"validation_passed={str(validation['status'].eq('PASS').all()).lower()}")
    return 0 if bool(validation["status"].eq("PASS").all()) else 1


def generate_f1bx(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, config_path)
    specs = dataset_specs(config)
    runtime = resource_preflight(project_root)
    materials = {spec.name: build_dataset_material(project_root, spec) for spec in specs}
    timestamp = datetime.now(timezone.utc).isoformat()

    baseline_execution, recommendations, utility, beyond = execute_baselines(config, materials, timestamp)
    tuning_runs, validation_results, selected_configs, tuning_blocker = build_foodcom_tuning_gate(project_root, config, runtime, timestamp)
    integrity = build_integrity(project_root, materials, baseline_execution, tuning_runs, selected_configs)
    validation = build_validation(
        project_root,
        config,
        materials,
        baseline_execution,
        recommendations,
        utility,
        beyond,
        tuning_runs,
        validation_results,
        selected_configs,
        tuning_blocker,
        integrity,
    )
    decision = DECISION_VALIDATED if int(selected_configs["selected"].astype(str).str.lower().eq("true").sum()) == 4 else DECISION_BLOCKED
    execution_manifest = build_execution_manifest(
        project_root,
        decision,
        config,
        runtime,
        baseline_execution,
        recommendations,
        utility,
        beyond,
        tuning_runs,
        selected_configs,
        tuning_blocker,
        integrity,
        validation,
    )
    selection_freeze = build_selection_freeze_manifest(project_root, decision, selected_configs, tuning_runs, tuning_blocker)
    no_llm = build_no_llm_manifest(project_root, decision, validation, integrity)
    report = render_report(decision, runtime, baseline_execution, utility, beyond, tuning_runs, selected_configs, tuning_blocker, integrity, validation)
    return {
        "decision_token": decision,
        "runtime_provenance": runtime,
        "baseline_execution": baseline_execution,
        "baseline_recommendations": recommendations,
        "baseline_utility_results": utility,
        "baseline_beyond_utility_results": beyond,
        "foodcom_tuning_runs": tuning_runs,
        "foodcom_validation_results": validation_results,
        "foodcom_selected_configs": selected_configs,
        "tuning_blocker": tuning_blocker,
        "integrity": integrity,
        "validation": validation,
        "execution_manifest": execution_manifest,
        "foodcom_selection_freeze": selection_freeze,
        "no_llm_inference": no_llm,
        "report": report,
    }


def load_config(root: Path, config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def dataset_specs(config: Mapping[str, Any]) -> list[DatasetSpec]:
    specs = []
    for row in config["datasets"]:
        specs.append(
            DatasetSpec(
                name=str(row["name"]),
                display_name=str(row["display_name"]),
                cohort_id=str(row["cohort_id"]),
                evaluation_users=int(row["evaluation_users"]),
                candidate_source=Path(str(row["candidate_source"])),
                cohort_filter_source=Path(str(row["cohort_filter_source"])),
                interaction_source=Path(str(row["interaction_source"])),
                catalog_source=Path(str(row["catalog_source"])),
                feature_source=str(row["feature_source"]),
                bm25_text_source=str(row["bm25_text_source"]),
            )
        )
    return specs


def build_dataset_material(root: Path, spec: DatasetSpec) -> DatasetMaterial:
    candidates = load_candidates(root, spec)
    train_counts, history_items, history_lengths = load_training_material(root, spec, candidates["user_id"].tolist())
    catalog_items, terms_by_item, doc_lengths, idf, avgdl, features = load_catalog_material(root, spec)
    candidate_set_hash = frame_hash(candidates[["user_id", "candidate_position", "candidate_item_id", "is_ground_truth", "positive_item_id"]])
    target_leakage_count = count_target_leakage(candidates, history_items)
    candidate_items = set(candidates["candidate_item_id"].astype(str))
    catalog_missing = len(candidate_items - set(catalog_items))
    return DatasetMaterial(
        spec=spec,
        candidates=candidates,
        train_counts=train_counts,
        total_train_interactions=sum(train_counts.values()),
        history_items_by_user=history_items,
        history_lengths_by_user=history_lengths,
        catalog_items=catalog_items,
        bm25_terms_by_item=terms_by_item,
        bm25_doc_lengths=doc_lengths,
        bm25_idf=idf,
        bm25_average_doc_length=avgdl,
        item_features=features,
        candidate_set_hash=candidate_set_hash,
        target_leakage_count=target_leakage_count,
        catalog_missing_candidate_count=catalog_missing,
    )


def load_candidates(root: Path, spec: DatasetSpec) -> pd.DataFrame:
    source = root / spec.candidate_source
    if spec.name in {"beauty", "sports", "ml1m"}:
        manifest = pd.read_csv(source, dtype=str, keep_default_na=False)
        cohort_users = canonical_primary_users(root, spec)
        frame = manifest.loc[manifest["user_id"].astype(str).isin(cohort_users)].copy()
        frame = frame.rename(columns={"item_id": "candidate_item_id", "is_positive": "is_ground_truth"})
        positives = frame.loc[bool_series(frame["is_ground_truth"]), ["user_id", "candidate_item_id"]].rename(columns={"candidate_item_id": "positive_item_id"})
        frame = frame.merge(positives, on="user_id", how="left")
    elif spec.name == "lastfm":
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        sample = pd.read_csv(root / spec.cohort_filter_source, dtype=str, keep_default_na=False)
        cohort_users = set(sample["user_id"].astype(str))
        frame = frame.loc[frame["user_id"].astype(str).isin(cohort_users)].copy()
        frame["positive_item_id"] = frame["ground_truth_item_id"].astype(str)
    elif spec.name == "foodcom":
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported dataset: {spec.name}")

    frame = normalize_candidate_frame(frame, spec)
    validate_candidate_contract(frame, spec)
    return frame


def canonical_primary_users(root: Path, spec: DatasetSpec) -> set[str]:
    preflight = pd.read_csv(root / spec.cohort_filter_source, dtype=str, keep_default_na=False)
    selected = preflight.loc[
        preflight["dataset"].eq(spec.display_name)
        & preflight["task_type"].eq("ranking")
        & preflight["primary_prompt_compatible"].astype(str).str.lower().eq("true")
    ].copy()
    return set(selected["user_id"].astype(str))


def normalize_candidate_frame(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    required = {"user_id", "candidate_item_id", "candidate_position", "is_ground_truth"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"{spec.name} candidate frame missing columns: {sorted(missing)}")
    if "positive_item_id" not in normalized.columns:
        positives = normalized.loc[bool_series(normalized["is_ground_truth"]), ["user_id", "candidate_item_id"]].rename(columns={"candidate_item_id": "positive_item_id"})
        normalized = normalized.merge(positives, on="user_id", how="left")
    normalized["dataset"] = spec.name
    normalized["cohort_id"] = spec.cohort_id
    normalized["user_id"] = normalized["user_id"].astype(str)
    normalized["candidate_item_id"] = normalized["candidate_item_id"].astype(str)
    normalized["positive_item_id"] = normalized["positive_item_id"].astype(str)
    normalized["candidate_position"] = pd.to_numeric(normalized["candidate_position"], errors="raise").astype(int)
    normalized["is_ground_truth"] = bool_series(normalized["is_ground_truth"])
    normalized = normalized.sort_values(["user_id", "candidate_position"], kind="mergesort").reset_index(drop=True)
    return normalized[
        [
            "dataset",
            "cohort_id",
            "user_id",
            "candidate_position",
            "candidate_item_id",
            "is_ground_truth",
            "positive_item_id",
        ]
    ]


def validate_candidate_contract(frame: pd.DataFrame, spec: DatasetSpec) -> None:
    users = frame["user_id"].nunique()
    if users != spec.evaluation_users:
        raise ValueError(f"{spec.name} expected {spec.evaluation_users} users, observed {users}")
    per_user = frame.groupby("user_id", sort=False).size()
    if not per_user.eq(20).all():
        raise ValueError(f"{spec.name} candidate pool must have exactly 20 rows per user")
    positives = frame.groupby("user_id", sort=False)["is_ground_truth"].sum()
    if not positives.eq(1).all():
        raise ValueError(f"{spec.name} candidate pool must have exactly one positive per user")


def load_training_material(root: Path, spec: DatasetSpec, selected_users: Sequence[str]) -> tuple[Counter[str], dict[str, list[str]], dict[str, int]]:
    selected = set(str(user) for user in selected_users)
    if spec.name == "foodcom":
        return load_foodcom_train_material(root, spec, selected)
    path = root / spec.interaction_source
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=lambda column: column in {"user_id", "item_id", "split", "timestamp"})
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).str.lower().eq("train")].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    if "timestamp" in frame.columns:
        frame["__time"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(-1)
    else:
        frame["__time"] = -1
    frame = frame.sort_values(["user_id", "__time"], kind="mergesort")
    counts = Counter(frame["item_id"].tolist())
    history: dict[str, list[str]] = {
        user_id: group["item_id"].astype(str).tolist()
        for user_id, group in frame.loc[frame["user_id"].isin(selected)].groupby("user_id", sort=False)
    }
    lengths = {user_id: len(history.get(user_id, [])) + 1 for user_id in selected}
    return counts, history, lengths


def load_foodcom_train_material(root: Path, spec: DatasetSpec, selected_users: set[str]) -> tuple[Counter[str], dict[str, list[str]], dict[str, int]]:
    path = root / spec.interaction_source
    frame = pd.read_csv(
        path,
        dtype={"user_id": "string", "recipe_id": "string", "date": "string"},
        keep_default_na=False,
        usecols=["user_id", "recipe_id", "date"],
        low_memory=False,
    )
    frame["user_id"] = frame["user_id"].astype(str)
    frame["recipe_id"] = frame["recipe_id"].astype(str)
    frame["__date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["__input_order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["user_id", "__date", "__input_order"], kind="mergesort").reset_index(drop=True)
    grouped = frame.groupby("user_id", sort=False)
    group_size = grouped["recipe_id"].transform("size").astype(int)
    within_user = grouped.cumcount()
    train = frame.loc[within_user < group_size - 1, ["user_id", "recipe_id"]].copy()
    counts = Counter(train["recipe_id"].tolist())
    selected_train = train.loc[train["user_id"].isin(selected_users)].copy()
    history = {
        user_id: group["recipe_id"].astype(str).tolist()
        for user_id, group in selected_train.groupby("user_id", sort=False)
    }
    lengths = {user_id: len(history.get(user_id, [])) + 1 for user_id in selected_users}
    return counts, history, lengths


def load_catalog_material(root: Path, spec: DatasetSpec) -> tuple[list[str], dict[str, Counter[str]], dict[str, int], dict[str, float], float, dict[str, list[str]]]:
    path = root / spec.catalog_source
    if spec.name == "foodcom":
        catalog = pd.read_csv(path, dtype={"id": "string", "tags": "string"}, keep_default_na=False, usecols=["id", "tags"], low_memory=False)
        catalog = pd.DataFrame(
            {
                "item_id": catalog["id"].astype(str),
                "bm25_text": [tags_to_text(value) for value in catalog["tags"]],
                "features": [parse_feature_list(value) for value in catalog["tags"]],
            }
        )
    else:
        catalog = pd.read_csv(path, dtype=str, keep_default_na=False)
        catalog.columns = [str(column).strip() for column in catalog.columns]
        text_col = "normalized_text" if "normalized_text" in catalog.columns else "title"
        feature_col = "tag_values" if spec.feature_source == "tag_values" and "tag_values" in catalog.columns else "categories"
        catalog = pd.DataFrame(
            {
                "item_id": catalog["item_id"].astype(str),
                "bm25_text": catalog[text_col].astype(str),
                "features": [parse_pipe_features(value) for value in catalog.get(feature_col, pd.Series("", index=catalog.index))],
            }
        )
    catalog = catalog.drop_duplicates("item_id", keep="first").reset_index(drop=True)
    terms_by_item: dict[str, Counter[str]] = {}
    doc_lengths: dict[str, int] = {}
    document_frequency: Counter[str] = Counter()
    features: dict[str, list[str]] = {}
    for row in catalog.itertuples(index=False):
        item_id = str(row.item_id)
        terms = Counter(tokenize(row.bm25_text))
        terms_by_item[item_id] = terms
        doc_lengths[item_id] = int(sum(terms.values()))
        document_frequency.update(terms.keys())
        features[item_id] = sorted({str(value).strip().lower() for value in row.features if str(value).strip()})
    document_count = len(terms_by_item)
    avgdl = float(sum(doc_lengths.values()) / document_count) if document_count else 0.0
    idf = {
        term: math.log(1.0 + ((document_count - df + 0.5) / (df + 0.5)))
        for term, df in document_frequency.items()
    }
    return list(catalog["item_id"].astype(str)), terms_by_item, doc_lengths, idf, avgdl, features


def execute_baselines(
    config: Mapping[str, Any],
    materials: Mapping[str, DatasetMaterial],
    timestamp: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recommendation_frames: list[pd.DataFrame] = []
    execution_rows: list[dict[str, Any]] = []
    per_run_user_stats: dict[tuple[str, str], pd.DataFrame] = {}
    runtimes: dict[tuple[str, str], float] = {}

    for dataset in ACTIVE_DATASETS:
        material = materials[dataset]
        for method in BASELINE_METHODS:
            started = time.perf_counter()
            run_id = f"f1bx_{dataset}_{method_slug(method)}"
            recs = rank_baseline(material, method, config["baselines"][method], run_id, timestamp)
            runtime_seconds = time.perf_counter() - started
            runtimes[(dataset, method)] = runtime_seconds
            recommendation_frames.append(recs)
            user_stats = per_user_stats(recs, material)
            per_run_user_stats[(dataset, method)] = user_stats
            utility_values = utility_metric_values(user_stats)
            execution_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "method": method,
                    "cohort_id": material.spec.cohort_id,
                    "cohort_N": material.spec.evaluation_users,
                    "seed": baseline_seed(method, config["baselines"][method]),
                    "candidate_set_identity": material.candidate_set_hash,
                    "candidate_source": material.spec.candidate_source.as_posix(),
                    "recommendation_output_hash": frame_hash(recs),
                    "hr_at_5": format_float(utility_values["hr_at_5"]),
                    "ndcg_at_5": format_float(utility_values["ndcg_at_5"]),
                    "mrr": format_float(utility_values["mrr_diagnostic"]),
                    "runtime_seconds": format_float(runtime_seconds),
                    "status": "COMPUTED",
                    "reason": "F1BX deterministic baseline executed on frozen candidate-constrained cohort.",
                }
            )

    recommendations = pd.concat(recommendation_frames, ignore_index=True)
    baseline_execution = pd.DataFrame(execution_rows)
    utility = build_utility_results(baseline_execution, per_run_user_stats)
    beyond = build_beyond_results(baseline_execution, per_run_user_stats, materials)
    return baseline_execution, recommendations, utility, beyond


def rank_baseline(material: DatasetMaterial, method: str, method_config: Mapping[str, Any], run_id: str, timestamp: str) -> pd.DataFrame:
    frame = material.candidates.copy()
    if method == "Random":
        seed = int(method_config["seed"])
        frame["score"] = [
            stable_random_score(seed, material.spec.name, user_id, item_id)
            for user_id, item_id in zip(frame["user_id"], frame["candidate_item_id"])
        ]
    elif method == "MostPop":
        frame["score"] = [float(material.train_counts.get(str(item_id), 0)) for item_id in frame["candidate_item_id"]]
    elif method == "BM25-like/content":
        profiles = bm25_profiles(material)
        frame["score"] = [
            bm25_score(
                profiles.get(str(user_id), Counter()),
                material.bm25_terms_by_item.get(str(item_id), Counter()),
                material.bm25_doc_lengths.get(str(item_id), 0),
                material,
            )
            for user_id, item_id in zip(frame["user_id"], frame["candidate_item_id"])
        ]
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    sort_cols = item_sort_columns(frame["candidate_item_id"])
    for column, values in sort_cols.items():
        frame[column] = values
    frame = frame.sort_values(
        ["user_id", "score", "candidate_position", "__item_group", "__item_number", "__item_text"],
        ascending=[True, False, True, True, True, True],
        kind="mergesort",
    ).copy()
    frame["rank"] = frame.groupby("user_id", sort=False).cumcount() + 1
    return pd.DataFrame(
        {
            "run_id": run_id,
            "dataset": material.spec.name,
            "method": method,
            "cohort_id": material.spec.cohort_id,
            "user_id": frame["user_id"].astype(str),
            "rank": frame["rank"].astype(int),
            "recommended_item_id": frame["candidate_item_id"].astype(str),
            "score": frame["score"].astype(float),
            "candidate_position": frame["candidate_position"].astype(int),
            "is_ground_truth": frame["is_ground_truth"].astype(bool),
            "positive_item_id": frame["positive_item_id"].astype(str),
            "candidate_set_hash": material.candidate_set_hash,
            "candidate_source": material.spec.candidate_source.as_posix(),
            "timestamp": timestamp,
        }
    )


def bm25_profiles(material: DatasetMaterial) -> dict[str, Counter[str]]:
    profiles: dict[str, Counter[str]] = {}
    for user_id, history in material.history_items_by_user.items():
        profile: Counter[str] = Counter()
        for item_id in history:
            terms = material.bm25_terms_by_item.get(str(item_id))
            if terms:
                profile.update(terms)
        profiles[user_id] = profile
    return profiles


def bm25_score(profile_terms: Counter[str], item_terms: Counter[str], item_length: int, material: DatasetMaterial) -> float:
    if not profile_terms or not item_terms or material.bm25_average_doc_length <= 0:
        return 0.0
    k1 = 1.5
    b = 0.75
    length_norm = 1.0 - b + b * (item_length / material.bm25_average_doc_length)
    score = 0.0
    for term, tf in item_terms.items():
        qtf = profile_terms.get(term, 0)
        if qtf <= 0:
            continue
        denominator = tf + k1 * length_norm
        if denominator > 0:
            score += material.bm25_idf.get(term, 0.0) * qtf * ((tf * (k1 + 1.0)) / denominator)
    return float(score)


def per_user_stats(recommendations: pd.DataFrame, material: DatasetMaterial) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for user_id, group in recommendations.sort_values(["user_id", "rank"], kind="mergesort").groupby("user_id", sort=False):
        ranked = group.sort_values("rank", kind="mergesort")
        top_items = ranked["recommended_item_id"].astype(str).tolist()
        positive = str(ranked["positive_item_id"].iloc[0])
        rank = None
        for _, row in ranked.iterrows():
            if bool_value(row["is_ground_truth"]) or str(row["recommended_item_id"]) == positive:
                rank = int(row["rank"])
                break
        rank_zero = None if rank is None else rank - 1
        rows.append(
            {
                "dataset": material.spec.name,
                "method": str(ranked["method"].iloc[0]),
                "user_id": str(user_id),
                "positive_item_id": positive,
                "rank_one_based": "" if rank is None else rank,
                "hit_at_5": 1 if rank is not None and rank <= TOP_K else 0,
                "ndcg_at_5": ndcg_at_k_from_rank(rank_zero, TOP_K),
                "mrr_diagnostic": reciprocal_rank(rank_zero),
                "history_length": int(material.history_lengths_by_user.get(str(user_id), 0)),
                "top_items": top_items[:TOP_K],
            }
        )
    return pd.DataFrame(rows)


def utility_metric_values(user_stats: pd.DataFrame) -> dict[str, float]:
    return {
        "hr_at_5": float(pd.to_numeric(user_stats["hit_at_5"]).mean()),
        "ndcg_at_5": float(pd.to_numeric(user_stats["ndcg_at_5"]).mean()),
        "mrr_diagnostic": float(pd.to_numeric(user_stats["mrr_diagnostic"]).mean()),
    }


def build_utility_results(baseline_execution: pd.DataFrame, per_run_user_stats: Mapping[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for execution in baseline_execution.to_dict("records"):
        stats = per_run_user_stats[(execution["dataset"], execution["method"])]
        values = utility_metric_values(stats)
        for metric in UTILITY_METRICS:
            rows.append(metric_row(execution, metric, values[metric], "COMPUTED", "computed from F1BX baseline recommendations", len(stats), 0, {}))
    return pd.DataFrame(rows)


def build_beyond_results(
    baseline_execution: pd.DataFrame,
    per_run_user_stats: Mapping[tuple[str, str], pd.DataFrame],
    materials: Mapping[str, DatasetMaterial],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mostpop_hits = {
        dataset: stats.set_index("user_id")["hit_at_5"].astype(float).to_dict()
        for (dataset, method), stats in per_run_user_stats.items()
        if method == "MostPop"
    }
    for execution in baseline_execution.to_dict("records"):
        dataset = execution["dataset"]
        method = execution["method"]
        material = materials[dataset]
        stats = per_run_user_stats[(dataset, method)]
        top_lists = [list(items) for items in stats["top_items"]]
        ranks = [int(rank) if clean(rank) else TOP_K + 1 for rank in stats["rank_one_based"]]
        hits = [int(value) for value in stats["hit_at_5"]]
        targets = stats["positive_item_id"].astype(str).tolist()
        histories = [material.history_items_by_user.get(str(user_id), []) for user_id in stats["user_id"]]
        values = beyond_metric_values(material, stats, top_lists, ranks, hits, targets, histories, mostpop_hits[dataset])
        for metric in BEYOND_METRICS:
            value, status, reason, undefined_count, value_json = values[metric]
            rows.append(metric_row(execution, metric, value, status, reason, len(stats), undefined_count, value_json))
    return pd.DataFrame(rows)


def beyond_metric_values(
    material: DatasetMaterial,
    stats: pd.DataFrame,
    top_lists: Sequence[Sequence[str]],
    ranks: Sequence[int],
    hits: Sequence[int],
    targets: Sequence[str],
    histories: Sequence[Sequence[str]],
    mostpop_hit_by_user: Mapping[str, float],
) -> dict[str, tuple[float | str, str, str, int, Mapping[str, Any]]]:
    train_values = list(material.train_counts.values())
    long_tail_threshold = float(np.quantile(train_values, 0.8)) if train_values else math.nan
    candidate_union = set(material.candidates["candidate_item_id"].astype(str))
    catalog_union = set(material.catalog_items)
    unique_top = {item for row in top_lists for item in row}
    exposure = exposure_counts(top_lists, TOP_K)
    exposure_universe = sorted(catalog_union.union(unique_top), key=sort_key)
    exposure_vector = [float(exposure.get(item, 0)) for item in exposure_universe]

    self_info_values = [self_information_at_k(row, material.train_counts, material.total_train_interactions, TOP_K) for row in top_lists]
    shannon_values = [shannon_category_entropy_at_k(row, material.item_features, TOP_K) for row in top_lists]
    ild_values = [ild_jaccard_at_k(row, material.item_features, TOP_K) for row in top_lists]
    calibration_values = [calibration_at_k(history, row, material.item_features, TOP_K) for history, row in zip(histories, top_lists)]
    ser_values = [
        float(hit) - (float(mostpop_hit_by_user.get(str(user_id), 0.0)) * float(hit))
        for user_id, hit in zip(stats["user_id"], hits)
    ]
    popreo = popreo_at_k_from_hits(hits, targets, material.train_counts)
    dpd = dpd_at_k_from_ranks(ranks, [material.history_lengths_by_user.get(str(user_id), 0) for user_id in stats["user_id"]], k=TOP_K)

    return {
        "self_information_at_5": aggregate_optional_metric(self_info_values, "mean over users with nonzero train-popularity support"),
        "serendipity_at_5": (float(np.mean(ser_values)), "COMPUTED", "binary relevance unexpectedness against aligned MostPop primitive", 0, {}),
        "arp_at_5": (mean_non_null([average_recommendation_popularity(row, material.train_counts, TOP_K) for row in top_lists]), "COMPUTED", "mean train interaction count over top-5", 0, {}),
        "aplt_at_5": (mean_non_null([long_tail_rate_at_k(row, material.train_counts, long_tail_threshold, TOP_K) for row in top_lists]), "COMPUTED", "fraction of top-5 items below train-popularity 0.8 quantile", 0, {"long_tail_threshold": long_tail_threshold}),
        "popreo_at_5": popreo_metric_tuple(popreo),
        "item_coverage_at_5": (len(unique_top) / len(candidate_union) if candidate_union else "", "COMPUTED" if candidate_union else "BLOCKED_DATA_REQUIREMENT", "unique top-5 items over prompt-compatible candidate union", 0, {"candidate_union_N": len(candidate_union)}),
        "catalog_coverage_at_5": (len(unique_top) / len(catalog_union) if catalog_union else "", "COMPUTED" if catalog_union else "BLOCKED_DATA_REQUIREMENT", "unique top-5 items over catalog item universe", 0, {"catalog_N": len(catalog_union)}),
        "oic_at_5": optional_scalar_tuple(oic_at_k(top_lists, TOP_K), "mean pairwise top-5 overlap over users"),
        "shannon_category_entropy_at_5": aggregate_optional_metric(shannon_values, "mean per-user top-5 feature entropy"),
        "gini_at_5": optional_scalar_tuple(gini_index(exposure_vector), "Gini over full catalog exposure vector including zero exposure"),
        "dpd_at_5": dpd_metric_tuple(dpd),
        "jain_index_at_5": optional_scalar_tuple(jain_index(exposure_vector), "Jain index over full catalog exposure vector including zero exposure"),
        "ild_at_5": aggregate_optional_metric(ild_values, "mean per-user Jaccard ILD over feature-backed top-5 items"),
        "calibration_at_5": aggregate_optional_metric(calibration_values, "mean over users with train-history and recommendation feature distributions"),
    }


def aggregate_optional_metric(values: Sequence[float | None], reason: str) -> tuple[float | str, str, str, int, Mapping[str, Any]]:
    defined = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    undefined = len(values) - len(defined)
    if not defined:
        return "", "BLOCKED_DATA_REQUIREMENT", reason + "; no defined user rows", undefined, {"defined_user_count": 0}
    return float(np.mean(defined)), "COMPUTED", reason, undefined, {"defined_user_count": len(defined)}


def optional_scalar_tuple(value: float | None, reason: str) -> tuple[float | str, str, str, int, Mapping[str, Any]]:
    if value is None or not math.isfinite(float(value)):
        return "", "NOT_APPLICABLE", reason + "; scalar undefined", 0, {}
    return float(value), "COMPUTED", reason, 0, {}


def popreo_metric_tuple(result: Mapping[str, Any]) -> tuple[float | str, str, str, int, Mapping[str, Any]]:
    status = clean(result.get("computation_status"))
    reason = clean(result.get("reason")) or "repository PopREO protocol over target popularity quintiles"
    if status == "COMPUTED":
        return float(result["metric_value"]), "COMPUTED", reason, 0, result
    return "", "DEGENERATE_PROTOCOL_INPUT", reason, 0, result


def dpd_metric_tuple(result: Mapping[str, Any]) -> tuple[float | str, str, str, int, Mapping[str, Any]]:
    status = clean(result.get("computation_status"))
    reason = clean(result.get("reason")) or "repository DPD median-history split"
    if status == "COMPUTED":
        return float(result["metric_value"]), "COMPUTED", reason, 0, result
    return "", "DEGENERATE_PROTOCOL_INPUT", reason, 0, result


def metric_row(
    execution: Mapping[str, Any],
    metric: str,
    value: float | str,
    status: str,
    reason: str,
    evaluated_users: int,
    undefined_user_count: int,
    value_json: Mapping[str, Any],
) -> dict[str, Any]:
    value_text = "" if value == "" else format_float(float(value))
    identity = {
        "run_id": execution["run_id"],
        "metric": metric,
        "status": status,
        "value": value_text,
    }
    return {
        "result_id": "F1BX-METRIC-" + sha256_json(identity).upper()[:16],
        "run_id": execution["run_id"],
        "dataset": execution["dataset"],
        "method": execution["method"],
        "cohort_id": execution["cohort_id"],
        "metric": metric,
        "k": TOP_K if metric.endswith("_at_5") else "",
        "value": value_text,
        "value_json": stable_json(value_json) if value_json else "",
        "status": status,
        "evaluated_users": evaluated_users,
        "undefined_user_count": int(undefined_user_count),
        "source_recommendation_hash": execution["recommendation_output_hash"],
        "reason": reason,
    }


def build_foodcom_tuning_gate(
    root: Path,
    config: Mapping[str, Any],
    runtime: pd.DataFrame,
    timestamp: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tuning_config = config["foodcom_tuning"]
    search = pd.read_csv(root / Path(tuning_config["frozen_search_space_source"]), dtype=str, keep_default_na=False)
    contract = pd.read_csv(root / Path(tuning_config["frozen_contract_source"]), dtype=str, keep_default_na=False)
    split_hashes = {
        key: sha256_file(root / Path(path))
        for key, path in tuning_config["protected_split_sources"].items()
    }
    blocker = {
        "status": "BLOCKED_NOT_EXECUTED_MISSING_VALIDATION_CANDIDATE_CONTRACT",
        "reason": "F1BR freezes Food.com search space, tuning seed, validation metric, and no-test-tuning policy, but no separate frozen Food.com validation candidate/cohort contract is materialized; using the F0 primary final-test candidate set for tuning would leak final-test evidence.",
        "missing_artifact": "frozen Food.com validation candidate set or all-item validation evaluation contract",
        "protected_train_hash": split_hashes["train"],
        "protected_validation_hash": split_hashes["validation"],
        "protected_test_hash": split_hashes["test"],
    }
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for index, row in enumerate(search.to_dict("records"), start=1):
        params = hyperparameter_json(row)
        config_hash = sha256_json({"search_config_id": row["search_config_id"], "params": params})
        run_id = "f1bx_foodcom_tuning_" + row["search_config_id"].lower()
        base = {
            "run_id": run_id,
            "config_id": row["search_config_id"],
            "model": row["method"],
            "dataset": "foodcom",
            "seed": int(tuning_config["tuning_seed"]),
            "search_order": index,
            "hyperparameters": stable_json(params),
            "config_hash": config_hash,
            "train_hash": split_hashes["train"],
            "validation_hash": split_hashes["validation"],
            "test_hash_protected": split_hashes["test"],
            "validation_hr_at_5": "",
            "validation_ndcg_at_5": "",
            "start_time": "",
            "end_time": "",
            "runtime_seconds": "",
            "device": runtime_value(runtime, "torch_device"),
            "terminal": "true",
            "training_executed": "false",
            "final_test_metric_computed": "false",
            "status": blocker["status"],
            "failure_reason": blocker["reason"],
        }
        rows.append(base)
        validation_rows.append(
            {
                "run_id": run_id,
                "config_id": row["search_config_id"],
                "model": row["method"],
                "selection_metric": tuning_config["selection_metric"],
                "selection_metric_value": "",
                "selection_eligible": "false",
                "selected": "false",
                "status": "BLOCKED_NO_VALIDATION_SCORE",
                "reason": blocker["reason"],
            }
        )
    tuning_runs = pd.DataFrame(rows)
    validation_results = pd.DataFrame(validation_rows)
    selected_rows: list[dict[str, Any]] = []
    for method in TRAINABLE_MODELS:
        method_rows = search.loc[search["method"].eq(method)].copy()
        selected_rows.append(
            {
                "dataset": "foodcom",
                "model": method,
                "selected_config_id": "",
                "selected": "false",
                "selection_rank": "",
                "validation_hr_at_5": "",
                "validation_ndcg_at_5": "",
                "config_hash": "",
                "hyperparameters": "",
                "candidate_selection_basis": "validation_only",
                "tie_break_evidence": tuning_config["tie_breaking"],
                "available_config_count": len(method_rows),
                "valid_completed_config_count": 0,
                "status": "BLOCKED_NO_VALID_WINNER",
                "reason": blocker["reason"],
            }
        )
    return tuning_runs, validation_results, pd.DataFrame(selected_rows), blocker


def hyperparameter_json(row: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in ("lr", "l2", "embedding", "hidden", "layers", "heads"):
        value = clean(row.get(key))
        if value == "":
            continue
        try:
            numeric = float(value)
        except ValueError:
            params[key] = value
        else:
            params[key] = int(numeric) if numeric.is_integer() and key not in {"lr", "l2"} else numeric
    return params


def build_integrity(
    root: Path,
    materials: Mapping[str, DatasetMaterial],
    baseline_execution: pd.DataFrame,
    tuning_runs: pd.DataFrame,
    selected_configs: pd.DataFrame,
) -> dict[str, Any]:
    dataset_matrix = read_csv(root / "results/tables/f0_final_dataset_matrix.csv")
    raw_universe = read_csv(root / "results/tables/f0_final_raw_experiment_universe.csv")
    result_cells = read_csv(root / "results/tables/f0_final_result_cell_universe.csv")
    traditional = read_csv(root / "results/tables/f0_final_traditional_model_matrix.csv")
    f0_budget = {row["budget_item"]: int(row["request_count"]) for row in read_csv(root / "results/tables/f0_final_request_budget.csv").to_dict("records")}
    h0_manifest = read_json(root / "results/manifests/extension_h0_no_inference_manifest.json")
    foodcom_manifest = read_json(root / "results/manifests/d0a_foodcom_extracted_file_hashes.json")
    active = dataset_matrix.loc[dataset_matrix["active"].astype(str).str.lower().eq("true"), "dataset"].tolist()
    active_video_games = video_games_rows(raw_universe, result_cells, traditional, baseline_execution, tuning_runs, selected_configs)
    return {
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_head": EXPECTED_HEAD,
        "head_is_expected": git_output(root, ["rev-parse", "HEAD"]) == EXPECTED_HEAD,
        "active_datasets": active,
        "active_video_games_rows": active_video_games,
        "f0_budget": f0_budget,
        "f0_budget_unchanged": f0_budget_unchanged(f0_budget),
        "foodcom_hashes_unchanged": verify_foodcom_hashes(foodcom_manifest),
        "h0_hashes_unchanged": verify_manifest_hashes(root, h0_manifest.get("artifact_hashes", {})),
        "original_repo_status": git_output(root, ["status", "--short", "--untracked-files=all", "--", "original_repo"]),
        "paper_status": git_output(root, ["status", "--short", "--untracked-files=all", "--", "paper"]),
        "target_leakage_count": sum(material.target_leakage_count for material in materials.values()),
        "catalog_missing_candidate_count": sum(material.catalog_missing_candidate_count for material in materials.values()),
        "baseline_runs_executed": int(baseline_execution["status"].eq("COMPUTED").sum()),
        "foodcom_tuning_runs_executed": int(tuning_runs["training_executed"].astype(str).str.lower().eq("true").sum()),
        "foodcom_tuning_terminal_runs": int(tuning_runs["terminal"].astype(str).str.lower().eq("true").sum()),
        "foodcom_winners_selected": int(selected_configs["selected"].astype(str).str.lower().eq("true").sum()),
        "original_four_final_runs_executed": 0,
        "foodcom_final_five_seed_runs_executed": 0,
        "final_training_runs_executed": 0,
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "historical_full_suite_executed": 0,
        "commit_created": 0,
    }


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    materials: Mapping[str, DatasetMaterial],
    baseline_execution: pd.DataFrame,
    recommendations: pd.DataFrame,
    utility: pd.DataFrame,
    beyond: pd.DataFrame,
    tuning_runs: pd.DataFrame,
    validation_results: pd.DataFrame,
    selected_configs: pd.DataFrame,
    tuning_blocker: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def check(name: str, condition: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})

    check("head_is_expected_f1br_checkpoint", bool(integrity["head_is_expected"]), clean(integrity["head"]))
    check("active_datasets_exact", tuple(integrity["active_datasets"]) == ACTIVE_DATASETS, ",".join(integrity["active_datasets"]))
    check("operational_video_games_rows_zero", int(integrity["active_video_games_rows"]) == 0, str(integrity["active_video_games_rows"]))
    check("candidate_contracts_match_f1br_cohorts", all(material.candidates["user_id"].nunique() == material.spec.evaluation_users for material in materials.values()), "all datasets")
    check("target_leakage_zero", int(integrity["target_leakage_count"]) == 0, str(integrity["target_leakage_count"]))
    check("baseline_execution_rows_15", len(baseline_execution) == 15, str(len(baseline_execution)))
    check("baseline_execution_computed_15", int(baseline_execution["status"].eq("COMPUTED").sum()) == 15, str(int(baseline_execution["status"].eq("COMPUTED").sum())))
    for method in BASELINE_METHODS:
        check(f"{method_slug(method)}_rows_5", int(baseline_execution["method"].eq(method).sum()) == 5, str(int(baseline_execution["method"].eq(method).sum())))
    check("recommendation_rows_complete", len(recommendations) == int((baseline_execution["cohort_N"].astype(int) * 20).sum()), str(len(recommendations)))
    check("utility_metric_rows_45", len(utility) == 45 and utility["status"].eq("COMPUTED").all(), str(len(utility)))
    check("beyond_metric_rows_210", len(beyond) == 210, str(len(beyond)))
    check("bm25_foodcom_tag_policy", materials["foodcom"].spec.bm25_text_source == "RAW_recipes.tags", materials["foodcom"].spec.bm25_text_source)
    check("foodcom_search_configs_40", len(tuning_runs) == 40, str(len(tuning_runs)))
    check("foodcom_tuning_seed_42", tuning_runs["seed"].astype(int).eq(42).all(), "42")
    check("foodcom_tuning_terminal_40", int(tuning_runs["terminal"].astype(str).str.lower().eq("true").sum()) == 40, str(int(tuning_runs["terminal"].astype(str).str.lower().eq("true").sum())))
    check("foodcom_tuning_executed_zero_due_blocker", int(integrity["foodcom_tuning_runs_executed"]) == 0, str(integrity["foodcom_tuning_runs_executed"]))
    check("validation_candidate_contract_blocker_explicit", clean(tuning_blocker["status"]) == "BLOCKED_NOT_EXECUTED_MISSING_VALIDATION_CANDIDATE_CONTRACT", clean(tuning_blocker["reason"]))
    check("validation_results_do_not_use_test_metric", validation_results["selection_eligible"].eq("false").all() and tuning_runs["final_test_metric_computed"].eq("false").all(), "final_test_metric_computed=false")
    check("selected_config_rows_4", len(selected_configs) == 4, str(len(selected_configs)))
    check("selected_winners_zero_when_blocked", int(integrity["foodcom_winners_selected"]) == 0, str(integrity["foodcom_winners_selected"]))
    check("no_final_five_seed_training", int(integrity["foodcom_final_five_seed_runs_executed"]) == 0, "0")
    check("no_original_four_final_training", int(integrity["original_four_final_runs_executed"]) == 0, "0")
    check("f0_request_budget_unchanged", bool(integrity["f0_budget_unchanged"]), stable_json(integrity["f0_budget"]))
    check("foodcom_hashes_unchanged", bool(integrity["foodcom_hashes_unchanged"]), "true")
    check("h0_hashes_unchanged", bool(integrity["h0_hashes_unchanged"]), "true")
    check("original_repo_unchanged", clean(integrity["original_repo_status"]) == "", clean(integrity["original_repo_status"]) or "clean")
    check("paper_unchanged", clean(integrity["paper_status"]) == "", clean(integrity["paper_status"]) or "clean")
    check("zero_llm_inference", int(integrity["llm_inference_calls_executed"]) == 0, "0")
    check("zero_ollama_recommendation_calls", int(integrity["ollama_recommendation_calls_executed"]) == 0, "0")
    check("zero_external_api_download_network", int(integrity["external_api_calls_executed"]) == 0 and int(integrity["downloads_executed"]) == 0 and int(integrity["network_calls_executed"]) == 0, "0")
    check("historical_full_suite_not_run", int(integrity["historical_full_suite_executed"]) == 0, "0")
    check("config_forbids_final_training_and_llm", config["execution"]["allow_final_five_seed_training"] is False and config["execution"]["allow_llm_inference"] is False, "false")
    return pd.DataFrame(checks)


def build_execution_manifest(
    root: Path,
    decision: str,
    config: Mapping[str, Any],
    runtime: pd.DataFrame,
    baseline_execution: pd.DataFrame,
    recommendations: pd.DataFrame,
    utility: pd.DataFrame,
    beyond: pd.DataFrame,
    tuning_runs: pd.DataFrame,
    selected_configs: pd.DataFrame,
    tuning_blocker: Mapping[str, Any],
    integrity: Mapping[str, Any],
    validation: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": decision,
        "blocked": decision == DECISION_BLOCKED,
        "blocker": dict(tuning_blocker) if decision == DECISION_BLOCKED else {},
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_head": clean(config["phase"]["expected_head"]),
        "phase_start_clean_confirmation": clean(config["phase"]["phase_start_clean_confirmation"]),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS"), "check"].tolist(),
        "baseline_rows": len(baseline_execution),
        "baseline_computed_rows": int(baseline_execution["status"].eq("COMPUTED").sum()),
        "baseline_recommendation_rows": len(recommendations),
        "baseline_utility_result_rows": len(utility),
        "baseline_beyond_utility_result_rows": len(beyond),
        "foodcom_tuning_planned": len(tuning_runs),
        "foodcom_tuning_terminal": int(tuning_runs["terminal"].astype(str).str.lower().eq("true").sum()),
        "foodcom_tuning_executed": int(tuning_runs["training_executed"].astype(str).str.lower().eq("true").sum()),
        "foodcom_winners_selected": int(selected_configs["selected"].astype(str).str.lower().eq("true").sum()),
        "final_training_runs_executed": 0,
        "original_four_final_runs_executed": 0,
        "foodcom_final_five_seed_runs_executed": 0,
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "runtime_preflight": {row["resource"]: row["value"] for row in runtime.to_dict("records")},
        "integrity": dict(integrity),
        "commit_created": 0,
    }


def build_selection_freeze_manifest(root: Path, decision: str, selected_configs: pd.DataFrame, tuning_runs: pd.DataFrame, tuning_blocker: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": decision,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "selection_metric": "validation_ndcg_at_5",
        "selection_basis": "validation_only",
        "final_test_metrics_used": False,
        "tuning_runs_planned": len(tuning_runs),
        "tuning_runs_executed": int(tuning_runs["training_executed"].astype(str).str.lower().eq("true").sum()),
        "selected_rows": selected_configs.to_dict("records"),
        "winner_count": int(selected_configs["selected"].astype(str).str.lower().eq("true").sum()),
        "blocker": dict(tuning_blocker),
    }


def build_no_llm_manifest(root: Path, decision: str, validation: pd.DataFrame, integrity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": decision,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "raw_llm_outputs_created": 0,
        "traditional_final_training_outputs_created": 0,
        "foodcom_tuning_training_runs_executed": int(integrity["foodcom_tuning_runs_executed"]),
        "deterministic_baseline_runs_executed": int(integrity["baseline_runs_executed"]),
    }


def render_report(
    decision: str,
    runtime: pd.DataFrame,
    baseline_execution: pd.DataFrame,
    utility: pd.DataFrame,
    beyond: pd.DataFrame,
    tuning_runs: pd.DataFrame,
    selected_configs: pd.DataFrame,
    tuning_blocker: Mapping[str, Any],
    integrity: Mapping[str, Any],
    validation: pd.DataFrame,
) -> str:
    method_counts = baseline_execution.groupby("method")["status"].apply(lambda values: int((values == "COMPUTED").sum())).to_dict()
    degenerate_beyond = int(beyond["status"].eq("DEGENERATE_PROTOCOL_INPUT").sum())
    runtime_map = {row["resource"]: row["value"] for row in runtime.to_dict("records")}
    return "\n".join(
        [
            "# F1BX Baselines and Food.com Validation Tuning",
            "",
            f"Decision token: `{decision}`",
            "",
            "## Deterministic Baselines",
            f"- Baseline rows computed: `{int(baseline_execution['status'].eq('COMPUTED').sum())}/15`.",
            "\n".join(f"- `{method}` computed rows: `{count}`." for method, count in method_counts.items()),
            "- Food.com `BM25-like/content` text source: `RAW_recipes.tags` only.",
            f"- Utility metric rows: `{len(utility)}`.",
            f"- Beyond-utility metric rows: `{len(beyond)}`.",
            f"- Degenerate PopREO/DPD result rows after observed recommendations: `{degenerate_beyond}`.",
            "",
            "## Food.com Tuning",
            f"- Planned tuning configs: `{len(tuning_runs)}`.",
            f"- Training runs executed: `{int(tuning_runs['training_executed'].astype(str).str.lower().eq('true').sum())}`.",
            f"- Winners selected: `{int(selected_configs['selected'].astype(str).str.lower().eq('true').sum())}`.",
            f"- Blocker: `{clean(tuning_blocker['status'])}`.",
            f"- Reason: {clean(tuning_blocker['reason'])}",
            "",
            "## Resource Preflight",
            f"- Python: `{runtime_map.get('python', '')}`.",
            f"- Torch: `{runtime_map.get('torch', '')}`.",
            f"- CUDA: `{runtime_map.get('cuda_available', '')}`.",
            f"- Free disk bytes: `{runtime_map.get('free_disk_bytes', '')}`.",
            f"- RAM bytes: `{runtime_map.get('total_ram_bytes', '')}`.",
            "",
            "## Integrity",
            f"- Active datasets: `{','.join(integrity['active_datasets'])}`.",
            f"- Video Games rows: `{integrity['active_video_games_rows']}`.",
            f"- F0 budget unchanged: `{str(integrity['f0_budget_unchanged']).lower()}`.",
            f"- Food.com hashes unchanged: `{str(integrity['foodcom_hashes_unchanged']).lower()}`.",
            f"- H0 hashes unchanged: `{str(integrity['h0_hashes_unchanged']).lower()}`.",
            "- LLM/API/download/network counters: `0`.",
            "",
            "## Validation",
            f"- Focused validation checks passed: `{str(validation['status'].eq('PASS').all()).lower()}`.",
        ]
    ) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / RAW_OUTPUT_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    outputs["baseline_recommendations"].to_csv(root / RECOMMENDATIONS_PATH, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")

    artifact_paths = [*OUTPUT_TABLES.values(), RECOMMENDATIONS_PATH, REPORT_PATH, *SUPPORT_ARTIFACTS]
    table_hashes = artifact_hashes(root, artifact_paths)
    for key, rel_path in OUTPUT_MANIFESTS.items():
        manifest = dict(outputs[key])
        manifest["artifact_hashes"] = dict(table_hashes)
        (root / rel_path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        table_hashes[rel_path.as_posix()] = sha256_file(root / rel_path)


def resource_preflight(root: Path) -> pd.DataFrame:
    usage = shutil.disk_usage(root)
    memory = memory_status()
    torch_info = torch_status()
    rows = [
        ("python", platform.python_version(), "RECORDED", ""),
        ("platform", platform.platform(), "RECORDED", ""),
        ("free_disk_bytes", usage.free, "RECORDED", ""),
        ("total_disk_bytes", usage.total, "RECORDED", ""),
        ("total_ram_bytes", memory.get("total_ram_bytes", ""), memory.get("status", "UNAVAILABLE"), memory.get("reason", "")),
        ("available_ram_bytes", memory.get("available_ram_bytes", ""), memory.get("status", "UNAVAILABLE"), memory.get("reason", "")),
        ("torch", torch_info.get("torch", ""), torch_info.get("status", "UNAVAILABLE"), torch_info.get("reason", "")),
        ("cuda_available", torch_info.get("cuda_available", ""), torch_info.get("status", "UNAVAILABLE"), ""),
        ("cuda_version", torch_info.get("cuda_version", ""), torch_info.get("status", "UNAVAILABLE"), ""),
        ("gpu", torch_info.get("gpu", ""), torch_info.get("status", "UNAVAILABLE"), ""),
        ("vram_bytes", torch_info.get("vram_bytes", ""), torch_info.get("status", "UNAVAILABLE"), ""),
        ("torch_device", torch_info.get("device", "cpu"), torch_info.get("status", "UNAVAILABLE"), ""),
    ]
    return pd.DataFrame([{"resource": key, "value": clean(value), "status": status, "notes": notes} for key, value, status, notes in rows])


def memory_status() -> dict[str, Any]:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            raise OSError("GlobalMemoryStatusEx failed")
        return {"total_ram_bytes": int(status.ullTotalPhys), "available_ram_bytes": int(status.ullAvailPhys), "status": "RECORDED", "reason": ""}
    except Exception as exc:  # pragma: no cover - platform-specific fallback
        return {"total_ram_bytes": "", "available_ram_bytes": "", "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"}


def torch_status() -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        info: dict[str, Any] = {
            "torch": torch.__version__,
            "cuda_available": str(cuda_available).lower(),
            "cuda_version": clean(getattr(torch.version, "cuda", "")),
            "gpu": "",
            "vram_bytes": "",
            "device": "cuda" if cuda_available else "cpu",
            "status": "RECORDED",
            "reason": "",
        }
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["vram_bytes"] = int(props.total_memory)
        return info
    except Exception as exc:  # pragma: no cover - torch may be unavailable
        return {"torch": "", "cuda_available": "false", "cuda_version": "", "gpu": "", "vram_bytes": "", "device": "cpu", "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"}


def count_target_leakage(candidates: pd.DataFrame, history_items_by_user: Mapping[str, Sequence[str]]) -> int:
    positives = candidates.loc[candidates["is_ground_truth"], ["user_id", "positive_item_id"]].copy()
    leakage = 0
    for row in positives.to_dict("records"):
        leakage += int(str(row["positive_item_id"]) in set(str(item) for item in history_items_by_user.get(str(row["user_id"]), [])))
    return leakage


def f0_budget_unchanged(budget: Mapping[str, int]) -> bool:
    return (
        int(budget.get("CORE_MANDATORY_NOW_REQUESTS", -1)) == 29292
        and int(budget.get("REQUIRED_SENSITIVITY_REQUESTS", -1)) == 21000
        and int(budget.get("MANDATORY_RAW_REQUESTS_TOTAL", -1)) == 50292
        and int(budget.get("TRAINING_DEPENDENT_REQUESTS", -1)) == 29292
        and int(budget.get("CONDITIONAL_REQUESTS", -1)) == 9000
        and int(budget.get("OPTIONAL_REQUESTS", -1)) == 12000
        and int(budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL", -1)) == 100584
        and int(budget.get("EXISTING_REUSABLE_REQUESTS", -1)) == 0
    )


def verify_manifest_hashes(root: Path, hashes: Mapping[str, str]) -> bool:
    for rel_path, expected in hashes.items():
        path = root / rel_path
        if not path.exists() or sha256_file(path) != expected:
            return False
    return True


def verify_foodcom_hashes(manifest: Mapping[str, Any]) -> bool:
    for record in manifest.get("files", {}).values():
        path = Path(record["relative_extracted_path"])
        if not path.exists() or sha256_file(path) != record["sha256"]:
            return False
    return True


def video_games_rows(*frames: pd.DataFrame) -> int:
    total = 0
    for frame in frames:
        mask = frame.astype(str).apply(lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True, na=False))
        total += int(mask.any(axis=1).sum())
    return total


def runtime_value(runtime: pd.DataFrame, resource: str) -> str:
    matched = runtime.loc[runtime["resource"].eq(resource), "value"]
    return clean(matched.iloc[0]) if not matched.empty else ""


def stable_random_score(seed: int, dataset: str, user_id: str, item_id: str) -> float:
    digest = hashlib.sha256(f"{seed}|{dataset}|{user_id}|{item_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def tokenize(value: Any) -> list[str]:
    return TOKEN_RE.findall(clean(value).lower())


def tags_to_text(value: Any) -> str:
    return " ".join(parse_feature_list(value))


def parse_feature_list(value: Any) -> list[str]:
    text = clean(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return tokenize(text)
    if isinstance(parsed, (list, tuple, set)):
        return [clean(item).strip().lower() for item in parsed if clean(item).strip()]
    return tokenize(parsed)


def parse_pipe_features(value: Any) -> list[str]:
    text = clean(value)
    if not text:
        return []
    return [part.strip().lower() for part in text.replace("|", ";").split(";") if part.strip()]


def item_sort_columns(values: pd.Series) -> dict[str, list[Any]]:
    groups: list[int] = []
    numbers: list[int] = []
    texts: list[str] = []
    for value in values.astype(str):
        try:
            number = int(value)
        except ValueError:
            groups.append(1)
            numbers.append(0)
            texts.append(value)
        else:
            groups.append(0)
            numbers.append(number)
            texts.append("")
    return {"__item_group": groups, "__item_number": numbers, "__item_text": texts}


def sort_key(value: Any) -> tuple[int, int, str]:
    text = clean(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def bool_value(value: Any) -> bool:
    return bool_series(pd.Series([value])).iloc[0].item()


def baseline_seed(method: str, config: Mapping[str, Any]) -> str:
    return str(config.get("seed", "")) if method == "Random" else ""


def method_slug(method: str) -> str:
    return method.lower().replace("/", "_").replace("-", "").replace(" ", "_")


def mean_non_null(values: Sequence[float | None]) -> float:
    defined = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(defined)) if defined else math.nan


def frame_hash(frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def artifact_hashes(root: Path, rel_paths: Sequence[Path]) -> dict[str, str]:
    return {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in rel_paths
        if (root / rel_path).exists()
    }


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


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

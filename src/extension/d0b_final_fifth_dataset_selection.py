"""D0B final fifth-dataset selection audit.

This module compares the frozen Food.com D0A evidence with the frozen Amazon
Video Games 2014 E2B evidence and selects exactly one final fifth dataset. It
does not run models, train recommenders, download data, or purge files.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.extension.e0_protocol import FUTURE_MODELS, HISTORY_LENGTHS
from src.utils.config import find_project_root


PHASE_ID = "d0b_final_fifth_dataset_selection"
FOODCOM_WINNER = "FOODCOM_SELECTED_FINAL_FIFTH_DATASET"
VIDEO_GAMES_WINNER = "VIDEO_GAMES_SELECTED_FINAL_FIFTH_DATASET"
BLOCKED_SELECTION = "FIFTH_DATASET_SELECTION_REMAINS_BLOCKED"

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")

D0A_ARCHIVE_MANIFEST = MANIFEST_DIR / "d0a_foodcom_raw_archive_manifest.json"
D0A_EXTRACTED_HASHES = MANIFEST_DIR / "d0a_foodcom_extracted_file_hashes.json"
D0A_NO_INFERENCE = MANIFEST_DIR / "d0a_foodcom_no_inference_manifest.json"
D0A_PRIMARY = TABLE_DIR / "d0a_foodcom_primary1000_feasibility.csv"
D0A_HISTORY = TABLE_DIR / "d0a_foodcom_history200_feasibility.csv"
D0A_METADATA = TABLE_DIR / "d0a_foodcom_metadata_coverage.csv"
D0A_FEATURES = TABLE_DIR / "d0a_foodcom_feature_coverage.csv"
D0A_METRICS = TABLE_DIR / "d0a_foodcom_metric_capability.csv"

VG_PRIMARY = TABLE_DIR / "extension_e2b0_video_games_2014_primary_1000_feasibility.csv"
VG_HISTORY = TABLE_DIR / "extension_e2b0_video_games_2014_history_200_feasibility.csv"
VG_TITLE_SUMMARY = TABLE_DIR / "extension_e2b0_video_games_2014_title_provenance_summary.csv"
VG_CANDIDATE_TITLE = TABLE_DIR / "extension_e2b0_video_games_2014_candidate_strict_title_summary.csv"
VG_JOINT_SUMMARY = TABLE_DIR / "extension_e2b0j_joint_recovery_summary.csv"

H0_MANIFEST = MANIFEST_DIR / "extension_h0_no_inference_manifest.json"
H0_BUDGET = TABLE_DIR / "extension_h0_exact_history_request_budget.csv"
H0_MATRIX = TABLE_DIR / "extension_h0_72_cell_history_matrix.csv"
E2A_SUMMARY = TABLE_DIR / "extension_cell_completeness_summary.csv"

OUTPUT_INPUT_EVIDENCE = TABLE_DIR / "d0b_authoritative_input_evidence.csv"
OUTPUT_REQUIREMENT_MATRIX = TABLE_DIR / "d0b_fifth_dataset_complete_requirement_matrix.csv"
OUTPUT_COMPLETENESS_COUNTS = TABLE_DIR / "d0b_completeness_counts.csv"
OUTPUT_BLOCKERS = TABLE_DIR / "d0b_dataset_specific_blockers.csv"
OUTPUT_FUTURE_BUDGET = TABLE_DIR / "d0b_future_inference_budget.csv"
OUTPUT_DECISION = TABLE_DIR / "d0b_final_selection_decision.csv"
OUTPUT_NO_INFERENCE = MANIFEST_DIR / "d0b_no_inference_manifest.json"
OUTPUT_PURGE = MANIFEST_DIR / "d0b_loser_purge_inventory.json"
OUTPUT_REPORT = REPORT_DIR / "d0b_final_fifth_dataset_selection.md"

STATUS_VALUES = {
    "READY",
    "READY_WITH_ADAPTATION",
    "READY_AFTER_TRAINING",
    "BLOCKED_DATASET",
    "BLOCKED_GLOBAL_PROTOCOL",
    "NOT_APPLICABLE",
}

COUNT_STATUSES = [
    "READY",
    "READY_WITH_ADAPTATION",
    "READY_AFTER_TRAINING",
    "BLOCKED_DATASET",
    "BLOCKED_GLOBAL_PROTOCOL",
    "NOT_APPLICABLE",
]

MODELS = tuple(FUTURE_MODELS)
PROMPTS = ("BASE", "LLMRANK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D0B final fifth-dataset selection audit.")
    parser.add_argument("--output-root", default=".")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_d0b(root)
    write_outputs(root, outputs)
    decision = outputs["decision"].iloc[0]
    print(f"decision={decision['winner_token']}")
    print(f"foodcom_dataset_blockers={decision['foodcom_dataset_blockers']}")
    print(f"videogames_dataset_blockers={decision['videogames_dataset_blockers']}")
    print("llm_requests=0")
    return 0


def generate_d0b(root: Path) -> dict[str, Any]:
    evidence = load_evidence(root)
    validate_authoritative_facts(evidence)

    requirement_matrix = build_requirement_matrix(evidence)
    counts = build_completeness_counts(requirement_matrix)
    blockers = build_dataset_blockers(requirement_matrix)
    winner = select_winner(requirement_matrix)
    future_budget = build_future_budget(root, evidence, winner)
    input_evidence = build_input_evidence(evidence)
    decision = build_decision(requirement_matrix, blockers, winner)
    purge_inventory = build_loser_purge_inventory(root, winner)
    manifest = build_no_inference_manifest(root, evidence, requirement_matrix, counts, decision, future_budget)
    return {
        "input_evidence": input_evidence,
        "requirement_matrix": requirement_matrix,
        "completeness_counts": counts,
        "blockers": blockers,
        "future_budget": future_budget,
        "decision": decision,
        "purge_inventory": purge_inventory,
        "manifest": manifest,
    }


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "input_evidence": root / OUTPUT_INPUT_EVIDENCE,
        "requirement_matrix": root / OUTPUT_REQUIREMENT_MATRIX,
        "completeness_counts": root / OUTPUT_COMPLETENESS_COUNTS,
        "blockers": root / OUTPUT_BLOCKERS,
        "future_budget": root / OUTPUT_FUTURE_BUDGET,
        "decision": root / OUTPUT_DECISION,
    }
    for key, path in table_paths.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(path, index=False)

    purge_inventory = dict(outputs["purge_inventory"])
    purge_inventory["artifact_hashes"] = {path.relative_to(root).as_posix(): sha256_file(path) for path in table_paths.values()}
    (root / OUTPUT_PURGE).write_text(json.dumps(purge_inventory, indent=2, sort_keys=True), encoding="utf-8")

    manifest = dict(outputs["manifest"])
    manifest["artifact_hashes"] = {
        **{path.relative_to(root).as_posix(): sha256_file(path) for path in table_paths.values()},
        OUTPUT_PURGE.as_posix(): sha256_file(root / OUTPUT_PURGE),
    }
    (root / OUTPUT_NO_INFERENCE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / OUTPUT_REPORT).write_text(render_report(outputs), encoding="utf-8")


def load_evidence(root: Path) -> dict[str, Any]:
    return {
        "food_archive": read_json(root / D0A_ARCHIVE_MANIFEST),
        "food_hashes": read_json(root / D0A_EXTRACTED_HASHES),
        "food_no_inference": read_json(root / D0A_NO_INFERENCE),
        "food_primary": read_csv(root / D0A_PRIMARY).iloc[0].to_dict(),
        "food_history": read_csv(root / D0A_HISTORY).iloc[0].to_dict(),
        "food_metadata": read_csv(root / D0A_METADATA),
        "food_features": read_csv(root / D0A_FEATURES),
        "food_metrics": read_csv(root / D0A_METRICS),
        "vg_primary": read_csv(root / VG_PRIMARY).iloc[0].to_dict(),
        "vg_history": read_csv(root / VG_HISTORY).iloc[0].to_dict(),
        "vg_title_summary": read_csv(root / VG_TITLE_SUMMARY),
        "vg_candidate_title": read_csv(root / VG_CANDIDATE_TITLE).iloc[0].to_dict(),
        "vg_joint": read_csv(root / VG_JOINT_SUMMARY).iloc[0].to_dict(),
        "h0_manifest": read_json(root / H0_MANIFEST),
        "h0_budget": read_csv(root / H0_BUDGET),
        "h0_matrix": read_csv(root / H0_MATRIX),
        "e2a_summary": read_csv(root / E2A_SUMMARY),
    }


def validate_authoritative_facts(evidence: Mapping[str, Any]) -> None:
    food_archive = evidence["food_archive"]
    food_hashes = evidence["food_hashes"]
    if food_archive["sha256"] != "e7a43e623c616799f78a0479b550483450fd7ede4e0c40979db3458a82b5b913":
        raise ValueError("Food.com archive hash mismatch")
    if food_hashes["files"]["RAW_interactions.csv"]["sha256"] != "4e278d2f687b1020f5694e51003a7d6174344b51aebbdf4530de88b5c6a78ee4":
        raise ValueError("Food.com RAW_interactions hash mismatch")
    if food_hashes["files"]["RAW_recipes.csv"]["sha256"] != "6a8136d1da9e03396a9f52d72200b3fddb8fe103f11e1496effca30ac4e0539f":
        raise ValueError("Food.com RAW_recipes hash mismatch")
    if evidence["vg_primary"]["selected_user_hash"] != "90f4f4694d97d50e72ddd6475b85b911d29cae24a556a0799a2a751f78e3a66b":
        raise ValueError("Video Games primary user hash mismatch")
    if evidence["vg_primary"]["selected_candidate_hash"] != "c48f48a0353e73c44589965cb48463df4bdd0397355b358d46917d10b812fbad":
        raise ValueError("Video Games primary candidate hash mismatch")
    if int(evidence["h0_manifest"]["final_cells"]) != 72 or int(evidence["h0_manifest"]["blocked_cells"]) != 0:
        raise ValueError("H0 final matrix evidence is not ready")


def build_requirement_matrix(evidence: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for prompt in PROMPTS:
            rows.append(
                req(
                    f"PRIMARY_{model}_{prompt}",
                    "LOCAL_REPRODUCTION_PARITY",
                    "primary",
                    f"primary N=1000 {model} {prompt}",
                    "MANDATORY",
                    "READY",
                    "D0A primary strict-name 1000 feasible",
                    "READY",
                    "E2B-0 primary strict-title 1000 feasible",
                    "NONE",
                    "TIE",
                    1000,
                    "false",
                    "NONE",
                    "D0A_PRIMARY;E2B0_PRIMARY",
                )
            )
    for model in MODELS:
        for length in HISTORY_LENGTHS:
            rows.append(
                req(
                    f"HISTORY_{model}_L{length}",
                    "LOCAL_REPRODUCTION_PARITY",
                    "history",
                    f"history N=200 BASE {model} L{length}",
                    "MANDATORY",
                    "READY",
                    "D0A strict paired L20 count 6387; selected 200",
                    "BLOCKED_DATASET",
                    "Video Games strict L20 users 35, below required 200",
                    "DATASET",
                    "Food.com",
                    200,
                    "false",
                    "FOODCOM_TIE_POLICY;VIDEO_GAMES_TITLE_BLOCKER",
                    "D0A_HISTORY;E2B0_HISTORY;E2B0J",
                )
            )
    metric_specs = [
        ("HR@5", "PAPER_METRIC", "utility", "READY", "READY", "NONE", "NONE"),
        ("NDCG@5", "PAPER_METRIC", "utility", "READY", "READY", "NONE", "NONE"),
        ("MRR diagnostic", "DIAGNOSTIC_RELIABILITY", "utility", "READY", "READY", "NONE", "DIAGNOSTIC"),
        ("SelfInformation@5", "PAPER_METRIC", "beyond utility", "READY", "READY", "NONE", "TRAIN_POPULARITY"),
        ("Serendipity@5", "PAPER_METRIC", "beyond utility", "READY_WITH_ADAPTATION", "READY_WITH_ADAPTATION", "NONE", "MOSTPOP_COMPARISON_ADAPTATION"),
        ("ARP@5", "PAPER_METRIC", "beyond utility", "READY", "READY", "NONE", "TRAIN_POPULARITY"),
        ("APLT@5", "PAPER_METRIC", "beyond utility", "READY", "READY", "NONE", "TRAIN_POPULARITY"),
        ("PopREO@5", "PAPER_METRIC", "beyond utility", "BLOCKED_GLOBAL_PROTOCOL", "BLOCKED_GLOBAL_PROTOCOL", "GLOBAL_PROTOCOL", "UNRECOVERED_GROUPING_POLICY"),
        ("ItemCoverage@5", "PAPER_METRIC", "diversity", "READY", "READY", "NONE", "NONE"),
        ("Catalog Coverage@5", "PAPER_METRIC", "diversity", "READY", "READY", "NONE", "NONE"),
        ("OIC@5", "PAPER_METRIC", "diversity", "READY", "READY", "NONE", "NONE"),
        ("Shannon category entropy@5", "PAPER_METRIC", "diversity", "READY_WITH_ADAPTATION", "BLOCKED_DATASET", "DATASET", "FOODCOM_TAG_JACCARD;VIDEO_GAMES_NO_FEATURE_AUDIT"),
        ("ILD@5", "PAPER_METRIC", "diversity", "READY_WITH_ADAPTATION", "BLOCKED_DATASET", "DATASET", "FOODCOM_TAG_JACCARD;VIDEO_GAMES_NO_FEATURE_AUDIT"),
        ("Calibration@5", "PAPER_METRIC", "fairness", "READY_WITH_ADAPTATION", "BLOCKED_DATASET", "DATASET", "FOODCOM_TAG_JACCARD;VIDEO_GAMES_NO_FEATURE_AUDIT"),
        ("Gini@5", "PAPER_METRIC", "fairness", "READY", "READY", "NONE", "EXPOSURE_DISTRIBUTION"),
        ("DPD@5", "PAPER_METRIC", "fairness", "BLOCKED_GLOBAL_PROTOCOL", "BLOCKED_GLOBAL_PROTOCOL", "GLOBAL_PROTOCOL", "UNRECOVERED_GROUPING_POLICY"),
        ("Jain's Index@5", "PAPER_METRIC", "fairness", "READY", "READY", "NONE", "EXPOSURE_DISTRIBUTION"),
        ("hallucination", "DIAGNOSTIC_RELIABILITY", "reliability", "READY", "READY", "NONE", "STRICT_CANDIDATE_NAME_MATCH"),
        ("strict parse validity", "DIAGNOSTIC_RELIABILITY", "reliability", "READY", "READY", "NONE", "STRICT_TOP10_PARSER"),
        ("duplicate rate", "DIAGNOSTIC_RELIABILITY", "reliability", "READY", "READY", "NONE", "STRICT_TOP10_PARSER"),
        ("invalid-reason distribution", "DIAGNOSTIC_RELIABILITY", "reliability", "READY", "READY", "NONE", "STRICT_TOP10_PARSER"),
        ("backend failure separation", "DIAGNOSTIC_RELIABILITY", "reliability", "READY", "READY", "NONE", "GLOBAL_RUNTIME_MANIFEST_POLICY"),
        ("raw-response preservation", "DIAGNOSTIC_RELIABILITY", "reliability", "READY", "READY", "NONE", "GLOBAL_RUNTIME_MANIFEST_POLICY"),
        ("valid-rate reporting", "DIAGNOSTIC_RELIABILITY", "reliability", "READY", "READY", "NONE", "STRICT_TOP10_PARSER"),
        ("latency", "EFFICIENCY", "efficiency", "NOT_APPLICABLE", "NOT_APPLICABLE", "NONE", "EXECUTION_TELEMETRY_ONLY"),
        ("throughput", "EFFICIENCY", "efficiency", "NOT_APPLICABLE", "NOT_APPLICABLE", "NONE", "EXECUTION_TELEMETRY_ONLY"),
        ("runtime quantiles", "EFFICIENCY", "efficiency", "NOT_APPLICABLE", "NOT_APPLICABLE", "NONE", "EXECUTION_TELEMETRY_ONLY"),
        ("token counts where obtainable", "EFFICIENCY", "efficiency", "NOT_APPLICABLE", "NOT_APPLICABLE", "NONE", "EXECUTION_TELEMETRY_ONLY"),
        ("monetary cost only where meaningful", "EFFICIENCY", "efficiency", "NOT_APPLICABLE", "NOT_APPLICABLE", "NONE", "LOCAL_MODELS_NO_MONETARY_COST"),
    ]
    for metric, evidence_family, family, food_status, vg_status, blocker_scope, adaptation in metric_specs:
        rows.append(
            req(
                normalize_id(metric),
                evidence_family,
                family,
                metric,
                "MANDATORY" if evidence_family != "EFFICIENCY" else "REQUIRED_CONDITIONAL",
                food_status,
                reason_for_status("Food.com", metric, food_status),
                vg_status,
                reason_for_status("Video Games", metric, vg_status),
                blocker_scope,
                preferred(food_status, vg_status),
                0,
                "false",
                adaptation,
                "D0A_METRIC_CAPABILITY;E2B_TITLE_AUDIT;E2A_METRIC_CONTRACT",
            )
        )
    for model in MODELS:
        rows.append(
            req(
                f"CANDDIF_{model}",
                "PAPER_ANALYSIS_DIMENSION",
                "CandDif",
                f"CandDif / candidate-position bias {model}",
                "MANDATORY",
                "READY_WITH_ADAPTATION",
                "requires deterministic order-variant materialization over Food.com primary 1000",
                "READY_WITH_ADAPTATION",
                "primary 1000 feasible; deterministic order variants still future work",
                "NONE",
                "TIE",
                1000,
                "false",
                "DETERMINISTIC_ORDER_VARIANT",
                "D0A_PRIMARY;E2B0_PRIMARY;E2A_CANDDIF",
            )
        )
    profile_specs = [
        ("history-only", "MANDATORY", "READY", "BLOCKED_DATASET", 600),
        ("profile generation", "REQUIRED_CONDITIONAL", "READY_WITH_ADAPTATION", "BLOCKED_DATASET", 600),
        ("profile-only recommendation", "REQUIRED_CONDITIONAL", "READY_WITH_ADAPTATION", "BLOCKED_DATASET", 600),
        ("profile + history recommendation", "REQUIRED_CONDITIONAL", "READY_WITH_ADAPTATION", "BLOCKED_DATASET", 600),
    ]
    for name, mandatory, food_status, vg_status, requests in profile_specs:
        rows.append(
            req(
                f"PROFILE_{normalize_id(name)}",
                "PAPER_ANALYSIS_DIMENSION",
                "profiles",
                name,
                mandatory,
                food_status,
                "Food.com has names, tags, ingredients, and paired history; profile prompts require domain adaptation",
                vg_status,
                "Video Games paired L20 history blocked and canonical feature/profile representation not frozen",
                "DATASET" if vg_status == "BLOCKED_DATASET" else "NONE",
                "Food.com",
                requests,
                "false",
                "DOCUMENTED_DOMAIN_ADAPTATION",
                "D0A_FEATURES;E2B0_HISTORY;E2B0J",
            )
        )
    reranking_specs = [
        "traditional source ranking availability",
        "candidate recall",
        "unconditional HR/NDCG",
        "positive-present conditional HR/NDCG",
    ]
    for name in reranking_specs:
        rows.append(
            req(
                f"RERANKING_{normalize_id(name)}",
                "PAPER_ANALYSIS_DIMENSION",
                "reranking",
                name,
                "REQUIRED_CONDITIONAL",
                "READY_AFTER_TRAINING",
                "Food.com train/candidate inputs are available; source rankers must be trained or computed later",
                "READY_AFTER_TRAINING",
                "Video Games primary candidates are feasible; source rankers must be trained or computed later",
                "NONE",
                "TIE",
                3000 if name == "traditional source ranking availability" else 0,
                "true",
                "SOURCE_RANKING_REQUIRED",
                "D0A_PRIMARY;E2B0_PRIMARY;E2A_RERANKING",
            )
        )
    trad_specs = [
        ("Random", "READY", "READY", "false", "NONE"),
        ("MostPop", "READY", "READY", "false", "NONE"),
        ("BM25-like", "READY_WITH_ADAPTATION", "READY_WITH_ADAPTATION", "false", "TEXT_FIELD_ADAPTATION"),
        ("BPRMF", "READY_AFTER_TRAINING", "READY_AFTER_TRAINING", "true", "TRAINING_REQUIRED"),
        ("GRU4Rec", "READY_AFTER_TRAINING", "READY_AFTER_TRAINING", "true", "TRAINING_REQUIRED"),
        ("SASRec", "READY_AFTER_TRAINING", "READY_AFTER_TRAINING", "true", "TRAINING_REQUIRED"),
        ("LightGCN", "READY_AFTER_TRAINING", "READY_AFTER_TRAINING", "true", "TRAINING_REQUIRED"),
    ]
    for model, food_status, vg_status, training, adaptation in trad_specs:
        rows.append(
            req(
                f"TRADITIONAL_{normalize_id(model)}",
                "LOCAL_REPRODUCTION_PARITY",
                "traditional recommenders",
                model,
                "MANDATORY",
                food_status,
                traditional_reason("Food.com", model, food_status),
                vg_status,
                traditional_reason("Video Games", model, vg_status),
                "NONE",
                "TIE",
                0,
                training,
                adaptation,
                "D0A_TRADITIONAL_CAPABILITY;E2A_TRADITIONAL_PARITY",
            )
        )
    rows.append(
        req(
            "KS_REPRESENTATIVENESS",
            "ORIGINAL_EXTENSION",
            "beyond utility",
            "K-S representativeness if globally recoverable",
            "REQUIRED_CONDITIONAL",
            "BLOCKED_GLOBAL_PROTOCOL",
            "global operational protocol remains underspecified",
            "BLOCKED_GLOBAL_PROTOCOL",
            "global operational protocol remains underspecified",
            "GLOBAL_PROTOCOL",
            "TIE",
            0,
            "false",
            "UNRECOVERED_GLOBAL_POLICY",
            "D0A_PAPER_DIMENSION;E2A_POLICY_BLOCKERS",
        )
    )
    frame = pd.DataFrame(rows)
    validate_requirement_matrix(frame)
    return frame


def req(
    requirement_id: str,
    evidence_family: str,
    experiment_family: str,
    metric_or_analysis: str,
    mandatory_status: str,
    foodcom_status: str,
    foodcom_reason: str,
    videogames_status: str,
    videogames_reason: str,
    blocker_scope: str,
    preferred_dataset_for_requirement: str,
    additional_llm_requests: int,
    additional_training: str,
    adaptation_class: str,
    evidence_source: str,
) -> dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "evidence_family": evidence_family,
        "experiment_family": experiment_family,
        "metric_or_analysis": metric_or_analysis,
        "mandatory_status": mandatory_status,
        "foodcom_status": foodcom_status,
        "foodcom_reason": foodcom_reason,
        "videogames_status": videogames_status,
        "videogames_reason": videogames_reason,
        "blocker_scope": blocker_scope,
        "preferred_dataset_for_requirement": preferred_dataset_for_requirement,
        "additional_llm_requests": int(additional_llm_requests),
        "additional_training": additional_training,
        "adaptation_class": adaptation_class,
        "evidence_source": evidence_source,
    }


def reason_for_status(dataset: str, metric: str, status: str) -> str:
    if status == "READY":
        return f"{dataset} has the final candidate/target data needed for {metric}"
    if status == "READY_WITH_ADAPTATION":
        return f"{dataset} can compute {metric} with a documented domain adaptation"
    if status == "BLOCKED_DATASET":
        return f"{dataset} lacks frozen dataset metadata/feature evidence for {metric}"
    if status == "BLOCKED_GLOBAL_PROTOCOL":
        return f"{metric} is blocked by unrecovered global policy, not by {dataset}"
    if status == "NOT_APPLICABLE":
        return f"{metric} is execution telemetry, not a raw dataset-selection capability"
    if status == "READY_AFTER_TRAINING":
        return f"{dataset} can support {metric} after source model training"
    return status


def traditional_reason(dataset: str, model: str, status: str) -> str:
    if status == "READY":
        return f"{dataset} can compute {model} from frozen candidate/train data without training"
    if status == "READY_WITH_ADAPTATION":
        return f"{dataset} can compute {model} after declaring a text representation"
    if status == "READY_AFTER_TRAINING":
        return f"{dataset} has required train inputs; {model} training is future work"
    return f"{dataset} status for {model}: {status}"


def preferred(food_status: str, vg_status: str) -> str:
    food_score = status_score(food_status)
    vg_score = status_score(vg_status)
    if food_score < vg_score:
        return "Food.com"
    if vg_score < food_score:
        return "Video Games"
    return "TIE"


def status_score(status: str) -> int:
    order = {
        "READY": 0,
        "READY_WITH_ADAPTATION": 1,
        "READY_AFTER_TRAINING": 2,
        "NOT_APPLICABLE": 3,
        "BLOCKED_GLOBAL_PROTOCOL": 4,
        "BLOCKED_DATASET": 5,
    }
    return order[status]


def validate_requirement_matrix(frame: pd.DataFrame) -> None:
    required = [
        "requirement_id",
        "evidence_family",
        "experiment_family",
        "metric_or_analysis",
        "mandatory_status",
        "foodcom_status",
        "foodcom_reason",
        "videogames_status",
        "videogames_reason",
        "blocker_scope",
        "preferred_dataset_for_requirement",
        "additional_llm_requests",
        "additional_training",
        "adaptation_class",
        "evidence_source",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"requirement matrix missing columns: {missing}")
    if frame[required].astype(str).eq("").any().any():
        raise ValueError("requirement matrix contains empty required fields")
    invalid_food = set(frame["foodcom_status"]) - STATUS_VALUES
    invalid_vg = set(frame["videogames_status"]) - STATUS_VALUES
    if invalid_food or invalid_vg:
        raise ValueError(f"invalid statuses: {invalid_food}; {invalid_vg}")
    if frame["requirement_id"].duplicated().any():
        raise ValueError("duplicate requirement_id values")


def build_completeness_counts(requirement_matrix: pd.DataFrame) -> pd.DataFrame:
    mandatory = requirement_matrix.loc[~requirement_matrix["mandatory_status"].eq("OPTIONAL")].copy()
    rows: list[dict[str, Any]] = []
    for dataset, status_column in (("Food.com", "foodcom_status"), ("Video Games", "videogames_status")):
        rows.append(count_row(dataset, "mandatory_total", "ALL", mandatory, status_column))
        for family, group in mandatory.groupby("experiment_family", sort=True):
            rows.append(count_row(dataset, "family", str(family), group, status_column))
    return pd.DataFrame(rows)


def count_row(dataset: str, scope: str, family: str, frame: pd.DataFrame, status_column: str) -> dict[str, Any]:
    counts = frame[status_column].value_counts().to_dict()
    row = {"dataset_candidate": dataset, "scope": scope, "family": family, "total": int(len(frame))}
    for status in COUNT_STATUSES:
        row[status] = int(counts.get(status, 0))
    return row


def build_dataset_blockers(requirement_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mandatory = requirement_matrix.loc[~requirement_matrix["mandatory_status"].eq("OPTIONAL")]
    for dataset, status_column, reason_column in (
        ("Food.com", "foodcom_status", "foodcom_reason"),
        ("Video Games", "videogames_status", "videogames_reason"),
    ):
        blocked = mandatory.loc[mandatory[status_column].eq("BLOCKED_DATASET")]
        if blocked.empty:
            rows.append(
                {
                    "dataset_candidate": dataset,
                    "requirement_id": "NONE",
                    "experiment_family": "ALL",
                    "metric_or_analysis": "none",
                    "blocker_scope": "NONE",
                    "reason": "no mandatory dataset-specific blockers",
                }
            )
        else:
            for row in blocked.itertuples(index=False):
                rows.append(
                    {
                        "dataset_candidate": dataset,
                        "requirement_id": row.requirement_id,
                        "experiment_family": row.experiment_family,
                        "metric_or_analysis": row.metric_or_analysis,
                        "blocker_scope": "DATASET",
                        "reason": getattr(row, reason_column),
                    }
                )
    return pd.DataFrame(rows)


def select_winner(requirement_matrix: pd.DataFrame) -> str:
    mandatory = requirement_matrix.loc[~requirement_matrix["mandatory_status"].eq("OPTIONAL")]
    food_blockers = int(mandatory["foodcom_status"].eq("BLOCKED_DATASET").sum())
    vg_blockers = int(mandatory["videogames_status"].eq("BLOCKED_DATASET").sum())
    if food_blockers < vg_blockers:
        return FOODCOM_WINNER
    if vg_blockers < food_blockers:
        return VIDEO_GAMES_WINNER
    food_ready_like = int(mandatory["foodcom_status"].isin(["READY", "READY_WITH_ADAPTATION", "READY_AFTER_TRAINING"]).sum())
    vg_ready_like = int(mandatory["videogames_status"].isin(["READY", "READY_WITH_ADAPTATION", "READY_AFTER_TRAINING"]).sum())
    if food_ready_like > vg_ready_like:
        return FOODCOM_WINNER
    if vg_ready_like > food_ready_like:
        return VIDEO_GAMES_WINNER
    return BLOCKED_SELECTION


def build_future_budget(root: Path, evidence: Mapping[str, Any], winner: str) -> pd.DataFrame:
    e2a = evidence["e2a_summary"]
    gemma_primary = e2a_request_count(e2a, "gemma_original_four_parity", 15528)
    gemma_canddif = e2a_request_count(e2a, "gemma_canddif_parity", 600)
    original_four_history = int(
        evidence["h0_budget"].loc[evidence["h0_budget"]["scope"].eq("grand_total"), "to_compute_raw_requests"].iloc[0]
    )
    fifth_primary = 3 * 2 * 1000
    fifth_history = 3 * len(HISTORY_LENGTHS) * 200 if winner == FOODCOM_WINNER else 0
    fifth_canddif = 3 * 1000 if winner == FOODCOM_WINNER else 0
    profile_generation = 3 * 200 if winner == FOODCOM_WINNER else 0
    profile_only = 3 * 200 if winner == FOODCOM_WINNER else 0
    profile_history = 3 * 200 if winner == FOODCOM_WINNER else 0
    rows = [
        budget_row("original_four_history", original_four_history, "MANDATORY", "H0 final history matrix; 72 cells"),
        budget_row("fifth_dataset_primary", fifth_primary, "MANDATORY", "3 models x 2 prompts x 1000 users"),
        budget_row("fifth_dataset_history", fifth_history, "MANDATORY", "3 models x 6 lengths x 200 users; feasible only for Food.com"),
        budget_row("gemma_original_four_primary_parity", gemma_primary, "MANDATORY", "derived from E2A gemma_original_four_parity raw rows"),
        budget_row("required_original_four_gemma_canddif", gemma_canddif, "MANDATORY", "derived from E2A gemma_canddif_parity raw rows"),
        budget_row("fifth_dataset_canddif_primary", fifth_canddif, "MANDATORY", "3 models x one order-variant x 1000 users"),
        budget_row("other_sensitivities_predeclared", 0, "NOT_PREDECLARED", "no additional mandatory sensitivity budget is frozen in D0B"),
        budget_row("profile_generation", profile_generation, "REQUIRED_CONDITIONAL", "3 models x 200 paired users"),
        budget_row("profile_only_recommendation", profile_only, "REQUIRED_CONDITIONAL", "3 models x 200 paired users"),
        budget_row("profile_plus_history_recommendation", profile_history, "REQUIRED_CONDITIONAL", "3 models x 200 paired users"),
        budget_row("derived_metrics", 0, "DERIVED", "derived metrics add zero raw LLM requests"),
    ]
    total = sum(row["raw_llm_requests"] for row in rows if row["budget_class"] in {"MANDATORY", "REQUIRED_CONDITIONAL"})
    rows.append(budget_row("grand_total_mandatory_and_conditional", total, "TOTAL", "excludes non-predeclared optional sensitivities"))
    return pd.DataFrame(rows)


def e2a_request_count(summary: pd.DataFrame, claim_family: str, fallback: int) -> int:
    matches = summary.loc[
        summary["summary_scope"].eq("raw_experiment")
        & summary["claim_family"].eq(claim_family)
        & summary["status"].astype(str).isin(["TO_INFER", "TO_COMPUTE_OFFLINE", "TO_TRAIN"])
    ]
    if matches.empty:
        return fallback
    return int(pd.to_numeric(matches["llm_request_count"], errors="coerce").fillna(0).sum())


def budget_row(component: str, raw_llm_requests: int, budget_class: str, evidence: str) -> dict[str, Any]:
    return {
        "component": component,
        "raw_llm_requests": int(raw_llm_requests),
        "budget_class": budget_class,
        "training_required": False,
        "derived_metric_extra_requests": 0,
        "evidence": evidence,
    }


def build_input_evidence(evidence: Mapping[str, Any]) -> pd.DataFrame:
    food_name = evidence["food_metadata"].loc[evidence["food_metadata"]["field"].eq("name")].iloc[0]
    vg_title = evidence["vg_title_summary"].set_index("metric")["value"].to_dict()
    rows = [
        input_row("Food.com", "archive_sha256", evidence["food_archive"]["sha256"], "D0A archive manifest"),
        input_row("Food.com", "raw_interactions_sha256", evidence["food_hashes"]["files"]["RAW_interactions.csv"]["sha256"], "D0A extracted hashes"),
        input_row("Food.com", "raw_recipes_sha256", evidence["food_hashes"]["files"]["RAW_recipes.csv"]["sha256"], "D0A extracted hashes"),
        input_row("Food.com", "primary_strict_users", evidence["food_primary"]["final_strict_name_eligible_users"], "D0A primary feasibility"),
        input_row("Food.com", "history_strict_l20_users", evidence["food_history"]["strict_paired_l20_users"], "D0A history feasibility"),
        input_row("Food.com", "usable_name_coverage", food_name["coverage"], "D0A metadata coverage"),
        input_row("Video Games", "primary_strict_users", evidence["vg_primary"]["strict_title_candidate_users"], "E2B-0 primary feasibility"),
        input_row("Video Games", "primary_user_hash", evidence["vg_primary"]["selected_user_hash"], "E2B-0 primary feasibility"),
        input_row("Video Games", "primary_candidate_hash", evidence["vg_primary"]["selected_candidate_hash"], "E2B-0 primary feasibility"),
        input_row("Video Games", "history_strict_l20_users", evidence["vg_history"]["strict_history_l20_users"], "E2B-0 history feasibility"),
        input_row("Video Games", "title_coverage", vg_title["final_coverage"], "E2B-0 title provenance"),
        input_row("H0", "original_four_history_to_compute", evidence["h0_manifest"]["to_compute_requests"], "H0 no-inference manifest"),
    ]
    return pd.DataFrame(rows)


def input_row(dataset: str, metric: str, value: Any, evidence_source: str) -> dict[str, Any]:
    return {"dataset": dataset, "metric": metric, "value": value, "evidence_source": evidence_source}


def build_decision(requirement_matrix: pd.DataFrame, blockers: pd.DataFrame, winner: str) -> pd.DataFrame:
    mandatory = requirement_matrix.loc[~requirement_matrix["mandatory_status"].eq("OPTIONAL")]
    return pd.DataFrame(
        [
            {
                "winner_token": winner,
                "selected_dataset": "Food.com" if winner == FOODCOM_WINNER else "Video Games" if winner == VIDEO_GAMES_WINNER else "NONE",
                "foodcom_dataset_blockers": int(mandatory["foodcom_status"].eq("BLOCKED_DATASET").sum()),
                "videogames_dataset_blockers": int(mandatory["videogames_status"].eq("BLOCKED_DATASET").sum()),
                "foodcom_ready_like": int(mandatory["foodcom_status"].isin(["READY", "READY_WITH_ADAPTATION", "READY_AFTER_TRAINING"]).sum()),
                "videogames_ready_like": int(mandatory["videogames_status"].isin(["READY", "READY_WITH_ADAPTATION", "READY_AFTER_TRAINING"]).sum()),
                "global_protocol_blocker_requirements": int(
                    (
                        mandatory["foodcom_status"].eq("BLOCKED_GLOBAL_PROTOCOL")
                        & mandatory["videogames_status"].eq("BLOCKED_GLOBAL_PROTOCOL")
                    ).sum()
                ),
                "selection_rule": "fewer mandatory dataset-specific blockers, then more complete final matrix",
                "loser": "Video Games" if winner == FOODCOM_WINNER else "Food.com" if winner == VIDEO_GAMES_WINNER else "NONE",
                "blocker_rows_recorded": len(blockers),
            }
        ]
    )


def build_loser_purge_inventory(root: Path, winner: str) -> dict[str, Any]:
    loser = "video_games" if winner == FOODCOM_WINNER else "foodcom"
    entries: list[dict[str, Any]] = []
    if loser == "video_games":
        remove_paths = [
            Path("data/raw/video_games_2014"),
            Path("data/processed/video_games"),
            Path("configs/video_games.yaml"),
            Path("src/data/prepare_video_games.py"),
            Path("src/experiments/run_video_games_gemma2.py"),
            Path("src/extension/e2b0_video_games_2014_feasibility.py"),
            Path("src/extension/e2b0h_video_games_history_title_blockers.py"),
            Path("src/extension/e2b0j_joint_title_recovery_frontier.py"),
            Path("tests/test_extension_e2b0_video_games_2014_feasibility.py"),
            Path("tests/test_extension_e2b0h_video_games_history_title_blockers.py"),
            Path("tests/test_extension_e2b0j_joint_title_recovery_frontier.py"),
            Path("report/extension_dataset_pivot_video_games.md"),
            Path("report/extension_e2b0_video_games_2014_feasibility.md"),
            Path("report/extension_e2b0h_video_games_history_title_blocker_audit.md"),
            Path("report/extension_e2b0j_joint_title_recovery_frontier.md"),
        ]
        remove_paths.extend(glob_relative(root, "results/tables", "extension_e2b0*.csv"))
        remove_paths.extend(glob_relative(root, "results/tables", "*video_games*.csv"))
        remove_paths.extend(glob_relative(root, "results/manifests", "extension_e2b0*.json"))
        remove_paths.extend(glob_relative(root, "results/manifests", "*video_games*.json"))
        remove_paths.extend(glob_relative(root, "results/extension", "*video_games*.csv"))
        remove_paths.extend(glob_relative(root, "results/raw_outputs", "*video_games*"))
        remove_paths.extend(glob_relative(root, "results/metrics", "*video_games*.csv"))
        seen: set[str] = set()
        for path in remove_paths:
            rel_path = rel_path_string(path)
            if rel_path in seen:
                continue
            seen.add(rel_path)
            entries.append(purge_entry(root, path, "REMOVE_DATASET_SPECIFIC", artifact_type_for_path(path)))
        for path in [
            Path("src/extension/e0_protocol.py"),
            Path("src/extension/e0_candidates.py"),
            Path("src/extension/e0_history.py"),
            Path("src/extension/e0_prompts.py"),
            Path("src/extension/e0_parser.py"),
            Path("tests/test_extension_h0_history_final_parity_audit.py"),
        ]:
            entries.append(purge_entry(root, path, "RETAIN_GENERIC_INFRASTRUCTURE", "generic_extension_infrastructure"))
        for virtual_path, artifact_type in [
            ("results/tables/extension_cell_universe.csv rows where dataset=video_games", "cell_universe_rows"),
            ("results/tables/extension_raw_experiment_universe.csv rows where dataset=video_games", "raw_experiment_rows"),
            ("report/notes.md Video Games D0/E2B notes", "notes_rows"),
            (".gitignore extension_e2b0/video_games visibility rules", "gitignore_rules"),
        ]:
            entries.append(
                {
                    "path": virtual_path,
                    "exists": "",
                    "is_dir": "",
                    "classification": "REVIEW_MANUALLY",
                    "artifact_type": artifact_type,
                    "reason": "D1 cleanup must edit shared files without removing generic extension evidence",
                }
            )
        entries.append(
            {
                "path": "original_repo/EvaLLMasRecommender",
                "exists": str((root / "original_repo/EvaLLMasRecommender").exists()).lower(),
                "is_dir": "true",
                "classification": "RETAIN_HISTORICAL_REPRODUCTION",
                "artifact_type": "historical_reproduction",
                "reason": "not a fifth-dataset operational artifact",
            }
        )
    else:
        for path in [Path("data/raw/foodcom"), Path("configs/foodcom_extension.yaml"), Path("src/extension/d0a_foodcom_material_feasibility.py")]:
            entries.append(purge_entry(root, path, "REMOVE_DATASET_SPECIFIC", artifact_type_for_path(path)))
    return {
        "phase_id": PHASE_ID,
        "winner": winner,
        "loser": "Video Games" if loser == "video_games" else "Food.com",
        "no_purge_executed": True,
        "entries": entries,
        "created_at": utc_now(),
    }


def purge_entry(root: Path, path: Path, classification: str, artifact_type: str) -> dict[str, Any]:
    full = root / path
    return {
        "path": rel_path_string(path),
        "exists": str(full.exists()).lower(),
        "is_dir": str(full.is_dir()).lower() if full.exists() else "false",
        "classification": classification,
        "artifact_type": artifact_type,
        "reason": "losing fifth-dataset operational artifact" if classification == "REMOVE_DATASET_SPECIFIC" else "retain for winner/generic/historical use",
    }


def glob_relative(root: Path, directory: str, pattern: str) -> list[Path]:
    base = root / directory
    if not base.exists():
        return []
    return sorted(path.relative_to(root) for path in base.glob(pattern))


def artifact_type_for_path(path: Path) -> str:
    text = rel_path_string(path)
    if text.startswith("data/raw"):
        return "raw_canonical_or_enrichment_data"
    if text.startswith("data/processed"):
        return "processed_dataset"
    if text.startswith("configs"):
        return "config"
    if text.startswith("src"):
        return "source_module"
    if text.startswith("tests"):
        return "test"
    if text.startswith("report"):
        return "report"
    if "manifests" in text:
        return "manifest"
    if "tables" in text:
        return "table"
    if "raw_outputs" in text:
        return "raw_output"
    if "metrics" in text:
        return "metrics"
    return "other"


def build_no_inference_manifest(
    root: Path,
    evidence: Mapping[str, Any],
    requirement_matrix: pd.DataFrame,
    counts: pd.DataFrame,
    decision: pd.DataFrame,
    future_budget: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "created_at": utc_now(),
        "source_commit": git_head(root),
        "winner_token": decision.iloc[0]["winner_token"],
        "requirement_rows": int(len(requirement_matrix)),
        "foodcom_dataset_blockers": int(decision.iloc[0]["foodcom_dataset_blockers"]),
        "videogames_dataset_blockers": int(decision.iloc[0]["videogames_dataset_blockers"]),
        "future_budget_total_mandatory_and_conditional": int(
            future_budget.loc[future_budget["component"].eq("grand_total_mandatory_and_conditional"), "raw_llm_requests"].iloc[0]
        ),
        "llm_requests_executed": 0,
        "training_runs_executed": 0,
        "remote_api_calls_executed": 0,
        "downloads_executed": 0,
        "purge_actions_executed": 0,
        "input_hashes": {
            "foodcom_archive_sha256": evidence["food_archive"]["sha256"],
            "foodcom_raw_interactions_sha256": evidence["food_hashes"]["files"]["RAW_interactions.csv"]["sha256"],
            "foodcom_raw_recipes_sha256": evidence["food_hashes"]["files"]["RAW_recipes.csv"]["sha256"],
            "videogames_primary_user_hash": evidence["vg_primary"]["selected_user_hash"],
            "videogames_primary_candidate_hash": evidence["vg_primary"]["selected_candidate_hash"],
            "h0_manifest_hash": sha256_file(root / H0_MANIFEST),
            "h0_matrix_hash": sha256_file(root / H0_MATRIX),
        },
        "count_rows": int(len(counts)),
    }


def render_report(outputs: Mapping[str, Any]) -> str:
    decision = outputs["decision"]
    counts = outputs["completeness_counts"]
    blockers = outputs["blockers"]
    future_budget = outputs["future_budget"]
    evidence = outputs["input_evidence"]
    matrix = outputs["requirement_matrix"]
    purge = outputs["purge_inventory"]
    decision_row = decision.iloc[0]
    return f"""# D0B Final Fifth-Dataset Selection

Decision: `{decision_row['winner_token']}`

No LLM inference, recommender training, downloads, purge, commit, or Git
history rewrite was performed.

## Authoritative Input Evidence

{evidence.to_markdown(index=False)}

## Completeness Counts

{counts.to_markdown(index=False)}

## Dataset-Specific Blockers

{blockers.to_markdown(index=False)}

## Requirement Matrix Summary

{matrix.groupby(['experiment_family', 'foodcom_status', 'videogames_status']).size().rename('requirements').reset_index().to_markdown(index=False)}

## Future Inference Budget

{future_budget.to_markdown(index=False)}

## Loser Purge Inventory

Loser: `{purge['loser']}`. Entries: `{len(purge['entries'])}`. No purge executed.
"""


def normalize_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.upper()).strip("_")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_path_string(path: Path | str) -> str:
    return Path(path).as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head(root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

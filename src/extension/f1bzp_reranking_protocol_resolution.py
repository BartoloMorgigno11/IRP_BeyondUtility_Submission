"""F1BZP reranking protocol resolution.

This phase resolves the reranking source protocol without executing new
traditional scoring, training, tuning, LLM inference, or authoritative pool
generation. It audits checkpoint-representable catalog coverage and freezes a
balanced adapted pool policy because the exact paper/repository aggregation is
not locally recoverable.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd
import torch
import yaml

from src.extension.e0_protocol import sha256_json, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1bx_execution_stage1 import (
    ACTIVE_DATASETS,
    TRAINABLE_MODELS,
    DatasetMaterial,
    build_dataset_material,
    dataset_specs as f1bx_dataset_specs,
    frame_hash,
    read_csv,
    read_json,
    sort_key,
)
from src.extension.f1bz_final_traditional_execution import FINAL_SEEDS, build_final_data
from src.utils.config import find_project_root


PHASE_ID = "f1bzp_reranking_protocol_resolution"
DECISION_PAPER_RECOVERED = "F1BZP_PAPER_POOL_PROTOCOL_RECOVERED"
DECISION_ADAPTED_FROZEN = "F1BZP_ADAPTED_POOL_PROTOCOL_FROZEN"
DECISION_BLOCKED = "F1BZP_RERANKING_PROTOCOL_BLOCKED"

MODEL_READY = "MODEL_REPRESENTABLE_FULL_CATALOG_READY"
TARGET_REPRESENTABLE = "REPRESENTABLE_TARGET"
TARGET_OOV_RECALL_MISS = "OOV_TARGET_RECALL_MISS"
EXACT_UNRECOVERED = "PAPER_REPOSITORY_RERANKING_POOL_UNRECOVERED"
ADAPTED_POLICY = "ADAPTED_BALANCED_MULTI_MODEL_RERANKING_POOL"
ADAPTED_ALGORITHM = "BALANCED_RANK_ROUND_ROBIN_UNIQUE_20"
F1BR_DEFECT = "F1BR_PROTOCOL_DEFECT_DISCOVERED_POST_EXECUTION"
NON_AUTHORITATIVE_RECALL = "NON_AUTHORITATIVE_DIAGNOSTIC_SOURCE_RECALL"
PROTOCOL_ONLY = "PROTOCOL_ONLY_NOT_EXECUTED"

CONFIG_PATH = Path("configs/f1bzp_reranking_protocol_resolution.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_PATH = Path("report/f1bzp_reranking_protocol_resolution.md")
MANIFEST_PATH = MANIFEST_DIR / "f1bzp_reranking_protocol_resolution.json"

OUTPUT_TABLES = {
    "checkpoint_vocab_audit": TABLE_DIR / "f1bzp_checkpoint_vocab_audit.csv",
    "target_representability": TABLE_DIR / "f1bzp_target_representability.csv",
    "representable_catalog_readiness": TABLE_DIR / "f1bzp_representable_catalog_readiness.csv",
    "paper_pool_evidence": TABLE_DIR / "f1bzp_paper_pool_evidence.csv",
    "official_artifact_pool_forensics": TABLE_DIR / "f1bzp_official_artifact_pool_forensics.csv",
    "top5x4_hypothesis_audit": TABLE_DIR / "f1bzp_top5x4_hypothesis_audit.csv",
    "table5_consistency_audit": TABLE_DIR / "f1bzp_table5_consistency_audit.csv",
    "pool_policy_resolution": TABLE_DIR / "f1bzp_pool_policy_resolution.csv",
    "corrective_scope": TABLE_DIR / "f1bzp_corrective_scope.csv",
    "validation": TABLE_DIR / "f1bzp_validation.csv",
}

SUPPORT_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/f1bzp_reranking_protocol_resolution.py"),
    Path("tests/test_extension_f1bzp_reranking_protocol_resolution.py"),
    REPORT_PATH,
    Path(".gitignore"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1BZP reranking protocol resolution.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    if args.validate_only:
        outputs = load_existing_outputs(root)
    else:
        outputs = generate_f1bzp(root, Path(args.config))
        write_outputs(root, outputs)
    validation = outputs["validation"]
    manifest = outputs["manifest"]
    print(f"decision={manifest['decision_token']}")
    print(f"model_representable_ready_cells={manifest['model_representable_ready_cells']}")
    print(f"target_rows={manifest['target_representability_rows']}")
    print(f"checkpoint_retraining_required={manifest['checkpoint_retraining_required_cells']}")
    print(f"future_reranking_requests_unlocked={manifest['future_reranking_requests_unlocked']}")
    print(f"validation_passed={str(validation['status'].eq('PASS').all()).lower()}")
    return 0 if bool(validation["status"].eq("PASS").all()) else 1


def generate_f1bzp(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, Path(config_path))
    f1bz_config = load_config(project_root, Path(str(config["inputs"]["f1bz_config"])))
    f1bx_config = load_config(project_root, Path(str(f1bz_config["inputs"]["f1bx_config"])))
    specs = {spec.name: spec for spec in f1bx_dataset_specs(f1bx_config)}

    run_manifest = read_csv(project_root / "results/tables/f1bz_final_run_manifest.csv")
    f1bz_manifest = read_json(project_root / str(config["inputs"]["f1bz_manifest"]))
    f1bza_manifest = read_json(project_root / str(config["inputs"]["f1bza_manifest"]))
    f1bzc_manifest = read_json(project_root / str(config["inputs"]["f1bzc_manifest"]))
    f1bza_readiness = read_csv(project_root / str(config["inputs"]["f1bza_readiness"]))
    f1bza_zero_history = read_csv(project_root / str(config["inputs"]["f1bza_zero_history_summary"]))
    f1bzc_adapter = read_csv(project_root / str(config["inputs"]["f1bzc_adapter_audit"]))
    f1bzc_recall = read_csv(project_root / str(config["inputs"]["f1bzc_candidate_recall"]))
    f1bzc_pool_validation = read_csv(project_root / str(config["inputs"]["f1bzc_pool_validation"]))
    f1bzc_contributions = read_csv(project_root / str(config["inputs"]["f1bzc_source_contributions"]))
    f1br_protocol = read_csv(project_root / str(config["inputs"]["f1br_reranking_source_protocol"]))
    f0_budget = read_csv(project_root / str(config["inputs"]["f0_request_budget"]))

    materials: dict[str, DatasetMaterial] = {}
    final_data: dict[str, Any] = {}
    for dataset in ACTIVE_DATASETS:
        material = build_dataset_material(project_root, specs[dataset])
        materials[dataset] = material
        final_data[dataset] = build_final_data(project_root, specs[dataset], material, f1bz_config)

    checkpoint_vocab = build_checkpoint_vocab_audit(
        project_root,
        config,
        run_manifest,
        materials,
        final_data,
        f1bzc_adapter,
    )
    target_representability = build_target_representability(materials, final_data)
    readiness = build_representable_catalog_readiness(checkpoint_vocab, target_representability)
    paper_evidence = build_paper_pool_evidence(project_root, config, f1br_protocol, f1bzc_manifest)
    artifact_forensics = build_official_artifact_pool_forensics(project_root, f1bzc_pool_validation, f1bzc_contributions)
    top5x4 = build_top5x4_hypothesis_audit()
    table5 = build_table5_consistency_audit()
    pool_policy = build_pool_policy_resolution(config, f1br_protocol, f1bzc_recall, f1bzc_pool_validation, f1bzc_contributions)
    corrective_scope = build_corrective_scope(checkpoint_vocab, pool_policy)
    validation = build_validation(
        project_root,
        config,
        f1bz_manifest,
        f1bza_manifest,
        f1bzc_manifest,
        f1bza_readiness,
        f1bza_zero_history,
        f1bzc_recall,
        f0_budget,
        run_manifest,
        checkpoint_vocab,
        target_representability,
        readiness,
        paper_evidence,
        artifact_forensics,
        top5x4,
        pool_policy,
        corrective_scope,
    )
    decision = DECISION_ADAPTED_FROZEN if bool(validation["status"].eq("PASS").all()) else DECISION_BLOCKED
    manifest = build_manifest(
        project_root,
        config,
        decision,
        f1bz_manifest,
        f1bza_manifest,
        f1bzc_manifest,
        checkpoint_vocab,
        target_representability,
        readiness,
        paper_evidence,
        artifact_forensics,
        top5x4,
        table5,
        pool_policy,
        corrective_scope,
        validation,
    )
    report = render_report(
        decision,
        checkpoint_vocab,
        target_representability,
        readiness,
        paper_evidence,
        top5x4,
        table5,
        pool_policy,
        corrective_scope,
        validation,
    )
    return {
        "checkpoint_vocab_audit": checkpoint_vocab,
        "target_representability": target_representability,
        "representable_catalog_readiness": readiness,
        "paper_pool_evidence": paper_evidence,
        "official_artifact_pool_forensics": artifact_forensics,
        "top5x4_hypothesis_audit": top5x4,
        "table5_consistency_audit": table5,
        "pool_policy_resolution": pool_policy,
        "corrective_scope": corrective_scope,
        "validation": validation,
        "manifest": manifest,
        "report": report,
    }


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def build_checkpoint_vocab_audit(
    root: Path,
    config: Mapping[str, Any],
    run_manifest: pd.DataFrame,
    materials: Mapping[str, DatasetMaterial],
    final_data: Mapping[str, Any],
    f1bzc_adapter: pd.DataFrame,
) -> pd.DataFrame:
    old_status = {
        (str(row["dataset"]), str(row["model"])): str(row["adapter_status"])
        for row in f1bzc_adapter.to_dict("records")
    }
    rows = []
    for dataset in ACTIVE_DATASETS:
        material = materials[dataset]
        data = final_data[dataset]
        raw_catalog = set(map(str, material.catalog_items))
        checkpoint_vocab = set(map(str, data.tuning.item_index))
        representable_catalog = [item for item in material.catalog_items if str(item) in checkpoint_vocab]
        for model in TRAINABLE_MODELS:
            method_runs = run_manifest.loc[
                run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(model)
            ].sort_values("seed", key=lambda series: pd.to_numeric(series, errors="coerce"), kind="mergesort")
            representative = method_runs.iloc[0].to_dict()
            checkpoint_path = root / str(representative["checkpoint_path"])
            shape = checkpoint_item_shape(checkpoint_path)
            expected_rows = int(data.tuning.num_items + (1 if model in {"GRU4Rec", "SASRec"} else 0))
            shape_status = "PASS" if shape["item_embedding_rows"] == expected_rows else "FAIL"
            checkpoint_hash_matches = sha256_file(checkpoint_path) == str(representative["checkpoint_hash"])
            can_score = shape_status == "PASS" and checkpoint_hash_matches and len(representable_catalog) > 0
            prior_status = old_status.get((dataset, model), "")
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "raw_catalog_N": int(len(raw_catalog)),
                    "eligible_catalog_N": int(len(raw_catalog)),
                    "checkpoint_item_vocab_N": int(data.tuning.num_items),
                    "checkpoint_item_embedding_rows": int(shape["item_embedding_rows"]),
                    "checkpoint_shape_expected_rows": expected_rows,
                    "checkpoint_shape_status": shape_status,
                    "representative_checkpoint_path": str(representative["checkpoint_path"]),
                    "representative_checkpoint_hash_matches_manifest": bool_text(checkpoint_hash_matches),
                    "eligible_and_representable_N": int(len(representable_catalog)),
                    "eligible_but_OOV_N": int(len(raw_catalog - checkpoint_vocab)),
                    "train_item_N": int(data.train_item_count),
                    "evaluation_user_N": int(len(data.tuning.users)),
                    "can_score_eligible_catalog_intersection": bool_text(can_score),
                    "requires_scoring_raw_catalog_oov_items": bool_text(False),
                    "checkpoint_retraining_required": bool_text(False),
                    "f1bzc_raw_catalog_status": prior_status,
                    "f1bzc_raw_catalog_blocker_reassessed": (
                        "RESOLVED_BY_MODEL_REPRESENTABLE_POLICY"
                        if prior_status.startswith("BLOCKED")
                        else "PRESERVED_READY_CELL"
                    ),
                    "model_representable_status": MODEL_READY if can_score else "REAL_BLOCKER",
                    "reason": (
                        "The frozen checkpoint can score every item in the raw-catalog/checkpoint-vocabulary intersection."
                        if can_score
                        else "The checkpoint shape or hash does not match its frozen F1BZ manifest."
                    ),
                }
            )
    return pd.DataFrame(rows)


def checkpoint_item_shape(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    if "item_embedding.weight" not in state_dict:
        raise ValueError(f"missing item embedding in checkpoint: {checkpoint_path}")
    weight = state_dict["item_embedding.weight"]
    return {"item_embedding_rows": int(weight.shape[0]), "item_embedding_dim": int(weight.shape[1])}


def build_target_representability(
    materials: Mapping[str, DatasetMaterial],
    final_data: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for dataset in ACTIVE_DATASETS:
        material = materials[dataset]
        data = final_data[dataset]
        raw_catalog = set(map(str, material.catalog_items))
        checkpoint_vocab = set(map(str, data.tuning.item_index))
        representable_catalog = raw_catalog & checkpoint_vocab
        item_by_index = {index: item for item, index in data.tuning.item_index.items()}
        for user_row, user_id in enumerate(data.tuning.users):
            positive = str(data.positive_item_ids[user_row])
            positive_in_raw = positive in raw_catalog
            positive_in_vocab = positive in checkpoint_vocab
            positive_representable = positive_in_raw and positive_in_vocab
            user_index = int(data.eval_user_indices[user_row])
            seen_indices = set(data.tuning.user_positive_items.get(user_index, set()))
            seen_items = {str(item_by_index[index]) for index in seen_indices if index in item_by_index}
            history_excluded = positive in seen_items
            eligible_count = len(representable_catalog - seen_items)
            for model in TRAINABLE_MODELS:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "user_id": str(user_id),
                        "positive_item": positive,
                        "positive_in_raw_catalog": bool_text(positive_in_raw),
                        "positive_in_checkpoint_vocab": bool_text(positive_in_vocab),
                        "positive_representable": bool_text(positive_representable),
                        "history_excluded": bool_text(history_excluded),
                        "eligible_candidate_count": int(eligible_count),
                        "target_forced": bool_text(False),
                        "candidate_recall_if_oov": "positive_present=false" if not positive_representable else "",
                        "status": TARGET_REPRESENTABLE if positive_representable else TARGET_OOV_RECALL_MISS,
                        "reason": (
                            "Target is in the model-representable raw catalog."
                            if positive_representable
                            else "Target is not model-representable and must count as absent in future Candidate Recall."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_representable_catalog_readiness(
    checkpoint_vocab: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    grouped_targets = targets.groupby(["dataset", "model"], sort=False)
    rows = []
    for row in checkpoint_vocab.to_dict("records"):
        dataset = str(row["dataset"])
        model = str(row["model"])
        group = grouped_targets.get_group((dataset, model))
        ready = str(row["model_representable_status"]) == MODEL_READY
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "evaluation_user_N": int(len(group)),
                "representable_target_N": int(group["positive_representable"].eq("true").sum()),
                "oov_target_N": int(group["positive_representable"].ne("true").sum()),
                "history_excluded_target_N": int(group["history_excluded"].eq("true").sum()),
                "min_eligible_candidate_count": int(group["eligible_candidate_count"].astype(int).min()),
                "model_representable_status": row["model_representable_status"],
                "full_catalog_scoring_scope": "eligible_catalog_intersect_checkpoint_item_vocabulary",
                "checkpoint_retraining_required": bool_text(False),
                "full_catalog_regeneration_required": bool_text(True),
                "status": "READY_FOR_FUTURE_PROTOCOL_CONTROLLED_REGENERATION" if ready else "REAL_BLOCKER",
                "reason": (
                    "The prior raw-catalog blocker is removed by using the representable catalog intersection."
                    if ready
                    else "A real checkpoint or vocabulary blocker remains."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_paper_pool_evidence(
    root: Path,
    config: Mapping[str, Any],
    f1br_protocol: pd.DataFrame,
    f1bzc_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    paper_path = Path(str(config["paper_pool_evidence"]["local_pdf"]))
    exact_recovered = f1br_protocol["exact_union_order_recovered"].astype(str).str.lower().eq("true").all()
    rows = [
        {
            "evidence_id": "paper_section_4_1_3",
            "source_path": paper_path.as_posix(),
            "source_hash": sha256_file(root / paper_path) if (root / paper_path).exists() else "",
            "specified_candidate_pool_size": "20",
            "specified_source_models": stable_json(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]),
            "positive_may_be_absent": "true",
            "specified_source_depth": "UNSPECIFIED",
            "specified_model_order": "UNSPECIFIED",
            "specified_duplicate_handling": "UNSPECIFIED",
            "specified_refill_behavior": "UNSPECIFIED",
            "classification": "PAPER_PARTIAL_POOL_SPECIFICATION",
            "notes": "Local prior controlled evidence records the paper statement, but direct PDF text extraction is unavailable in this environment.",
        },
        {
            "evidence_id": "paper_section_4_8",
            "source_path": paper_path.as_posix(),
            "source_hash": sha256_file(root / paper_path) if (root / paper_path).exists() else "",
            "specified_candidate_pool_size": "UNSPECIFIED_IN_TABLE_SECTION",
            "specified_source_models": "UNSPECIFIED_IN_TABLE_SECTION",
            "positive_may_be_absent": "NOT_TABLE_EVIDENCE",
            "specified_source_depth": "UNSPECIFIED",
            "specified_model_order": "UNSPECIFIED",
            "specified_duplicate_handling": "UNSPECIFIED",
            "specified_refill_behavior": "UNSPECIFIED",
            "classification": "CONSISTENCY_CHECK_ONLY",
            "notes": "Beauty Table 5 utility values cannot determine pool aggregation by themselves.",
        },
        {
            "evidence_id": "f1br_protocol_freeze",
            "source_path": "results/tables/f1br_reranking_source_protocol.csv",
            "source_hash": sha256_file(root / "results/tables/f1br_reranking_source_protocol.csv"),
            "specified_candidate_pool_size": "20",
            "specified_source_models": stable_json(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]),
            "positive_may_be_absent": "true",
            "specified_source_depth": "20_ADAPTED",
            "specified_model_order": stable_json(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]),
            "specified_duplicate_handling": "ADAPTED_KEEP_FIRST",
            "specified_refill_behavior": "ADAPTED_CONCATENATION",
            "classification": EXACT_UNRECOVERED if not exact_recovered else "PAPER_REPOSITORY_RERANKING_POOL_RECOVERED",
            "notes": "F1BR recovered the source model set but explicitly did not recover exact union order.",
        },
        {
            "evidence_id": "f1bzc_empirical_defect",
            "source_path": "results/manifests/f1bzc_full_catalog_recovery.json",
            "source_hash": sha256_file(root / "results/manifests/f1bzc_full_catalog_recovery.json"),
            "specified_candidate_pool_size": "20",
            "specified_source_models": stable_json(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]),
            "positive_may_be_absent": "true",
            "specified_source_depth": "20_ADAPTED",
            "specified_model_order": stable_json(["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]),
            "specified_duplicate_handling": "ADAPTED_KEEP_FIRST",
            "specified_refill_behavior": "ADAPTED_CONCATENATION",
            "classification": F1BR_DEFECT,
            "notes": f"F1BZC future requests unlocked: {f1bzc_manifest.get('future_requests_unlocked')}.",
        },
    ]
    return pd.DataFrame(rows)


def build_official_artifact_pool_forensics(
    root: Path,
    f1bzc_pool_validation: pd.DataFrame,
    f1bzc_contributions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    known_artifacts = [
        (
            "template_candidate_first",
            Path("original_repo/EvaLLMasRecommender/eva/templates/template_candidate_first.txt"),
            "Single candidate_list prompt asks for top 10; no aggregation contract.",
            "JOINT_CANDIDATE_LIST_PROMPT_ONLY",
        ),
        (
            "template",
            Path("original_repo/EvaLLMasRecommender/eva/templates/template.txt"),
            "Single candidate_list prompt asks for top 10; no aggregation contract.",
            "JOINT_CANDIDATE_LIST_PROMPT_ONLY",
        ),
        (
            "llmrank_template",
            Path("original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt"),
            "Ranks 20 supplied candidates; no source-model aggregation contract.",
            "LLMRANK_PROMPT_ONLY",
        ),
        (
            "llms_code",
            Path("original_repo/EvaLLMasRecommender/eva/codes/llms.py"),
            "System instruction asks for 10 items from the candidate list; no pool construction code.",
            "OUTPUT_COUNT_CODE_ONLY",
        ),
        (
            "output_refine_code",
            Path("original_repo/EvaLLMasRecommender/eva/codes/output_refine.py"),
            "Output parser/refinement evidence concerns returned recommendations, not candidate-pool construction.",
            "OUTPUT_PARSING_ONLY",
        ),
        (
            "llm_metrics_code",
            Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"),
            "Metric code consumes result lists and does not specify source-model contribution policy.",
            "METRIC_CONSUMPTION_ONLY",
        ),
        (
            "readme_task_statement",
            Path("original_repo/EvaLLMasRecommender/README.md"),
            "README states ranking and re-ranking differ by candidate-set formation, without mechanics.",
            "HIGH_LEVEL_TASK_DESCRIPTION_ONLY",
        ),
    ]
    for artifact_id, rel_path, observation, classification in known_artifacts:
        rows.append(
            {
                "artifact_id": artifact_id,
                "source_path": rel_path.as_posix(),
                "source_hash": sha256_file(root / rel_path) if (root / rel_path).exists() else "",
                "observed_content": observation,
                "per_model_contribution_counts": "UNSPECIFIED",
                "pool_order": "UNSPECIFIED",
                "duplicate_handling": "UNSPECIFIED",
                "source_top_k_depth": "UNSPECIFIED",
                "source_specific_or_joint": "JOINT_SUPPLIED_CANDIDATE_LIST" if "template" in artifact_id or artifact_id == "llms_code" else "UNSPECIFIED",
                "gpt35_pool_mode": "UNSPECIFIED",
                "classification": classification,
            }
        )
    for rel_path in sorted((root / "original_repo/EvaLLMasRecommender/llm_results").glob("*.zip")):
        rows.append(
            {
                "artifact_id": f"released_{rel_path.stem}_llm_results",
                "source_path": rel_path.relative_to(root).as_posix(),
                "source_hash": sha256_file(rel_path),
                "observed_content": "Valid zip archive" if zipfile.is_zipfile(rel_path) else "Not a valid inspectable zip archive in local checkout.",
                "per_model_contribution_counts": "UNSPECIFIED",
                "pool_order": "UNSPECIFIED",
                "duplicate_handling": "UNSPECIFIED",
                "source_top_k_depth": "UNSPECIFIED",
                "source_specific_or_joint": "UNSPECIFIED",
                "gpt35_pool_mode": "UNSPECIFIED",
                "classification": "NO_POOL_FORENSIC_CONTENT_AVAILABLE",
            }
        )
    contribution_summary = (
        f1bzc_contributions.groupby("source_model")["contribution_count"].apply(lambda s: int(s.astype(int).sum())).to_dict()
        if not f1bzc_contributions.empty
        else {}
    )
    pool_status = f1bzc_pool_validation["status"].value_counts().to_dict() if not f1bzc_pool_validation.empty else {}
    rows.append(
        {
            "artifact_id": "f1bzc_attempted_pool_forensics",
            "source_path": "results/tables/f1bzc_reranking_source_contributions.csv",
            "source_hash": sha256_file(root / "results/tables/f1bzc_reranking_source_contributions.csv"),
            "observed_content": "F1BR adapted concatenation filled every attempted LastFM pool from BPRMF only.",
            "per_model_contribution_counts": stable_json(contribution_summary),
            "pool_order": "BPRMF_EXHAUSTS_POOL_BEFORE_LATER_MODELS",
            "duplicate_handling": "KEEP_FIRST_IN_CONCATENATION",
            "source_top_k_depth": "20",
            "source_specific_or_joint": "JOINT_POOL",
            "gpt35_pool_mode": "NOT_EXECUTED",
            "classification": F1BR_DEFECT,
        }
    )
    rows.append(
        {
            "artifact_id": "f1bzc_pool_validation",
            "source_path": "results/tables/f1bzc_reranking_pool_validation.csv",
            "source_hash": sha256_file(root / "results/tables/f1bzc_reranking_pool_validation.csv"),
            "observed_content": stable_json(pool_status),
            "per_model_contribution_counts": "SEE_F1BZC_SOURCE_CONTRIBUTIONS",
            "pool_order": "BLOCKED",
            "duplicate_handling": "BLOCKED",
            "source_top_k_depth": "20",
            "source_specific_or_joint": "JOINT_POOL_ATTEMPTED",
            "gpt35_pool_mode": "NOT_EXECUTED",
            "classification": "BLOCKED_BEFORE_LLM_RERANKING",
        }
    )
    return pd.DataFrame(rows)


def build_top5x4_hypothesis_audit() -> pd.DataFrame:
    rows = [
        {
            "evidence_id": "arithmetic_fit",
            "observation": "Four source models times evaluation cutoff five equals twenty candidates.",
            "supports_top5x4": "weak",
            "classification": "PLAUSIBLE_BUT_UNRECOVERED",
            "reason": "Arithmetic compatibility is not protocol evidence.",
        },
        {
            "evidence_id": "paper_wording",
            "observation": "Available controlled evidence says top-K items from four models and a 20-item pool.",
            "supports_top5x4": "ambiguous",
            "classification": "PLAUSIBLE_BUT_UNRECOVERED",
            "reason": "The audited evidence does not specify K=5 per model, order, duplicates, or refill.",
        },
        {
            "evidence_id": "official_prompt_templates",
            "observation": "Original templates consume one candidate list and ask for top 10 or ranking 20 supplied candidates.",
            "supports_top5x4": "no",
            "classification": "PLAUSIBLE_BUT_UNRECOVERED",
            "reason": "Prompt templates do not expose how the upstream 20 candidates were assembled.",
        },
        {
            "evidence_id": "historical_predictions",
            "observation": "Local historical predictions demonstrate non-tautological full-catalog retrieval but not aggregation mechanics.",
            "supports_top5x4": "ambiguous",
            "classification": "PLAUSIBLE_BUT_UNRECOVERED",
            "reason": "Historical files lack recovered source contribution counts for the exact paper pool.",
        },
        {
            "evidence_id": "final_classification",
            "observation": "Top-5 x four-model pooling remains plausible but unrecovered.",
            "supports_top5x4": "not_enough",
            "classification": "PLAUSIBLE_BUT_UNRECOVERED",
            "reason": "F1BZP must not claim this as paper-exact.",
        },
    ]
    return pd.DataFrame(rows)


def build_table5_consistency_audit() -> pd.DataFrame:
    values = {
        "beauty_bprmf_hr5": "0.0238",
        "beauty_gru4rec_hr5": "0.0216",
        "beauty_sasrec_hr5": "0.0242",
        "beauty_lightgcn_hr5": "0.0244",
        "beauty_gpt35_hr5": "0.0320",
    }
    rows = []
    for hypothesis, assessment in [
        ("ONE_JOINT_20_ITEM_MULTI_MODEL_POOL", "COMPATIBLE_NOT_IDENTIFYING"),
        ("FOUR_SOURCE_SPECIFIC_TOPK_POOLS_THEN_AGGREGATION", "COMPATIBLE_NOT_IDENTIFYING"),
        ("ANOTHER_RECOVERED_INTERPRETATION", "NO_LOCAL_RECOVERY"),
    ]:
        rows.append(
            {
                "hypothesis": hypothesis,
                **values,
                "consistency_assessment": assessment,
                "classification": "INCONCLUSIVE_CONSISTENCY_CHECK",
                "reason": "Beauty Table 5 can check plausibility but cannot recover candidate-pool assembly mechanics.",
            }
        )
    return pd.DataFrame(rows)


def build_pool_policy_resolution(
    config: Mapping[str, Any],
    f1br_protocol: pd.DataFrame,
    f1bzc_recall: pd.DataFrame,
    f1bzc_pool_validation: pd.DataFrame,
    f1bzc_contributions: pd.DataFrame,
) -> pd.DataFrame:
    model_order = list(config["adapted_pool_policy"]["model_order"])
    requested_depth = int(config["adapted_pool_policy"]["requested_source_depth"])
    final_size = int(config["adapted_pool_policy"]["final_pool_size"])
    single_model_dominated = (
        int(f1bzc_pool_validation.loc[f1bzc_pool_validation["check_name"].eq("multi_model_contribution"), "observed"].astype(int).iloc[0])
        if not f1bzc_pool_validation.empty
        and f1bzc_pool_validation["check_name"].eq("multi_model_contribution").any()
        else 0
    )
    lastfm_recall = f1bzc_recall.loc[f1bzc_recall["dataset"].eq("lastfm")].iloc[0]
    contribution_summary = (
        f1bzc_contributions.groupby("source_model")["contribution_count"].apply(lambda s: int(s.astype(int).sum())).to_dict()
        if not f1bzc_contributions.empty
        else {}
    )
    exact_recovered = f1br_protocol["exact_union_order_recovered"].astype(str).str.lower().eq("true").all()
    rows = [
        {
            "component": "exact_paper_repository_pool_aggregation",
            "policy_id": "PAPER_REPOSITORY_EXACT",
            "algorithm_id": "UNRECOVERED",
            "resolution_status": EXACT_UNRECOVERED if not exact_recovered else "PAPER_REPOSITORY_RERANKING_POOL_RECOVERED",
            "paper_exact": bool_text(bool(exact_recovered)),
            "requested_source_depth": "UNRECOVERED",
            "actual_max_depth_used": PROTOCOL_ONLY,
            "users_unable_to_fill_20": PROTOCOL_ONLY,
            "model_order": "UNRECOVERED",
            "final_pool_size": "20",
            "force_positive": "false",
            "target_identity_used": "false",
            "candidate_order_immutable_for_llm": "true",
            "source_order_preservation": "UNRECOVERED",
            "candidate_recall_classification": "NOT_COMPUTED_PROTOCOL_ONLY",
            "reason": "Paper/repository evidence specifies the source model set and 20 candidates, but not the exact aggregation mechanics.",
        },
        {
            "component": "f1br_pool_aggregation_component",
            "policy_id": "F1BR_ADAPTED_ORDERED_DEDUP_TOP20_UNION_V1",
            "algorithm_id": "MODEL_CONCATENATION_DEDUP_TOP20",
            "resolution_status": "SUPERSEDED_POOL_COMPONENT_ONLY",
            "paper_exact": "false",
            "requested_source_depth": "20",
            "actual_max_depth_used": "20_ON_LASTFM_DIAGNOSTIC",
            "users_unable_to_fill_20": "0_ON_ATTEMPTED_LASTFM_DIAGNOSTIC",
            "model_order": stable_json(model_order),
            "final_pool_size": "20",
            "force_positive": "false",
            "target_identity_used": "false",
            "candidate_order_immutable_for_llm": "true",
            "source_order_preservation": "within_model_only",
            "candidate_recall_classification": NON_AUTHORITATIVE_RECALL,
            "reason": f"{F1BR_DEFECT}; contribution totals {stable_json(contribution_summary)}; dominated pools {single_model_dominated}.",
        },
        {
            "component": "f1bzp_adapted_pool_policy",
            "policy_id": ADAPTED_POLICY,
            "algorithm_id": ADAPTED_ALGORITHM,
            "resolution_status": "FROZEN",
            "paper_exact": "false",
            "requested_source_depth": str(requested_depth),
            "actual_max_depth_used": PROTOCOL_ONLY,
            "users_unable_to_fill_20": PROTOCOL_ONLY,
            "model_order": stable_json(model_order),
            "final_pool_size": str(final_size),
            "force_positive": bool_text(bool(config["adapted_pool_policy"]["force_positive"])),
            "target_identity_used": bool_text(bool(config["adapted_pool_policy"]["target_identity_used"])),
            "candidate_order_immutable_for_llm": bool_text(bool(config["adapted_pool_policy"]["immutable_candidate_order_for_llm"])),
            "source_order_preservation": bool_text(bool(config["adapted_pool_policy"]["preserve_source_internal_order"])),
            "candidate_recall_classification": "NOT_COMPUTED_PROTOCOL_ONLY",
            "reason": "Balanced rank round-robin prevents first-model domination while preserving source-internal order and deterministic duplicate refill.",
        },
        {
            "component": "lastfm_diagnostic_recall_preservation",
            "policy_id": "F1BZC_FULL_CATALOG_FIVE_SEED_MEAN",
            "algorithm_id": "DIAGNOSTIC_ONLY_INVALID_POOL_AGGREGATION",
            "resolution_status": NON_AUTHORITATIVE_RECALL,
            "paper_exact": "false",
            "requested_source_depth": "20",
            "actual_max_depth_used": "20_ON_LASTFM_DIAGNOSTIC",
            "users_unable_to_fill_20": "0_ON_ATTEMPTED_LASTFM_DIAGNOSTIC",
            "model_order": stable_json(model_order),
            "final_pool_size": "20",
            "force_positive": "false",
            "target_identity_used": "false",
            "candidate_order_immutable_for_llm": "true",
            "source_order_preservation": "within_model_only",
            "candidate_recall_classification": NON_AUTHORITATIVE_RECALL,
            "reason": f"Preserve LastFM {lastfm_recall['positive_present_N']}/{lastfm_recall['evaluation_N']} as diagnostic only.",
        },
    ]
    return pd.DataFrame(rows)


def build_corrective_scope(checkpoint_vocab: pd.DataFrame, pool_policy: pd.DataFrame) -> pd.DataFrame:
    policy_ready = pool_policy.loc[
        pool_policy["policy_id"].eq(ADAPTED_POLICY) & pool_policy["resolution_status"].eq("FROZEN")
    ]
    rows = []
    for row in checkpoint_vocab.to_dict("records"):
        checkpoint_valid = (
            str(row["checkpoint_shape_status"]) == "PASS"
            and str(row["representative_checkpoint_hash_matches_manifest"]) == "true"
        )
        representable_ready = str(row["model_representable_status"]) == MODEL_READY
        pool_ready = not policy_ready.empty
        real_blocker = "NONE" if checkpoint_valid and representable_ready and pool_ready else "REAL_BLOCKER_PRESENT"
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "CHECKPOINT_VALID": bool_text(checkpoint_valid),
                "REPRESENTABLE_CATALOG_SCORING_READY": bool_text(representable_ready),
                "POOL_POLICY_READY": bool_text(pool_ready),
                "FULL_CATALOG_REGENERATION_REQUIRED": bool_text(True),
                "REAL_BLOCKER": real_blocker,
                "checkpoint_retraining_required": bool_text(False),
                "future_authoritative_pool_generation_in_f1bzp": bool_text(False),
                "next_step_status": "READY_FOR_FUTURE_FULL_CATALOG_REGENERATION",
                "reason": "Future execution must regenerate source top20 over the representable catalog and then apply the frozen balanced pool policy.",
            }
        )
    return pd.DataFrame(rows)


def balanced_round_robin_unique_pool(
    source_lists: Mapping[str, Sequence[str]],
    model_order: Sequence[str] = TRAINABLE_MODELS,
    pool_size: int = 20,
    source_depth: int = 20,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build a deterministic unique pool by rank round-robin over source models."""
    pool: list[str] = []
    seen: set[str] = set()
    lineage: list[dict[str, Any]] = []
    max_rank = min(
        source_depth,
        max((len(source_lists.get(model, ())) for model in model_order), default=0),
    )
    for rank_index in range(max_rank):
        for model in model_order:
            items = source_lists.get(model, ())
            if rank_index >= len(items) or rank_index >= source_depth:
                continue
            item = str(items[rank_index])
            duplicate = item in seen
            if not duplicate:
                seen.add(item)
                pool.append(item)
                lineage.append(
                    {
                        "candidate_position": len(pool),
                        "item_id": item,
                        "source_model": str(model),
                        "source_rank": rank_index + 1,
                        "duplicate_skipped": bool_text(False),
                    }
                )
                if len(pool) == pool_size:
                    return pool, lineage
    return pool, lineage


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    f1bz_manifest: Mapping[str, Any],
    f1bza_manifest: Mapping[str, Any],
    f1bzc_manifest: Mapping[str, Any],
    f1bza_readiness: pd.DataFrame,
    f1bza_zero_history: pd.DataFrame,
    f1bzc_recall: pd.DataFrame,
    f0_budget: pd.DataFrame,
    run_manifest: pd.DataFrame,
    checkpoint_vocab: pd.DataFrame,
    targets: pd.DataFrame,
    readiness: pd.DataFrame,
    paper_evidence: pd.DataFrame,
    artifact_forensics: pd.DataFrame,
    top5x4: pd.DataFrame,
    pool_policy: pd.DataFrame,
    corrective_scope: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    budget = {str(row["budget_item"]): int(row["request_count"]) for row in f0_budget.to_dict("records")}
    expected_users = {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}
    target_counts = targets.groupby("dataset")["user_id"].nunique().astype(int).to_dict()
    target_rows_expected = sum(expected_users.values()) * len(TRAINABLE_MODELS)
    adapted = pool_policy.loc[pool_policy["policy_id"].eq(ADAPTED_POLICY)]
    exact = pool_policy.loc[pool_policy["policy_id"].eq("PAPER_REPOSITORY_EXACT")]
    old_f1br = pool_policy.loc[pool_policy["policy_id"].eq("F1BR_ADAPTED_ORDERED_DEDUP_TOP20_UNION_V1")]
    no_dup_sources = {
        "BPRMF": [f"b{index}" for index in range(1, 21)],
        "GRU4Rec": [f"g{index}" for index in range(1, 21)],
        "SASRec": [f"s{index}" for index in range(1, 21)],
        "LightGCN": [f"l{index}" for index in range(1, 21)],
    }
    no_dup_pool, no_dup_lineage = balanced_round_robin_unique_pool(no_dup_sources)
    duplicate_sources = {
        "BPRMF": ["shared1", "shared2", "b3", "b4", "b5", "b6"],
        "GRU4Rec": ["shared1", "g2", "g3", "g4", "g5", "g6"],
        "SASRec": ["s1", "shared2", "s3", "s4", "s5", "s6"],
        "LightGCN": ["l1", "l2", "shared1", "l4", "l5", "l6"],
    }
    dup_pool_a, dup_lineage_a = balanced_round_robin_unique_pool(duplicate_sources, pool_size=12, source_depth=6)
    dup_pool_b, dup_lineage_b = balanced_round_robin_unique_pool(duplicate_sources, pool_size=12, source_depth=6)

    check("precondition_f1bz_validated", f1bz_manifest.get("decision_token") == "F1BZ_FINAL_TRADITIONAL_EXECUTION_VALIDATED", f1bz_manifest.get("decision_token"))
    check("precondition_f1bza_correction_required", f1bza_manifest.get("decision_token") == "F1BZA_CORRECTION_REQUIRED", f1bza_manifest.get("decision_token"))
    check("precondition_f1bzc_blocked", f1bzc_manifest.get("decision_token") == "F1BZC_RERANKING_SOURCE_BLOCKED", f1bzc_manifest.get("decision_token"))
    check("active_datasets_exact", f1bz_manifest.get("active_datasets") == list(ACTIVE_DATASETS), f1bz_manifest.get("active_datasets"))
    check("video_games_absent", int(f1bz_manifest.get("integrity", {}).get("active_video_games_rows", -1)) == 0, f1bz_manifest.get("integrity", {}).get("active_video_games_rows"))
    check("run_manifest_100_completed", len(run_manifest) == 100 and run_manifest["training_status"].eq("COMPLETED").all(), len(run_manifest))
    check("checkpoint_vocab_intersection_20_ready", len(checkpoint_vocab) == 20 and checkpoint_vocab["model_representable_status"].eq(MODEL_READY).all(), checkpoint_vocab["model_representable_status"].value_counts().to_dict())
    check("old_16_raw_catalog_blockers_reassessed", int(checkpoint_vocab["f1bzc_raw_catalog_blocker_reassessed"].eq("RESOLVED_BY_MODEL_REPRESENTABLE_POLICY").sum()) == 16, checkpoint_vocab["f1bzc_raw_catalog_blocker_reassessed"].value_counts().to_dict())
    check("checkpoint_hash_shape_valid", checkpoint_vocab["checkpoint_shape_status"].eq("PASS").all() and checkpoint_vocab["representative_checkpoint_hash_matches_manifest"].eq("true").all(), checkpoint_vocab[["checkpoint_shape_status", "representative_checkpoint_hash_matches_manifest"]].value_counts().to_dict())
    check("target_representability_rows", len(targets) == target_rows_expected and target_counts == expected_users, {"rows": len(targets), "users": target_counts})
    check("oov_targets_are_recall_misses_not_forced", targets["target_forced"].eq("false").all() and set(targets["status"]).issubset({TARGET_REPRESENTABLE, TARGET_OOV_RECALL_MISS}), targets["status"].value_counts().to_dict())
    check("representable_readiness_20_cells", len(readiness) == 20 and readiness["status"].eq("READY_FOR_FUTURE_PROTOCOL_CONTROLLED_REGENERATION").all(), readiness["status"].value_counts().to_dict())
    zero_overall = f1bza_zero_history.loc[f1bza_zero_history["scope"].eq("overall_foodcom_final_cohort")].iloc[0]
    check("f1bza_foodcom_zero_history_preserved", zero_overall["decision"] == "FOODCOM_HISTORY_VALID" and zero_overall["actual_zero_history_users"] == "727" and zero_overall["pipeline_zero_history_users"] == "0", zero_overall.to_dict())
    check("exact_paper_evidence_unrecovered", not exact.empty and exact["resolution_status"].iloc[0] == EXACT_UNRECOVERED, exact.to_dict("records"))
    check("official_forensics_no_exact_aggregation", len(artifact_forensics) >= 9 and not artifact_forensics["classification"].eq("PAPER_REPOSITORY_RERANKING_POOL_RECOVERED").any(), artifact_forensics["classification"].value_counts().to_dict())
    check("top5x4_not_claimed_supported", top5x4["classification"].eq("PLAUSIBLE_BUT_UNRECOVERED").all() and "SUPPORTED" not in set(top5x4["classification"]), top5x4.to_dict("records"))
    check("f1br_defect_explicit_and_superseded", not old_f1br.empty and old_f1br["resolution_status"].iloc[0] == "SUPERSEDED_POOL_COMPONENT_ONLY" and F1BR_DEFECT in old_f1br["reason"].iloc[0], old_f1br.to_dict("records"))
    check("adapted_pool_policy_frozen", not adapted.empty and adapted["resolution_status"].iloc[0] == "FROZEN" and adapted["paper_exact"].iloc[0] == "false", adapted.to_dict("records"))
    check("balanced_policy_anti_domination_invariant", len(no_dup_pool) == 20 and {row["source_model"] for row in no_dup_lineage[:4]} == set(TRAINABLE_MODELS) and no_dup_lineage[0]["source_model"] == "BPRMF" and no_dup_lineage[4]["source_rank"] == 2, no_dup_lineage[:8])
    check("dedupe_refill_determinism", dup_pool_a == dup_pool_b and dup_lineage_a == dup_lineage_b and len(dup_pool_a) == len(set(dup_pool_a)) == 12, dup_lineage_a)
    check("balanced_policy_signature_no_positive_argument", "positive_item" not in inspect.signature(balanced_round_robin_unique_pool).parameters, str(inspect.signature(balanced_round_robin_unique_pool)))
    check("pool_size_20_source_order_preserved", adapted["final_pool_size"].iloc[0] == "20" and adapted["source_order_preservation"].iloc[0] == "true", adapted.to_dict("records"))
    check("no_authoritative_pool_generation", bool(config["execution"]["allow_authoritative_pool_generation"]) is False and adapted["actual_max_depth_used"].iloc[0] == PROTOCOL_ONLY, adapted.to_dict("records"))
    check("source_depth_20_frozen_before_recall", adapted["requested_source_depth"].iloc[0] == "20" and adapted["users_unable_to_fill_20"].iloc[0] == PROTOCOL_ONLY, adapted.to_dict("records"))
    check("lastfm_409_preserved_as_non_authoritative", f1bzc_recall.loc[f1bzc_recall["dataset"].eq("lastfm"), "positive_present_N"].iloc[0] == "409" and pool_policy["resolution_status"].eq(NON_AUTHORITATIVE_RECALL).any(), f1bzc_recall.loc[f1bzc_recall["dataset"].eq("lastfm")].to_dict("records"))
    check("corrective_scope_20_no_real_blocker", len(corrective_scope) == 20 and corrective_scope["REAL_BLOCKER"].eq("NONE").all() and corrective_scope["checkpoint_retraining_required"].eq("false").all(), corrective_scope["REAL_BLOCKER"].value_counts().to_dict())
    check("f0_request_budget_invariant", budget.get("CORE_MANDATORY_NOW_REQUESTS") == 29292 and budget.get("REQUIRED_SENSITIVITY_REQUESTS") == 21000 and budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292 and budget.get("TRAINING_DEPENDENT_REQUESTS") == 29292 and budget.get("CONDITIONAL_REQUESTS") == 9000 and budget.get("OPTIONAL_REQUESTS") == 12000 and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584, budget)
    check("future_requests_unlocked_remain_zero", int(config["request_budget"]["reranking_requests_unlocked_in_f1bzp"]) == 0 and int(f1bzc_manifest.get("future_requests_unlocked", -1)) == 0, {"f1bzp": config["request_budget"]["reranking_requests_unlocked_in_f1bzp"], "f1bzc": f1bzc_manifest.get("future_requests_unlocked")})
    check("no_training_no_llm_no_network", not any(bool(config["execution"][key]) for key in ["allow_traditional_scoring", "allow_traditional_training", "allow_baseline_execution", "allow_tuning", "allow_final_test_evaluation", "allow_llm_inference", "allow_ollama_recommendation_calls", "allow_external_api_calls", "allow_downloads", "allow_network"]), config["execution"])
    check("original_repo_paper_unchanged", git_output(root, ["status", "--short", "--", "original_repo", "paper"]) == "", "clean")
    return pd.DataFrame(rows)


def build_manifest(
    root: Path,
    config: Mapping[str, Any],
    decision: str,
    f1bz_manifest: Mapping[str, Any],
    f1bza_manifest: Mapping[str, Any],
    f1bzc_manifest: Mapping[str, Any],
    checkpoint_vocab: pd.DataFrame,
    targets: pd.DataFrame,
    readiness: pd.DataFrame,
    paper_evidence: pd.DataFrame,
    artifact_forensics: pd.DataFrame,
    top5x4: pd.DataFrame,
    table5: pd.DataFrame,
    pool_policy: pd.DataFrame,
    corrective_scope: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    budget = dict(config["request_budget"])
    artifact_hash_inputs = {
        "checkpoint_vocab_audit_hash": checkpoint_vocab,
        "target_representability_hash": targets,
        "representable_catalog_readiness_hash": readiness,
        "paper_pool_evidence_hash": paper_evidence,
        "official_artifact_pool_forensics_hash": artifact_forensics,
        "top5x4_hypothesis_audit_hash": top5x4,
        "table5_consistency_audit_hash": table5,
        "pool_policy_resolution_hash": pool_policy,
        "corrective_scope_hash": corrective_scope,
        "validation_hash": validation,
    }
    return {
        "phase_id": PHASE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_token": decision,
        "precondition_f1bz": f1bz_manifest.get("decision_token", ""),
        "precondition_f1bza": f1bza_manifest.get("decision_token", ""),
        "precondition_f1bzc": f1bzc_manifest.get("decision_token", ""),
        "active_datasets": list(ACTIVE_DATASETS),
        "models": list(TRAINABLE_MODELS),
        "seeds": list(FINAL_SEEDS),
        "model_representable_ready_cells": int(checkpoint_vocab["model_representable_status"].eq(MODEL_READY).sum()),
        "prior_f1bzc_raw_catalog_blockers_reassessed": int(checkpoint_vocab["f1bzc_raw_catalog_blocker_reassessed"].eq("RESOLVED_BY_MODEL_REPRESENTABLE_POLICY").sum()),
        "target_representability_rows": int(len(targets)),
        "oov_positive_target_rows": int(targets["positive_representable"].ne("true").sum()),
        "checkpoint_retraining_required_cells": int(corrective_scope["checkpoint_retraining_required"].ne("false").sum()),
        "exact_paper_pool_protocol_recovered": False,
        "adapted_pool_policy_frozen": True,
        "adapted_pool_policy_id": ADAPTED_POLICY,
        "adapted_pool_algorithm_id": ADAPTED_ALGORITHM,
        "requested_source_depth": int(config["adapted_pool_policy"]["requested_source_depth"]),
        "actual_max_depth_used": PROTOCOL_ONLY,
        "users_unable_to_fill_20": PROTOCOL_ONLY,
        "future_reranking_requests_unlocked": int(config["request_budget"]["reranking_requests_unlocked_in_f1bzp"]),
        "non_authoritative_lastfm_source_recall": "409/1000",
        "foodcom_true_zero_history_users": 727,
        "foodcom_pipeline_zero_history_users": 0,
        "training_runs_executed_in_f1bzp": 0,
        "traditional_scoring_executed_in_f1bzp": 0,
        "checkpoint_parameter_updates": 0,
        "hyperparameter_changes": 0,
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_llm_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "authoritative_pools_generated": 0,
        "commit_created": 0,
        "request_budget": budget,
        "f0_request_budget_unchanged": True,
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
        **{key: frame_hash(frame) for key, frame in artifact_hash_inputs.items()},
        "config_hash": sha256_json(config),
    }


def render_report(
    decision: str,
    checkpoint_vocab: pd.DataFrame,
    targets: pd.DataFrame,
    readiness: pd.DataFrame,
    paper_evidence: pd.DataFrame,
    top5x4: pd.DataFrame,
    table5: pd.DataFrame,
    pool_policy: pd.DataFrame,
    corrective_scope: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    old_reassessed = checkpoint_vocab["f1bzc_raw_catalog_blocker_reassessed"].value_counts().to_dict()
    target_counts = targets["status"].value_counts().to_dict()
    readiness_counts = readiness["status"].value_counts().to_dict()
    corrective_counts = corrective_scope["REAL_BLOCKER"].value_counts().to_dict()
    policy = pool_policy.loc[pool_policy["policy_id"].eq(ADAPTED_POLICY)].iloc[0]
    lines = [
        "# F1BZP Reranking Protocol Resolution",
        "",
        f"Decision token: `{decision}`",
        "",
        "## Checkpoint-Representable Catalog",
        "",
        f"- Audited dataset/model cells: `{len(checkpoint_vocab)}`.",
        f"- Ready under model-representable catalog policy: `{int(checkpoint_vocab['model_representable_status'].eq(MODEL_READY).sum())}` / `20`.",
        f"- Prior F1BZC raw-catalog blocker reassessment: `{stable_json(old_reassessed)}`.",
        "- The retrievable universe is the raw metadata catalog intersected with the frozen checkpoint item vocabulary.",
        "- OOV raw metadata items are not scored and do not require invented embeddings or retraining.",
        "",
        "## Target Representability",
        "",
        f"- Rows: `{len(targets)}`.",
        f"- Status counts: `{stable_json(target_counts)}`.",
        "- If a future target is outside the representable catalog, it is a Candidate Recall miss and is never forced into the vocabulary.",
        "",
        "## Paper And Artifact Evidence",
        "",
        f"- Paper evidence rows: `{len(paper_evidence)}`.",
        f"- Top-5 x four-model classification: `{top5x4.loc[top5x4['evidence_id'].eq('final_classification'), 'classification'].iloc[0]}`.",
        f"- Table 5 consistency classification: `{table5['classification'].iloc[0]}`.",
        "- Local evidence recovers the 20-item pool size and four source models, but not source depth, order, duplicate handling, or refill mechanics.",
        "",
        "## Superseded F1BR Pool Component",
        "",
        f"- Defect: `{F1BR_DEFECT}`.",
        "- F1BZC showed the frozen F1BR concatenation policy fills attempted LastFM pools entirely from BPRMF.",
        "- Only the pool aggregation component is superseded; the source seed policy and other F1BR decisions remain frozen.",
        "",
        "## Frozen Adapted Policy",
        "",
        f"- Policy: `{policy['policy_id']}`.",
        f"- Algorithm: `{policy['algorithm_id']}`.",
        f"- Model order: `{policy['model_order']}`.",
        f"- Requested source depth: `{policy['requested_source_depth']}`.",
        f"- Actual max depth used: `{policy['actual_max_depth_used']}`.",
        f"- Users unable to fill 20: `{policy['users_unable_to_fill_20']}`.",
        "- The policy visits models by rank round, skips duplicates, preserves each source ranking, and stops at 20 unique items.",
        "- It is adapted, not paper-exact, and does not use target identity or positive forcing.",
        "",
        "## Corrective Scope",
        "",
        f"- Scope rows: `{len(corrective_scope)}`.",
        f"- Real blockers: `{stable_json(corrective_counts)}`.",
        f"- Readiness statuses: `{stable_json(readiness_counts)}`.",
        "- F1BZP generates no authoritative pools; future execution must regenerate source top20 over the representable catalog first.",
        "",
        "## Validation",
        "",
        f"- Checks: `{len(validation)}`.",
        f"- Failures: `{int(validation['status'].ne('PASS').sum())}`.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    ensure_output_dirs(root)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")
    payload = dict(outputs["manifest"])
    artifact_paths = [*OUTPUT_TABLES.values(), *SUPPORT_ARTIFACTS]
    payload["artifact_hashes"] = {
        path.as_posix(): sha256_file(root / path) for path in artifact_paths if (root / path).exists()
    }
    (root / MANIFEST_PATH).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_existing_outputs(root: Path) -> dict[str, Any]:
    return {
        **{key: read_csv(root / rel_path) for key, rel_path in OUTPUT_TABLES.items()},
        "manifest": read_json(root / MANIFEST_PATH),
        "report": (root / REPORT_PATH).read_text(encoding="utf-8"),
    }


def ensure_output_dirs(root: Path) -> None:
    for rel_path in (TABLE_DIR, MANIFEST_DIR, REPORT_PATH.parent):
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

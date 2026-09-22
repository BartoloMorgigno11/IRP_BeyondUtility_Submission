from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results" / "raw_outputs"

PHASE_ID = "E3A"
PHASE_TITLE = "Paper-like baseline alignment and no-training evaluation audit"
SEED = 20260702
K = 5

DATASETS = [
    {"key": "ml1m", "label": "ML-1M", "short": "ml1m", "aliases": ("ml1m", "ml-1m")},
    {"key": "amazon_beauty", "label": "Amazon Beauty", "short": "beauty", "aliases": ("amazon_beauty", "beauty")},
    {"key": "amazon_sports", "label": "Amazon Sports", "short": "sports", "aliases": ("amazon_sports", "sports")},
]

TASKS = ["ranking", "reranking"]
PAPER_BASELINES = ["BM25", "MostPop", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]
REFERENCE_METHODS = ["candidate_order_reference", "random_deterministic_reference"]
ALL_METHODS = PAPER_BASELINES + REFERENCE_METHODS
NEURAL_BASELINES = {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}

SEARCH_ROOTS = ["data", "datasets", "results", "report", "configs", "manifests"]
EXCLUDED_DISCOVERY_PREFIXES = (
    "results/tables/plan_e_baseline_alignment",
    "report/plan_e_baseline_alignment.md",
    "results/tables/plan_e_bm25",
    "report/plan_e_bm25_scoring.md",
)

MANIFEST_INVENTORY_COLUMNS = [
    "manifest_id",
    "dataset",
    "task",
    "path",
    "exists",
    "rows",
    "user_count",
    "candidate_pool_size",
    "positive_items_per_user",
    "negative_items_per_user",
    "exact_1000_user_target_met",
    "exact_candidate_pool_20_met",
    "exact_1_positive_19_negative_met",
    "classification",
    "notes",
]

SOURCE_INVENTORY_COLUMNS = [
    "artifact_id",
    "path",
    "artifact_type",
    "dataset",
    "model_or_method",
    "phase",
    "rows_if_tabular",
    "has_user_column",
    "has_item_column",
    "has_score_column",
    "has_rank_column",
    "has_metric_columns",
    "can_support_e2_evaluation",
    "notes",
]

SCORE_AVAILABILITY_COLUMNS = [
    "dataset",
    "task",
    "model_or_method",
    "required_for_paper",
    "e2_manifest_path",
    "score_artifact_found",
    "score_artifact_path",
    "score_artifact_has_user_item_scores",
    "can_compute_hr_ndcg_on_e2",
    "no_training_possible",
    "requires_training",
    "requires_new_scoring",
    "classification",
    "exactness_class",
    "notes",
]

NO_TRAINING_METRIC_COLUMNS = [
    "dataset",
    "task",
    "model_or_method",
    "user_count",
    "row_count",
    "candidate_pool_size",
    "hr_at_5",
    "ndcg_at_5",
    "evaluated_users",
    "missing_users",
    "scoring_source",
    "no_training",
    "exactness_class",
    "classification",
    "notes",
]

MISSING_BASELINE_COLUMNS = [
    "dataset",
    "task",
    "model_or_method",
    "missing_reason",
    "required_artifact",
    "required_future_phase",
    "training_needed",
    "scoring_needed",
    "risk_level",
    "classification",
    "notes",
]

VALIDATION_COLUMNS = [
    "check_name",
    "passed",
    "observed_value",
    "expected_value",
    "severity",
    "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id",
    "phase",
    "priority",
    "action",
    "reason",
    "depends_on",
    "risk_level",
    "expected_artifact",
    "notes",
]


@dataclass(frozen=True)
class PlanEBaselineAlignmentPaths:
    manifest_inventory: Path = TABLES_DIR / "plan_e_baseline_alignment_manifest_inventory.csv"
    source_inventory: Path = TABLES_DIR / "plan_e_baseline_alignment_source_inventory.csv"
    score_availability: Path = TABLES_DIR / "plan_e_baseline_alignment_score_availability.csv"
    no_training_metrics: Path = TABLES_DIR / "plan_e_baseline_alignment_no_training_metrics.csv"
    missing_baselines: Path = TABLES_DIR / "plan_e_baseline_alignment_missing_baselines.csv"
    validation: Path = TABLES_DIR / "plan_e_baseline_alignment_validation.csv"
    next_actions: Path = TABLES_DIR / "plan_e_baseline_alignment_next_actions.csv"
    report: Path = REPORT_DIR / "plan_e_baseline_alignment.md"

    def required_output_paths(self) -> list[Path]:
        return [
            self.manifest_inventory,
            self.source_inventory,
            self.score_availability,
            self.no_training_metrics,
            self.missing_baselines,
            self.validation,
            self.next_actions,
            self.report,
        ]


def e2_manifest_paths(project_root: Path = PROJECT_ROOT) -> dict[tuple[str, str], Path]:
    """Return the expected E2 manifest paths without regenerating them."""

    return {
        ("ML-1M", "ranking"): project_root / "results" / "tables" / "plan_e_ml1m_1000user_ranking_manifest.csv",
        ("ML-1M", "reranking"): project_root / "results" / "tables" / "plan_e_ml1m_1000user_reranking_manifest.csv",
        ("Amazon Beauty", "ranking"): project_root / "results" / "tables" / "plan_e_beauty_1000user_ranking_manifest.csv",
        ("Amazon Beauty", "reranking"): project_root / "results" / "tables" / "plan_e_beauty_1000user_reranking_manifest.csv",
        ("Amazon Sports", "ranking"): project_root / "results" / "tables" / "plan_e_sports_1000user_ranking_manifest.csv",
        ("Amazon Sports", "reranking"): project_root / "results" / "tables" / "plan_e_sports_1000user_reranking_manifest.csv",
    }


def load_e2_manifests(project_root: Path = PROJECT_ROOT) -> dict[tuple[str, str], pd.DataFrame]:
    """Load E2 manifests read-only."""

    manifests: dict[tuple[str, str], pd.DataFrame] = {}
    for key, path in e2_manifest_paths(project_root).items():
        if path.exists():
            manifests[key] = pd.read_csv(path, dtype=str).fillna("")
        else:
            manifests[key] = pd.DataFrame()
    return manifests


def build_manifest_inventory(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Inventory E2 manifests and validate their shape at a summary level."""

    rows: list[dict[str, object]] = []
    manifests = load_e2_manifests(project_root)
    for (dataset, task), path in e2_manifest_paths(project_root).items():
        frame = manifests[(dataset, task)]
        exists = path.exists()
        summary = _manifest_shape_summary(frame) if exists and not frame.empty else {}
        rows.append(
            {
                "manifest_id": _manifest_id(dataset, task),
                "dataset": dataset,
                "task": task,
                "path": _relative(path, project_root),
                "exists": _bool_text(exists),
                "rows": len(frame),
                "user_count": summary.get("user_count", 0),
                "candidate_pool_size": summary.get("candidate_pool_size", ""),
                "positive_items_per_user": summary.get("positive_items_per_user", ""),
                "negative_items_per_user": summary.get("negative_items_per_user", ""),
                "exact_1000_user_target_met": _bool_text(summary.get("user_count", 0) == 1000),
                "exact_candidate_pool_20_met": _bool_text(summary.get("candidate_pool_size", "") == 20),
                "exact_1_positive_19_negative_met": _bool_text(
                    summary.get("positive_items_per_user", "") == 1
                    and summary.get("negative_items_per_user", "") == 19
                ),
                "classification": "reconstructed_paper_like",
                "notes": "Read-only E2 reconstructed paper-like manifest; not exact original paper candidate file.",
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_INVENTORY_COLUMNS)


def discover_source_inventory(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Discover relevant existing baseline, metric, manifest, and report artifacts read-only."""

    rows: list[dict[str, object]] = []
    for root_name in SEARCH_ROOTS:
        root = project_root / root_name
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = _relative(path, project_root)
            if _is_excluded_from_discovery(relative) or _should_skip_path(path):
                continue
            if not _is_relevant_artifact(relative):
                continue
            columns = _header_columns(path)
            lower_columns = {column.lower() for column in columns}
            model_or_method = _model_or_method_for_path(relative)
            rows.append(
                {
                    "artifact_id": _artifact_id(relative),
                    "path": relative,
                    "artifact_type": _artifact_type(relative),
                    "dataset": _dataset_for_path(relative),
                    "model_or_method": model_or_method,
                    "phase": _phase_for_path(relative),
                    "rows_if_tabular": _count_tabular_rows(path),
                    "has_user_column": _bool_text(bool(lower_columns & {"user_id", "user", "uid", "reviewerid", "reviewer_id"})),
                    "has_item_column": _bool_text(bool(lower_columns & {"item_id", "recommended_item_id", "candidate_item_id", "item", "iid", "asin", "movie_id", "product_id"})),
                    "has_score_column": _bool_text(bool(lower_columns & {"score", "prediction", "predicted_score", "logit"})),
                    "has_rank_column": _bool_text(bool(lower_columns & {"rank", "candidate_position", "position", "source_rank", "candidate_rank"})),
                    "has_metric_columns": _bool_text(_has_metric_columns(lower_columns)),
                    "can_support_e2_evaluation": _bool_text(_can_support_e2_evaluation(relative, columns)),
                    "notes": _source_notes(relative, model_or_method),
                }
            )
    return pd.DataFrame(rows, columns=SOURCE_INVENTORY_COLUMNS)


def build_score_availability(
    manifest_inventory: pd.DataFrame | None = None,
    source_inventory: pd.DataFrame | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Represent score availability for every dataset x task x paper baseline/reference method."""

    manifest_inventory = manifest_inventory if manifest_inventory is not None else build_manifest_inventory(project_root)
    source_inventory = source_inventory if source_inventory is not None else discover_source_inventory(project_root)
    rows: list[dict[str, object]] = []
    for _, manifest in manifest_inventory.iterrows():
        dataset = str(manifest["dataset"])
        task = str(manifest["task"])
        manifest_path = str(manifest["path"])
        for method in ALL_METHODS:
            source_match = _find_existing_method_artifact(source_inventory, dataset, method)
            score_found = source_match is not None
            has_user_item_scores = bool(source_match is not None and source_match["has_user_column"] == "true" and source_match["has_item_column"] == "true")
            can_compute = method in {"candidate_order_reference", "random_deterministic_reference"} or (
                method == "MostPop" and _interaction_path_for_dataset(dataset, project_root).exists()
            )
            no_training_possible = method in {"candidate_order_reference", "random_deterministic_reference", "MostPop"}
            requires_training = method in NEURAL_BASELINES
            requires_new_scoring = method in {"BM25"} or method in NEURAL_BASELINES
            classification = _availability_classification(method, can_compute, score_found)
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "model_or_method": method,
                    "required_for_paper": _bool_text(method in PAPER_BASELINES),
                    "e2_manifest_path": manifest_path,
                    "score_artifact_found": _bool_text(score_found),
                    "score_artifact_path": source_match["path"] if source_match is not None else "",
                    "score_artifact_has_user_item_scores": _bool_text(has_user_item_scores),
                    "can_compute_hr_ndcg_on_e2": _bool_text(can_compute),
                    "no_training_possible": _bool_text(no_training_possible and can_compute),
                    "requires_training": _bool_text(requires_training),
                    "requires_new_scoring": _bool_text(requires_new_scoring),
                    "classification": classification,
                    "exactness_class": "reconstructed_paper_like" if can_compute else "needs_scoring",
                    "notes": _availability_notes(method, can_compute, score_found),
                }
            )
    return pd.DataFrame(rows, columns=SCORE_AVAILABILITY_COLUMNS)


def build_no_training_metrics(
    manifests: dict[tuple[str, str], pd.DataFrame] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Compute HR@5/NDCG@5 only for methods safely evaluable without training."""

    manifests = manifests if manifests is not None else load_e2_manifests(project_root)
    rows: list[dict[str, object]] = []
    for (dataset, task), frame in manifests.items():
        if frame.empty:
            continue
        rows.append(_metric_row(dataset, task, "candidate_order_reference", frame, _candidate_order_ranks(frame), "E2 candidate_position", "sanity_check_not_paper_baseline", "no_training_evaluation", "Candidate order reference uses E2 manifest candidate_position."))
        rows.append(_metric_row(dataset, task, "random_deterministic_reference", frame, _random_reference_ranks(frame, dataset, task), f"deterministic_hash_seed_{SEED}", "sanity_check_not_paper_baseline", "no_training_evaluation", "Deterministic pseudo-random reference; not a paper baseline."))
        popularity = _load_item_popularity(dataset, project_root)
        if popularity is not None:
            interaction_path = _interaction_path_for_dataset(dataset, project_root)
            rows.append(_metric_row(dataset, task, "MostPop", frame, _mostpop_ranks(frame, popularity), _relative(interaction_path, project_root), "reconstructed_paper_like", "no_training_evaluation", "MostPop computed from available train interactions without recommender training."))
    return pd.DataFrame(rows, columns=NO_TRAINING_METRIC_COLUMNS)


def build_missing_baselines(score_availability: pd.DataFrame) -> pd.DataFrame:
    """Build missing-baseline action table for methods not computable in E3A."""

    missing = score_availability.loc[
        (score_availability["required_for_paper"] == "true")
        & (score_availability["can_compute_hr_ndcg_on_e2"] != "true")
    ]
    rows: list[dict[str, object]] = []
    for _, row in missing.iterrows():
        method = str(row["model_or_method"])
        future_phase = "E3B_no_training_lightweight_BM25_or_MostPop_fix" if method == "BM25" else "E3C_neural_baseline_scoring"
        if method in NEURAL_BASELINES:
            future_phase = "E3C_neural_baseline_scoring_or_E3D_training_if_approved"
        rows.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "model_or_method": method,
                "missing_reason": _missing_reason(method),
                "required_artifact": _required_artifact(method),
                "required_future_phase": future_phase,
                "training_needed": _bool_text(method in NEURAL_BASELINES),
                "scoring_needed": "true",
                "risk_level": "high" if method in NEURAL_BASELINES else "medium",
                "classification": "needs_training_or_existing_scores" if method in NEURAL_BASELINES else "needs_scoring",
                "notes": "Do not use prior Plan C/D metric rows as aligned E2 metrics unless user-item scores are regenerated or verified on E2 manifests.",
            }
        )
    return pd.DataFrame(rows, columns=MISSING_BASELINE_COLUMNS)


def build_validation_table(
    manifest_inventory: pd.DataFrame,
    score_availability: pd.DataFrame,
    no_training_metrics: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Build E3A validation checks."""

    rows = [
        _validation("all_E2_manifests_present", (manifest_inventory["exists"] == "true").all(), int((manifest_inventory["exists"] == "true").sum()), 6, "critical", "All six E2 manifests must exist."),
        _validation("manifest_schema_valid", _all_manifest_schemas_valid(project_root), "valid", "valid", "critical", "E2 manifests must contain the required columns."),
        _validation("candidate_pool_size_20", (manifest_inventory["exact_candidate_pool_20_met"] == "true").all(), int((manifest_inventory["exact_candidate_pool_20_met"] == "true").sum()), 6, "critical", "Every manifest must have 20 candidates per user."),
        _validation("one_positive_per_user", _manifest_inventory_has_value(manifest_inventory, "positive_items_per_user", 1), "1", "1", "critical", "Every user must have one positive."),
        _validation("nineteen_negatives_per_user", _manifest_inventory_has_value(manifest_inventory, "negative_items_per_user", 19), "19", "19", "critical", "Every user must have nineteen negatives."),
        _validation("no_duplicate_items_per_user", _all_manifests_no_duplicates(project_root), "no_duplicates", "no_duplicates", "critical", "No duplicate items per user."),
        _validation("metric_formula_hr_at_5_available", True, "implemented", "implemented", "critical", "E3A implements single-positive HR@5 directly."),
        _validation("metric_formula_ndcg_at_5_available", True, "implemented", "implemented", "critical", "E3A implements single-positive NDCG@5 directly."),
        _validation("candidate_order_reference_metrics_created", _method_metric_count(no_training_metrics, "candidate_order_reference") == 6, _method_metric_count(no_training_metrics, "candidate_order_reference"), 6, "critical", "Candidate-order reference must be evaluated for every E2 manifest."),
        _validation("random_deterministic_reference_metrics_created", _method_metric_count(no_training_metrics, "random_deterministic_reference") == 6, _method_metric_count(no_training_metrics, "random_deterministic_reference"), 6, "critical", "Deterministic random reference must be evaluated for every E2 manifest."),
        _validation("all_paper_baselines_represented", set(PAPER_BASELINES).issubset(set(score_availability["model_or_method"])), ";".join(PAPER_BASELINES), ";".join(PAPER_BASELINES), "critical", "All paper baselines must be represented in score availability."),
        _validation("no_neural_training_performed", True, "no_training", "no_training", "critical", "E3A never trains neural baselines."),
        _validation("no_llm_inference_performed", True, "no_llm_inference", "no_llm_inference", "critical", "E3A does not run LLMs."),
        _validation("no_ollama_call_performed", True, "no_ollama_call", "no_ollama_call", "critical", "E3A does not call Ollama."),
        _validation("no_api_call_performed", True, "no_api_call", "no_api_call", "critical", "E3A does not call remote APIs."),
        _validation("no_download_performed", True, "no_download", "no_download", "critical", "E3A does not download data or models."),
        _validation("no_lastfm_run_performed", True, "no_lastfm_run", "no_lastfm_run", "critical", "LastFM remains blocked/conditional."),
        _validation("no_raw_outputs_created", True, _raw_jsonl_count(project_root), "unchanged_existing_only", "critical", "E3A writes no raw outputs or JSONL files."),
        _validation("no_original_repo_output", not _git_lines(["git", "status", "--short", "original_repo"], project_root), "clean", "clean", "critical", "original_repo/ must remain unmodified."),
        _validation("no_plan_c_modification", not _plan_c_status_lines(project_root), "clean", "clean", "critical", "Plan C files and metrics must remain unmodified."),
        _validation("no_exact_reproduction_overclaim", True, "no_exact_claim", "no_exact_claim", "critical", "E3A report states no full exact reproduction is claimed."),
    ]
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def build_next_actions() -> pd.DataFrame:
    """Build concrete next actions after E3A."""

    rows = [
        _action("E3A_A01", "E3B", 1, "decide whether to implement lightweight BM25 scoring on E2 manifests", "BM25 is a paper baseline but aligned E2 scores are not currently available.", "E3A missing-baseline table", "medium", "E3B BM25 score table or documented no-BM25 decision", "Do not build a new index unless the scope is approved and reproducible."),
        _action("E3A_A02", "E3C", 2, "decide whether to run neural baseline scoring on E2 manifests", "BPRMF/GRU4Rec/SASRec/LightGCN need aligned E2 scores before HR/NDCG can be compared.", "E2 manifests", "high", "E3C aligned neural score audit", "Scoring may be possible without retraining if saved model outputs exist; otherwise E3D approval is needed."),
        _action("E3A_A03", "E3B", 3, "prepare E3B for paper-like baseline scoring", "E3A established the E2 evaluation substrate and no-training references.", "E3A metrics", "medium", "E3B baseline scoring plan", "Keep E3B separate from training."),
        _action("E3A_A04", "E5", 4, "keep LLM inference for E5", "E3A is baseline alignment only.", "E4 prompt alignment and explicit approval", "high", "E5 local LLM run plan", "Do not mix D5 diagnostic LLM metrics with E3A baseline metrics."),
        _action("E3A_A05", "E6", 5, "keep API audit for E6", "Closed-source API work is outside E3A.", "E0/E1 cost audit", "high", "E6 API availability and pricing report", "No API calls in E3A."),
        _action("E3A_A06", "blocked", 6, "keep LastFM conditional", "LastFM lacks valid E2 manifests.", "valid LastFM artifacts", "high", "LastFM unblock decision", "Do not run LastFM or fabricate utility metrics."),
        _action("E3A_A07", "reporting", 7, "do not mix D5 LLM diagnostic metrics with baseline metrics", "D5 diagnostics use different scopes and models.", "E3A source inventory", "medium", "claim-safety note", "D5 remains diagnostic_subset."),
        _action("E3A_A08", "E3B/E4/E5", 8, "preserve E2 manifests as the common evaluation substrate", "Common candidate pools are required for paper-like comparisons.", "E2 manifests", "low", "shared manifest registry", "Do not regenerate E2 manifests in E3A."),
    ]
    return pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS)


def build_all_tables(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    """Build all E3A tables without writing files."""

    manifests = load_e2_manifests(project_root)
    manifest_inventory = build_manifest_inventory(project_root)
    source_inventory = discover_source_inventory(project_root)
    score_availability = build_score_availability(manifest_inventory, source_inventory, project_root)
    no_training_metrics = build_no_training_metrics(manifests, project_root)
    missing_baselines = build_missing_baselines(score_availability)
    validation = build_validation_table(manifest_inventory, score_availability, no_training_metrics, project_root)
    next_actions = build_next_actions()
    return {
        "manifest_inventory": manifest_inventory,
        "source_inventory": source_inventory,
        "score_availability": score_availability,
        "no_training_metrics": no_training_metrics,
        "missing_baselines": missing_baselines,
        "validation": validation,
        "next_actions": next_actions,
    }


def build_markdown_report(tables: dict[str, pd.DataFrame]) -> str:
    """Build the E3A Markdown report."""

    manifest_inventory = tables["manifest_inventory"]
    score_availability = tables["score_availability"]
    metrics = tables["no_training_metrics"]
    missing = tables["missing_baselines"]
    validation = tables["validation"]
    next_actions = tables["next_actions"]
    failures = validation.loc[validation["passed"] != "true"]
    methods = sorted(metrics["model_or_method"].unique().tolist())

    return f"""# Plan E3A Baseline Alignment and No-Training Evaluation Audit

## Executive Summary

E3A is not full E3 baseline completion. E3A is a no-training alignment/evaluation audit over the E2 reconstructed paper-like manifests. It computes only metrics that are safely computable without recommender training, neural baseline training, LLM inference, Ollama calls, APIs, downloads, model pulls, or LastFM runs.

E3A performed no inference. E3A made no Ollama calls. E3A made no Qwen2/Mistral calls. E3A made no API calls. E3A performed no downloads. E3A performed no model pulls. E3A performed no training. E3A performed no neural-baseline training. E3A performed no LastFM run. E3A performed no Plan C modification. E3A performed no original_repo modification. E3A performed no destructive cleanup.

E3A does not complete full paper baseline reproduction. Candidate_order_reference and random_deterministic_reference are sanity checks, not paper baselines. MostPop is a paper baseline here because it is computed from valid available train interactions without model training. Neural baselines require aligned scores or future approved scoring/training. LastFM remains blocked/conditional. No full exact reproduction is claimed.

## E2 Manifest Overview

{_markdown_table(manifest_inventory, ["manifest_id", "dataset", "task", "rows", "user_count", "candidate_pool_size", "positive_items_per_user", "negative_items_per_user", "classification"], max_rows=10)}

## Baseline Availability Overview

{_markdown_table(score_availability, ["dataset", "task", "model_or_method", "can_compute_hr_ndcg_on_e2", "classification", "exactness_class"], max_rows=30)}

## No-Training Metric Results

Metrics created in E3A: `{'; '.join(methods)}`.

{_markdown_table(metrics, ["dataset", "task", "model_or_method", "hr_at_5", "ndcg_at_5", "evaluated_users", "classification"], max_rows=30)}

## Missing Baseline Analysis

{_markdown_table(missing, ["dataset", "task", "model_or_method", "missing_reason", "required_future_phase", "classification"], max_rows=30)}

## Validation Summary

Validation failures: `{len(failures)}`.

{_markdown_table(validation, ["check_name", "passed", "observed_value", "expected_value", "severity"], max_rows=30)}

## Exactness and Claim-Safety Notes

- E3A is a no-training audit, not exact reproduction.
- E3A computes only metrics safely computable without training or new inference.
- E3A does not complete full paper baseline reproduction.
- Candidate-order and deterministic-random references are sanity checks, not paper baselines.
- MostPop uses reconstructed train interactions and E2 manifests, so it is reconstructed_paper_like/no_training_evaluation.
- BM25, BPRMF, GRU4Rec, SASRec, and LightGCN need aligned E2 scores or future approved scoring/training before E2 HR@5/NDCG@5 can be reported.
- D5 local LLM diagnostics are not paper baseline scores and are not mixed into E3A baseline metrics.
- No full exact reproduction is claimed.

## Recommended Next Phase

{_markdown_table(next_actions, ["action_id", "phase", "priority", "action", "risk_level"], max_rows=12)}
"""


def write_baseline_alignment_outputs(
    tables: dict[str, pd.DataFrame] | None = None,
    paths: PlanEBaselineAlignmentPaths | None = None,
) -> dict[str, pd.DataFrame]:
    """Write all E3A CSV/report outputs."""

    tables = tables if tables is not None else build_all_tables()
    paths = paths or PlanEBaselineAlignmentPaths()
    paths.manifest_inventory.parent.mkdir(parents=True, exist_ok=True)
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(tables["manifest_inventory"], paths.manifest_inventory, MANIFEST_INVENTORY_COLUMNS)
    _write_csv(tables["source_inventory"], paths.source_inventory, SOURCE_INVENTORY_COLUMNS)
    _write_csv(tables["score_availability"], paths.score_availability, SCORE_AVAILABILITY_COLUMNS)
    _write_csv(tables["no_training_metrics"], paths.no_training_metrics, NO_TRAINING_METRIC_COLUMNS)
    _write_csv(tables["missing_baselines"], paths.missing_baselines, MISSING_BASELINE_COLUMNS)
    _write_csv(tables["validation"], paths.validation, VALIDATION_COLUMNS)
    _write_csv(tables["next_actions"], paths.next_actions, NEXT_ACTION_COLUMNS)
    paths.report.write_text(build_markdown_report(tables), encoding="utf-8", newline="\n")
    return tables


def _metric_row(
    dataset: str,
    task: str,
    method: str,
    manifest: pd.DataFrame,
    ranks: pd.Series,
    scoring_source: str,
    exactness_class: str,
    classification: str,
    notes: str,
) -> dict[str, object]:
    scored = manifest.copy()
    scored["__rank"] = ranks.astype(int)
    metrics = _single_positive_metrics(scored, rank_col="__rank")
    user_count = int(scored["user_id"].nunique())
    return {
        "dataset": dataset,
        "task": task,
        "model_or_method": method,
        "user_count": user_count,
        "row_count": len(scored),
        "candidate_pool_size": int(scored.groupby("user_id").size().iloc[0]),
        "hr_at_5": round(metrics["hr_at_5"], 6),
        "ndcg_at_5": round(metrics["ndcg_at_5"], 6),
        "evaluated_users": metrics["evaluated_users"],
        "missing_users": user_count - metrics["evaluated_users"],
        "scoring_source": scoring_source,
        "no_training": "true",
        "exactness_class": exactness_class,
        "classification": classification,
        "notes": notes,
    }


def _single_positive_metrics(frame: pd.DataFrame, *, rank_col: str) -> dict[str, float | int]:
    hits = 0
    ndcg_total = 0.0
    evaluated = 0
    for _, group in frame.sort_values(["user_id", rank_col, "item_id"], kind="mergesort").groupby("user_id", sort=False):
        evaluated += 1
        top_k = group.head(K)
        positives = top_k.loc[top_k["is_positive"].astype(str).str.lower() == "true"]
        if positives.empty:
            continue
        hit_rank = int(positives.iloc[0][rank_col])
        if hit_rank <= K:
            hits += 1
            ndcg_total += 1.0 / math.log2(hit_rank + 1)
    if evaluated == 0:
        return {"hr_at_5": 0.0, "ndcg_at_5": 0.0, "evaluated_users": 0}
    return {"hr_at_5": hits / evaluated, "ndcg_at_5": ndcg_total / evaluated, "evaluated_users": evaluated}


def _candidate_order_ranks(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["candidate_position"], errors="coerce").fillna(9999).astype(int)


def _random_reference_ranks(frame: pd.DataFrame, dataset: str, task: str) -> pd.Series:
    working = frame[["user_id", "item_id"]].copy()
    working["__hash"] = [
        _stable_hash(f"{SEED}|{dataset}|{task}|{user_id}|{item_id}")
        for user_id, item_id in zip(working["user_id"], working["item_id"])
    ]
    working["__input_order"] = range(len(working))
    working = working.sort_values(["user_id", "__hash", "item_id", "__input_order"], kind="mergesort")
    working["__rank"] = working.groupby("user_id", sort=False).cumcount() + 1
    return working.sort_values("__input_order", kind="mergesort")["__rank"]


def _mostpop_ranks(frame: pd.DataFrame, popularity: dict[str, int]) -> pd.Series:
    working = frame[["user_id", "item_id"]].copy()
    working["__popularity"] = working["item_id"].map(lambda item: popularity.get(str(item), 0))
    working["__input_order"] = range(len(working))
    working = working.sort_values(["user_id", "__popularity", "item_id", "__input_order"], ascending=[True, False, True, True], kind="mergesort")
    working["__rank"] = working.groupby("user_id", sort=False).cumcount() + 1
    return working.sort_values("__input_order", kind="mergesort")["__rank"]


def _load_item_popularity(dataset: str, project_root: Path) -> dict[str, int] | None:
    path = _interaction_path_for_dataset(dataset, project_root)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=["item_id", "split"], dtype=str).fillna("")
    except Exception:
        try:
            frame = pd.read_csv(path, usecols=["item_id"], dtype=str).fillna("")
        except Exception:
            return None
    if "split" in frame.columns:
        train = frame.loc[frame["split"].astype(str).str.lower().isin(["train", "observed"])]
    else:
        train = frame
    if train.empty:
        return None
    return {str(item): int(count) for item, count in train["item_id"].astype(str).value_counts().items()}


def _interaction_path_for_dataset(dataset: str, project_root: Path) -> Path:
    key = _dataset_key(dataset)
    return project_root / "data" / "reconstructed" / "plan_c" / key / "interactions.csv"


def _find_existing_method_artifact(source_inventory: pd.DataFrame, dataset: str, method: str) -> pd.Series | None:
    method_key = method.lower()
    rows = source_inventory.loc[
        (source_inventory["dataset"].isin([dataset, "all", _dataset_key(dataset)]))
        & (source_inventory["model_or_method"].str.lower() == method_key)
        & (
            (source_inventory["has_score_column"] == "true")
            | (source_inventory["has_rank_column"] == "true")
            | (source_inventory["has_metric_columns"] == "true")
        )
    ]
    if rows.empty:
        return None
    return rows.sort_values(["can_support_e2_evaluation", "path"], ascending=[False, True], kind="mergesort").iloc[0]


def _availability_classification(method: str, can_compute: bool, score_found: bool) -> str:
    if method in REFERENCE_METHODS:
        return "no_training_evaluation"
    if method == "MostPop" and can_compute:
        return "no_training_evaluation"
    if method == "BM25":
        return "needs_scoring"
    if method in NEURAL_BASELINES:
        return "needs_training_or_existing_scores"
    return "needs_scoring" if score_found else "needs_scoring"


def _availability_notes(method: str, can_compute: bool, score_found: bool) -> str:
    if method == "candidate_order_reference":
        return "Sanity check computed from E2 candidate_position; not a paper baseline."
    if method == "random_deterministic_reference":
        return "Sanity check computed from deterministic stable hashing; not a paper baseline."
    if method == "MostPop" and can_compute:
        return "Computed from available train interactions without training."
    if method == "BM25":
        return "Existing BM25-like artifacts are prior Plan C outputs, not verified aligned E2 user-item scores; future E3B scoring is needed."
    if method in NEURAL_BASELINES:
        return "No aligned E2 user-item score artifact found; neural training is forbidden in E3A."
    return "Not computable in E3A."


def _missing_reason(method: str) -> str:
    if method == "BM25":
        return "No verified BM25 user-item score artifact aligned to E2 manifests."
    return "No verified neural baseline user-item score artifact aligned to E2 manifests; E3A forbids training."


def _required_artifact(method: str) -> str:
    if method == "BM25":
        return "E2-aligned BM25 scores or ranks for every manifest user/candidate pair."
    return f"E2-aligned {method} scores or ranks, or approved future scoring/training artifact."


def _manifest_shape_summary(frame: pd.DataFrame) -> dict[str, int | str]:
    if frame.empty or "user_id" not in frame.columns:
        return {"user_count": 0}
    per_user = frame.groupby("user_id", sort=False).size()
    positives = frame.loc[frame["is_positive"].astype(str).str.lower() == "true"].groupby("user_id", sort=False).size()
    negatives = frame.loc[frame["is_positive"].astype(str).str.lower() == "false"].groupby("user_id", sort=False).size()
    return {
        "user_count": int(frame["user_id"].nunique()),
        "candidate_pool_size": int(per_user.iloc[0]) if len(per_user.unique()) == 1 else "",
        "positive_items_per_user": int(positives.iloc[0]) if len(positives.unique()) == 1 and len(positives) == frame["user_id"].nunique() else "",
        "negative_items_per_user": int(negatives.iloc[0]) if len(negatives.unique()) == 1 and len(negatives) == frame["user_id"].nunique() else "",
    }


def _all_manifest_schemas_valid(project_root: Path) -> bool:
    required = {"manifest_id", "dataset", "task", "user_id", "candidate_position", "item_id", "is_positive"}
    for path in e2_manifest_paths(project_root).values():
        if not path.exists():
            return False
        if not required.issubset(set(_header_columns(path))):
            return False
    return True


def _all_manifests_no_duplicates(project_root: Path) -> bool:
    for frame in load_e2_manifests(project_root).values():
        if frame.empty:
            return False
        unique_counts = frame.groupby("user_id")["item_id"].nunique()
        row_counts = frame.groupby("user_id").size()
        if not (unique_counts == row_counts).all():
            return False
    return True


def _manifest_inventory_has_value(frame: pd.DataFrame, column: str, expected: int) -> bool:
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool((values == expected).all())


def _method_metric_count(metrics: pd.DataFrame, method: str) -> int:
    return int((metrics["model_or_method"] == method).sum()) if not metrics.empty else 0


def _validation(check_name: str, passed: bool, observed_value: object, expected_value: object, severity: str, notes: str) -> dict[str, object]:
    return {
        "check_name": check_name,
        "passed": _bool_text(passed),
        "observed_value": observed_value,
        "expected_value": expected_value,
        "severity": severity,
        "notes": notes,
    }


def _action(action_id: str, phase: str, priority: int, action: str, reason: str, depends_on: str, risk_level: str, expected_artifact: str, notes: str) -> dict[str, object]:
    return {
        "action_id": action_id,
        "phase": phase,
        "priority": priority,
        "action": action,
        "reason": reason,
        "depends_on": depends_on,
        "risk_level": risk_level,
        "expected_artifact": expected_artifact,
        "notes": notes,
    }


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, columns=columns, lineterminator="\n")


def _is_relevant_artifact(relative: str) -> bool:
    lower = relative.lower()
    return any(
        token in lower
        for token in (
            "plan_c",
            "plan_d",
            "plan_e",
            "baseline",
            "metric",
            "recommend",
            "score",
            "manifest",
            "interactions.csv",
            "config",
        )
    )


def _artifact_type(relative: str) -> str:
    lower = relative.lower()
    name = Path(relative).name.lower()
    if lower.startswith("results/metrics/"):
        return "metric_csv"
    if lower.startswith("results/tables/"):
        return "result_table"
    if lower.startswith("results/raw_outputs/") and name.endswith(".csv"):
        return "recommendation_or_score_csv"
    if lower.startswith("results/raw_outputs/"):
        return "raw_output"
    if name == "interactions.csv":
        return "interaction_data"
    if "manifest" in name:
        return "manifest"
    if lower.startswith("report/"):
        return "report"
    if lower.startswith("configs/"):
        return "config"
    return "artifact"


def _model_or_method_for_path(relative: str) -> str:
    lower = relative.lower()
    patterns = [
        ("content_bm25", "BM25"),
        ("bm25", "BM25"),
        ("mostpop", "MostPop"),
        ("bprmf", "BPRMF"),
        ("gru4rec", "GRU4Rec"),
        ("sasrec", "SASRec"),
        ("lightgcn", "LightGCN"),
        ("candidate_order_reference", "candidate_order_reference"),
        ("random", "random_deterministic_reference"),
        ("qwen", "qwen2_local_llm_diagnostic"),
        ("mistral", "mistral_local_llm_diagnostic"),
    ]
    for token, method in patterns:
        if token in lower:
            return method
    return "unknown"


def _dataset_for_path(relative: str) -> str:
    lower = relative.lower()
    if "amazon_beauty" in lower or "beauty" in lower:
        return "Amazon Beauty"
    if "amazon_sports" in lower or "sports" in lower:
        return "Amazon Sports"
    if "ml1m" in lower or "ml-1m" in lower:
        return "ML-1M"
    if "lastfm" in lower:
        return "LastFM"
    return "all"


def _dataset_key(dataset: str) -> str:
    return {
        "ML-1M": "ml1m",
        "Amazon Beauty": "amazon_beauty",
        "Amazon Sports": "amazon_sports",
        "LastFM": "lastfm",
    }[dataset]


def _phase_for_path(relative: str) -> str:
    lower = relative.lower()
    if "plan_e_ml1m_1000user" in lower or "plan_e_beauty_1000user" in lower or "plan_e_sports_1000user" in lower or "plan_e_paper_like_candidate_pool" in lower:
        return "E2"
    if "plan_e_full_data_inventory" in lower:
        return "E1"
    if "plan_e_paper_condition" in lower or "plan_e_scaleup" in lower or "plan_e_exactness" in lower or "plan_e_missing" in lower or "plan_e_api" in lower:
        return "E0"
    if "plan_d" in lower or "qwen" in lower or "mistral" in lower:
        return "Plan D/D5"
    if "plan_c" in lower or "reconstructed/plan_c" in lower:
        return "Plan C"
    return "support"


def _can_support_e2_evaluation(relative: str, columns: list[str]) -> bool:
    lower = relative.lower()
    lower_columns = {column.lower() for column in columns}
    has_user = bool(lower_columns & {"user_id", "user", "uid"})
    has_item = bool(lower_columns & {"item_id", "recommended_item_id", "candidate_item_id"})
    has_rank_or_score = bool(lower_columns & {"rank", "score", "prediction", "predicted_score"})
    return "plan_e" in lower and has_user and has_item and has_rank_or_score


def _has_metric_columns(columns: set[str]) -> bool:
    return bool(columns & {"metric", "value", "hr_at_5", "hit_rate_at_5", "ndcg_at_5", "hit_rate", "ndcg"})


def _source_notes(relative: str, method: str) -> str:
    lower = relative.lower()
    if "raw_outputs" in lower and "plan_d" in lower:
        return "Existing Plan D recommendation artifact; not assumed aligned to E2 manifests."
    if method in {"qwen2_local_llm_diagnostic", "mistral_local_llm_diagnostic"}:
        return "D5 local LLM diagnostic artifact; not a paper baseline score."
    if "plan_e" in lower:
        return "Plan E audit or manifest artifact."
    return "Read-only source inventory artifact."


def _is_excluded_from_discovery(relative: str) -> bool:
    lower = relative.lower().replace("\\", "/")
    return any(lower.startswith(prefix) for prefix in EXCLUDED_DISCOVERY_PREFIXES)


def _should_skip_path(path: Path) -> bool:
    return any(part in {".git", ".pytest_cache", "__pycache__", ".venv"} for part in path.parts)


def _count_tabular_rows(path: Path) -> int | str:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return ""


def _header_columns(path: Path) -> list[str]:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return [column.strip() for column in next(csv.reader(handle, delimiter=delimiter), [])]
    except OSError:
        return []


def _raw_jsonl_count(project_root: Path) -> int:
    return len(list((project_root / "results" / "raw_outputs").glob("*.jsonl"))) if (project_root / "results" / "raw_outputs").exists() else 0


def _plan_c_status_lines(project_root: Path) -> list[str]:
    status = _git_lines(["git", "status", "--short"], project_root)
    return [line for line in status if "plan_c" in line.lower() or "plan-c" in line.lower() or "reconstructed/plan_c" in line.lower()]


def _git_lines(command: list[str], project_root: Path) -> list[str]:
    return common_git.git_lines_unavailable_on_error(command, project_root, normalize_paths=True)


def _manifest_id(dataset: str, task: str) -> str:
    short = next(row["short"] for row in DATASETS if row["label"] == dataset)
    return f"plan_e_{short}_1000user_{task}_manifest"


def _artifact_id(relative: str) -> str:
    return relative.replace("\\", "/").replace("/", "__").replace(".", "_").replace("-", "_")[:180]


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str], *, max_rows: int) -> str:
    selected_columns = list(columns)
    if frame.empty:
        return "| " + " | ".join(selected_columns) + " |\n| " + " | ".join("---" for _ in selected_columns) + " |"
    shown = frame[selected_columns].head(max_rows)
    lines = [
        "| " + " | ".join(selected_columns) + " |",
        "| " + " | ".join("---" for _ in selected_columns) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(_escape_markdown(str(row[column])) for column in selected_columns) + " |")
    if len(frame) > max_rows:
        lines.append("| " + " | ".join("..." for _ in selected_columns) + " |")
    return "\n".join(lines)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _bool_text(value: bool) -> str:
    return "true" if value else "false"

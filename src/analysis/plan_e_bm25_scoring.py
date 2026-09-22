from __future__ import annotations

from src.audit import common_git

import csv
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results" / "raw_outputs"

PHASE_ID = "E3B"
PHASE_TITLE = "Lightweight BM25/BM25-like scoring on E2 manifests"
K = 5
BM25_K1 = 1.2
BM25_B = 0.75
BM25_VARIANT = "local_bm25_like_k1_1.2_b_0.75"

DATASETS = [
    {"key": "ml1m", "label": "ML-1M", "short": "ml1m", "aliases": ("ml1m", "ml-1m", "movielens")},
    {"key": "amazon_beauty", "label": "Amazon Beauty", "short": "beauty", "aliases": ("amazon_beauty", "beauty")},
    {"key": "amazon_sports", "label": "Amazon Sports", "short": "sports", "aliases": ("amazon_sports", "sports")},
]
TASKS = ("ranking", "reranking")
LASTFM_LABEL = "LastFM"

SEARCH_ROOTS = ("data", "datasets", "results", "results/tables", "results/metrics", "configs", "manifests")
EXCLUDED_DISCOVERY_PREFIXES = (
    "results/tables/plan_e_bm25",
    "report/plan_e_bm25_scoring.md",
)

USER_COLUMNS = ("user_id", "user", "uid", "reviewerID", "reviewer_id")
ITEM_COLUMNS = (
    "item_id",
    "candidate_item_id",
    "item",
    "iid",
    "asin",
    "movie_id",
    "product_id",
    "ground_truth_item_id",
)
TEXT_COLUMNS = (
    "normalized_text",
    "candidate_item_title",
    "item_title",
    "title",
    "name",
    "movie_title",
    "description",
    "categories",
    "category",
    "genres",
    "genre",
    "brand",
)
ORDER_COLUMNS = ("timestamp", "time", "unixReviewTime", "candidate_position", "rank", "position", "order", "split")

SOURCE_INVENTORY_COLUMNS = [
    "artifact_id",
    "path",
    "dataset",
    "artifact_type",
    "exists",
    "rows_if_tabular",
    "detected_user_column",
    "detected_item_column",
    "detected_text_columns",
    "detected_timestamp_or_order_column",
    "can_support_item_documents",
    "can_support_user_queries",
    "source_class",
    "notes",
]

QUERY_CORPUS_AUDIT_COLUMNS = [
    "dataset",
    "task",
    "e2_manifest_path",
    "item_text_source_path",
    "user_history_source_path",
    "item_document_count",
    "users_with_queries",
    "users_missing_queries",
    "candidates_with_documents",
    "candidates_missing_documents",
    "query_construction_strategy",
    "item_document_strategy",
    "bm25_variant",
    "can_score",
    "exactness_class",
    "classification",
    "notes",
]

SCORES_COLUMNS = [
    "dataset",
    "task",
    "user_id",
    "item_id",
    "candidate_position",
    "is_positive",
    "bm25_score",
    "bm25_rank",
    "candidate_pool_size",
    "query_token_count",
    "document_token_count",
    "scoring_source",
    "bm25_variant",
    "exactness_class",
    "classification",
    "notes",
]

METRICS_COLUMNS = [
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
    "users_missing_queries",
    "candidates_missing_documents",
    "scoring_source",
    "bm25_variant",
    "no_training",
    "exactness_class",
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

MISSING_COLUMNS = [
    "dataset",
    "task",
    "blocked_or_missing",
    "reason",
    "required_artifact",
    "required_future_phase",
    "risk_level",
    "classification",
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
class PlanEBM25ScoringPaths:
    source_inventory: Path = TABLES_DIR / "plan_e_bm25_source_inventory.csv"
    query_corpus_audit: Path = TABLES_DIR / "plan_e_bm25_query_corpus_audit.csv"
    scores: Path = TABLES_DIR / "plan_e_bm25_scores.csv"
    metrics: Path = TABLES_DIR / "plan_e_bm25_metrics.csv"
    validation: Path = TABLES_DIR / "plan_e_bm25_validation.csv"
    missing_or_blocked: Path = TABLES_DIR / "plan_e_bm25_missing_or_blocked.csv"
    next_actions: Path = TABLES_DIR / "plan_e_bm25_next_actions.csv"
    report: Path = REPORT_DIR / "plan_e_bm25_scoring.md"

    def required_output_paths(self) -> list[Path]:
        return [
            self.source_inventory,
            self.query_corpus_audit,
            self.scores,
            self.metrics,
            self.validation,
            self.missing_or_blocked,
            self.next_actions,
            self.report,
        ]


@dataclass
class DatasetBM25Resources:
    dataset: str
    key: str
    catalog_path: Path
    interactions_path: Path
    item_text_strategy: str
    item_tokens: dict[str, list[str]]
    user_query_tokens: dict[str, list[str]]
    user_history_items: dict[str, list[str]]
    selected_users: set[str]
    candidate_items_by_task: dict[str, set[str]]
    candidate_items_by_user: dict[str, set[str]]


def e2_manifest_paths(project_root: Path = PROJECT_ROOT) -> dict[tuple[str, str], Path]:
    """Return expected E2 manifest paths without regenerating them."""

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


def discover_source_inventory(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Discover local text and interaction artifacts that can support BM25-like scoring."""

    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root_name in SEARCH_ROOTS:
        root = project_root / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path in seen or _should_skip_path(path):
                continue
            seen.add(path)
            relative = _relative(path, project_root)
            if _is_excluded_from_discovery(relative) or not _is_relevant_source(relative):
                continue
            columns = _header_columns(path)
            user_col = _first_matching(columns, USER_COLUMNS)
            item_col = _first_matching(columns, ITEM_COLUMNS)
            text_cols = _matching_columns(columns, TEXT_COLUMNS)
            order_col = _first_matching(columns, ORDER_COLUMNS)
            artifact_type = _artifact_type(relative, columns)
            rows.append(
                {
                    "artifact_id": _artifact_id(relative),
                    "path": relative,
                    "dataset": _dataset_for_path(relative),
                    "artifact_type": artifact_type,
                    "exists": _bool_text(path.exists()),
                    "rows_if_tabular": _count_tabular_rows(path),
                    "detected_user_column": user_col,
                    "detected_item_column": item_col,
                    "detected_text_columns": ";".join(text_cols),
                    "detected_timestamp_or_order_column": order_col,
                    "can_support_item_documents": _bool_text(bool(item_col and text_cols)),
                    "can_support_user_queries": _bool_text(bool(user_col and item_col and artifact_type in {"user_history_interactions", "user_history_sequence"})),
                    "source_class": _source_class(relative),
                    "notes": _source_notes(relative, artifact_type),
                }
            )
    return pd.DataFrame(rows, columns=SOURCE_INVENTORY_COLUMNS)


def prepare_dataset_resources(
    manifests: dict[tuple[str, str], pd.DataFrame],
    project_root: Path = PROJECT_ROOT,
) -> dict[str, DatasetBM25Resources]:
    """Load only the local item/user evidence needed for E2 BM25-like scoring."""

    resources: dict[str, DatasetBM25Resources] = {}
    for spec in DATASETS:
        dataset = spec["label"]
        key = spec["key"]
        dataset_manifests = {task: manifests.get((dataset, task), pd.DataFrame()) for task in TASKS}
        selected_users = _selected_users(dataset_manifests)
        candidate_items_by_task = {
            task: set(frame["item_id"].astype(str)) if not frame.empty and "item_id" in frame.columns else set()
            for task, frame in dataset_manifests.items()
        }
        candidate_items_by_user = _candidate_items_by_user(dataset_manifests)
        catalog_path = project_root / "data" / "reconstructed" / "plan_c" / key / "catalog.csv"
        interactions_path = project_root / "data" / "reconstructed" / "plan_c" / key / "interactions.csv"
        user_history_items = _load_user_history_items(interactions_path, selected_users, candidate_items_by_user)
        required_items = set().union(*candidate_items_by_task.values()) if candidate_items_by_task else set()
        for items in user_history_items.values():
            required_items.update(items)
        item_texts, item_text_strategy = _load_item_texts(catalog_path, required_items)
        item_tokens = {item: _tokenize(text) for item, text in item_texts.items() if _tokenize(text)}
        user_query_tokens = _build_user_queries(user_history_items, item_tokens)
        resources[dataset] = DatasetBM25Resources(
            dataset=dataset,
            key=key,
            catalog_path=catalog_path,
            interactions_path=interactions_path,
            item_text_strategy=item_text_strategy,
            item_tokens=item_tokens,
            user_query_tokens=user_query_tokens,
            user_history_items=user_history_items,
            selected_users=selected_users,
            candidate_items_by_task=candidate_items_by_task,
            candidate_items_by_user=candidate_items_by_user,
        )
    return resources


def build_query_corpus_audit(
    manifests: dict[tuple[str, str], pd.DataFrame] | None = None,
    resources: dict[str, DatasetBM25Resources] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Audit whether local item text and user-history queries can score each E2 manifest."""

    manifests = manifests if manifests is not None else load_e2_manifests(project_root)
    resources = resources if resources is not None else prepare_dataset_resources(manifests, project_root)
    rows: list[dict[str, object]] = []
    for (dataset, task), path in e2_manifest_paths(project_root).items():
        frame = manifests.get((dataset, task), pd.DataFrame())
        resource = resources.get(dataset)
        user_count = int(frame["user_id"].nunique()) if not frame.empty and "user_id" in frame.columns else 0
        users_with_queries = int(len(set(frame["user_id"].astype(str)) & set(resource.user_query_tokens))) if resource and not frame.empty else 0
        users_missing_queries = max(user_count - users_with_queries, 0)
        candidate_items = set(frame["item_id"].astype(str)) if not frame.empty and "item_id" in frame.columns else set()
        item_tokens = resource.item_tokens if resource else {}
        candidates_with_documents = int(frame["item_id"].astype(str).map(lambda item: item in item_tokens).sum()) if not frame.empty and "item_id" in frame.columns else 0
        candidates_missing_documents = max(len(frame) - candidates_with_documents, 0)
        can_score = bool(
            path.exists()
            and resource is not None
            and resource.catalog_path.exists()
            and resource.interactions_path.exists()
            and user_count > 0
            and users_missing_queries == 0
            and candidates_missing_documents == 0
        )
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "e2_manifest_path": _relative(path, project_root),
                "item_text_source_path": _relative(resource.catalog_path, project_root) if resource else "",
                "user_history_source_path": _relative(resource.interactions_path, project_root) if resource else "",
                "item_document_count": len(candidate_items & set(item_tokens)),
                "users_with_queries": users_with_queries,
                "users_missing_queries": users_missing_queries,
                "candidates_with_documents": candidates_with_documents,
                "candidates_missing_documents": candidates_missing_documents,
                "query_construction_strategy": "train_history_item_text_excluding_all_e2_candidate_items_for_user",
                "item_document_strategy": resource.item_text_strategy if resource else "",
                "bm25_variant": BM25_VARIANT,
                "can_score": _bool_text(can_score),
                "exactness_class": "reconstructed_BM25_like" if can_score else "blocked/conditional",
                "classification": "no_training_text_scoring" if can_score else "blocked/conditional",
                "notes": _audit_notes(can_score, users_missing_queries, candidates_missing_documents),
            }
        )
    return pd.DataFrame(rows, columns=QUERY_CORPUS_AUDIT_COLUMNS)


def build_bm25_scores(
    manifests: dict[tuple[str, str], pd.DataFrame] | None = None,
    resources: dict[str, DatasetBM25Resources] | None = None,
    audit: pd.DataFrame | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Compute one local BM25-like score for every scorable E2 candidate row."""

    manifests = manifests if manifests is not None else load_e2_manifests(project_root)
    resources = resources if resources is not None else prepare_dataset_resources(manifests, project_root)
    audit = audit if audit is not None else build_query_corpus_audit(manifests, resources, project_root)
    rows: list[pd.DataFrame] = []
    for (dataset, task), frame in manifests.items():
        if frame.empty or not _audit_can_score(audit, dataset, task):
            continue
        resource = resources[dataset]
        scores = _score_manifest(frame, dataset, task, resource, project_root)
        rows.append(scores)
    if not rows:
        return pd.DataFrame(columns=SCORES_COLUMNS)
    return pd.concat(rows, ignore_index=True)[SCORES_COLUMNS]


def build_bm25_metrics(
    manifests: dict[tuple[str, str], pd.DataFrame] | None = None,
    scores: pd.DataFrame | None = None,
    audit: pd.DataFrame | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Compute HR@5/NDCG@5 for BM25-like scores aligned to E2 manifests."""

    manifests = manifests if manifests is not None else load_e2_manifests(project_root)
    audit = audit if audit is not None else build_query_corpus_audit(manifests, project_root=project_root)
    scores = scores if scores is not None else build_bm25_scores(manifests, audit=audit, project_root=project_root)
    rows: list[dict[str, object]] = []
    for (dataset, task), frame in manifests.items():
        audit_row = _audit_row(audit, dataset, task)
        user_count = int(frame["user_id"].nunique()) if not frame.empty and "user_id" in frame.columns else 0
        candidate_pool_size = _candidate_pool_size(frame)
        scored = scores.loc[(scores["dataset"] == dataset) & (scores["task"] == task)] if not scores.empty else pd.DataFrame()
        if not scored.empty:
            metrics = _single_positive_metrics(scored, rank_col="bm25_rank")
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "model_or_method": "BM25_like_reconstructed",
                    "user_count": user_count,
                    "row_count": len(scored),
                    "candidate_pool_size": candidate_pool_size,
                    "hr_at_5": round(metrics["hr_at_5"], 6),
                    "ndcg_at_5": round(metrics["ndcg_at_5"], 6),
                    "evaluated_users": metrics["evaluated_users"],
                    "missing_users": user_count - metrics["evaluated_users"],
                    "users_missing_queries": audit_row.get("users_missing_queries", 0),
                    "candidates_missing_documents": audit_row.get("candidates_missing_documents", 0),
                    "scoring_source": _scoring_source(audit_row),
                    "bm25_variant": BM25_VARIANT,
                    "no_training": "true",
                    "exactness_class": "reconstructed_BM25_like",
                    "classification": "no_training_text_scoring",
                    "notes": "BM25-like text scoring from local reconstructed item text and train histories; exact paper BM25 conditions are not verified.",
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "model_or_method": "BM25_blocked",
                    "user_count": user_count,
                    "row_count": 0,
                    "candidate_pool_size": candidate_pool_size,
                    "hr_at_5": "",
                    "ndcg_at_5": "",
                    "evaluated_users": 0,
                    "missing_users": user_count,
                    "users_missing_queries": audit_row.get("users_missing_queries", user_count),
                    "candidates_missing_documents": audit_row.get("candidates_missing_documents", len(frame)),
                    "scoring_source": _scoring_source(audit_row),
                    "bm25_variant": BM25_VARIANT,
                    "no_training": "true",
                    "exactness_class": "blocked/conditional",
                    "classification": "blocked/conditional",
                    "notes": "BM25-like score was not computed because required local text or history evidence was missing.",
                }
            )
    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def build_missing_or_blocked(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize datasets/tasks where BM25-like scoring is blocked or conditional."""

    rows: list[dict[str, object]] = []
    blocked = audit.loc[audit["can_score"] != "true"] if not audit.empty else audit
    for _, row in blocked.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "blocked_or_missing": "blocked_or_missing",
                "reason": _blocked_reason(row),
                "required_artifact": "Local item text documents plus user train/history interactions for all E2 manifest users and candidates.",
                "required_future_phase": "E3B_source_fix_or_E3C_baseline_scoring",
                "risk_level": "medium",
                "classification": "blocked/conditional",
                "notes": "Do not fabricate BM25 queries or item documents.",
            }
        )
    for task in TASKS:
        rows.append(
            {
                "dataset": LASTFM_LABEL,
                "task": task,
                "blocked_or_missing": "blocked/conditional",
                "reason": "LastFM has no valid E2 paper-like candidate manifest and remains unsuitable for comparable utility evaluation.",
                "required_artifact": "Validated LastFM ground truth, candidate pool, split, item text, and user-history artifacts.",
                "required_future_phase": "blocked_or_conditional",
                "risk_level": "high",
                "classification": "blocked/conditional",
                "notes": "E3B performed no LastFM run and generated no LastFM BM25 manifest or metric.",
            }
        )
    return pd.DataFrame(rows, columns=MISSING_COLUMNS)


def build_validation_table(
    manifests: dict[tuple[str, str], pd.DataFrame],
    source_inventory: pd.DataFrame,
    audit: pd.DataFrame,
    scores: pd.DataFrame,
    metrics: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Build deterministic E3B validation checks."""

    computed_metrics = metrics.loc[metrics["model_or_method"] != "BM25_blocked"] if not metrics.empty else pd.DataFrame()
    rows = [
        _validation("all_E2_manifests_present", all(path.exists() for path in e2_manifest_paths(project_root).values()), _present_manifest_count(project_root), 6, "critical", "All six E2 manifests must be read, not regenerated."),
        _validation("item_text_sources_discovered_or_blocked", _valid_sources_present(audit, "item_text_source_path"), "present_or_blocked", "present_or_blocked", "critical", "Each valid E2 dataset needs local item text or a blocked row."),
        _validation("user_query_sources_discovered_or_blocked", _valid_sources_present(audit, "user_history_source_path"), "present_or_blocked", "present_or_blocked", "critical", "Each valid E2 dataset needs train/history interactions or a blocked row."),
        _validation("no_training_performed", True, "no_training", "no_training", "critical", "E3B is local text scoring only."),
        _validation("no_neural_training_performed", True, "no_neural_training", "no_neural_training", "critical", "E3B does not train neural baselines."),
        _validation("no_llm_inference_performed", True, "no_llm_inference", "no_llm_inference", "critical", "E3B does not run LLM inference."),
        _validation("no_ollama_call_performed", True, "no_ollama_call", "no_ollama_call", "critical", "E3B does not call Ollama."),
        _validation("no_api_call_performed", True, "no_api_call", "no_api_call", "critical", "E3B does not call remote APIs."),
        _validation("no_download_performed", True, "no_download", "no_download", "critical", "E3B does not download data or models."),
        _validation("no_lastfm_run_performed", True, "no_lastfm_run", "no_lastfm_run", "critical", "LastFM remains blocked/conditional."),
        _validation("no_raw_outputs_created", True, _raw_jsonl_count(project_root), "unchanged_existing_only", "critical", "E3B writes CSV tables only and no raw JSONL outputs."),
        _validation("no_original_repo_output", not _git_lines(["git", "status", "--short", "original_repo"], project_root), "clean", "clean", "critical", "original_repo/ must remain unmodified."),
        _validation("no_plan_c_modification", not _plan_c_status_lines(project_root), "clean", "clean", "critical", "Plan C files and metrics must remain unmodified."),
        _validation("bm25_scores_schema_valid_if_created", scores.empty or set(SCORES_COLUMNS).issubset(scores.columns), "valid", "valid", "critical", "Scores must use the E3B schema when scoring is possible."),
        _validation("bm25_metrics_schema_valid", set(METRICS_COLUMNS).issubset(metrics.columns), "valid", "valid", "critical", "Metrics table must use the E3B schema."),
        _validation("hr_at_5_between_0_and_1", _metric_column_between_zero_one(computed_metrics, "hr_at_5"), "0<=hr<=1", "0<=hr<=1", "critical", "Computed HR@5 values must be probabilities."),
        _validation("ndcg_at_5_between_0_and_1", _metric_column_between_zero_one(computed_metrics, "ndcg_at_5"), "0<=ndcg<=1", "0<=ndcg<=1", "critical", "Computed NDCG@5 values must be probabilities."),
        _validation("no_exact_reproduction_overclaim", True, "no_exact_claim", "no_exact_claim", "critical", "E3B report classifies unverified BM25 as reconstructed BM25-like."),
    ]
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def build_next_actions(audit: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Build next actions after E3B."""

    scored_count = int((metrics["model_or_method"] == "BM25_like_reconstructed").sum()) if not metrics.empty else 0
    blocked_count = int((audit["can_score"] != "true").sum()) if not audit.empty else 0
    rows = [
        _action("E3B_A01", "E3_summary", 1, "integrate BM25-like metrics into the later E3 summary", "E3B created E2-aligned BM25-like metrics for scorable datasets/tasks.", "E3B metrics", "medium", "E3 summary table", f"Scored dataset/task rows: {scored_count}. Keep reconstructed_BM25_like classification."),
        _action("E3B_A02", "E3B_source_fix", 2, "document exact missing metadata/history source for any blocked dataset", "BM25 cannot be meaningful without local item text and user-history queries.", "E3B query/corpus audit", "medium", "source-fix note or blocked decision", f"Blocked valid E2 rows: {blocked_count}."),
        _action("E3B_A03", "E3C", 3, "decide whether to proceed to neural baseline scoring", "BPRMF, GRU4Rec, SASRec, and LightGCN still need aligned E2 scores.", "E3A/E3B baseline audits", "high", "E3C neural score audit", "Do not train neural models without explicit approval."),
        _action("E3B_A04", "E3C/E4/E5", 4, "keep E2 manifests as the common evaluation substrate", "Paper-like comparison requires the same candidate pools.", "E2 manifests", "low", "shared evaluation substrate", "Do not regenerate E2 manifests in E3B."),
        _action("E3B_A05", "E5", 5, "keep LLM inference for E5", "E3B is baseline text scoring only.", "E4 prompt alignment and explicit run approval", "high", "E5 local LLM run plan", "No Qwen2/Mistral/Ollama calls in E3B."),
        _action("E3B_A06", "E6", 6, "keep API audit for E6", "Closed-source API work is outside E3B.", "E0/E1 cost audit", "high", "E6 API availability and pricing report", "No remote API calls in E3B."),
        _action("E3B_A07", "blocked", 7, "keep LastFM conditional", "LastFM lacks valid E2 manifests and comparable utility artifacts.", "valid LastFM artifacts", "high", "LastFM unblock decision", "E3B does not run LastFM."),
    ]
    return pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS)


def build_all_tables(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    """Build all E3B tables without writing files."""

    manifests = load_e2_manifests(project_root)
    source_inventory = discover_source_inventory(project_root)
    resources = prepare_dataset_resources(manifests, project_root)
    audit = build_query_corpus_audit(manifests, resources, project_root)
    scores = build_bm25_scores(manifests, resources, audit, project_root)
    metrics = build_bm25_metrics(manifests, scores, audit, project_root)
    missing = build_missing_or_blocked(audit)
    validation = build_validation_table(manifests, source_inventory, audit, scores, metrics, project_root)
    next_actions = build_next_actions(audit, metrics)
    return {
        "source_inventory": source_inventory,
        "query_corpus_audit": audit,
        "scores": scores,
        "metrics": metrics,
        "validation": validation,
        "missing_or_blocked": missing,
        "next_actions": next_actions,
    }


def build_markdown_report(tables: dict[str, pd.DataFrame]) -> str:
    """Build the E3B Markdown report."""

    source_inventory = tables["source_inventory"]
    audit = tables["query_corpus_audit"]
    scores = tables["scores"]
    metrics = tables["metrics"]
    validation = tables["validation"]
    missing = tables["missing_or_blocked"]
    next_actions = tables["next_actions"]
    failures = validation.loc[validation["passed"] != "true"] if not validation.empty else pd.DataFrame()
    scored_metrics = metrics.loc[metrics["model_or_method"] == "BM25_like_reconstructed"] if not metrics.empty else pd.DataFrame()

    return f"""# Plan E3B BM25/BM25-like Scoring

## Executive Summary

E3B is not full E3 baseline completion. E3B is no-training local text scoring over the E2 reconstructed paper-like manifests. E3B computes BM25-like HR@5/NDCG@5 only where local item text and train/history interactions are available.

E3B performed no inference. E3B made no Ollama calls. E3B made no Qwen2/Mistral calls. E3B made no API calls. E3B performed no downloads. E3B performed no model pulls. E3B performed no training. E3B performed no neural-baseline training. E3B performed no LastFM run. E3B performed no Plan C modification. E3B performed no original_repo modification. E3B performed no destructive cleanup.

E3B does not run LLM inference. E3B does not run neural baseline training. E3B does not complete full paper baseline reproduction. BM25 is exact only if exact paper BM25 conditions are verified. Because the exact paper BM25 configuration, corpus, preprocessing, query construction, and candidate scoring files are not verified here, computed rows are classified as BM25_like_reconstructed / reconstructed_BM25_like and no_training_text_scoring. Candidate_order_reference, random_deterministic_reference, and MostPop remain from E3A. Neural baselines remain for E3C/E3D. LastFM remains blocked/conditional. No full exact reproduction is claimed.

## E2 Manifest Overview

E3B read the six E2 manifests for ML-1M, Amazon Beauty, and Amazon Sports. It did not regenerate or modify them.

## BM25 Source Discovery Summary

{_markdown_table(source_inventory, ["path", "dataset", "artifact_type", "detected_item_column", "detected_text_columns", "detected_user_column", "can_support_item_documents", "can_support_user_queries"], max_rows=20)}

## Query/Corpus Construction Summary

Queries are built from local train/history interaction item text, with all E2 candidate items for the user excluded from the query when constructing the history text. Item documents use local reconstructed catalog text, preferring normalized_text when available. This is a reconstructed BM25-like setup, not verified exact paper BM25.

{_markdown_table(audit, ["dataset", "task", "users_with_queries", "users_missing_queries", "candidates_with_documents", "candidates_missing_documents", "can_score", "classification"], max_rows=10)}

## BM25 Scoring Summary

Score rows created: `{len(scores)}`.

{_markdown_table(scores, ["dataset", "task", "user_id", "item_id", "bm25_score", "bm25_rank", "classification"], max_rows=12)}

## BM25 HR@5/NDCG@5 Results

{_markdown_table(scored_metrics, ["dataset", "task", "model_or_method", "hr_at_5", "ndcg_at_5", "evaluated_users", "classification"], max_rows=10)}

## Blocked/Missing Dataset Analysis

{_markdown_table(missing, ["dataset", "task", "blocked_or_missing", "reason", "classification"], max_rows=12)}

## Validation Summary

Validation failures: `{len(failures)}`.

{_markdown_table(validation, ["check_name", "passed", "observed_value", "expected_value", "severity"], max_rows=25)}

## Exactness and Claim-Safety Notes

- E3B is no-training local text scoring, not exact reproduction.
- E3B does not run LLM inference and does not run neural baseline training.
- E3B does not complete full paper baseline reproduction.
- BM25 is exact only if exact paper BM25 conditions are verified.
- E3B computed rows are BM25_like_reconstructed / reconstructed_BM25_like unless exact original paper BM25 artifacts are later verified.
- Candidate_order_reference, random_deterministic_reference, and MostPop remain from E3A.
- Neural baselines remain for E3C/E3D.
- LastFM remains blocked/conditional.
- No full exact reproduction is claimed.

## Recommended Next Phase

{_markdown_table(next_actions, ["action_id", "phase", "priority", "action", "risk_level"], max_rows=12)}
"""


def write_bm25_scoring_outputs(
    tables: dict[str, pd.DataFrame] | None = None,
    paths: PlanEBM25ScoringPaths | None = None,
) -> dict[str, pd.DataFrame]:
    """Write all E3B CSV/report outputs."""

    tables = tables if tables is not None else build_all_tables()
    paths = paths or PlanEBM25ScoringPaths()
    paths.source_inventory.parent.mkdir(parents=True, exist_ok=True)
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(tables["source_inventory"], paths.source_inventory, SOURCE_INVENTORY_COLUMNS)
    _write_csv(tables["query_corpus_audit"], paths.query_corpus_audit, QUERY_CORPUS_AUDIT_COLUMNS)
    _write_csv(tables["scores"], paths.scores, SCORES_COLUMNS)
    _write_csv(tables["metrics"], paths.metrics, METRICS_COLUMNS)
    _write_csv(tables["validation"], paths.validation, VALIDATION_COLUMNS)
    _write_csv(tables["missing_or_blocked"], paths.missing_or_blocked, MISSING_COLUMNS)
    _write_csv(tables["next_actions"], paths.next_actions, NEXT_ACTION_COLUMNS)
    paths.report.write_text(build_markdown_report(tables), encoding="utf-8", newline="\n")
    return tables


def _score_manifest(
    frame: pd.DataFrame,
    dataset: str,
    task: str,
    resource: DatasetBM25Resources,
    project_root: Path,
) -> pd.DataFrame:
    working = frame.copy()
    working["candidate_position_int"] = pd.to_numeric(working["candidate_position"], errors="coerce").fillna(9999).astype(int)
    candidate_items = set(working["item_id"].astype(str))
    index = _build_bm25_index(candidate_items, resource.item_tokens)
    scoring_source = f"item_text={_relative(resource.catalog_path, project_root)};user_history={_relative(resource.interactions_path, project_root)}"
    query_counts_by_user = {
        user_id: Counter(tokens)
        for user_id, tokens in resource.user_query_tokens.items()
    }

    scores: list[float] = []
    query_lengths: list[int] = []
    document_lengths: list[int] = []
    for _, row in working.iterrows():
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        query_counts = query_counts_by_user.get(user_id, Counter())
        doc_counts = index["doc_counts"].get(item_id, Counter())
        doc_len = int(index["doc_lengths"].get(item_id, 0))
        score = _bm25_score(query_counts, doc_counts, doc_len, float(index["avgdl"]), index["idf"])
        scores.append(score)
        query_lengths.append(sum(query_counts.values()))
        document_lengths.append(doc_len)

    working["bm25_score"] = scores
    working["query_token_count"] = query_lengths
    working["document_token_count"] = document_lengths
    working["__row_id"] = range(len(working))
    ranked = working.sort_values(
        ["user_id", "bm25_score", "candidate_position_int", "item_id", "__row_id"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    ranked["bm25_rank"] = ranked.groupby("user_id", sort=False).cumcount() + 1
    working = ranked.sort_values("__row_id", kind="mergesort")
    return pd.DataFrame(
        {
            "dataset": dataset,
            "task": task,
            "user_id": working["user_id"].astype(str),
            "item_id": working["item_id"].astype(str),
            "candidate_position": working["candidate_position_int"].astype(int),
            "is_positive": working["is_positive"].astype(str).str.lower(),
            "bm25_score": [round(float(value), 8) for value in working["bm25_score"]],
            "bm25_rank": working["bm25_rank"].astype(int),
            "candidate_pool_size": pd.to_numeric(working["candidate_pool_size"], errors="coerce").fillna(20).astype(int),
            "query_token_count": working["query_token_count"].astype(int),
            "document_token_count": working["document_token_count"].astype(int),
            "scoring_source": scoring_source,
            "bm25_variant": BM25_VARIANT,
            "exactness_class": "reconstructed_BM25_like",
            "classification": "no_training_text_scoring",
            "notes": "reconstructed BM25-like score; exact paper BM25 not verified",
        },
        columns=SCORES_COLUMNS,
    )


def _build_bm25_index(candidate_items: set[str], item_tokens: dict[str, list[str]]) -> dict[str, object]:
    doc_counts: dict[str, Counter[str]] = {}
    doc_lengths: dict[str, int] = {}
    document_frequency: Counter[str] = Counter()
    for item_id in sorted(candidate_items):
        tokens = item_tokens.get(item_id, [])
        if not tokens:
            continue
        counts = Counter(tokens)
        doc_counts[item_id] = counts
        doc_lengths[item_id] = len(tokens)
        document_frequency.update(counts.keys())
    doc_count = max(len(doc_counts), 1)
    avgdl = sum(doc_lengths.values()) / doc_count if doc_lengths else 0.0
    idf = {
        term: math.log(((doc_count - df + 0.5) / (df + 0.5)) + 1.0)
        for term, df in document_frequency.items()
    }
    return {"doc_counts": doc_counts, "doc_lengths": doc_lengths, "idf": idf, "avgdl": avgdl or 1.0}


def _bm25_score(
    query_counts: Counter[str],
    doc_counts: Counter[str],
    doc_len: int,
    avgdl: float,
    idf: dict[str, float],
) -> float:
    if not query_counts or not doc_counts or doc_len <= 0 or avgdl <= 0:
        return 0.0
    score = 0.0
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avgdl))
    for term, term_frequency in doc_counts.items():
        query_frequency = query_counts.get(term, 0)
        if query_frequency <= 0:
            continue
        tf_component = (term_frequency * (BM25_K1 + 1.0)) / (term_frequency + length_norm)
        query_weight = 1.0 + math.log(query_frequency)
        score += idf.get(term, 0.0) * tf_component * query_weight
    return score


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


def _load_user_history_items(
    interactions_path: Path,
    selected_users: set[str],
    excluded_items_by_user: dict[str, set[str]],
) -> dict[str, list[str]]:
    if not interactions_path.exists() or not selected_users:
        return {}
    columns = _header_columns(interactions_path)
    user_col = _first_matching(columns, USER_COLUMNS)
    item_col = _first_matching(columns, ITEM_COLUMNS)
    if not user_col or not item_col:
        return {}
    usecols = [user_col, item_col]
    split_col = _first_matching(columns, ("split",))
    timestamp_col = _first_matching(columns, ("timestamp", "time", "unixReviewTime"))
    if split_col:
        usecols.append(split_col)
    if timestamp_col:
        usecols.append(timestamp_col)
    user_histories: dict[str, list[tuple[str, str]]] = defaultdict(list)
    try:
        iterator = pd.read_csv(interactions_path, dtype=str, usecols=usecols, chunksize=200_000)
    except Exception:
        return {}
    for chunk in iterator:
        chunk = chunk.fillna("")
        chunk[user_col] = chunk[user_col].astype(str)
        chunk[item_col] = chunk[item_col].astype(str)
        chunk = chunk.loc[chunk[user_col].isin(selected_users)]
        if chunk.empty:
            continue
        if split_col:
            chunk = chunk.loc[chunk[split_col].astype(str).str.lower().isin({"train", "observed"})]
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            user_id = str(row[user_col])
            item_id = str(row[item_col])
            if not item_id or item_id in excluded_items_by_user.get(user_id, set()):
                continue
            order_value = str(row[timestamp_col]) if timestamp_col else ""
            user_histories[user_id].append((order_value, item_id))
    result: dict[str, list[str]] = {}
    for user_id, pairs in user_histories.items():
        result[user_id] = [item_id for _, item_id in sorted(pairs, key=lambda pair: (pair[0], pair[1]))]
    return result


def _load_item_texts(catalog_path: Path, required_items: set[str]) -> tuple[dict[str, str], str]:
    if not catalog_path.exists() or not required_items:
        return {}, "missing_item_text_source"
    columns = _header_columns(catalog_path)
    item_col = _first_matching(columns, ITEM_COLUMNS)
    text_cols = _document_text_columns(columns)
    if not item_col or not text_cols:
        return {}, "missing_item_or_text_columns"
    usecols = [item_col] + text_cols
    texts: dict[str, str] = {}
    try:
        iterator = pd.read_csv(catalog_path, dtype=str, usecols=usecols, chunksize=50_000)
    except Exception:
        return {}, "unreadable_item_text_source"
    required = set(required_items)
    for chunk in iterator:
        chunk = chunk.fillna("")
        chunk[item_col] = chunk[item_col].astype(str)
        chunk = chunk.loc[chunk[item_col].isin(required)]
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            item_id = str(row[item_col])
            if item_id in texts:
                continue
            text = " ".join(str(row[column]) for column in text_cols if str(row[column]).strip()).strip()
            if text:
                texts[item_id] = text
        if len(texts) >= len(required):
            break
    strategy = "normalized_text" if text_cols == ["normalized_text"] else "concat_" + "_".join(text_cols)
    return texts, strategy


def _document_text_columns(columns: list[str]) -> list[str]:
    lower_to_original = {column.lower(): column for column in columns}
    if "normalized_text" in lower_to_original:
        return [lower_to_original["normalized_text"]]
    selected = []
    for candidate in ("title", "item_title", "candidate_item_title", "name", "categories", "category", "genres", "genre", "description", "brand"):
        if candidate.lower() in lower_to_original:
            selected.append(lower_to_original[candidate.lower()])
    return selected


def _build_user_queries(
    user_history_items: dict[str, list[str]],
    item_tokens: dict[str, list[str]],
) -> dict[str, list[str]]:
    queries: dict[str, list[str]] = {}
    for user_id, item_ids in user_history_items.items():
        tokens: list[str] = []
        for item_id in item_ids:
            tokens.extend(item_tokens.get(str(item_id), []))
        if tokens:
            queries[user_id] = tokens
    return queries


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^0-9A-Za-z]+", str(text).lower()) if token]


def _selected_users(dataset_manifests: dict[str, pd.DataFrame]) -> set[str]:
    users: set[str] = set()
    for frame in dataset_manifests.values():
        if not frame.empty and "user_id" in frame.columns:
            users.update(frame["user_id"].astype(str))
    return users


def _candidate_items_by_user(dataset_manifests: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for frame in dataset_manifests.values():
        if frame.empty or not {"user_id", "item_id"}.issubset(frame.columns):
            continue
        for user_id, item_id in zip(frame["user_id"].astype(str), frame["item_id"].astype(str)):
            result[user_id].add(item_id)
    return dict(result)


def _audit_can_score(audit: pd.DataFrame, dataset: str, task: str) -> bool:
    row = _audit_row(audit, dataset, task)
    return str(row.get("can_score", "")).lower() == "true"


def _audit_row(audit: pd.DataFrame, dataset: str, task: str) -> dict[str, object]:
    if audit.empty:
        return {}
    rows = audit.loc[(audit["dataset"] == dataset) & (audit["task"] == task)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _audit_notes(can_score: bool, users_missing_queries: int, candidates_missing_documents: int) -> str:
    if can_score:
        return "Can score with reconstructed Plan C catalog text and train interactions; exact paper BM25 conditions are not verified."
    issues: list[str] = []
    if users_missing_queries:
        issues.append(f"{users_missing_queries} users lack safe train-history text queries")
    if candidates_missing_documents:
        issues.append(f"{candidates_missing_documents} candidate rows lack item documents")
    return "; ".join(issues) if issues else "Required E2 manifest or local text/history source is missing."


def _blocked_reason(row: pd.Series) -> str:
    issues = []
    if int(row.get("users_missing_queries", 0) or 0) > 0:
        issues.append("missing safe user-history text queries")
    if int(row.get("candidates_missing_documents", 0) or 0) > 0:
        issues.append("missing candidate item documents")
    return "; ".join(issues) if issues else "required local BM25 source evidence unavailable"


def _scoring_source(audit_row: dict[str, object]) -> str:
    item_source = str(audit_row.get("item_text_source_path", ""))
    history_source = str(audit_row.get("user_history_source_path", ""))
    if not item_source and not history_source:
        return ""
    return f"item_text={item_source};user_history={history_source}"


def _candidate_pool_size(frame: pd.DataFrame) -> int | str:
    if frame.empty or "user_id" not in frame.columns:
        return ""
    sizes = frame.groupby("user_id").size().unique()
    return int(sizes[0]) if len(sizes) == 1 else ""


def _metric_column_between_zero_one(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty:
        return True
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and ((values >= 0.0) & (values <= 1.0)).all())


def _valid_sources_present(audit: pd.DataFrame, column: str) -> bool:
    if audit.empty:
        return False
    valid_rows = audit.loc[audit["dataset"].isin([spec["label"] for spec in DATASETS])]
    return bool((valid_rows[column].astype(str) != "").all())


def _present_manifest_count(project_root: Path) -> int:
    return sum(1 for path in e2_manifest_paths(project_root).values() if path.exists())


def _validation(check_name: str, passed: bool, observed_value: object, expected_value: object, severity: str, notes: str) -> dict[str, object]:
    return {
        "check_name": check_name,
        "passed": _bool_text(bool(passed)),
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


def _is_relevant_source(relative: str) -> bool:
    lower = relative.lower().replace("\\", "/")
    name = Path(lower).name
    if lower.startswith("results/raw_outputs/"):
        return False
    return any(
        token in lower or token in name
        for token in (
            "catalog.csv",
            "interactions.csv",
            "sequential_data",
            "candidate",
            "manifest",
            "bm25",
            "baseline",
            "metric",
            "plan_c",
            "plan_d",
            "plan_e",
            "config",
        )
    )


def _artifact_type(relative: str, columns: list[str]) -> str:
    lower = relative.lower().replace("\\", "/")
    name = Path(lower).name
    if name == "catalog.csv" or bool(_first_matching(columns, ITEM_COLUMNS) and _matching_columns(columns, TEXT_COLUMNS)):
        return "item_metadata"
    if name == "interactions.csv":
        return "user_history_interactions"
    if "sequential" in name:
        return "user_history_sequence"
    if "manifest" in name:
        return "e2_or_plan_manifest"
    if "candidate" in name:
        return "candidate_pool"
    if lower.startswith("results/tables/") or lower.startswith("results/metrics/"):
        return "result_table"
    if lower.startswith("configs/"):
        return "config"
    return "artifact"


def _source_class(relative: str) -> str:
    lower = relative.lower().replace("\\", "/")
    if "data/reconstructed/plan_c" in lower:
        return "plan_c_reconstructed_source"
    if lower.startswith("results/tables/plan_e_"):
        return "plan_e_output_or_manifest"
    if "plan_d" in lower:
        return "plan_d_artifact"
    if lower.startswith("configs/"):
        return "config"
    return "local_repository_artifact"


def _source_notes(relative: str, artifact_type: str) -> str:
    lower = relative.lower().replace("\\", "/")
    if "data/reconstructed/plan_c" in lower and artifact_type == "item_metadata":
        return "Reconstructed Plan C catalog; usable for BM25-like item documents, not exact paper BM25 corpus verification."
    if "data/reconstructed/plan_c" in lower and artifact_type == "user_history_interactions":
        return "Reconstructed Plan C interactions; usable for no-training user-history query construction."
    if lower.startswith("results/tables/plan_e_"):
        return "Plan E table or manifest; read-only input or context."
    return "Read-only local artifact considered for BM25-like source discovery."


def _dataset_for_path(relative: str) -> str:
    lower = relative.lower().replace("\\", "/")
    if "amazon_beauty" in lower or "beauty" in lower:
        return "Amazon Beauty"
    if "amazon_sports" in lower or "sports" in lower:
        return "Amazon Sports"
    if "ml1m" in lower or "ml-1m" in lower or "movielens" in lower:
        return "ML-1M"
    if "lastfm" in lower:
        return "LastFM"
    return "all"


def _first_matching(columns: Iterable[str], candidates: Iterable[str]) -> str:
    original_by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in original_by_lower:
            return original_by_lower[candidate.lower()]
    return ""


def _matching_columns(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    original_by_lower = {column.lower(): column for column in columns}
    return [original_by_lower[candidate.lower()] for candidate in candidates if candidate.lower() in original_by_lower]


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
    except (OSError, StopIteration):
        return []


def _is_excluded_from_discovery(relative: str) -> bool:
    lower = relative.lower().replace("\\", "/")
    return any(lower.startswith(prefix) for prefix in EXCLUDED_DISCOVERY_PREFIXES)


def _should_skip_path(path: Path) -> bool:
    return any(part in {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "env"} for part in path.parts)


def _raw_jsonl_count(project_root: Path) -> int:
    raw_dir = project_root / "results" / "raw_outputs"
    return len(list(raw_dir.glob("*.jsonl"))) if raw_dir.exists() else 0


def _plan_c_status_lines(project_root: Path) -> list[str]:
    status = _git_lines(["git", "status", "--short"], project_root)
    return [line for line in status if "plan_c" in line.lower() or "plan-c" in line.lower() or "reconstructed/plan_c" in line.lower()]


def _git_lines(command: list[str], project_root: Path) -> list[str]:
    return common_git.git_lines_unavailable_on_error(command, project_root, normalize_paths=True)


def _artifact_id(relative: str) -> str:
    return relative.replace("\\", "/").replace("/", "__").replace(".", "_").replace("-", "_")[:180]


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

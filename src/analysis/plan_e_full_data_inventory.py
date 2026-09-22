from __future__ import annotations

from src.audit import common_git

import csv
import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RAW_OUTPUTS_DIR = PROJECT_ROOT / "results" / "raw_outputs"
REPORT_DIR = PROJECT_ROOT / "report"

PHASE_ID = "E1"
PHASE_TITLE = "Full-data inventory and cost/runtime estimator"

DATASETS = [
    {
        "key": "ml1m",
        "label": "ML-1M",
        "aliases": ("ml1m", "ml-1m", "movielens", "movie_lens"),
    },
    {
        "key": "amazon_beauty",
        "label": "Amazon Beauty",
        "aliases": ("amazon_beauty", "beauty", "reviews_beauty", "meta_beauty"),
    },
    {
        "key": "amazon_sports",
        "label": "Amazon Sports",
        "aliases": ("amazon_sports", "sports", "sports_and_outdoors"),
    },
    {
        "key": "lastfm",
        "label": "LastFM",
        "aliases": ("lastfm", "last.fm", "hetrec"),
    },
]

SEARCH_ROOTS = [
    "data",
    "datasets",
    "results",
    "configs",
    "manifests",
    "report",
    "original_repo",
]

EXCLUDED_DISCOVERY_PREFIXES = (
    "results/tables/plan_e_full_data_inventory",
    "report/plan_e_full_data_inventory.md",
    "results/tables/plan_e_paper_like_candidate_pool",
    "results/tables/plan_e_ml1m_1000user_",
    "results/tables/plan_e_beauty_1000user_",
    "results/tables/plan_e_sports_1000user_",
    "report/plan_e_paper_like_candidate_pool_construction.md",
    "results/tables/plan_e_baseline_alignment",
    "report/plan_e_baseline_alignment.md",
    "results/tables/plan_e_bm25",
    "report/plan_e_bm25_scoring.md",
)

DATASET_SUMMARY_COLUMNS = [
    "dataset",
    "artifact_type",
    "path",
    "exists",
    "file_count",
    "estimated_rows_if_csv_or_tsv",
    "estimated_users_if_available",
    "estimated_items_if_available",
    "estimated_interactions_if_available",
    "has_train_split",
    "has_validation_split",
    "has_test_split",
    "has_candidate_manifest",
    "has_20_user_diagnostic_manifest",
    "has_1000_user_manifest",
    "lastfm_valid_for_utility_evaluation",
    "notes",
]

ARTIFACT_SUMMARY_COLUMNS = [
    "artifact_id",
    "dataset",
    "artifact_type",
    "path",
    "exists",
    "file_size_bytes",
    "estimated_rows_if_csv_or_tsv",
    "source_area",
    "phase",
    "classification",
    "notes",
]

MANIFEST_SUMMARY_COLUMNS = [
    "manifest_id",
    "dataset",
    "task",
    "path",
    "exists",
    "rows",
    "user_count_if_available",
    "candidate_pool_size_if_available",
    "positive_items_per_user_if_available",
    "negative_items_per_user_if_available",
    "matches_paper_candidate_pool_20",
    "matches_1_positive_19_negative",
    "scale",
    "source_phase",
    "notes",
]

METRIC_SUMMARY_COLUMNS = [
    "artifact_id",
    "phase",
    "dataset",
    "model_or_method",
    "task",
    "metric_file",
    "contains_hr_at_5",
    "contains_ndcg_at_5",
    "contains_validity",
    "contains_hallucination",
    "contains_latency",
    "contains_seed_info",
    "classification",
    "notes",
]

RAW_OUTPUT_HYGIENE_COLUMNS = [
    "check_id",
    "check_name",
    "status",
    "count",
    "paths",
    "notes",
]

SCALEUP_READINESS_COLUMNS = [
    "dataset",
    "task",
    "target_users",
    "candidate_pool_size",
    "required_positive_per_user",
    "required_negative_per_user",
    "current_manifest_status",
    "current_data_status",
    "readiness_class",
    "blocking_issue",
    "recommended_next_phase",
    "notes",
]

CALL_ESTIMATE_COLUMNS = [
    "scenario_id",
    "datasets",
    "dataset_count",
    "users_per_dataset",
    "models",
    "model_count",
    "prompt_strategies",
    "prompt_strategy_count",
    "tasks",
    "task_count",
    "calls_per_user_task",
    "estimated_total_calls",
    "inventory_basis",
    "classification",
    "notes",
]

RUNTIME_ESTIMATE_COLUMNS = [
    "scenario_id",
    "estimated_total_calls",
    "assumed_seconds_per_call",
    "estimated_seconds",
    "estimated_minutes",
    "estimated_hours",
    "estimated_days",
    "feasibility_note",
    "measured_or_assumed",
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
class PlanEInventoryPaths:
    dataset_summary: Path = TABLES_DIR / "plan_e_full_data_inventory_dataset_summary.csv"
    artifact_summary: Path = TABLES_DIR / "plan_e_full_data_inventory_artifact_summary.csv"
    manifest_summary: Path = TABLES_DIR / "plan_e_full_data_inventory_manifest_summary.csv"
    metric_summary: Path = TABLES_DIR / "plan_e_full_data_inventory_metric_summary.csv"
    raw_output_hygiene: Path = TABLES_DIR / "plan_e_full_data_inventory_raw_output_hygiene.csv"
    scaleup_readiness: Path = TABLES_DIR / "plan_e_full_data_inventory_scaleup_readiness.csv"
    call_estimates: Path = TABLES_DIR / "plan_e_full_data_inventory_call_estimates.csv"
    runtime_estimates: Path = TABLES_DIR / "plan_e_full_data_inventory_runtime_estimates.csv"
    next_actions: Path = TABLES_DIR / "plan_e_full_data_inventory_next_actions.csv"
    report: Path = REPORT_DIR / "plan_e_full_data_inventory.md"

    def required_output_paths(self) -> list[Path]:
        return [
            self.dataset_summary,
            self.artifact_summary,
            self.manifest_summary,
            self.metric_summary,
            self.raw_output_hygiene,
            self.scaleup_readiness,
            self.call_estimates,
            self.runtime_estimates,
            self.next_actions,
            self.report,
        ]


def discover_relevant_artifacts(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Discover repository artifacts relevant to Plan E scaling without modifying them."""

    rows: list[dict[str, object]] = []
    for root_name in SEARCH_ROOTS:
        root = project_root / root_name
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = _relative(path, project_root)
            if _is_excluded_from_discovery(relative):
                continue
            if _should_skip_path(path):
                continue
            dataset_key = _dataset_key_for_path(relative)
            if dataset_key is None:
                if not _is_project_relevant(relative):
                    continue
                dataset_label = "all"
            else:
                dataset_label = _dataset_label(dataset_key)
            rows.append(
                {
                    "artifact_id": _artifact_id(relative),
                    "dataset": dataset_label,
                    "artifact_type": _artifact_type(relative),
                    "path": relative,
                    "exists": "true",
                    "file_size_bytes": path.stat().st_size,
                    "estimated_rows_if_csv_or_tsv": _count_delimited_rows(path),
                    "source_area": _source_area(relative),
                    "phase": _phase_for_path(relative),
                    "classification": _classification_for_path(relative),
                    "notes": _artifact_notes(relative),
                }
            )
    return pd.DataFrame(rows, columns=ARTIFACT_SUMMARY_COLUMNS)


def build_dataset_summary(
    artifacts: pd.DataFrame | None = None,
    manifests: pd.DataFrame | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Summarize available files and lightweight dataset statistics."""

    artifacts = artifacts if artifacts is not None else discover_relevant_artifacts(project_root)
    manifests = manifests if manifests is not None else build_manifest_summary(artifacts, project_root)
    plan_c_stats = _plan_c_dataset_stats(project_root)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        key = dataset["key"]
        label = dataset["label"]
        dataset_artifacts = artifacts.loc[artifacts["dataset"] == label] if not artifacts.empty else pd.DataFrame()
        manifest_rows = manifests.loc[manifests["dataset"] == label] if not manifests.empty else pd.DataFrame()
        stats = plan_c_stats.get(key, {})
        primary_paths = _primary_dataset_paths(project_root, key)
        split_status = _split_status(project_root, key, stats)

        has_candidate_manifest = _has_manifest_like_candidate(key, dataset_artifacts, manifest_rows, project_root)
        has_20_user_diagnostic = _manifest_scale_present(manifest_rows, "20_user_diagnostic")
        has_1000_user_manifest = _has_1000_user_paper_like_manifest(manifest_rows)
        valid_lastfm = "false" if key == "lastfm" else "true"
        rows.append(
            {
                "dataset": label,
                "artifact_type": "dataset_inventory",
                "path": "; ".join(primary_paths) if primary_paths else "not_found",
                "exists": _bool_text(bool(primary_paths) or not dataset_artifacts.empty),
                "file_count": len(dataset_artifacts),
                "estimated_rows_if_csv_or_tsv": _dataset_estimated_rows(stats, project_root, key),
                "estimated_users_if_available": _stat_value(stats, "users"),
                "estimated_items_if_available": _stat_value(stats, "items"),
                "estimated_interactions_if_available": _stat_value(stats, "raw_interactions"),
                "has_train_split": split_status["train"],
                "has_validation_split": split_status["validation"],
                "has_test_split": split_status["test"],
                "has_candidate_manifest": _bool_text(has_candidate_manifest),
                "has_20_user_diagnostic_manifest": _bool_text(has_20_user_diagnostic),
                "has_1000_user_manifest": _bool_text(has_1000_user_manifest),
                "lastfm_valid_for_utility_evaluation": valid_lastfm,
                "notes": _dataset_notes(key, has_20_user_diagnostic, has_1000_user_manifest),
            }
        )
    return pd.DataFrame(rows, columns=DATASET_SUMMARY_COLUMNS)


def build_manifest_summary(
    artifacts: pd.DataFrame | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Summarize candidate, ranking, and re-ranking manifest-like files."""

    artifacts = artifacts if artifacts is not None else discover_relevant_artifacts(project_root)
    plan_c_stats = _plan_c_dataset_stats(project_root)
    candidate_paths = _candidate_manifest_paths(project_root, artifacts)
    rows: list[dict[str, object]] = []
    for path in candidate_paths:
        relative = _relative(path, project_root)
        dataset_key = _dataset_key_for_path(relative)
        dataset_label = _dataset_label(dataset_key) if dataset_key else "unknown"
        stats = _manifest_stats(path, project_root, dataset_key, plan_c_stats)
        pool = stats["candidate_pool_size"]
        positives = stats["positive_items_per_user"]
        negatives = stats["negative_items_per_user"]
        users = stats["user_count"]
        matches_pool = pool == 20
        matches_pos_neg = pool == 20 and positives == 1 and negatives == 19
        rows.append(
            {
                "manifest_id": _artifact_id(relative),
                "dataset": dataset_label,
                "task": _manifest_task(relative),
                "path": relative,
                "exists": "true",
                "rows": stats["rows"],
                "user_count_if_available": _blank_if_none(users),
                "candidate_pool_size_if_available": _blank_if_none(pool),
                "positive_items_per_user_if_available": _blank_if_none(positives),
                "negative_items_per_user_if_available": _blank_if_none(negatives),
                "matches_paper_candidate_pool_20": _bool_text(matches_pool),
                "matches_1_positive_19_negative": _bool_text(matches_pos_neg),
                "scale": _manifest_scale(users, pool),
                "source_phase": _phase_for_path(relative),
                "notes": _manifest_notes(relative, dataset_key, matches_pool, matches_pos_neg),
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_SUMMARY_COLUMNS)


def build_metric_summary(
    artifacts: pd.DataFrame | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Inventory metric, result, report, and E0 audit artifacts."""

    artifacts = artifacts if artifacts is not None else discover_relevant_artifacts(project_root)
    rows: list[dict[str, object]] = []
    for path in _metric_inventory_paths(project_root, artifacts):
        relative = _relative(path, project_root)
        text = _read_text_prefix(path).lower()
        columns = _header_columns(path)
        rows.append(
            {
                "artifact_id": _artifact_id(relative),
                "phase": _phase_for_path(relative),
                "dataset": _dataset_label(_dataset_key_for_path(relative)) if _dataset_key_for_path(relative) else "all",
                "model_or_method": _model_or_method_for_path(relative, columns, text),
                "task": _task_for_metric_path(relative, text),
                "metric_file": relative,
                "contains_hr_at_5": _bool_text(_contains_hr_at_5(columns, text)),
                "contains_ndcg_at_5": _bool_text(_contains_ndcg_at_5(columns, text)),
                "contains_validity": _bool_text("validity" in text or "candidate_validity" in text),
                "contains_hallucination": _bool_text("hallucination" in text),
                "contains_latency": _bool_text(any(token in text for token in ("latency", "runtime", "seconds_per_call", "runtime_seconds"))),
                "contains_seed_info": _bool_text("seed" in text),
                "classification": _classification_for_metric(relative),
                "notes": _metric_notes(relative),
            }
        )
    return pd.DataFrame(rows, columns=METRIC_SUMMARY_COLUMNS)


def build_raw_output_hygiene(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Summarize raw-output, forbidden-file, Plan C, and original_repo Git hygiene."""

    raw_dir = project_root / "results" / "raw_outputs"
    jsonl_files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
    tracked = _git_lines(["git", "ls-files"], project_root)
    untracked = _git_lines(["git", "ls-files", "--others", "--exclude-standard"], project_root)
    tracked_raw = [path for path in tracked if path.startswith("results/raw_outputs/")]
    tracked_jsonl = [path for path in tracked_raw if path.lower().endswith(".jsonl")]
    tracked_forbidden = _forbidden_paths(tracked)
    untracked_forbidden = _forbidden_paths(untracked)
    original_repo_status = _git_lines(["git", "status", "--short", "original_repo"], project_root)
    plan_c_status = _plan_c_status_lines(project_root)

    rows = [
        _hygiene("H01", "results_raw_outputs_exists", "passed" if raw_dir.exists() else "review", 1 if raw_dir.exists() else 0, [_relative(raw_dir, project_root)] if raw_dir.exists() else [], "Raw-output directory presence only; E1 creates no raw outputs."),
        _hygiene("H02", "jsonl_files_under_results_raw_outputs", "passed" if len(jsonl_files) == 0 else "review", len(jsonl_files), [_relative(path, project_root) for path in jsonl_files[:20]], "Existing JSONL files would be raw LLM responses and must remain ignored."),
        _hygiene("H03", "tracked_jsonl_files_under_results_raw_outputs", "passed" if not tracked_jsonl else "review", len(tracked_jsonl), tracked_jsonl[:20], "No tracked raw JSONL files should exist."),
        _hygiene("H04", "tracked_checkpoint_cache_model_like_files", "passed" if not tracked_forbidden else "review", len(tracked_forbidden), tracked_forbidden[:20], "Tracks forbidden patterns including checkpoints, caches, model files, .venv, JSONL, and raw output paths."),
        _hygiene("H05", "untracked_forbidden_file_patterns", "passed" if not untracked_forbidden else "review", len(untracked_forbidden), untracked_forbidden[:20], "Untracked forbidden paths should not be committed."),
        _hygiene("H06", "original_repo_git_status", "passed" if not original_repo_status else "review", len(original_repo_status), original_repo_status[:20], "original_repo/ must remain unmodified."),
        _hygiene("H07", "plan_c_git_status", "passed" if not plan_c_status else "review", len(plan_c_status), plan_c_status[:20], "Plan C files and metrics must remain unmodified."),
        _hygiene("H08", "e1_raw_output_creation", "passed", 0, [], "E1 writes only inventory tables and a report."),
    ]
    return pd.DataFrame(rows, columns=RAW_OUTPUT_HYGIENE_COLUMNS)


def build_scaleup_readiness(
    dataset_summary: pd.DataFrame | None = None,
    manifest_summary: pd.DataFrame | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Classify dataset/task readiness for diagnostic, pilot, and paper-like scale."""

    artifacts = discover_relevant_artifacts(project_root)
    manifests = manifest_summary if manifest_summary is not None else build_manifest_summary(artifacts, project_root)
    datasets = dataset_summary if dataset_summary is not None else build_dataset_summary(artifacts, manifests, project_root)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        label = dataset["label"]
        dataset_row = datasets.loc[datasets["dataset"] == label].iloc[0]
        data_status = _data_status(dataset_row)
        for task in ("ranking", "reranking"):
            for target_users in (20, 100, 1000):
                manifest_status = _current_manifest_status(manifests, label, task, target_users)
                readiness, blocker = _readiness_for(label, data_status, manifest_status, target_users)
                rows.append(
                    {
                        "dataset": label,
                        "task": task,
                        "target_users": target_users,
                        "candidate_pool_size": 20,
                        "required_positive_per_user": 1,
                        "required_negative_per_user": 19,
                        "current_manifest_status": manifest_status,
                        "current_data_status": data_status,
                        "readiness_class": readiness,
                        "blocking_issue": blocker,
                        "recommended_next_phase": "E2" if readiness != "blocked" else "E1",
                        "notes": _readiness_notes(label, task, target_users, readiness),
                    }
                )
    return pd.DataFrame(rows, columns=SCALEUP_READINESS_COLUMNS)


def build_call_estimates(
    dataset_summary: pd.DataFrame | None = None,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Build no-inference local and paper-matrix call estimates from inventory status."""

    dataset_summary = dataset_summary if dataset_summary is not None else build_dataset_summary(project_root=project_root)
    valid_labels = _valid_scaleup_dataset_labels(dataset_summary)
    valid_dataset_text = "; ".join(valid_labels)
    lastfm_dataset_text = valid_dataset_text + "; LastFM"
    rows = [
        _call("local_Qwen2_Mistral_20_users_ranking", "ML-1M", 1, 20, "Qwen2-7B; Mistral-7B", 2, "aligned_prompt_placeholder", 1, "ranking", 1, 1, "D5 shows local ML-1M 20-user diagnostics exist.", "diagnostic_subset", "Local diagnostic scenario; E1 does not run LLMs."),
        _call("local_Qwen2_Mistral_100_users_ranking", "ML-1M", 1, 100, "Qwen2-7B; Mistral-7B", 2, "aligned_prompt_placeholder", 1, "ranking", 1, 1, "Inventory shows ML-1M data exists; 100-user manifest still needs E2 construction.", "needs_manifest_construction", "Pilot scenario only."),
        _call("local_Qwen2_Mistral_1000_users_ranking_ML1M", "ML-1M", 1, 1000, "Qwen2-7B; Mistral-7B", 2, "aligned_prompt_placeholder", 1, "ranking", 1, 1, "ML-1M reconstructed data exists; paper-like 20-candidate 1000-user manifest not complete.", "needs_manifest_construction", "Local 1000-user ML-1M ranking estimate."),
        _call("local_Qwen2_Mistral_1000_users_ranking_3_valid_datasets", valid_dataset_text, len(valid_labels), 1000, "Qwen2-7B; Mistral-7B", 2, "aligned_prompt_placeholder", 1, "ranking", 1, 1, "Inventory supports three valid reconstructed datasets: ML-1M, Beauty, Sports.", "needs_manifest_construction", "Excludes LastFM."),
        _call("local_Qwen2_Mistral_1000_users_ranking_plus_reranking_3_valid_datasets", valid_dataset_text, len(valid_labels), 1000, "Qwen2-7B; Mistral-7B", 2, "aligned_prompt_placeholder", 1, "ranking; reranking", 2, 1, "Inventory supports three valid reconstructed datasets but manifests need E2.", "needs_manifest_construction", "Ranking plus reranking local estimate."),
        _call("paper_7LLM_1000_users_3_valid_datasets_ranking", valid_dataset_text, len(valid_labels), 1000, "GPT-3.5; GPT-4o-mini; GPT-4o; Claude 3 Haiku; Llama3-70B; Qwen2-7B; Mistral-7B", 7, "aligned_prompt_placeholder", 1, "ranking", 1, 1, "Inventory supports three valid datasets; model availability remains conditional.", "conditional/API-dependent", "Paper LLM model matrix ranking estimate."),
        _call("paper_7LLM_1000_users_3_valid_datasets_ranking_plus_reranking", valid_dataset_text, len(valid_labels), 1000, "GPT-3.5; GPT-4o-mini; GPT-4o; Claude 3 Haiku; Llama3-70B; Qwen2-7B; Mistral-7B", 7, "aligned_prompt_placeholder", 1, "ranking; reranking", 2, 1, "Inventory supports three valid datasets; reranking manifests need E2.", "conditional/API-dependent", "Paper LLM model matrix ranking plus reranking estimate."),
        _call("paper_7LLM_3_prompt_strategies_1000_users_3_valid_datasets_ranking", valid_dataset_text, len(valid_labels), 1000, "GPT-3.5; GPT-4o-mini; GPT-4o; Claude 3 Haiku; Llama3-70B; Qwen2-7B; Mistral-7B", 7, "OpenP5-like; LLMRank-like; LLMSRec/ICL", 3, "ranking", 1, 1, "E0/E1 inventory identifies three prompt families; exact prompt alignment remains E4.", "conditional/API-dependent", "Prompt-matrix ranking estimate."),
        _call("paper_7LLM_3_prompt_strategies_1000_users_3_valid_datasets_ranking_plus_reranking", valid_dataset_text, len(valid_labels), 1000, "GPT-3.5; GPT-4o-mini; GPT-4o; Claude 3 Haiku; Llama3-70B; Qwen2-7B; Mistral-7B", 7, "OpenP5-like; LLMRank-like; LLMSRec/ICL", 3, "ranking; reranking", 2, 1, "E0/E1 inventory identifies the full feasibility scenario for three valid datasets.", "conditional/API-dependent", "Full paper-like feasibility estimate excluding LastFM."),
        _call("local_Qwen2_Mistral_1000_users_ranking_4_datasets_LastFM_conditional", lastfm_dataset_text, len(valid_labels) + 1, 1000, "Qwen2-7B; Mistral-7B", 2, "aligned_prompt_placeholder", 1, "ranking", 1, 1, "Adds LastFM only as conditional because valid utility artifacts are missing.", "conditional", "Do not run LastFM until valid artifacts exist."),
        _call("paper_7LLM_1000_users_4_datasets_ranking_plus_reranking_LastFM_conditional", lastfm_dataset_text, len(valid_labels) + 1, 1000, "GPT-3.5; GPT-4o-mini; GPT-4o; Claude 3 Haiku; Llama3-70B; Qwen2-7B; Mistral-7B", 7, "aligned_prompt_placeholder", 1, "ranking; reranking", 2, 1, "Adds LastFM only as conditional; closed-source models remain API-dependent.", "conditional", "Conditional four-dataset paper matrix estimate."),
        _call("paper_7LLM_3_prompt_strategies_1000_users_4_datasets_ranking_plus_reranking_LastFM_conditional", lastfm_dataset_text, len(valid_labels) + 1, 1000, "GPT-3.5; GPT-4o-mini; GPT-4o; Claude 3 Haiku; Llama3-70B; Qwen2-7B; Mistral-7B", 7, "OpenP5-like; LLMRank-like; LLMSRec/ICL", 3, "ranking; reranking", 2, 1, "Same full matrix with LastFM marked conditional.", "conditional", "Conditional only; not an executable recommendation."),
    ]
    return pd.DataFrame(rows, columns=CALL_ESTIMATE_COLUMNS)


def build_runtime_estimates(call_estimates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Estimate runtime using assumed seconds per call; E1 does not measure inference."""

    call_estimates = call_estimates if call_estimates is not None else build_call_estimates()
    rows: list[dict[str, object]] = []
    for _, scenario in call_estimates.iterrows():
        calls = int(scenario["estimated_total_calls"])
        for seconds_per_call in (10, 20, 45, 90):
            estimated_seconds = calls * seconds_per_call
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "estimated_total_calls": calls,
                    "assumed_seconds_per_call": seconds_per_call,
                    "estimated_seconds": estimated_seconds,
                    "estimated_minutes": round(estimated_seconds / 60, 6),
                    "estimated_hours": round(estimated_seconds / 3600, 6),
                    "estimated_days": round(estimated_seconds / 86400, 6),
                    "feasibility_note": "Planning estimate only; E1 did not run inference or benchmark local models.",
                    "measured_or_assumed": "assumed_planning_estimate",
                }
            )
    return pd.DataFrame(rows, columns=RUNTIME_ESTIMATE_COLUMNS)


def build_next_actions() -> pd.DataFrame:
    """Build concrete next actions leading into Plan E2 and later phases."""

    rows = [
        _action("E1_A01", "E2", 1, "validate ML-1M 1000 eligible test users", "ML-1M is the strongest current local LLM diagnostic base and needs paper-like eligibility validation.", "E1 inventory", "medium", "results/tables/plan_e2_ml1m_eligible_users.csv", "Do not run LLMs during validation."),
        _action("E1_A02", "E2", 2, "validate Beauty 1000 eligible test users", "Beauty is one of the three valid reconstructed datasets for scale-up.", "E1 inventory", "medium", "results/tables/plan_e2_beauty_eligible_users.csv", "Use reconstructed Plan C data only."),
        _action("E1_A03", "E2", 3, "validate Sports 1000 eligible test users", "Sports is one of the three valid reconstructed datasets for scale-up.", "E1 inventory", "medium", "results/tables/plan_e2_sports_eligible_users.csv", "Use reconstructed Plan C data only."),
        _action("E1_A04", "E2", 4, "keep LastFM conditional unless valid artifacts exist", "LastFM lacks valid comparable utility-evaluation artifacts.", "E1 inventory", "high", "results/tables/plan_e2_lastfm_validity_decision.csv", "Do not run LastFM or fabricate utility metrics."),
        _action("E1_A05", "E2", 5, "build 1000-user ranking manifests", "Paper-like scale requires deterministic 20-candidate ranking manifests.", "eligible user validation", "medium", "results/tables/plan_e2_ranking_manifest_summary.csv", "Preserve 20 candidates per user."),
        _action("E1_A06", "E2", 6, "build 1000-user reranking manifests", "Re-ranking needs separate paper-like candidate-order manifests.", "ranking manifest construction", "medium", "results/tables/plan_e2_reranking_manifest_summary.csv", "Keep task separation explicit."),
        _action("E1_A07", "E2", 7, "preserve 20-candidate pool", "Paper LLM condition uses 20 items per user.", "manifest construction", "low", "manifest validation table", "Validate every user has exactly 20 candidates."),
        _action("E1_A08", "E2", 8, "preserve 1 positive + 19 negatives", "Paper candidate condition requires one positive target and nineteen negatives.", "manifest construction", "low", "manifest validation table", "Validate positives and negatives separately."),
        _action("E1_A09", "E2", 9, "add deterministic seed handling", "Candidate sampling and user selection must be reproducible.", "manifest construction design", "medium", "config and manifest metadata", "Do not rely on implicit random state."),
        _action("E1_A10", "E5", 10, "design resumable local LLM runner for E5", "Full local scale-up will need resumability and raw-output hygiene.", "E2 manifests and E4 prompts", "high", "src/experiments/run_plan_e_local_llm_scaleup.py", "Design only until E5; do not run LLMs in E1/E2."),
        _action("E1_A11", "E6", 11, "keep API pricing audit for E6, not E1", "Closed-source costs require current official price checks later.", "E1 estimates", "high", "report/plan_e_api_availability_cost_audit.md", "E1 uses no current price lookup."),
        _action("E1_A12", "E5", 12, "do not run LLMs until E5", "E1 and E2 are audit/manifest phases.", "explicit user approval", "high", "approval record before inference", "No Ollama, Qwen2, Mistral, API, or model-pull actions before approval."),
    ]
    return pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS)


def build_all_tables(project_root: Path = PROJECT_ROOT) -> dict[str, pd.DataFrame]:
    """Build every E1 inventory table without writing files."""

    artifacts = discover_relevant_artifacts(project_root)
    manifests = build_manifest_summary(artifacts, project_root)
    datasets = build_dataset_summary(artifacts, manifests, project_root)
    metrics = build_metric_summary(artifacts, project_root)
    hygiene = build_raw_output_hygiene(project_root)
    readiness = build_scaleup_readiness(datasets, manifests, project_root)
    calls = build_call_estimates(datasets, project_root)
    runtime = build_runtime_estimates(calls)
    next_actions = build_next_actions()
    return {
        "dataset_summary": datasets,
        "artifact_summary": artifacts,
        "manifest_summary": manifests,
        "metric_summary": metrics,
        "raw_output_hygiene": hygiene,
        "scaleup_readiness": readiness,
        "call_estimates": calls,
        "runtime_estimates": runtime,
        "next_actions": next_actions,
    }


def build_markdown_report(tables: dict[str, pd.DataFrame]) -> str:
    """Generate the E1 Markdown report content from inventory tables."""

    dataset_summary = tables["dataset_summary"]
    manifest_summary = tables["manifest_summary"]
    metric_summary = tables["metric_summary"]
    hygiene = tables["raw_output_hygiene"]
    readiness = tables["scaleup_readiness"]
    calls = tables["call_estimates"]
    runtime = tables["runtime_estimates"]
    next_actions = tables["next_actions"]
    full_scenario = calls.loc[
        calls["scenario_id"] == "paper_7LLM_3_prompt_strategies_1000_users_3_valid_datasets_ranking_plus_reranking"
    ].iloc[0]
    slow_runtime = runtime.loc[
        (runtime["scenario_id"] == full_scenario["scenario_id"])
        & (runtime["assumed_seconds_per_call"] == 90)
    ].iloc[0]
    lastfm_row = dataset_summary.loc[dataset_summary["dataset"] == "LastFM"].iloc[0]

    return f"""# Plan E1 Full-Data Inventory and Cost/Runtime Estimator

## Executive Summary

E1 is an inventory and estimation phase only. It reads repository artifacts, summarizes available datasets, manifests, metrics, raw-output hygiene, and scale-up readiness, and estimates call/runtime planning scenarios for future Plan E phases.

E1 performed no inference. E1 made no Ollama calls. E1 made no Qwen2/Mistral calls. E1 made no API calls. E1 performed no downloads. E1 performed no model pulls. E1 performed no training. E1 performed no LastFM run. E1 performed no Plan C modification. E1 performed no original_repo modification. E1 performed no destructive cleanup.

E1 does not prove that 1000-user paper-like evaluation is complete. E1 prepares E2 by identifying what manifests need to be built. No full exact reproduction is claimed.

## Dataset Inventory Summary

{_markdown_table(dataset_summary, ["dataset", "exists", "file_count", "estimated_users_if_available", "estimated_items_if_available", "estimated_interactions_if_available", "has_candidate_manifest", "has_20_user_diagnostic_manifest", "has_1000_user_manifest", "lastfm_valid_for_utility_evaluation"], max_rows=10)}

LastFM status remains {lastfm_row["lastfm_valid_for_utility_evaluation"]} for valid utility evaluation. LastFM remains conditional/blocked unless valid candidate, ground-truth, and split artifacts are found.

## Manifest Inventory Summary

{_markdown_table(manifest_summary, ["manifest_id", "dataset", "task", "rows", "user_count_if_available", "candidate_pool_size_if_available", "matches_paper_candidate_pool_20", "matches_1_positive_19_negative", "scale"], max_rows=18)}

The current repository contains 20-user D5 diagnostic manifests for ML-1M. Existing reconstructed Plan C candidate files are useful inputs, but their candidate pools are not the paper-like 20-item LLM manifests required for E2.

## Metric Inventory Summary

{_markdown_table(metric_summary, ["artifact_id", "phase", "dataset", "contains_hr_at_5", "contains_ndcg_at_5", "contains_validity", "contains_hallucination", "classification"], max_rows=18)}

The metric inventory represents Plan C reconstructed metrics, Plan D extended neural baselines, D5 local LLM diagnostics, and E0 audit outputs. Local Qwen2/Mistral status remains local diagnostic, not paper-exact.

## Raw-Output Hygiene Summary

{_markdown_table(hygiene, ["check_id", "check_name", "status", "count"], max_rows=20)}

Raw LLM outputs must remain under `results/raw_outputs/` and Git-ignored. E1 created no raw output and no JSONL file.

## Scale-Up Readiness Summary

{_markdown_table(readiness, ["dataset", "task", "target_users", "candidate_pool_size", "current_manifest_status", "current_data_status", "readiness_class", "blocking_issue"], max_rows=24)}

Readiness is strongest for existing ML-1M 20-user diagnostics. ML-1M, Amazon Beauty, and Amazon Sports need E2 paper-like manifest construction for 100-user pilots and 1000-user scale. LastFM remains conditional/blocked unless valid utility-evaluation artifacts are found.

## Local Call Estimate Interpretation

{_markdown_table(calls, ["scenario_id", "dataset_count", "users_per_dataset", "model_count", "prompt_strategy_count", "task_count", "estimated_total_calls", "classification"], max_rows=20)}

The largest three-valid-dataset paper-style scenario in E1 is `{full_scenario["scenario_id"]}` with `{full_scenario["estimated_total_calls"]}` estimated calls. This is a planning estimate only.

## Runtime Estimate Interpretation

Runtime estimates use assumed seconds-per-call values of 10, 20, 45, and 90. All rows are marked `assumed_planning_estimate`, not measured inference. At 90 seconds per call, `{full_scenario["scenario_id"]}` is estimated at `{slow_runtime["estimated_hours"]}` hours or `{slow_runtime["estimated_days"]}` days.

Closed-source model costs are not verified in E1. API pricing verification belongs to E6, not E1.

## Risks and Blockers

- No 1000-user paper-like 20-candidate manifests are complete yet.
- LastFM remains conditional/blocked unless valid utility artifacts are found.
- Closed-source/API-dependent models require E6 availability and pricing checks.
- Llama3-70B remains hardware/cloud/API-dependent.
- Local Qwen2/Mistral diagnostics are useful feasibility evidence but are not paper-exact.
- E1 does not run any experiment and does not validate utility at paper scale.

## Recommended E2 Scope

E2 should validate eligible users for ML-1M, Amazon Beauty, and Amazon Sports; keep LastFM conditional; build deterministic ranking and reranking manifests; preserve the 20-candidate pool; preserve 1 positive plus 19 negatives; and record deterministic seed metadata. E2 should still avoid LLM inference unless the project explicitly changes phase and approval scope.

## Claim-Safety Notes for Oral Defense

- Say E1 is inventory and planning, not reproduction.
- Say E1 does not complete 1000-user paper-like evaluation.
- Say Plan C remains reconstructed reproduction.
- Say Plan D local LLM diagnostics remain diagnostic_subset.
- Say strict candidate-ID-only prompting remains a project extension.
- Say no full exact reproduction is claimed.
- Say closed-source/API model costs are not verified in E1.
- Say API pricing verification belongs to E6.

## Generated Outputs

- `results/tables/plan_e_full_data_inventory_dataset_summary.csv`
- `results/tables/plan_e_full_data_inventory_artifact_summary.csv`
- `results/tables/plan_e_full_data_inventory_manifest_summary.csv`
- `results/tables/plan_e_full_data_inventory_metric_summary.csv`
- `results/tables/plan_e_full_data_inventory_raw_output_hygiene.csv`
- `results/tables/plan_e_full_data_inventory_scaleup_readiness.csv`
- `results/tables/plan_e_full_data_inventory_call_estimates.csv`
- `results/tables/plan_e_full_data_inventory_runtime_estimates.csv`
- `results/tables/plan_e_full_data_inventory_next_actions.csv`
- `report/plan_e_full_data_inventory.md`
"""


def write_inventory_outputs(
    tables: dict[str, pd.DataFrame] | None = None,
    paths: PlanEInventoryPaths | None = None,
) -> dict[str, pd.DataFrame]:
    """Write all E1 CSV/report outputs and return the generated tables."""

    tables = tables if tables is not None else build_all_tables()
    paths = paths or PlanEInventoryPaths()
    paths.dataset_summary.parent.mkdir(parents=True, exist_ok=True)
    paths.report.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(tables["dataset_summary"], paths.dataset_summary, DATASET_SUMMARY_COLUMNS)
    _write_csv(tables["artifact_summary"], paths.artifact_summary, ARTIFACT_SUMMARY_COLUMNS)
    _write_csv(tables["manifest_summary"], paths.manifest_summary, MANIFEST_SUMMARY_COLUMNS)
    _write_csv(tables["metric_summary"], paths.metric_summary, METRIC_SUMMARY_COLUMNS)
    _write_csv(tables["raw_output_hygiene"], paths.raw_output_hygiene, RAW_OUTPUT_HYGIENE_COLUMNS)
    _write_csv(tables["scaleup_readiness"], paths.scaleup_readiness, SCALEUP_READINESS_COLUMNS)
    _write_csv(tables["call_estimates"], paths.call_estimates, CALL_ESTIMATE_COLUMNS)
    _write_csv(tables["runtime_estimates"], paths.runtime_estimates, RUNTIME_ESTIMATE_COLUMNS)
    _write_csv(tables["next_actions"], paths.next_actions, NEXT_ACTION_COLUMNS)
    paths.report.write_text(build_markdown_report(tables), encoding="utf-8", newline="\n")
    return tables


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, columns=columns, lineterminator="\n")


def _candidate_manifest_paths(project_root: Path, artifacts: pd.DataFrame) -> list[Path]:
    candidates: set[Path] = set()
    for _, row in artifacts.iterrows():
        path = project_root / str(row["path"])
        relative = str(row["path"]).lower()
        if "artifact_manifest" in relative:
            continue
        if path.suffix.lower() != ".csv":
            continue
        name = path.name.lower()
        if name == "candidates.csv" or ("manifest" in name and "suggested_manifest" not in name):
            candidates.add(path)
    return sorted(candidates, key=lambda path: _relative(path, project_root))


def _manifest_stats(
    path: Path,
    project_root: Path,
    dataset_key: str | None,
    plan_c_stats: dict[str, dict[str, str]],
) -> dict[str, int | None]:
    relative = _relative(path, project_root)
    rows = _count_delimited_rows(path)
    stats = {
        "rows": rows,
        "user_count": None,
        "candidate_pool_size": None,
        "positive_items_per_user": None,
        "negative_items_per_user": None,
    }
    if path.name.lower() == "candidates.csv" and dataset_key in plan_c_stats:
        plan_c = plan_c_stats[dataset_key]
        users = _to_int(plan_c.get("users"))
        candidate_rows = _to_int(plan_c.get("candidate_rows"))
        ground_truth_rows = _to_int(plan_c.get("ground_truth_rows"))
        if users and candidate_rows:
            pool = candidate_rows // users if candidate_rows % users == 0 else None
            positives = ground_truth_rows // users if users and ground_truth_rows and ground_truth_rows % users == 0 else None
            stats.update(
                {
                    "rows": candidate_rows,
                    "user_count": users,
                    "candidate_pool_size": pool,
                    "positive_items_per_user": positives,
                    "negative_items_per_user": pool - positives if pool is not None and positives is not None else None,
                }
            )
            return stats
    if path.suffix.lower() != ".csv":
        return stats
    columns = _header_columns(path)
    try:
        frame = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return stats
    if "user_id" in frame.columns:
        stats["user_count"] = int(frame["user_id"].nunique())
    if "candidate_pool_size" in frame.columns:
        pool_values = _unique_ints(frame["candidate_pool_size"])
        stats["candidate_pool_size"] = pool_values[0] if len(pool_values) == 1 else None
    elif {"user_id", "candidate_item_id"}.issubset(frame.columns):
        pool_counts = frame.groupby("user_id", sort=False)["candidate_item_id"].count()
        stats["candidate_pool_size"] = int(pool_counts.iloc[0]) if len(pool_counts.unique()) == 1 else None
    elif "candidate_count" in frame.columns:
        pool_values = _unique_ints(frame["candidate_count"])
        stats["candidate_pool_size"] = pool_values[0] if len(pool_values) == 1 else None
    if "positive_item_id" in columns:
        stats["positive_items_per_user"] = 1
    elif "is_ground_truth" in frame.columns and "user_id" in frame.columns:
        positives = frame.assign(_is_gt=frame["is_ground_truth"].astype(str).str.lower().isin(["true", "1", "yes"]))
        pos_counts = positives.groupby("user_id", sort=False)["_is_gt"].sum()
        stats["positive_items_per_user"] = int(pos_counts.iloc[0]) if len(pos_counts.unique()) == 1 else None
    if stats["candidate_pool_size"] is not None and stats["positive_items_per_user"] is not None:
        stats["negative_items_per_user"] = int(stats["candidate_pool_size"]) - int(stats["positive_items_per_user"])
    if "reranking" in relative.lower() and stats["positive_items_per_user"] is None:
        stats["positive_items_per_user"] = 1
        if stats["candidate_pool_size"] is not None:
            stats["negative_items_per_user"] = int(stats["candidate_pool_size"]) - 1
    return stats


def _metric_inventory_paths(project_root: Path, artifacts: pd.DataFrame) -> list[Path]:
    paths: set[Path] = set()
    for _, row in artifacts.iterrows():
        relative = str(row["path"])
        lower = relative.lower()
        if _is_excluded_from_discovery(relative):
            continue
        if lower.startswith("results/metrics/") and lower.endswith(".csv"):
            paths.add(project_root / relative)
        elif lower.startswith("results/tables/") and lower.endswith(".csv") and (
            lower.startswith("results/tables/plan_c")
            or lower.startswith("results/tables/plan_d")
            or lower.startswith("results/tables/plan_e_")
            or "summary" in lower
            or "metrics" in lower
        ):
            paths.add(project_root / relative)
        elif lower.startswith("report/") and lower.endswith(".md") and (
            "plan_c" in lower or "plan_d" in lower or "plan_e" in lower
        ):
            paths.add(project_root / relative)
    return sorted(paths, key=lambda path: _relative(path, project_root))


def _plan_c_dataset_stats(project_root: Path) -> dict[str, dict[str, str]]:
    path = project_root / "results" / "tables" / "plan_c_reconstructed_dataset_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {str(row["dataset"]): {column: str(row[column]) for column in frame.columns} for _, row in frame.iterrows()}


def _primary_dataset_paths(project_root: Path, dataset_key: str) -> list[str]:
    candidates = [
        project_root / "data" / "reconstructed" / "plan_c" / dataset_key,
        project_root / "data" / "processed" / dataset_key,
        project_root / "data" / "raw" / dataset_key,
    ]
    if dataset_key == "ml1m":
        candidates.append(project_root / "data" / "raw" / "ml1m")
    relative_paths: list[str] = []
    for path in candidates:
        relative = _relative(path, project_root)
        if path.exists() and relative not in relative_paths:
            relative_paths.append(relative)
    return relative_paths


def _split_status(project_root: Path, dataset_key: str, stats: dict[str, str]) -> dict[str, str]:
    if stats:
        train = _to_int(stats.get("train_interactions")) or 0
        ground_truth = _to_int(stats.get("ground_truth_rows")) or 0
        return {
            "train": _bool_text(train > 0),
            "validation": "false",
            "test": _bool_text(ground_truth > 0),
        }
    for base in (
        project_root / "data" / "reconstructed" / "plan_c" / dataset_key,
        project_root / "data" / "processed" / dataset_key,
    ):
        interactions = base / "interactions.csv"
        if interactions.exists():
            values = _split_values(interactions)
            return {
                "train": _bool_text("train" in values or "observed" in values),
                "validation": _bool_text("validation" in values or "val" in values),
                "test": _bool_text("test" in values),
            }
    return {"train": "false", "validation": "false", "test": "false"}


def _dataset_estimated_rows(stats: dict[str, str], project_root: Path, dataset_key: str) -> str:
    if stats:
        total = sum(
            _to_int(stats.get(column)) or 0
            for column in ("raw_interactions", "catalog_rows", "ground_truth_rows", "candidate_rows")
        )
        return str(total)
    total = 0
    for path in (project_root / "data" / "processed" / dataset_key).glob("*.csv"):
        total += _count_delimited_rows(path)
    return str(total) if total else ""


def _has_manifest_like_candidate(
    dataset_key: str,
    dataset_artifacts: pd.DataFrame,
    manifest_rows: pd.DataFrame,
    project_root: Path,
) -> bool:
    if not manifest_rows.empty:
        return True
    if not dataset_artifacts.empty and any(str(value).endswith("candidates.csv") for value in dataset_artifacts["path"]):
        return True
    return (project_root / "data" / "reconstructed" / "plan_c" / dataset_key / "candidates.csv").exists()


def _manifest_scale_present(manifest_rows: pd.DataFrame, scale: str) -> bool:
    return not manifest_rows.empty and scale in set(manifest_rows["scale"])


def _has_1000_user_paper_like_manifest(manifest_rows: pd.DataFrame) -> bool:
    if manifest_rows.empty:
        return False
    matches = manifest_rows.loc[
        (manifest_rows["matches_paper_candidate_pool_20"] == "true")
        & (pd.to_numeric(manifest_rows["user_count_if_available"], errors="coerce") >= 1000)
    ]
    return not matches.empty


def _dataset_notes(dataset_key: str, has_20_user_diagnostic: bool, has_1000_user_manifest: bool) -> str:
    if dataset_key == "lastfm":
        return "LastFM remains extension_only_no_valid_utility_evaluation or blocked unless valid split/candidate/ground-truth artifacts are found."
    if has_1000_user_manifest:
        return "A paper-like 1000-user manifest appears to exist; validate before use."
    if has_20_user_diagnostic:
        return "20-user diagnostic manifest exists; 1000-user paper-like manifest still needs E2 construction."
    return "Dataset artifacts exist, but paper-like 20-candidate 1000-user manifests still need E2 construction."


def _current_manifest_status(manifests: pd.DataFrame, dataset_label: str, task: str, target_users: int) -> str:
    if manifests.empty:
        return "missing"
    task_rows = manifests.loc[(manifests["dataset"] == dataset_label) & (manifests["task"].isin([task, "ranking"]))]
    if task == "reranking":
        task_rows = manifests.loc[(manifests["dataset"] == dataset_label) & (manifests["task"] == "reranking")]
    if task_rows.empty:
        return "missing"
    numeric_users = pd.to_numeric(task_rows["user_count_if_available"], errors="coerce")
    paper_like = task_rows.loc[task_rows["matches_1_positive_19_negative"] == "true"]
    if target_users == 20 and not paper_like.empty and (pd.to_numeric(paper_like["user_count_if_available"], errors="coerce") >= 20).any():
        return "20_user_diagnostic_manifest_available"
    if target_users >= 1000 and not paper_like.empty and (pd.to_numeric(paper_like["user_count_if_available"], errors="coerce") >= target_users).any():
        return "paper_like_manifest_available"
    if (numeric_users >= target_users).any():
        return "nonpaper_candidate_manifest_available"
    return "needs_manifest_construction"


def _data_status(dataset_row: pd.Series) -> str:
    if dataset_row["dataset"] == "LastFM":
        return "conditional_or_blocked_for_valid_utility"
    if dataset_row["exists"] == "true" and dataset_row["has_test_split"] == "true":
        return "reconstructed_data_available"
    if dataset_row["exists"] == "true":
        return "partial_data_available"
    return "missing"


def _readiness_for(dataset_label: str, data_status: str, manifest_status: str, target_users: int) -> tuple[str, str]:
    if dataset_label == "LastFM":
        return "blocked", "LastFM lacks valid comparable utility-evaluation candidate/ground-truth artifacts."
    if data_status == "missing":
        return "needs_dataset_validation", "Dataset files are missing."
    if manifest_status in {"20_user_diagnostic_manifest_available", "paper_like_manifest_available"}:
        return "ready", ""
    if manifest_status == "nonpaper_candidate_manifest_available":
        return "needs_manifest_construction", "Existing candidates do not match the paper 20-item, 1-positive/19-negative LLM manifest."
    if target_users == 20 and data_status == "reconstructed_data_available":
        return "partially_ready", "Paper-like diagnostic manifest needs construction for this dataset/task."
    return "needs_manifest_construction", "Paper-like manifest is not yet available."


def _readiness_notes(dataset_label: str, task: str, target_users: int, readiness: str) -> str:
    if dataset_label == "LastFM":
        return "Keep conditional/blocked unless valid LastFM artifacts are found."
    if readiness == "ready":
        return f"{dataset_label} has a current {target_users}-user {task} manifest status sufficient for this planning target."
    return f"{dataset_label} {task} target {target_users} requires E2 validation and manifest construction before inference."


def _valid_scaleup_dataset_labels(dataset_summary: pd.DataFrame) -> list[str]:
    valid = dataset_summary.loc[
        (dataset_summary["dataset"].isin(["ML-1M", "Amazon Beauty", "Amazon Sports"]))
        & (dataset_summary["exists"] == "true")
    ]["dataset"].tolist()
    return valid or ["ML-1M", "Amazon Beauty", "Amazon Sports"]


def _call(
    scenario_id: str,
    datasets: str,
    dataset_count: int,
    users_per_dataset: int,
    models: str,
    model_count: int,
    prompt_strategies: str,
    prompt_strategy_count: int,
    tasks: str,
    task_count: int,
    calls_per_user_task: int,
    inventory_basis: str,
    classification: str,
    notes: str,
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "datasets": datasets,
        "dataset_count": dataset_count,
        "users_per_dataset": users_per_dataset,
        "models": models,
        "model_count": model_count,
        "prompt_strategies": prompt_strategies,
        "prompt_strategy_count": prompt_strategy_count,
        "tasks": tasks,
        "task_count": task_count,
        "calls_per_user_task": calls_per_user_task,
        "estimated_total_calls": dataset_count * users_per_dataset * model_count * prompt_strategy_count * task_count * calls_per_user_task,
        "inventory_basis": inventory_basis,
        "classification": classification,
        "notes": notes,
    }


def _action(
    action_id: str,
    phase: str,
    priority: int,
    action: str,
    reason: str,
    depends_on: str,
    risk_level: str,
    expected_artifact: str,
    notes: str,
) -> dict[str, object]:
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


def _hygiene(check_id: str, check_name: str, status: str, count: int, paths: Iterable[str], notes: str) -> dict[str, object]:
    return {
        "check_id": check_id,
        "check_name": check_name,
        "status": status,
        "count": count,
        "paths": "; ".join(paths),
        "notes": notes,
    }


def _forbidden_paths(paths: Iterable[str]) -> list[str]:
    forbidden: list[str] = []
    for path in paths:
        lower = path.lower()
        if (
            "raw_outputs" in lower
            or lower.endswith(".jsonl")
            or "checkpoint" in lower
            or "cache" in lower
            or lower.endswith((".pt", ".pth", ".bin", ".onnx"))
            or ".venv" in lower
            or _looks_like_model_file(lower)
        ):
            forbidden.append(path)
    return forbidden


def _looks_like_model_file(path: str) -> bool:
    name = Path(path).name.lower()
    return any(token in name for token in ("model", "weights")) and name.endswith((".pt", ".pth", ".bin", ".onnx", ".safetensors"))


def _plan_c_status_lines(project_root: Path) -> list[str]:
    status = _git_lines(["git", "status", "--short"], project_root)
    return [line for line in status if "plan_c" in line.lower() or "plan-c" in line.lower()]


def _git_lines(command: list[str], project_root: Path) -> list[str]:
    return common_git.git_lines_unavailable_on_error(command, project_root, normalize_paths=True)


def _is_excluded_from_discovery(relative: str) -> bool:
    lower = relative.lower().replace("\\", "/")
    return any(lower.startswith(prefix) for prefix in EXCLUDED_DISCOVERY_PREFIXES)


def _should_skip_path(path: Path) -> bool:
    skip_parts = {".git", ".pytest_cache", "__pycache__", ".venv"}
    return any(part in skip_parts for part in path.parts)


def _is_project_relevant(relative: str) -> bool:
    lower = relative.lower()
    return any(token in lower for token in ("plan_c", "plan_d", "plan_e", "baseline", "metric", "summary", "manifest", "reproduc"))


def _dataset_key_for_path(relative: str) -> str | None:
    lower = relative.lower().replace("\\", "/")
    for dataset in DATASETS:
        if any(alias in lower for alias in dataset["aliases"]):
            return dataset["key"]
    return None


def _dataset_label(dataset_key: str | None) -> str:
    for dataset in DATASETS:
        if dataset["key"] == dataset_key:
            return dataset["label"]
    return "unknown"


def _artifact_type(relative: str) -> str:
    lower = relative.lower()
    name = Path(relative).name.lower()
    if lower.startswith("configs/"):
        return "config"
    if lower.startswith("results/metrics/"):
        return "metric"
    if lower.startswith("results/tables/"):
        return "result_table"
    if lower.startswith("results/raw_outputs/"):
        return "raw_output"
    if lower.startswith("report/"):
        return "report"
    if lower.startswith("original_repo/"):
        return "original_repo_artifact"
    if name == "candidates.csv" or ("manifest" in name and "artifact_manifest" not in name):
        return "candidate_manifest"
    if "reconstructed" in lower:
        return "reconstructed_data"
    if "processed" in lower:
        return "processed_data"
    if "raw" in lower or "external" in lower:
        return "raw_or_external_data"
    return "artifact"


def _source_area(relative: str) -> str:
    return relative.split("/", 1)[0]


def _phase_for_path(relative: str) -> str:
    lower = relative.lower()
    if "plan_e_paper_condition" in lower or "plan_e_scaleup" in lower or "plan_e_exactness" in lower or "plan_e_missing" in lower or "plan_e_api" in lower:
        return "E0"
    if "plan_d" in lower or "d5" in lower or "qwen" in lower or "mistral" in lower:
        return "Plan D/D5"
    if "plan_c" in lower or "reconstructed/plan_c" in lower:
        return "Plan C"
    if lower.startswith("data/processed/ml1m") or lower.startswith("results/metrics/ml1m_"):
        return "Plan B/C support"
    return "unknown"


def _classification_for_path(relative: str) -> str:
    lower = relative.lower()
    if "lastfm" in lower:
        return "extension_only_or_blocked"
    if "plan_e" in lower:
        return "audit"
    if "plan_d" in lower:
        return "extended_subset_or_diagnostic_subset"
    if "plan_c" in lower or "reconstructed" in lower:
        return "reconstructed reproduction"
    return "supporting_artifact"


def _classification_for_metric(relative: str) -> str:
    lower = relative.lower()
    if "plan_e" in lower:
        return "audit"
    if "qwen" in lower or "mistral" in lower or "llm" in lower:
        return "diagnostic_subset"
    if "plan_d" in lower:
        return "extended_subset"
    if "plan_c" in lower or lower.startswith("results/metrics/ml1m_"):
        return "reconstructed reproduction"
    return "supporting_artifact"


def _manifest_task(relative: str) -> str:
    lower = relative.lower()
    if "rerank" in lower:
        return "reranking"
    if (
        "ranking" in lower
        or "candidate" in lower
        or "history_length" in lower
        or "position_bias" in lower
        or "strict_prompt" in lower
        or "qwen_vs_mistral" in lower
    ):
        return "ranking"
    return "unknown"


def _manifest_scale(users: int | None, pool: int | None) -> str:
    if users is None:
        return "unknown"
    if users == 20 and pool == 20:
        return "20_user_diagnostic"
    if users >= 1000 and pool == 20:
        return "1000_user_paper_like"
    if users >= 1000:
        return "1000_user_or_larger_nonpaper_pool"
    return "smaller_or_unknown"


def _manifest_notes(relative: str, dataset_key: str | None, matches_pool: bool, matches_pos_neg: bool) -> str:
    if dataset_key == "lastfm":
        return "LastFM manifest-like artifact is conditional unless valid utility evaluation artifacts are confirmed."
    if matches_pool and matches_pos_neg:
        return "Matches 20-candidate and 1-positive/19-negative condition at this manifest scale."
    if Path(relative).name.lower() == "candidates.csv":
        return "Candidate file is useful input but does not by itself establish a paper-like 20-item LLM manifest."
    return "Manifest-like artifact found; validate before any scale-up."


def _metric_notes(relative: str) -> str:
    lower = relative.lower()
    if "plan_e" in lower:
        return "Audit output; not an experiment metric."
    if "qwen" in lower or "mistral" in lower:
        return "Local LLM diagnostic_subset artifact."
    if "plan_d" in lower:
        return "Plan D extended baseline or diagnostic artifact."
    if "plan_c" in lower:
        return "Plan C reconstructed reproduction artifact."
    return "Supporting result artifact."


def _artifact_notes(relative: str) -> str:
    lower = relative.lower()
    if "original_repo" in lower:
        return "Read-only inventory only; original_repo is not modified."
    if "raw_outputs" in lower:
        return "Raw-output area; raw JSONL responses must remain ignored."
    if "plan_e_paper_condition" in lower:
        return "Plan E0 audit artifact."
    return "Repository artifact discovered for E1 inventory."


def _task_for_metric_path(relative: str, text: str) -> str:
    lower = relative.lower()
    if "rerank" in lower or "re-ranking" in text:
        return "reranking"
    if "ranking" in lower or "hit_rate" in text or "ndcg" in text:
        return "ranking"
    if "beyond" in lower or "hallucination" in text:
        return "beyond_utility"
    return "audit_or_summary"


def _model_or_method_for_path(relative: str, columns: list[str], text: str) -> str:
    lower = relative.lower()
    for token in ("qwen2", "mistral", "bprmf", "lightgcn", "sasrec", "gru4rec", "mostpop", "random", "bm25"):
        if token in lower or token in text:
            return token
    if "model" in columns or "method" in columns:
        return "see_file_columns"
    return "n/a"


def _contains_hr_at_5(columns: list[str], text: str) -> bool:
    joined = " ".join(columns).lower() + " " + text
    return "hr@5" in joined or "hr_at_5" in joined or "hit_rate_at_5" in joined or ("hit_rate" in joined and ",5," in joined)


def _contains_ndcg_at_5(columns: list[str], text: str) -> bool:
    joined = " ".join(columns).lower() + " " + text
    return "ndcg@5" in joined or "ndcg_at_5" in joined or ("ndcg" in joined and ",5," in joined)


def _count_delimited_rows(path: Path) -> int | str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if not suffixes or suffixes[-1] not in {".csv", ".tsv"}:
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            line_count = sum(1 for _ in handle)
    except OSError:
        return ""
    return max(line_count - 1, 0)


def _header_columns(path: Path) -> list[str]:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            return [column.strip() for column in next(reader, [])]
    except OSError:
        return []


def _read_text_prefix(path: Path, limit: int = 200_000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _split_values(path: Path) -> set[str]:
    try:
        frame = pd.read_csv(path, usecols=["split"], dtype=str)
    except Exception:
        return set()
    return {value.lower() for value in frame["split"].dropna().unique()}


def _unique_ints(series: pd.Series) -> list[int]:
    values: list[int] = []
    for value in series.dropna().unique():
        parsed = _to_int(value)
        if parsed is not None:
            values.append(parsed)
    return sorted(values)


def _stat_value(stats: dict[str, str], key: str) -> str:
    return stats.get(key, "") if stats else ""


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _blank_if_none(value: object) -> object:
    return "" if value is None else value


def _artifact_id(relative: str) -> str:
    safe = relative.replace("\\", "/").replace("/", "__").replace(".", "_").replace("-", "_")
    return safe[:180]


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str], *, max_rows: int) -> str:
    selected_columns = list(columns)
    shown = frame[selected_columns].head(max_rows).copy()
    lines = [
        "| " + " | ".join(selected_columns) + " |",
        "| " + " | ".join("---" for _ in selected_columns) + " |",
    ]
    for _, row in shown.iterrows():
        values = [_escape_markdown(str(row[column])) for column in selected_columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append("| " + " | ".join("..." for _ in selected_columns) + " |")
    return "\n".join(lines)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"

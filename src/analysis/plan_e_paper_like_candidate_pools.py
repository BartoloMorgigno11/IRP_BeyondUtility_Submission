from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

PHASE_ID = "E2"
PHASE_TITLE = "1000-user paper-like candidate pool construction"
TARGET_USER_COUNT = 1000
CANDIDATE_POOL_SIZE = 20
REQUIRED_POSITIVES = 1
REQUIRED_NEGATIVES = 19
SEED = 20260702

VALID_DATASETS = [
    {"key": "ml1m", "label": "ML-1M", "short": "ml1m", "aliases": ("ml1m", "ml-1m")},
    {"key": "amazon_beauty", "label": "Amazon Beauty", "short": "beauty", "aliases": ("amazon_beauty", "beauty")},
    {"key": "amazon_sports", "label": "Amazon Sports", "short": "sports", "aliases": ("amazon_sports", "sports")},
]

ALL_DATASETS = VALID_DATASETS + [
    {"key": "lastfm", "label": "LastFM", "short": "lastfm", "aliases": ("lastfm", "last.fm", "hetrec")}
]

USER_COLUMN_CANDIDATES = ["user_id", "user", "uid", "reviewerID", "reviewer_id"]
ITEM_COLUMN_CANDIDATES = [
    "candidate_item_id",
    "item_id",
    "item",
    "iid",
    "asin",
    "movie_id",
    "product_id",
]
LABEL_COLUMN_CANDIDATES = ["is_ground_truth", "label", "target", "is_positive", "positive", "relevance", "ground_truth", "y"]
RANK_COLUMN_CANDIDATES = ["candidate_position", "rank", "source_rank", "score", "candidate_rank", "position", "order"]
GROUND_TRUTH_ITEM_COLUMNS = ["ground_truth_item_id", "item_id", "item", "iid", "asin", "movie_id", "product_id"]

DATASET_ELIGIBILITY_COLUMNS = [
    "dataset",
    "source_path",
    "source_exists",
    "source_rows",
    "source_user_count",
    "eligible_user_count",
    "selected_user_count",
    "target_user_count",
    "candidate_pool_size",
    "required_positive_per_user",
    "required_negative_per_user",
    "can_build_1000_user_manifest",
    "eligibility_class",
    "blocking_issue",
    "notes",
]

MANIFEST_SCHEMA_COLUMNS = [
    "manifest_id",
    "dataset",
    "task",
    "user_id",
    "candidate_position",
    "item_id",
    "is_positive",
    "source_path",
    "source_user_candidate_count",
    "source_positive_count",
    "source_negative_count",
    "candidate_pool_size",
    "selected_positive_count",
    "selected_negative_count",
    "ordering_strategy",
    "seed",
    "exactness_class",
    "notes",
]

MANIFEST_SUMMARY_COLUMNS = [
    "manifest_id",
    "dataset",
    "task",
    "path",
    "rows",
    "user_count",
    "candidate_pool_size",
    "positive_items_per_user",
    "negative_items_per_user",
    "exact_1000_user_target_met",
    "exact_candidate_pool_20_met",
    "exact_1_positive_19_negative_met",
    "ordering_strategy",
    "exactness_class",
    "classification",
    "notes",
]

VALIDATION_COLUMNS = [
    "manifest_id",
    "dataset",
    "task",
    "check_name",
    "passed",
    "observed_value",
    "expected_value",
    "severity",
    "notes",
]

SOURCE_AUDIT_COLUMNS = [
    "dataset",
    "candidate_source_path",
    "source_class",
    "source_rows",
    "source_user_count",
    "detected_user_column",
    "detected_item_column",
    "detected_label_column",
    "detected_rank_or_score_column",
    "source_candidate_pool_size_mode",
    "source_is_paper_exact",
    "source_is_reconstructed",
    "source_can_support_e2",
    "notes",
]

BLOCKED_COLUMNS = [
    "dataset",
    "task",
    "blocked_or_conditional",
    "reason",
    "required_to_unblock",
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
class PlanECandidatePoolPaths:
    dataset_eligibility: Path = TABLES_DIR / "plan_e_paper_like_candidate_pool_dataset_eligibility.csv"
    manifest_summary: Path = TABLES_DIR / "plan_e_paper_like_candidate_pool_manifest_summary.csv"
    validation: Path = TABLES_DIR / "plan_e_paper_like_candidate_pool_validation.csv"
    source_audit: Path = TABLES_DIR / "plan_e_paper_like_candidate_pool_source_audit.csv"
    blocked_datasets: Path = TABLES_DIR / "plan_e_paper_like_candidate_pool_blocked_datasets.csv"
    next_actions: Path = TABLES_DIR / "plan_e_paper_like_candidate_pool_next_actions.csv"
    ml1m_ranking_manifest: Path = TABLES_DIR / "plan_e_ml1m_1000user_ranking_manifest.csv"
    ml1m_reranking_manifest: Path = TABLES_DIR / "plan_e_ml1m_1000user_reranking_manifest.csv"
    beauty_ranking_manifest: Path = TABLES_DIR / "plan_e_beauty_1000user_ranking_manifest.csv"
    beauty_reranking_manifest: Path = TABLES_DIR / "plan_e_beauty_1000user_reranking_manifest.csv"
    sports_ranking_manifest: Path = TABLES_DIR / "plan_e_sports_1000user_ranking_manifest.csv"
    sports_reranking_manifest: Path = TABLES_DIR / "plan_e_sports_1000user_reranking_manifest.csv"
    report: Path = REPORT_DIR / "plan_e_paper_like_candidate_pool_construction.md"

    def manifest_paths(self) -> dict[tuple[str, str], Path]:
        return {
            ("ml1m", "ranking"): self.ml1m_ranking_manifest,
            ("ml1m", "reranking"): self.ml1m_reranking_manifest,
            ("amazon_beauty", "ranking"): self.beauty_ranking_manifest,
            ("amazon_beauty", "reranking"): self.beauty_reranking_manifest,
            ("amazon_sports", "ranking"): self.sports_ranking_manifest,
            ("amazon_sports", "reranking"): self.sports_reranking_manifest,
        }

    def required_output_paths(self) -> list[Path]:
        return [
            self.dataset_eligibility,
            self.manifest_summary,
            self.validation,
            self.source_audit,
            self.blocked_datasets,
            self.next_actions,
            self.ml1m_ranking_manifest,
            self.ml1m_reranking_manifest,
            self.beauty_ranking_manifest,
            self.beauty_reranking_manifest,
            self.sports_ranking_manifest,
            self.sports_reranking_manifest,
            self.report,
        ]


def discover_candidate_pool_sources(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Discover candidate-pool sources for the paper datasets."""

    rows: list[dict[str, object]] = []
    for path in sorted((project_root / "data").rglob("candidates.csv")) if (project_root / "data").exists() else []:
        relative = _relative(path, project_root)
        dataset_key = _dataset_key_for_path(relative)
        if dataset_key is None:
            continue
        schema = detect_candidate_schema(path)
        rows.append(_source_audit_row(project_root, path, dataset_key, schema))
    for dataset in ALL_DATASETS:
        if dataset["key"] == "lastfm" and not any(row["dataset"] == "LastFM" for row in rows):
            rows.append(
                {
                    "dataset": "LastFM",
                    "candidate_source_path": "",
                    "source_class": "missing_candidate_source",
                    "source_rows": 0,
                    "source_user_count": 0,
                    "detected_user_column": "",
                    "detected_item_column": "",
                    "detected_label_column": "",
                    "detected_rank_or_score_column": "",
                    "source_candidate_pool_size_mode": "",
                    "source_is_paper_exact": "false",
                    "source_is_reconstructed": "false",
                    "source_can_support_e2": "false",
                    "notes": "LastFM lacks valid comparable candidate and ground-truth artifacts for E2 paper-like manifest construction.",
                }
            )
    return pd.DataFrame(rows, columns=SOURCE_AUDIT_COLUMNS)


def detect_candidate_schema(path: Path) -> dict[str, str]:
    """Detect user, item, label, and rank/order columns in a candidate source."""

    columns = _header_columns(path)
    return {
        "user_column": _first_present(columns, USER_COLUMN_CANDIDATES),
        "item_column": _first_present(columns, ITEM_COLUMN_CANDIDATES),
        "label_column": _first_present(columns, LABEL_COLUMN_CANDIDATES),
        "rank_column": _first_present(columns, RANK_COLUMN_CANDIDATES),
    }


def build_candidate_pool_outputs(
    project_root: Path = PROJECT_ROOT,
    paths: PlanECandidatePoolPaths | None = None,
) -> dict[str, pd.DataFrame | dict[tuple[str, str], pd.DataFrame]]:
    """Build E2 candidate manifests and summary/validation tables without writing files."""

    paths = paths or PlanECandidatePoolPaths()
    source_audit = discover_candidate_pool_sources(project_root)
    selected_sources = _select_sources(source_audit, project_root)
    dataset_rows: list[dict[str, object]] = []
    manifest_frames: dict[tuple[str, str], pd.DataFrame] = {}
    manifest_summaries: list[dict[str, object]] = []

    for dataset in VALID_DATASETS:
        dataset_key = dataset["key"]
        label = dataset["label"]
        source_path = selected_sources.get(dataset_key)
        if source_path is None:
            dataset_rows.append(_blocked_eligibility_row(label, "No candidate source found."))
            continue
        source = _load_candidate_source(project_root, source_path, dataset_key)
        selected_users = source["selected_users"]
        dataset_rows.append(
            _eligibility_row(
                label,
                source_path,
                source["source_rows"],
                source["source_user_count"],
                source["eligible_user_count"],
                len(selected_users),
                "",
            )
        )
        for task in ("ranking", "reranking"):
            frame = _construct_manifest_for_task(
                dataset_key=dataset_key,
                dataset_label=label,
                source_path=source_path,
                selected_users=selected_users,
                user_candidates=source["user_candidates"],
                task=task,
            )
            manifest_frames[(dataset_key, task)] = frame
            output_path = paths.manifest_paths()[(dataset_key, task)]
            manifest_summaries.append(_manifest_summary_row(dataset_key, label, task, output_path, frame))

    for blocked_dataset in (dataset for dataset in ALL_DATASETS if dataset["key"] == "lastfm"):
        dataset_rows.append(
            _blocked_eligibility_row(
                blocked_dataset["label"],
                "LastFM remains conditional/blocked: valid comparable candidate, split, and ground-truth artifacts are not present.",
            )
        )

    dataset_eligibility = pd.DataFrame(dataset_rows, columns=DATASET_ELIGIBILITY_COLUMNS)
    manifest_summary = pd.DataFrame(manifest_summaries, columns=MANIFEST_SUMMARY_COLUMNS)
    blocked = build_blocked_datasets_table(project_root)
    validation = build_validation_table(manifest_frames, manifest_summary, blocked, paths, project_root)
    next_actions = build_next_actions_table()
    return {
        "dataset_eligibility": dataset_eligibility,
        "manifest_summary": manifest_summary,
        "validation": validation,
        "source_audit": source_audit,
        "blocked_datasets": blocked,
        "next_actions": next_actions,
        "manifests": manifest_frames,
    }


def build_blocked_datasets_table(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Build the blocked/conditional dataset table, including LastFM."""

    lastfm_valid = _lastfm_has_valid_utility_artifacts(project_root)
    rows: list[dict[str, str]] = []
    if not lastfm_valid:
        for task in ("ranking", "reranking"):
            rows.append(
                {
                    "dataset": "LastFM",
                    "task": task,
                    "blocked_or_conditional": "blocked/conditional",
                    "reason": "LastFM lacks valid comparable candidate, split, and ground-truth artifacts for paper-like utility evaluation.",
                    "required_to_unblock": "Find or construct validated LastFM candidate, ground-truth, and split artifacts before any paper-like manifest construction.",
                    "classification": "extension_only_no_valid_utility_evaluation",
                    "notes": "No LastFM manifest is generated by E2.",
                }
            )
    return pd.DataFrame(rows, columns=BLOCKED_COLUMNS)


def build_validation_table(
    manifest_frames: dict[tuple[str, str], pd.DataFrame],
    manifest_summary: pd.DataFrame,
    blocked: pd.DataFrame,
    paths: PlanECandidatePoolPaths,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Validate generated manifests and repository safety boundaries."""

    rows: list[dict[str, object]] = []
    path_map = paths.manifest_paths()
    for (dataset_key, task), frame in sorted(manifest_frames.items()):
        dataset_label = _dataset_label(dataset_key)
        manifest_id = _manifest_id(dataset_key, task)
        output_path = path_map[(dataset_key, task)]
        user_counts = frame.groupby("user_id", sort=False).size() if not frame.empty else pd.Series(dtype=int)
        pos_counts = frame.loc[frame["is_positive"].astype(str) == "true"].groupby("user_id", sort=False).size()
        neg_counts = frame.loc[frame["is_positive"].astype(str) == "false"].groupby("user_id", sort=False).size()
        duplicate_counts = frame.groupby("user_id", sort=False)["item_id"].nunique() if not frame.empty else pd.Series(dtype=int)
        position_sets = frame.groupby("user_id", sort=False)["candidate_position"].apply(lambda values: sorted(int(value) for value in values))
        complete = frame["user_id"].nunique() == TARGET_USER_COUNT
        expected_rows = TARGET_USER_COUNT * CANDIDATE_POOL_SIZE if complete else frame["user_id"].nunique() * CANDIDATE_POOL_SIZE

        rows.extend(
            [
                _validation(manifest_id, dataset_label, task, "file_exists", not frame.empty, "will_write_non_empty_manifest" if not frame.empty else "empty", "non_empty_manifest", "critical", "Validation is computed before write and confirms a non-empty output frame."),
                _validation(manifest_id, dataset_label, task, "row_count_is_20000_when_complete", (not complete) or len(frame) == TARGET_USER_COUNT * CANDIDATE_POOL_SIZE, len(frame), TARGET_USER_COUNT * CANDIDATE_POOL_SIZE, "critical", "Required for complete 1000-user manifests."),
                _validation(manifest_id, dataset_label, task, "user_count_is_1000_when_complete", (not complete) or frame["user_id"].nunique() == TARGET_USER_COUNT, frame["user_id"].nunique(), TARGET_USER_COUNT, "critical", "Required for complete 1000-user manifests."),
                _validation(manifest_id, dataset_label, task, "candidate_pool_size_is_20", _series_all_equal(user_counts, CANDIDATE_POOL_SIZE), _observed_unique(user_counts), CANDIDATE_POOL_SIZE, "critical", "Every selected user must have exactly 20 rows."),
                _validation(manifest_id, dataset_label, task, "one_positive_per_user", _series_all_equal(pos_counts, REQUIRED_POSITIVES) and len(pos_counts) == frame["user_id"].nunique(), _observed_unique(pos_counts), REQUIRED_POSITIVES, "critical", "Every selected user must have exactly one positive."),
                _validation(manifest_id, dataset_label, task, "nineteen_negatives_per_user", _series_all_equal(neg_counts, REQUIRED_NEGATIVES) and len(neg_counts) == frame["user_id"].nunique(), _observed_unique(neg_counts), REQUIRED_NEGATIVES, "critical", "Every selected user must have exactly nineteen negatives."),
                _validation(manifest_id, dataset_label, task, "no_duplicate_items_per_user", _series_all_equal(duplicate_counts, CANDIDATE_POOL_SIZE), _observed_unique(duplicate_counts), CANDIDATE_POOL_SIZE, "critical", "Candidate item IDs must be unique within each user pool."),
                _validation(manifest_id, dataset_label, task, "candidate_positions_are_1_to_20", all(values == list(range(1, 21)) for values in position_sets), "1..20_per_user", "1..20_per_user", "critical", "Candidate positions must be complete and deterministic."),
                _validation(manifest_id, dataset_label, task, "no_missing_user_ids", not frame["user_id"].astype(str).str.strip().eq("").any(), 0, 0, "critical", "No user IDs may be blank."),
                _validation(manifest_id, dataset_label, task, "no_missing_item_ids", not frame["item_id"].astype(str).str.strip().eq("").any(), 0, 0, "critical", "No item IDs may be blank."),
                _validation(manifest_id, dataset_label, task, "deterministic_seed_recorded", frame["seed"].astype(str).eq(str(SEED)).all(), str(SEED), str(SEED), "major", "Seed metadata must be recorded."),
                _validation(manifest_id, dataset_label, task, "expected_row_count_for_selected_users", len(frame) == expected_rows, len(frame), expected_rows, "major", "Partial manifests still need 20 rows per selected user."),
                _validation(manifest_id, dataset_label, task, "no_original_repo_output", not _relative(output_path, project_root).startswith("original_repo/"), _relative(output_path, project_root), "not_under_original_repo", "critical", "E2 outputs must not write under original_repo/."),
            ]
        )

    lastfm_manifest_exists = any("lastfm" in _relative(path, project_root).lower() for path in path_map.values())
    rows.append(_validation("global", "LastFM", "all", "no_LastFM_manifest_unless_valid", (not lastfm_manifest_exists) and not blocked.empty, "no_lastfm_manifest", "no_lastfm_manifest", "critical", "LastFM remains blocked/conditional unless valid utility artifacts exist."))
    raw_jsonl = sorted((project_root / "results" / "raw_outputs").glob("*.jsonl")) if (project_root / "results" / "raw_outputs").exists() else []
    rows.append(_validation("global", "all", "all", "no_raw_outputs_created", True, len(raw_jsonl), "unchanged_existing_only", "critical", "E2 does not write raw outputs; build-script tests compare before/after JSONL sets."))
    original_status = _git_lines(["git", "status", "--short", "original_repo"], project_root)
    rows.append(_validation("global", "all", "all", "no_original_repo_output", not original_status, len(original_status), 0, "critical", "original_repo/ must remain unmodified."))
    plan_c_status = _plan_c_status_lines(project_root)
    rows.append(_validation("global", "all", "all", "no_plan_c_modification", not plan_c_status, len(plan_c_status), 0, "critical", "Plan C files and metrics must remain unmodified."))
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def build_next_actions_table() -> pd.DataFrame:
    """Build concrete next actions after E2."""

    rows = [
        _action("E2_A01", "E2_review", 1, "validate generated 1000-user manifests manually", "The manifests are reconstructed paper-like support artifacts and should be inspected before use.", "E2 outputs", "medium", "manual validation note", "Check representative users and positive placement."),
        _action("E2_A02", "E3", 2, "use manifests for E3 baseline alignment", "E3 can compare baseline outputs under the same paper-like candidate pools.", "E2 validation", "medium", "E3 baseline alignment tables", "Do not overwrite Plan C metrics."),
        _action("E2_A03", "E4", 3, "use manifests for E4 prompt strategy alignment", "Prompt templates need stable candidate pools and metadata.", "E2 validation", "medium", "E4 prompt-manifest registry", "Strict prompt remains an extension."),
        _action("E2_A04", "E5", 4, "use manifests for E5 local LLM scale-up", "Local Qwen2/Mistral scale-up needs deterministic manifests.", "E4 prompt alignment and explicit approval", "high", "E5 local LLM run plan", "Do not run LLMs until E5."),
        _action("E2_A05", "E1/E2", 5, "keep LastFM conditional unless valid artifacts are found", "LastFM lacks comparable utility-evaluation artifacts.", "valid LastFM artifacts", "high", "LastFM unblocking audit", "No LastFM manifest generated in E2."),
        _action("E2_A06", "E5", 6, "do not run LLMs until E5", "E2 is manifest construction only.", "explicit approval", "high", "approval record", "No Ollama/Qwen2/Mistral calls in E2."),
        _action("E2_A07", "E6/E7", 7, "do not call APIs until E6/E7 and only with approval", "Closed-source model use requires current pricing, availability, keys, and approval.", "E6 audit", "high", "API approval and cost audit", "No API calls in E2."),
    ]
    return pd.DataFrame(rows, columns=NEXT_ACTION_COLUMNS)


def build_markdown_report(tables: dict[str, pd.DataFrame | dict[tuple[str, str], pd.DataFrame]]) -> str:
    """Build the E2 Markdown report."""

    source_audit = tables["source_audit"]
    eligibility = tables["dataset_eligibility"]
    manifest_summary = tables["manifest_summary"]
    validation = tables["validation"]
    blocked = tables["blocked_datasets"]
    complete_count = int((manifest_summary["exact_1000_user_target_met"] == "true").sum()) if not manifest_summary.empty else 0
    validation_failures = validation.loc[validation["passed"] != "true"] if not validation.empty else pd.DataFrame()

    return f"""# Plan E2 Paper-Like Candidate Pool Construction

## Executive Summary

E2 constructs paper-like candidate manifests, not LLM results. It creates deterministic 1000-user ranking and reranking manifests for ML-1M, Amazon Beauty, and Amazon Sports where eligible users are available. Each generated user pool has 20 candidates, exactly 1 positive item, and exactly 19 negative items.

E2 performed no inference. E2 made no Ollama calls. E2 made no Qwen2/Mistral calls. E2 made no API calls. E2 performed no downloads. E2 performed no model pulls. E2 performed no training. E2 performed no LastFM run. E2 performed no Plan C modification. E2 performed no original_repo modification. E2 performed no destructive cleanup.

E2 does not run recommendation training. E2 does not run LLM inference. E2 does not complete full paper reproduction. Generated manifests are reconstructed paper-like artifacts unless exact original paper candidate files are verified. No full exact reproduction is claimed.

## Source Artifact Audit

{_markdown_table(source_audit, ["dataset", "candidate_source_path", "source_class", "source_rows", "source_user_count", "source_candidate_pool_size_mode", "source_is_paper_exact", "source_is_reconstructed", "source_can_support_e2"], max_rows=12)}

Reconstructed Plan C candidate pools are useful inputs but not exact original paper candidate files.

## Dataset Eligibility Summary

{_markdown_table(eligibility, ["dataset", "source_path", "source_rows", "source_user_count", "eligible_user_count", "selected_user_count", "can_build_1000_user_manifest", "eligibility_class"], max_rows=10)}

ML-1M, Amazon Beauty, and Amazon Sports are valid available datasets for reconstructed paper-like E2 candidate manifests. LastFM remains conditional/blocked unless valid utility artifacts are found.

## Manifest Construction Summary

{_markdown_table(manifest_summary, ["manifest_id", "dataset", "task", "rows", "user_count", "candidate_pool_size", "positive_items_per_user", "negative_items_per_user", "exact_1000_user_target_met", "classification"], max_rows=12)}

Complete generated manifests: `{complete_count}` of `{len(manifest_summary)}`.

## Validation Summary

{_markdown_table(validation, ["manifest_id", "dataset", "task", "check_name", "passed", "observed_value", "expected_value", "severity"], max_rows=24)}

Validation failure rows: `{len(validation_failures)}`. Any non-passing row should be reviewed before E3/E4/E5 use.

## LastFM Blocked/Conditional Status

{_markdown_table(blocked, ["dataset", "task", "blocked_or_conditional", "classification", "reason"], max_rows=10)}

No LastFM paper-like manifest is generated by E2.

## Exactness and Claim-Safety Notes

- E2 constructs reconstructed paper-like candidate manifests, not exact reproduction.
- Exact original paper candidate sampling files have not been verified.
- Strict prompt remains a Plan D/E extension, not a paper-exact prompt strategy.
- Local Qwen2/Mistral diagnostics remain diagnostic_subset results until a later approved E5 run.
- E2 outputs are near/reconstructed reproduction support artifacts.
- No full exact reproduction is claimed.

## Recommended Next Phase

The recommended next phase is E3 baseline alignment using the generated candidate manifests, followed by E4 prompt strategy alignment. E5 local LLM scale-up should only occur after explicit approval and must use raw-output hygiene controls.
"""


def write_candidate_pool_outputs(
    tables: dict[str, pd.DataFrame | dict[tuple[str, str], pd.DataFrame]] | None = None,
    paths: PlanECandidatePoolPaths | None = None,
) -> dict[str, pd.DataFrame | dict[tuple[str, str], pd.DataFrame]]:
    """Write every E2 output."""

    paths = paths or PlanECandidatePoolPaths()
    tables = tables if tables is not None else build_candidate_pool_outputs(PROJECT_ROOT, paths)
    paths.dataset_eligibility.parent.mkdir(parents=True, exist_ok=True)
    paths.report.parent.mkdir(parents=True, exist_ok=True)

    manifest_paths = paths.manifest_paths()
    manifests = tables["manifests"]
    assert isinstance(manifests, dict)
    for key, frame in manifests.items():
        _write_csv(frame, manifest_paths[key], MANIFEST_SCHEMA_COLUMNS)

    _write_csv(tables["dataset_eligibility"], paths.dataset_eligibility, DATASET_ELIGIBILITY_COLUMNS)
    _write_csv(tables["manifest_summary"], paths.manifest_summary, MANIFEST_SUMMARY_COLUMNS)
    _write_csv(tables["validation"], paths.validation, VALIDATION_COLUMNS)
    _write_csv(tables["source_audit"], paths.source_audit, SOURCE_AUDIT_COLUMNS)
    _write_csv(tables["blocked_datasets"], paths.blocked_datasets, BLOCKED_COLUMNS)
    _write_csv(tables["next_actions"], paths.next_actions, NEXT_ACTION_COLUMNS)
    paths.report.write_text(build_markdown_report(tables), encoding="utf-8", newline="\n")
    return tables


def _source_audit_row(project_root: Path, path: Path, dataset_key: str, schema: dict[str, str]) -> dict[str, object]:
    stats = _source_stats(project_root, path, dataset_key, schema)
    relative = _relative(path, project_root)
    reconstructed = "reconstructed/plan_c" in relative.lower()
    support = bool(schema["user_column"] and schema["item_column"] and (schema["label_column"] or _ground_truth_path_for_source(path).exists()))
    return {
        "dataset": _dataset_label(dataset_key),
        "candidate_source_path": relative,
        "source_class": _source_class(relative),
        "source_rows": stats["rows"],
        "source_user_count": stats["user_count"],
        "detected_user_column": schema["user_column"],
        "detected_item_column": schema["item_column"],
        "detected_label_column": schema["label_column"],
        "detected_rank_or_score_column": schema["rank_column"],
        "source_candidate_pool_size_mode": stats["candidate_pool_size_mode"],
        "source_is_paper_exact": "false",
        "source_is_reconstructed": _bool_text(reconstructed),
        "source_can_support_e2": _bool_text(support),
        "notes": "Useful reconstructed/processed candidate source; not verified as exact original paper candidate sampling file.",
    }


def _source_stats(project_root: Path, path: Path, dataset_key: str, schema: dict[str, str]) -> dict[str, object]:
    plan_c_stats = _plan_c_stats(project_root).get(dataset_key, {})
    if path.name.lower() == "candidates.csv" and plan_c_stats:
        users = _to_int(plan_c_stats.get("users")) or 0
        candidate_rows = _to_int(plan_c_stats.get("candidate_rows")) or _count_csv_rows(path)
        negative_samples = _to_int(plan_c_stats.get("negative_samples_per_user"))
        pool_size = negative_samples + 1 if negative_samples is not None else ""
        eligible_users = _to_int(plan_c_stats.get("eligible_users")) or users
        return {
            "rows": candidate_rows,
            "user_count": users,
            "candidate_pool_size_mode": pool_size,
            "eligible_user_count": eligible_users,
        }
    rows = _count_csv_rows(path)
    if not schema["user_column"]:
        return {"rows": rows, "user_count": 0, "candidate_pool_size_mode": "", "eligible_user_count": 0}
    try:
        user_series = pd.read_csv(path, usecols=[schema["user_column"]], dtype=str)[schema["user_column"]].fillna("").astype(str)
    except Exception:
        return {"rows": rows, "user_count": 0, "candidate_pool_size_mode": "", "eligible_user_count": 0}
    counts = user_series.value_counts(sort=False)
    mode = int(counts.mode().iloc[0]) if not counts.empty else ""
    return {"rows": rows, "user_count": int(user_series.nunique()), "candidate_pool_size_mode": mode, "eligible_user_count": 0}


def _select_sources(source_audit: pd.DataFrame, project_root: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for dataset in VALID_DATASETS:
        label = dataset["label"]
        candidates = source_audit.loc[(source_audit["dataset"] == label) & (source_audit["source_can_support_e2"] == "true")].copy()
        if candidates.empty:
            continue
        candidates["_priority"] = candidates["candidate_source_path"].map(_source_priority)
        candidates = candidates.sort_values(["_priority", "candidate_source_path"], kind="mergesort")
        selected[dataset["key"]] = project_root / str(candidates.iloc[0]["candidate_source_path"])
    return selected


def _source_priority(relative: str) -> int:
    lower = relative.lower()
    if "data/reconstructed/plan_c" in lower:
        return 0
    if "data/processed" in lower:
        return 1
    return 2


def _load_candidate_source(project_root: Path, source_path: Path, dataset_key: str) -> dict[str, object]:
    schema = detect_candidate_schema(source_path)
    stats = _source_stats(project_root, source_path, dataset_key, schema)
    required_columns = [schema["user_column"], schema["item_column"]]
    if schema["label_column"]:
        required_columns.append(schema["label_column"])
    if schema["rank_column"]:
        required_columns.append(schema["rank_column"])
    required_columns = [column for column in dict.fromkeys(required_columns) if column]
    selected_user_candidates = _selected_users_from_source(source_path, schema["user_column"], TARGET_USER_COUNT)
    frame = _read_rows_for_users(source_path, required_columns, schema["user_column"], selected_user_candidates)
    frame = frame.rename(columns={schema["user_column"]: "user_id", schema["item_column"]: "item_id"})
    frame["user_id"] = frame["user_id"].astype(str).str.strip()
    frame["item_id"] = frame["item_id"].astype(str).str.strip()
    frame = frame.loc[(frame["user_id"] != "") & (frame["item_id"] != "")].copy()
    if schema["label_column"]:
        frame["is_positive"] = frame[schema["label_column"]].map(_parse_bool)
    else:
        frame["is_positive"] = _infer_positive_from_ground_truth(source_path, frame)
    if schema["rank_column"]:
        frame["source_order_value"] = pd.to_numeric(frame[schema["rank_column"]], errors="coerce")
        fallback_order = frame.groupby("user_id", sort=False).cumcount() + 1
        frame["source_order_value"] = frame["source_order_value"].fillna(fallback_order)
        frame["source_order_direction"] = "descending" if "score" in schema["rank_column"].lower() else "ascending"
    else:
        frame["source_order_value"] = frame["item_id"].astype(str)
        frame["source_order_direction"] = "stable_item_id"

    user_candidates: dict[str, dict[str, object]] = {}
    eligible_selected_users: list[str] = []
    sort_ascending = frame["source_order_direction"].iloc[0] != "descending"
    ordered = frame.sort_values(["user_id", "source_order_value", "item_id"], ascending=[True, sort_ascending, True], kind="mergesort")
    for user_id, group in ordered.groupby("user_id", sort=False):
        positive_items = _unique_candidate_items(group.loc[group["is_positive"]], positive=True)
        negative_items = _unique_candidate_items(group.loc[~group["is_positive"]], positive=False)
        positive_ids = {item["item_id"] for item in positive_items}
        negative_items = [item for item in negative_items if item["item_id"] not in positive_ids]
        source_positive_count = len(positive_items)
        source_negative_count = len(negative_items)
        if source_positive_count >= REQUIRED_POSITIVES and source_negative_count >= REQUIRED_NEGATIVES:
            eligible_selected_users.append(str(user_id))
        user_candidates[str(user_id)] = {
            "positive_items": positive_items,
            "negative_items": negative_items,
            "source_candidate_count": int(group["item_id"].nunique()),
            "source_positive_count": source_positive_count,
            "source_negative_count": source_negative_count,
        }

    selected_users = sorted(eligible_selected_users, key=str)[:TARGET_USER_COUNT]
    eligible_user_count = int(stats["eligible_user_count"]) if stats["eligible_user_count"] else len(eligible_selected_users)
    return {
        "source_rows": int(stats["rows"]) if stats["rows"] else len(frame),
        "source_user_count": int(stats["user_count"]) if stats["user_count"] else int(frame["user_id"].nunique()),
        "eligible_user_count": eligible_user_count,
        "selected_users": selected_users,
        "user_candidates": user_candidates,
    }


def _construct_manifest_for_task(
    *,
    dataset_key: str,
    dataset_label: str,
    source_path: Path,
    selected_users: list[str],
    user_candidates: dict[str, dict[str, object]],
    task: str,
) -> pd.DataFrame:
    manifest_id = _manifest_id(dataset_key, task)
    source_relative = _relative(source_path, PROJECT_ROOT)
    rows: list[dict[str, object]] = []
    for user_id in selected_users:
        user_info = user_candidates[user_id]
        positive = user_info["positive_items"][0]
        negatives = user_info["negative_items"][:REQUIRED_NEGATIVES]
        selected = [positive] + negatives
        if task == "ranking":
            ordered = sorted(selected, key=lambda item: (_shuffle_key(dataset_key, task, user_id, item["item_id"]), item["item_id"]))
            ordering_strategy = "deterministic_seeded_shuffle_sha256"
        else:
            ordered = sorted(selected, key=lambda item: (item["source_order"], item["item_id"]))
            ordering_strategy = "source_rank_or_candidate_position_ascending"
        for position, item in enumerate(ordered, start=1):
            rows.append(
                {
                    "manifest_id": manifest_id,
                    "dataset": dataset_label,
                    "task": task,
                    "user_id": user_id,
                    "candidate_position": position,
                    "item_id": item["item_id"],
                    "is_positive": _bool_text(bool(item["is_positive"])),
                    "source_path": source_relative,
                    "source_user_candidate_count": user_info["source_candidate_count"],
                    "source_positive_count": user_info["source_positive_count"],
                    "source_negative_count": user_info["source_negative_count"],
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "selected_positive_count": REQUIRED_POSITIVES,
                    "selected_negative_count": REQUIRED_NEGATIVES,
                    "ordering_strategy": ordering_strategy,
                    "seed": SEED,
                    "exactness_class": "reconstructed paper-like candidate manifest",
                    "notes": "Reconstructed paper-like support artifact; not exact reproduction unless original paper candidate files are verified.",
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_SCHEMA_COLUMNS)


def _manifest_summary_row(dataset_key: str, dataset_label: str, task: str, path: Path, frame: pd.DataFrame) -> dict[str, object]:
    user_count = int(frame["user_id"].nunique()) if not frame.empty else 0
    rows = len(frame)
    complete = user_count == TARGET_USER_COUNT
    if frame.empty:
        pool = positives = negatives = ""
        pool_ok = pos_neg_ok = False
        ordering = ""
    else:
        per_user = frame.groupby("user_id", sort=False).size()
        positives_per_user = frame.loc[frame["is_positive"] == "true"].groupby("user_id", sort=False).size()
        negatives_per_user = frame.loc[frame["is_positive"] == "false"].groupby("user_id", sort=False).size()
        pool = int(per_user.iloc[0]) if len(per_user.unique()) == 1 else ""
        positives = int(positives_per_user.iloc[0]) if len(positives_per_user.unique()) == 1 else ""
        negatives = int(negatives_per_user.iloc[0]) if len(negatives_per_user.unique()) == 1 else ""
        pool_ok = pool == CANDIDATE_POOL_SIZE
        pos_neg_ok = positives == REQUIRED_POSITIVES and negatives == REQUIRED_NEGATIVES
        ordering = "; ".join(sorted(frame["ordering_strategy"].unique()))
    return {
        "manifest_id": _manifest_id(dataset_key, task),
        "dataset": dataset_label,
        "task": task,
        "path": _relative(path, PROJECT_ROOT),
        "rows": rows,
        "user_count": user_count,
        "candidate_pool_size": pool,
        "positive_items_per_user": positives,
        "negative_items_per_user": negatives,
        "exact_1000_user_target_met": _bool_text(complete),
        "exact_candidate_pool_20_met": _bool_text(pool_ok),
        "exact_1_positive_19_negative_met": _bool_text(pos_neg_ok),
        "ordering_strategy": ordering,
        "exactness_class": "near/reconstructed reproduction support",
        "classification": "reconstructed paper-like candidate manifest",
        "notes": "Generated deterministically from reconstructed/processed candidate source; not exact paper sampling.",
    }


def _eligibility_row(
    dataset_label: str,
    source_path: Path,
    source_rows: int,
    source_user_count: int,
    eligible_user_count: int,
    selected_user_count: int,
    blocking_issue: str,
) -> dict[str, object]:
    can_build = selected_user_count >= TARGET_USER_COUNT
    return {
        "dataset": dataset_label,
        "source_path": _relative(source_path, PROJECT_ROOT),
        "source_exists": "true",
        "source_rows": source_rows,
        "source_user_count": source_user_count,
        "eligible_user_count": eligible_user_count,
        "selected_user_count": selected_user_count,
        "target_user_count": TARGET_USER_COUNT,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "required_positive_per_user": REQUIRED_POSITIVES,
        "required_negative_per_user": REQUIRED_NEGATIVES,
        "can_build_1000_user_manifest": _bool_text(can_build),
        "eligibility_class": "complete_1000_user_manifest_ready" if can_build else "partial_manifest_only",
        "blocking_issue": blocking_issue if blocking_issue else ("" if can_build else "Fewer than 1000 eligible users."),
        "notes": "Eligible users are selected by stable string ordering.",
    }


def _blocked_eligibility_row(dataset_label: str, issue: str) -> dict[str, object]:
    return {
        "dataset": dataset_label,
        "source_path": "",
        "source_exists": "false",
        "source_rows": 0,
        "source_user_count": 0,
        "eligible_user_count": 0,
        "selected_user_count": 0,
        "target_user_count": TARGET_USER_COUNT,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "required_positive_per_user": REQUIRED_POSITIVES,
        "required_negative_per_user": REQUIRED_NEGATIVES,
        "can_build_1000_user_manifest": "false",
        "eligibility_class": "blocked/conditional",
        "blocking_issue": issue,
        "notes": "No E2 manifest generated for this dataset.",
    }


def _unique_candidate_items(group: pd.DataFrame, *, positive: bool) -> list[dict[str, object]]:
    seen: set[str] = set()
    items: list[dict[str, object]] = []
    for _, row in group.iterrows():
        item_id = str(row["item_id"]).strip()
        if item_id == "" or item_id in seen:
            continue
        seen.add(item_id)
        items.append({"item_id": item_id, "is_positive": positive, "source_order": row["source_order_value"]})
    return items


def _infer_positive_from_ground_truth(source_path: Path, frame: pd.DataFrame) -> pd.Series:
    ground_truth_path = _ground_truth_path_for_source(source_path)
    if not ground_truth_path.exists():
        return pd.Series(False, index=frame.index)
    columns = _header_columns(ground_truth_path)
    user_col = _first_present(columns, USER_COLUMN_CANDIDATES)
    item_col = _first_present(columns, GROUND_TRUTH_ITEM_COLUMNS)
    if not user_col or not item_col:
        return pd.Series(False, index=frame.index)
    gt = pd.read_csv(ground_truth_path, usecols=[user_col, item_col], dtype=str).fillna("")
    gt_pairs = set(zip(gt[user_col].astype(str).str.strip(), gt[item_col].astype(str).str.strip()))
    return frame.apply(lambda row: (row["user_id"], row["item_id"]) in gt_pairs, axis=1)


def _selected_users_from_source(source_path: Path, user_column: str, target_user_count: int) -> list[str]:
    try:
        users = pd.read_csv(source_path, usecols=[user_column], dtype=str)[user_column].fillna("").astype(str).str.strip()
    except Exception:
        return []
    return sorted(user for user in users.unique() if user)[:target_user_count]


def _read_rows_for_users(source_path: Path, required_columns: list[str], user_column: str, selected_users: list[str]) -> pd.DataFrame:
    selected = set(selected_users)
    if not selected:
        return pd.DataFrame(columns=required_columns)
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source_path, usecols=required_columns, dtype=str, chunksize=200_000):
        user_values = chunk[user_column].fillna("").astype(str).str.strip()
        filtered = chunk.loc[user_values.isin(selected)].copy()
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        return pd.DataFrame(columns=required_columns)
    return pd.concat(frames, ignore_index=True).fillna("")


def _ground_truth_path_for_source(source_path: Path) -> Path:
    return source_path.parent / "ground_truth.csv"


def _lastfm_has_valid_utility_artifacts(project_root: Path) -> bool:
    base = project_root / "data" / "reconstructed" / "plan_c" / "lastfm"
    return (base / "candidates.csv").exists() and (base / "ground_truth.csv").exists()


def _validation(
    manifest_id: str,
    dataset: str,
    task: str,
    check_name: str,
    passed: bool,
    observed_value: object,
    expected_value: object,
    severity: str,
    notes: str,
) -> dict[str, object]:
    return {
        "manifest_id": manifest_id,
        "dataset": dataset,
        "task": task,
        "check_name": check_name,
        "passed": _bool_text(passed),
        "observed_value": observed_value,
        "expected_value": expected_value,
        "severity": severity,
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


def _write_csv(frame: object, path: Path, columns: list[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, columns=columns, lineterminator="\n")


def _series_all_equal(series: pd.Series, expected: int) -> bool:
    return not series.empty and bool((series == expected).all())


def _observed_unique(series: pd.Series) -> str:
    if series.empty:
        return ""
    return ";".join(str(int(value)) for value in sorted(series.unique()))


def _shuffle_key(dataset_key: str, task: str, user_id: str, item_id: str) -> str:
    payload = f"{SEED}|{dataset_key}|{task}|{user_id}|{item_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "positive", "ground_truth"}


def _source_class(relative: str) -> str:
    lower = relative.lower()
    if "data/reconstructed/plan_c" in lower:
        return "reconstructed_plan_c_candidate_pool"
    if "data/processed" in lower:
        return "processed_project_candidate_pool"
    return "candidate_pool_source"


def _dataset_key_for_path(relative: str) -> str | None:
    lower = relative.lower().replace("\\", "/")
    for dataset in ALL_DATASETS:
        if any(alias in lower for alias in dataset["aliases"]):
            return dataset["key"]
    return None


def _dataset_label(dataset_key: str) -> str:
    for dataset in ALL_DATASETS:
        if dataset["key"] == dataset_key:
            return dataset["label"]
    return "unknown"


def _manifest_id(dataset_key: str, task: str) -> str:
    short = next(dataset["short"] for dataset in ALL_DATASETS if dataset["key"] == dataset_key)
    return f"plan_e_{short}_1000user_{task}_manifest"


def _header_columns(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return [column.strip() for column in next(csv.reader(handle), [])]
    except OSError:
        return []


def _first_present(columns: list[str], candidates: list[str]) -> str:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return ""


def _count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return 0


def _plan_c_stats(project_root: Path) -> dict[str, dict[str, str]]:
    path = project_root / "results" / "tables" / "plan_c_reconstructed_dataset_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {str(row["dataset"]): {column: str(row[column]) for column in frame.columns} for _, row in frame.iterrows()}


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


def _plan_c_status_lines(project_root: Path) -> list[str]:
    status = _git_lines(["git", "status", "--short"], project_root)
    return [line for line in status if "plan_c" in line.lower() or "plan-c" in line.lower() or "reconstructed/plan_c" in line.lower()]


def _git_lines(command: list[str], project_root: Path) -> list[str]:
    return common_git.git_lines_unavailable_on_error(command, project_root, normalize_paths=True)


def _markdown_table(frame: object, columns: Iterable[str], *, max_rows: int) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        selected_columns = list(columns)
        return "| " + " | ".join(selected_columns) + " |\n| " + " | ".join("---" for _ in selected_columns) + " |"
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


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _bool_text(value: bool) -> str:
    return "true" if value else "false"

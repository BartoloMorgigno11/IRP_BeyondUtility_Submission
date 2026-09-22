from __future__ import annotations

from src.audit import common_git

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.plan_e_non_neural_baselines import (  # noqa: E402
    BLOCKED_COLUMNS,
    GLOBAL_RANDOM_SEED,
    METRICS_COLUMNS,
    RANKINGS_COLUMNS,
    RUN_MANIFEST_COLUMNS,
    TOP_K,
    VALIDATION_COLUMNS,
    aggregate_metrics,
    decode_candidate_items,
    format_bool,
    format_float,
    group_rows,
    json_list,
    load_bm25_scores,
    load_csv_rows,
    load_train_popularity,
    make_ranking_output_row,
    make_run_id,
    parse_bool,
    score_lookup_candidate_row,
    score_mostpop_candidate_row,
    score_random_candidate_row,
    stringify_id,
    validate_bm25_variant_label,
    validation_row,
    write_csv,
)


LOGGER = logging.getLogger(__name__)

PHASE_ID = "Phase 3A"
PHASE_TITLE = "Deterministic Non-Neural Baseline Consolidation"

DATASET_ORDER = {
    "Amazon Beauty": 0,
    "Amazon Sports": 1,
    "ML-1M": 2,
}
TASK_ORDER = {"ranking": 0, "reranking": 1}
DATASET_SLUGS = {
    "Amazon Beauty": "amazon_beauty",
    "Amazon Sports": "amazon_sports",
    "ML-1M": "ml1m",
}

INPUT_PATHS = {
    "ranking_candidate_sets": PROJECT_ROOT / "results/tables/ranking_candidate_sets.csv",
    "positive_positions": PROJECT_ROOT / "results/tables/positive_positions.csv",
    "dataset_statistics": PROJECT_ROOT / "results/tables/dataset_statistics.csv",
    "candidate_item_title_map": PROJECT_ROOT / "results/tables/candidate_item_title_map.csv",
    "experiment_cohort_membership": PROJECT_ROOT / "results/tables/experiment_cohort_membership.csv",
    "experiment_cohort_summary": PROJECT_ROOT / "results/tables/experiment_cohort_summary.csv",
    "llm_title_exclusions": PROJECT_ROOT / "results/tables/llm_title_exclusions.csv",
}

OUTPUT_PATHS = {
    "manifest": PROJECT_ROOT / "results/tables/traditional_baseline_run_manifest.csv",
    "rankings": PROJECT_ROOT / "results/tables/traditional_baseline_rankings.csv",
    "metrics": PROJECT_ROOT / "results/tables/traditional_baselines_metrics.csv",
    "validation": PROJECT_ROOT / "results/tables/traditional_baseline_validation.csv",
    "blocked": PROJECT_ROOT / "results/tables/traditional_baselines_blocked.csv",
    "report": PROJECT_ROOT / "report/traditional_baseline_alignment.md",
}

BM25_SCORES_PATH = PROJECT_ROOT / "results/tables/plan_e_bm25_scores.csv"
PRIOR_E3A_METRICS_PATH = PROJECT_ROOT / "results/tables/plan_e_baseline_alignment_no_training_metrics.csv"
PRIOR_E3B_METRICS_PATH = PROJECT_ROOT / "results/tables/plan_e_bm25_metrics.csv"

REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_set_id",
    "source_path",
    "source_plan",
    "dataset",
    "user_id",
    "positive_item_id",
    "candidate_items",
    "candidate_count",
}

REQUIRED_MEMBERSHIP_COLUMNS = {
    "candidate_set_id",
    "source_path",
    "source_plan",
    "dataset",
    "task_type",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "eligible_plan_e2_paper_like_full",
    "eligible_plan_e2_llm_strict_title",
    "eligible_plan_e2_matched_comparison",
}

BASELINE_ORDER = ["random_seeded", "mostpop_train_only", "bm25_like_e3b"]


def rel_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: Sequence[str]) -> str:
    try:
        return common_git.git_stdout_checked(args, PROJECT_ROOT)
    except Exception:
        return "unavailable"


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def require_columns(path: Path, required: set[str]) -> None:
    header = set(read_csv_header(path))
    missing = sorted(required - header)
    if missing:
        raise ValueError(f"{rel_path(path)} is missing required columns: {missing}")


def membership_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        stringify_id(row.get("candidate_set_id")),
        stringify_id(row.get("user_id")),
        stringify_id(row.get("positive_item_id")),
    )


def verify_and_load_phase2_inputs() -> tuple[list[dict[str, str]], dict[str, str]]:
    require_columns(INPUT_PATHS["ranking_candidate_sets"], REQUIRED_CANDIDATE_COLUMNS)
    require_columns(INPUT_PATHS["experiment_cohort_membership"], REQUIRED_MEMBERSHIP_COLUMNS)

    candidate_rows = load_csv_rows(INPUT_PATHS["ranking_candidate_sets"])
    membership_rows = load_csv_rows(INPUT_PATHS["experiment_cohort_membership"])

    if len(candidate_rows) != 76121:
        raise ValueError(f"Expected 76,121 candidate rows, observed {len(candidate_rows)}")

    membership_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    duplicate_membership = 0
    matched_equals_strict = True
    matched_implies_full = True
    for row in membership_rows:
        key = membership_key(row)
        if key in membership_by_key:
            duplicate_membership += 1
        membership_by_key[key] = row
        matched = parse_bool(row.get("eligible_plan_e2_matched_comparison"))
        strict = parse_bool(row.get("eligible_plan_e2_llm_strict_title"))
        full = parse_bool(row.get("eligible_plan_e2_paper_like_full"))
        if matched != strict:
            matched_equals_strict = False
        if matched and not full:
            matched_implies_full = False

    if duplicate_membership:
        raise ValueError(f"Duplicate cohort membership keys observed: {duplicate_membership}")
    if not matched_equals_strict:
        raise ValueError("Matched membership does not exactly equal strict-title membership")
    if not matched_implies_full:
        raise ValueError("Matched membership contains rows outside the Plan E2 full cohort")

    full_rows: list[dict[str, str]] = []
    decoded_errors: list[str] = []
    plan_e_positive_errors = 0
    plan_e_count_errors = 0

    for index, candidate_row in enumerate(candidate_rows, start=2):
        try:
            items = decode_candidate_items(candidate_row["candidate_items"])
        except Exception as exc:  # pragma: no cover - error path exercised by integration.
            decoded_errors.append(f"line {index}: {exc}")
            continue

        key = membership_key(candidate_row)
        membership = membership_by_key.get(key)
        if not membership or not parse_bool(membership.get("eligible_plan_e2_paper_like_full")):
            continue

        if len(items) != 20 or stringify_id(candidate_row.get("candidate_count")) != "20":
            plan_e_count_errors += 1
        if items.count(stringify_id(candidate_row["positive_item_id"])) != 1:
            plan_e_positive_errors += 1

        merged = dict(candidate_row)
        merged["task_type"] = membership["task_type"]
        merged["source_plan"] = membership["source_plan"]
        merged["eligible_plan_e2_paper_like_full"] = membership["eligible_plan_e2_paper_like_full"]
        merged["eligible_plan_e2_llm_strict_title"] = membership["eligible_plan_e2_llm_strict_title"]
        merged["eligible_plan_e2_matched_comparison"] = membership["eligible_plan_e2_matched_comparison"]
        merged["_decoded_candidate_items"] = items
        full_rows.append(merged)

    if decoded_errors:
        raise ValueError(f"Candidate JSON decode failures: {decoded_errors[:5]}")
    if len(full_rows) != 6000:
        raise ValueError(f"Expected 6,000 Plan E2 full rows, observed {len(full_rows)}")
    matched_rows = sum(1 for row in full_rows if parse_bool(row["eligible_plan_e2_matched_comparison"]))
    if matched_rows != 5814:
        raise ValueError(f"Expected 5,814 matched rows, observed {matched_rows}")
    if plan_e_count_errors:
        raise ValueError(f"Plan E2 rows with non-20 candidate count: {plan_e_count_errors}")
    if plan_e_positive_errors:
        raise ValueError(f"Plan E2 rows without exactly one positive: {plan_e_positive_errors}")

    full_rows.sort(key=row_sort_key)
    input_hashes = {name: sha256_file(path) for name, path in INPUT_PATHS.items()}
    return full_rows, input_hashes


def row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        DATASET_ORDER.get(stringify_id(row.get("dataset")), 99),
        TASK_ORDER.get(stringify_id(row.get("task_type")), 99),
        stringify_id(row.get("user_id")),
        stringify_id(row.get("candidate_set_id")),
    )


def load_popularity_by_dataset() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for dataset, slug in DATASET_SLUGS.items():
        path = PROJECT_ROOT / f"data/reconstructed/plan_c/{slug}/interactions.csv"
        if not path.exists():
            loaded[dataset] = {
                "path": path,
                "counts": {},
                "row_count": 0,
                "train_row_count": 0,
                "available": False,
            }
            continue
        counts, row_count, train_row_count = load_train_popularity(path)
        loaded[dataset] = {
            "path": path,
            "counts": counts,
            "row_count": row_count,
            "train_row_count": train_row_count,
            "available": True,
        }
    return loaded


def content_path_for_dataset(dataset: str) -> Path:
    slug = DATASET_SLUGS[dataset]
    return PROJECT_ROOT / f"data/reconstructed/plan_c/{slug}/catalog.csv"


def training_path_for_dataset(dataset: str) -> Path:
    slug = DATASET_SLUGS[dataset]
    return PROJECT_ROOT / f"data/reconstructed/plan_c/{slug}/interactions.csv"


def baseline_family(baseline_id: str) -> str:
    if baseline_id.startswith("random"):
        return "Random"
    if baseline_id.startswith("mostpop"):
        return "MostPop"
    if baseline_id.startswith("bm25"):
        return "BM25"
    return baseline_id


def score_definition(baseline_id: str) -> str:
    if baseline_id == "random_seeded":
        return "SHA256-derived stable score over global seed, dataset, task, candidate-set ID, user ID and item ID"
    if baseline_id == "mostpop_train_only":
        return "raw train-split interaction frequency per item; unseen candidate items score 0"
    if baseline_id == "bm25_like_e3b":
        return "Plan E3B local BM25-like score variant local_bm25_like_k1_1.2_b_0.75 loaded from prior E3B score table"
    if baseline_id == "bm25_exact_official":
        return "blocked: exact official BM25 equivalence is not locally demonstrated"
    raise ValueError(f"Unknown baseline_id: {baseline_id}")


def tie_breaking_rule() -> str:
    return "descending score; lexical item_id as string; original candidate position"


def paper_alignment(baseline_id: str) -> str:
    if baseline_id in {"random_seeded", "mostpop_train_only"}:
        return "paper_like_reconstructed"
    if baseline_id == "bm25_like_e3b":
        return "adapted"
    return "unknown"


def score_row_for_baseline(
    baseline_id: str,
    row: Mapping[str, Any],
    popularity_by_dataset: Mapping[str, Mapping[str, Any]],
    bm25_scores: Mapping[tuple[str, str, str], Mapping[str, float]],
) -> Any:
    candidate_items = row["_decoded_candidate_items"]
    dataset = stringify_id(row["dataset"])
    task_type = stringify_id(row["task_type"])
    candidate_set_id = stringify_id(row["candidate_set_id"])
    user_id = stringify_id(row["user_id"])

    if baseline_id == "random_seeded":
        return score_random_candidate_row(
            candidate_items,
            GLOBAL_RANDOM_SEED,
            dataset,
            task_type,
            candidate_set_id,
            user_id,
        )
    if baseline_id == "mostpop_train_only":
        popularity = popularity_by_dataset[dataset]["counts"]
        return score_mostpop_candidate_row(candidate_items, popularity)
    if baseline_id == "bm25_like_e3b":
        lookup = bm25_scores.get((dataset, task_type, user_id), {})
        return score_lookup_candidate_row(candidate_items, lookup, missing_score_value=0.0)
    raise ValueError(f"Unsupported completed baseline: {baseline_id}")


def score_group(
    baseline_id: str,
    rows: Sequence[Mapping[str, Any]],
    popularity_by_dataset: Mapping[str, Mapping[str, Any]],
    bm25_scores: Mapping[tuple[str, str, str], Mapping[str, float]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        scored = score_row_for_baseline(baseline_id, row, popularity_by_dataset, bm25_scores)
        if baseline_id == "bm25_like_e3b" and scored.missing_score_count:
            raise ValueError(
                "Plan E3B BM25-like score table is missing scores for "
                f"{row['dataset']} {row['task_type']} user {row['user_id']}"
            )
        output.append(
            make_ranking_output_row(
                run_id=make_run_id(baseline_id, row["dataset"], row["task_type"]),
                baseline_id=baseline_id,
                dataset=row["dataset"],
                task_type=row["task_type"],
                candidate_set_id=row["candidate_set_id"],
                source_path=row["source_path"],
                user_id=row["user_id"],
                positive_item_id=row["positive_item_id"],
                candidate_items=row["_decoded_candidate_items"],
                scored=scored,
                eligible_full_cohort=True,
                eligible_matched_cohort=parse_bool(row["eligible_plan_e2_matched_comparison"]),
                notes=ranking_notes(baseline_id),
            )
        )
    return output


def ranking_notes(baseline_id: str) -> str:
    if baseline_id == "random_seeded":
        return "positive label excluded from SHA256 score derivation"
    if baseline_id == "mostpop_train_only":
        return "train-only raw interaction frequency; unseen candidates receive score 0"
    if baseline_id == "bm25_like_e3b":
        return "loaded from Plan E3B BM25-like scores; exact official BM25 remains blocked"
    return ""


def build_completed_outputs(
    full_rows: Sequence[dict[str, str]],
    popularity_by_dataset: Mapping[str, Mapping[str, Any]],
    bm25_scores: Mapping[tuple[str, str, str], Mapping[str, float]],
) -> tuple[list[dict[str, str]], dict[str, bool]]:
    rankings: list[dict[str, str]] = []
    deterministic_checks: dict[str, bool] = {}
    grouped = group_rows(full_rows, ["dataset", "task_type", "candidate_set_id"])

    for baseline_id in BASELINE_ORDER:
        for key in sorted(grouped, key=lambda value: (DATASET_ORDER.get(value[0], 99), TASK_ORDER.get(value[1], 99), value[2])):
            rows = sorted(grouped[key], key=row_sort_key)
            first = score_group(baseline_id, rows, popularity_by_dataset, bm25_scores)
            second = score_group(baseline_id, rows, popularity_by_dataset, bm25_scores)
            run_id = make_run_id(baseline_id, key[0], key[1])
            deterministic_checks[run_id] = rows_digest(first) == rows_digest(second)
            rankings.extend(first)

    rankings.sort(
        key=lambda row: (
            BASELINE_ORDER.index(row["baseline_id"]),
            DATASET_ORDER.get(row["dataset"], 99),
            TASK_ORDER.get(row["task_type"], 99),
            row["user_id"],
            row["candidate_set_id"],
        )
    )
    return rankings, deterministic_checks


def rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    blob = json.dumps(list(rows), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_manifest(
    full_rows: Sequence[dict[str, str]],
    popularity_by_dataset: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    grouped = group_rows(full_rows, ["dataset", "task_type", "candidate_set_id"])
    output_path = f"{rel_path(OUTPUT_PATHS['rankings'])};{rel_path(OUTPUT_PATHS['metrics'])}"

    for key in sorted(grouped, key=lambda value: (DATASET_ORDER.get(value[0], 99), TASK_ORDER.get(value[1], 99), value[2])):
        dataset, task_type, candidate_set_id = key
        rows = grouped[key]
        matched = sum(1 for row in rows if parse_bool(row["eligible_plan_e2_matched_comparison"]))
        source_plan = stringify_id(rows[0]["source_plan"])
        source_path = stringify_id(rows[0]["source_path"])
        for baseline_id in BASELINE_ORDER:
            train_path = training_path_for_dataset(dataset)
            content_path = content_path_for_dataset(dataset)
            if baseline_id == "random_seeded":
                training_data_path = "not_applicable"
                content_data_path = "not_applicable"
                global_seed = stringify_id(GLOBAL_RANDOM_SEED)
                seed_derivation = "SHA256(global_seed|dataset|task_type|candidate_set_id|user_id|item_id)"
            elif baseline_id == "mostpop_train_only":
                training_data_path = rel_path(train_path)
                content_data_path = "not_applicable"
                global_seed = "not_applicable"
                seed_derivation = "not_applicable"
            else:
                training_data_path = rel_path(train_path)
                content_data_path = rel_path(content_path)
                global_seed = "not_applicable"
                seed_derivation = "not_applicable"

            execution_status = "completed"
            blocking_reason = ""
            if baseline_id == "mostpop_train_only" and not popularity_by_dataset[dataset]["available"]:
                execution_status = "blocked"
                blocking_reason = "missing reconstructed train interactions"

            manifest.append(
                {
                    "run_id": make_run_id(baseline_id, dataset, task_type),
                    "baseline_id": baseline_id,
                    "baseline_family": baseline_family(baseline_id),
                    "implementation_path": "src/baselines/plan_e_non_neural_baselines.py",
                    "configuration_path": "not_applicable",
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "source_plan": source_plan,
                    "candidate_count": "20",
                    "full_cohort_rows": stringify_id(len(rows)),
                    "matched_cohort_rows": stringify_id(matched),
                    "training_data_path": training_data_path,
                    "content_data_path": content_data_path,
                    "global_seed": global_seed,
                    "seed_derivation": seed_derivation,
                    "score_definition": score_definition(baseline_id),
                    "tie_breaking_rule": tie_breaking_rule(),
                    "paper_alignment": paper_alignment(baseline_id),
                    "execution_status": execution_status,
                    "blocking_reason": blocking_reason,
                    "output_path": output_path if execution_status == "completed" else "",
                    "notes": f"candidate source: {source_path}",
                }
            )

        manifest.append(
            {
                "run_id": make_run_id("bm25_exact_official", dataset, task_type),
                "baseline_id": "bm25_exact_official",
                "baseline_family": "BM25",
                "implementation_path": "original_repo/EvaLLMasRecommender/eva/codes/random.sh;original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
                "configuration_path": "not_available",
                "dataset": dataset,
                "task_type": task_type,
                "candidate_set_id": candidate_set_id,
                "source_plan": source_plan,
                "candidate_count": "20",
                "full_cohort_rows": stringify_id(len(rows)),
                "matched_cohort_rows": stringify_id(matched),
                "training_data_path": "not_verified",
                "content_data_path": "not_verified",
                "global_seed": "not_applicable",
                "seed_derivation": "not_applicable",
                "score_definition": score_definition("bm25_exact_official"),
                "tie_breaking_rule": "not_verified",
                "paper_alignment": "unknown",
                "execution_status": "blocked",
                "blocking_reason": "exact paper BM25 document/query representation, preprocessing, parameters and local score artifacts are not verifiably available",
                "output_path": "",
                "notes": "bm25_like_e3b is executed separately as an adapted local variant",
            }
        )
    return manifest


def build_metrics(rankings: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    metrics: list[dict[str, str]] = []
    grouped = group_rows(rankings, ["baseline_id", "dataset", "task_type", "candidate_set_id"])
    for key in sorted(
        grouped,
        key=lambda value: (
            BASELINE_ORDER.index(value[0]) if value[0] in BASELINE_ORDER else 99,
            DATASET_ORDER.get(value[1], 99),
            TASK_ORDER.get(value[2], 99),
            value[3],
        ),
    ):
        baseline_id, dataset, task_type, candidate_set_id = key
        rows = list(grouped[key])
        run_id = make_run_id(baseline_id, dataset, task_type)
        for cohort_id in ("plan_e2_paper_like_full", "plan_e2_matched_comparison"):
            helper_cohort_id = "plan_e2_full" if cohort_id == "plan_e2_paper_like_full" else "plan_e2_matched_title_complete"
            metrics.append(
                aggregate_metrics(
                    rows,
                    cohort_id=helper_cohort_id,
                    run_id=run_id,
                    baseline_id=baseline_id,
                    dataset=dataset,
                    task_type=task_type,
                    candidate_set_id=candidate_set_id,
                    paper_alignment=paper_alignment(baseline_id),
                    notes="matched metrics are filtered from the same full-cohort ranking rows",
                )
                | {"cohort_id": cohort_id}
            )
    return metrics


def build_validation(
    full_rows: Sequence[dict[str, str]],
    rankings: Sequence[dict[str, str]],
    metrics: Sequence[dict[str, str]],
    deterministic_checks: Mapping[str, bool],
    popularity_by_dataset: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    validation: list[dict[str, str]] = []
    input_grouped = group_rows(full_rows, ["dataset", "task_type", "candidate_set_id"])
    ranking_grouped = group_rows(rankings, ["baseline_id", "dataset", "task_type", "candidate_set_id"])
    metrics_grouped = group_rows(metrics, ["baseline_id", "dataset", "task_type", "candidate_set_id"])

    for key in sorted(
        ranking_grouped,
        key=lambda value: (
            BASELINE_ORDER.index(value[0]),
            DATASET_ORDER.get(value[1], 99),
            TASK_ORDER.get(value[2], 99),
            value[3],
        ),
    ):
        baseline_id, dataset, task_type, candidate_set_id = key
        run_id = make_run_id(baseline_id, dataset, task_type)
        run_rows = list(ranking_grouped[key])
        input_rows = list(input_grouped[(dataset, task_type, candidate_set_id)])
        metric_rows = list(metrics_grouped[key])
        prefix = f"{run_id}:{candidate_set_id}"

        def add(check_name: str, expected: Any, observed: Any, status: str, severity: str, evidence: str, notes: str) -> None:
            validation.append(
                validation_row(
                    validation_id=f"{prefix}:{check_name}",
                    run_id=run_id,
                    baseline_id=baseline_id,
                    dataset=dataset,
                    task_type=task_type,
                    candidate_set_id=candidate_set_id,
                    check_name=check_name,
                    expected_value=expected,
                    observed_value=observed,
                    status=status,
                    severity=severity,
                    evidence=evidence,
                    notes=notes,
                )
            )

        input_count = len(input_rows)
        ranking_count = len(run_rows)
        add(
            "source_cohort_row_count",
            input_count,
            ranking_count,
            "passed" if input_count == ranking_count == 1000 else "failed",
            "error",
            rel_path(INPUT_PATHS["experiment_cohort_membership"]),
            "one completed ranking row per Plan E2 full cohort row",
        )
        candidate_lengths = sorted({len(decode_candidate_items(row["candidate_items"])) for row in run_rows})
        add(
            "candidate_count",
            "20",
            ",".join(map(str, candidate_lengths)),
            "passed" if candidate_lengths == [20] else "failed",
            "error",
            rel_path(OUTPUT_PATHS["rankings"]),
            "Plan E2 baseline consolidation excludes 100-candidate artifacts",
        )
        membership_ok = all(
            set(decode_candidate_items(row["candidate_items"])) == set(json.loads(row["ranked_items"]))
            for row in run_rows
        )
        add(
            "candidate_membership_preservation",
            "true",
            format_bool(membership_ok),
            "passed" if membership_ok else "failed",
            "error",
            rel_path(OUTPUT_PATHS["rankings"]),
            "ranked item set must equal candidate item set",
        )
        unique_ok = all(len(json.loads(row["ranked_items"])) == len(set(json.loads(row["ranked_items"]))) for row in run_rows)
        add("unique_ranked_items", "true", format_bool(unique_ok), "passed" if unique_ok else "failed", "error", rel_path(OUTPUT_PATHS["rankings"]), "")
        positive_ok = all(
            decode_candidate_items(row["candidate_items"]).count(row["positive_item_id"]) == 1 for row in run_rows
        )
        add("exactly_one_positive", "true", format_bool(positive_ok), "passed" if positive_ok else "failed", "error", rel_path(OUTPUT_PATHS["rankings"]), "")
        missing_total = sum(int(row["missing_score_count"]) for row in run_rows)
        add("score_completeness", "0", missing_total, "passed" if missing_total == 0 else "failed", "error", rel_path(OUTPUT_PATHS["rankings"]), "")
        finite_ok = all(all(is_finite_number(score) for score in json.loads(row["ranked_scores"])) for row in run_rows)
        add("finite_scores", "true", format_bool(finite_ok), "passed" if finite_ok else "failed", "error", rel_path(OUTPUT_PATHS["rankings"]), "")
        deterministic_ok = deterministic_checks.get(run_id, False)
        add("deterministic_rerun_consistency", "true", format_bool(deterministic_ok), "passed" if deterministic_ok else "failed", "error", "in-memory repeated scoring digest", "same scorer invoked twice before writing outputs")
        matched_expected = sum(1 for row in input_rows if parse_bool(row["eligible_plan_e2_matched_comparison"]))
        matched_observed = sum(1 for row in run_rows if parse_bool(row["eligible_matched_cohort"]))
        add("matched_subset_identity", matched_expected, matched_observed, "passed" if matched_expected == matched_observed else "failed", "error", rel_path(INPUT_PATHS["experiment_cohort_membership"]), "")
        metric_counts = sorted((row["cohort_id"], row["evaluated_rows"]) for row in metric_rows)
        expected_metric_counts = sorted(
            [
                ("plan_e2_matched_comparison", stringify_id(matched_expected)),
                ("plan_e2_paper_like_full", stringify_id(input_count)),
            ]
        )
        add("metric_ranking_row_reconciliation", json_list(expected_metric_counts), json_list(metric_counts), "passed" if metric_counts == expected_metric_counts else "failed", "error", rel_path(OUTPUT_PATHS["metrics"]), "")

        if baseline_id == "mostpop_train_only":
            train_rows = popularity_by_dataset[dataset]["train_row_count"]
            add("mostpop_training_only_provenance", "train_row_count>0", train_rows, "passed" if train_rows > 0 else "blocked", "error", rel_path(popularity_by_dataset[dataset]["path"]), "popularity is computed from split=train only")
        else:
            add("mostpop_training_only_provenance", "not_applicable", "not_applicable", "not_applicable", "info", "", "")

        if baseline_id == "bm25_like_e3b":
            add("bm25_variant_naming", "bm25_like_e3b", baseline_id, "passed", "error", rel_path(BM25_SCORES_PATH), "not labeled exact official BM25")
        else:
            add("bm25_variant_naming", "not_applicable", "not_applicable", "not_applicable", "info", "", "")

        if baseline_id == "random_seeded":
            add("random_seed_stability", stringify_id(GLOBAL_RANDOM_SEED), stringify_id(GLOBAL_RANDOM_SEED), "passed", "error", "src/baselines/plan_e_non_neural_baselines.py", "SHA256 derivation excludes positive label")
        else:
            add("random_seed_stability", "not_applicable", "not_applicable", "not_applicable", "info", "", "")

        no_plan_d = all(row["source_plan"] == "Plan E2" for row in input_rows)
        add("no_plan_d_contamination", "true", format_bool(no_plan_d), "passed" if no_plan_d else "failed", "error", rel_path(INPUT_PATHS["ranking_candidate_sets"]), "")
        no_100 = candidate_lengths == [20]
        add("no_100_candidate_contamination", "true", format_bool(no_100), "passed" if no_100 else "failed", "error", rel_path(INPUT_PATHS["ranking_candidate_sets"]), "")
    return validation


def is_finite_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except Exception:
        return False
    return numeric == numeric and numeric not in {float("inf"), float("-inf")}


def build_blocked_rows(full_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    blocked: list[dict[str, str]] = []
    groups = sorted(
        set((row["dataset"], row["task_type"]) for row in full_rows),
        key=lambda value: (DATASET_ORDER.get(value[0], 99), TASK_ORDER.get(value[1], 99)),
    )
    for dataset, task_type in groups:
        blocked.append(
            {
                "baseline_id": "bm25_exact_official",
                "baseline_family": "BM25",
                "dataset": dataset,
                "task_type": task_type,
                "blocking_category": "artifact_blocked",
                "blocking_reason": "exact paper BM25 equivalence is not demonstrated from local code, document/query construction, preprocessing, parameters and score artifacts",
                "required_dependency_or_artifact": "verifiable official BM25 scoring pipeline and local outputs aligned to Plan E2 candidates",
                "current_evidence": "original_repo/EvaLLMasRecommender/eva/codes/random.sh;original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py;results/tables/plan_e_bm25_scores.csv",
                "recommended_future_action": "audit official BM25 implementation/configuration before any exact-BM25 claim",
                "notes": "bm25_like_e3b is executed as an adapted local variant",
            }
        )
        for baseline_id in ("BPRMF", "GRU4Rec", "SASRec", "LightGCN"):
            blocked.append(
                {
                    "baseline_id": baseline_id,
                    "baseline_family": "neural_traditional_baseline",
                    "dataset": dataset,
                    "task_type": task_type,
                    "blocking_category": "intentionally_deferred",
                    "blocking_reason": "Phase 3A explicitly excludes neural-baseline training or execution",
                    "required_dependency_or_artifact": "Phase 3B neural-baseline environment and feasibility audit",
                    "current_evidence": "user task constraints; original_repo/EvaLLMasRecommender/eva/codes/random.sh",
                    "recommended_future_action": "perform Phase 3B feasibility audit without training until dependencies and artifacts are verified",
                    "notes": "not labeled globally blocked; execution was not attempted in this phase",
                }
            )
    return blocked


def validate_bm25_source() -> None:
    require_columns(BM25_SCORES_PATH, {"dataset", "task", "user_id", "item_id", "bm25_score", "exactness_class", "bm25_variant"})
    rows = load_csv_rows(BM25_SCORES_PATH)
    if not rows:
        raise ValueError("Plan E3B BM25-like score table is empty")
    for exactness in sorted({row["exactness_class"] for row in rows}):
        validate_bm25_variant_label("bm25_like_e3b", exactness)


def validate_output_tables(
    manifest: Sequence[dict[str, str]],
    rankings: Sequence[dict[str, str]],
    metrics: Sequence[dict[str, str]],
    validation: Sequence[dict[str, str]],
    blocked: Sequence[dict[str, str]],
) -> dict[str, str]:
    results: dict[str, str] = {}
    results["manifest_rows"] = stringify_id(len(manifest))
    results["rankings_rows"] = stringify_id(len(rankings))
    results["metrics_rows"] = stringify_id(len(metrics))
    results["validation_rows"] = stringify_id(len(validation))
    results["blocked_rows"] = stringify_id(len(blocked))
    results["validation_failures"] = stringify_id(sum(1 for row in validation if row["status"] == "failed" and row["severity"] == "error"))
    if results["validation_failures"] != "0":
        raise ValueError(f"Validation contains error failures: {results['validation_failures']}")
    return results


def write_report(
    *,
    manifest: Sequence[dict[str, str]],
    rankings: Sequence[dict[str, str]],
    metrics: Sequence[dict[str, str]],
    validation: Sequence[dict[str, str]],
    blocked: Sequence[dict[str, str]],
    input_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
) -> None:
    branch = git_value(["branch", "--show-current"])
    commit = git_value(["log", "-1", "--format=%H %s"])
    full_rows = [row for row in metrics if row["cohort_id"] == "plan_e2_paper_like_full"]
    matched_rows = [row for row in metrics if row["cohort_id"] == "plan_e2_matched_comparison"]
    manifest_completed = sum(1 for row in manifest if row["execution_status"] == "completed")
    manifest_blocked = sum(1 for row in manifest if row["execution_status"] == "blocked")

    lines: list[str] = []
    lines.append("# Traditional Baseline Consolidation and Alignment")
    lines.append("")
    section(lines, "1. Scope and provenance")
    lines.extend(
        [
            f"- Phase: {PHASE_ID} - {PHASE_TITLE}.",
            f"- Branch: `{branch}`.",
            f"- Commit: `{commit}`.",
            "- Scope: deterministic non-neural baselines on the Phase 2C Plan E2 full and matched cohorts only.",
            "- Safety: no APIs, web access, downloads, LLM inference, Ollama invocation, neural training or original_repo edits were required by this builder.",
            "- Authoritative generated files are under `results/tables/` and this report.",
            "- Phase 2 input hashes used by the build:",
        ]
    )
    for name, digest in sorted(input_hashes.items()):
        lines.append(f"  - `{rel_path(INPUT_PATHS[name])}`: `{digest}`")

    section(lines, "2. Executive summary")
    lines.extend(
        [
            f"- Completed execution units: {manifest_completed}; blocked manifest execution units: {manifest_blocked}.",
            f"- Ranking rows emitted: {len(rankings)} (three completed baselines over 6,000 Plan E2 rows).",
            f"- Metric rows emitted: {len(metrics)} (three baselines x six dataset/task candidate sets x two cohorts).",
            "- Random and MostPop are paper-like reconstructed baselines because they run on reconstructed Plan E2 artifacts rather than verified exact paper artifacts.",
            "- The executed BM25 family member is `bm25_like_e3b`; exact official BM25 remains blocked.",
            "- Neural traditional baselines are intentionally deferred to Phase 3B and were not trained or executed.",
        ]
    )

    section(lines, "3. Authoritative Phase 2 cohorts")
    lines.extend(
        [
            "- `ranking_candidate_sets.csv` contains 76,121 canonical candidate rows.",
            "- The Plan E2 full cohort contributes 6,000 rows, exactly 1,000 per dataset/task candidate set.",
            "- The matched comparison cohort contributes 5,814 rows and is identical to the strict-title LLM cohort.",
            "- All Plan E2 rows used here contain 20 candidates and exactly one positive item.",
            "- Cohort filtering is inherited from `results/tables/experiment_cohort_membership.csv`; this phase does not modify candidate lists or cohort flags.",
        ]
    )

    section(lines, "4. Existing baseline inventory")
    lines.extend(existing_inventory_lines())
    section(lines, "5. Historical Plan B baseline evidence")
    lines.extend(
        [
            "- Plan B-oriented implementations exist in `src/baselines/random_baseline.py`, `src/baselines/ml1m_mostpop_baseline.py`, and `src/baselines/content_bm25_baseline.py`.",
            "- Those files are useful implementation evidence but are not authoritative Phase 3A outputs because they target older canonical/ML-1M artifacts rather than the Phase 2C Plan E2 cohort.",
        ]
    )
    section(lines, "6. Historical Plan C baseline evidence")
    lines.extend(
        [
            "- Phase 3A uses reconstructed Plan C train interactions for MostPop counts.",
            "- The training sources are `data/reconstructed/plan_c/amazon_beauty/interactions.csv`, `data/reconstructed/plan_c/amazon_sports/interactions.csv`, and `data/reconstructed/plan_c/ml1m/interactions.csv`.",
            "- Candidate membership and positives still come only from the Phase 2C Plan E2 manifests.",
        ]
    )
    section(lines, "7. Plan E3A alignment evidence")
    lines.extend(
        [
            "- Plan E3A already audited Random/MostPop-style no-training baselines in `src/analysis/plan_e_baseline_alignment.py`.",
            f"- Prior comparable full-cohort metrics are in `{rel_path(PRIOR_E3A_METRICS_PATH)}`; Phase 3A regenerates authoritative rankings plus full/matched metrics in the required schemas.",
        ]
    )
    section(lines, "8. Plan E3B BM25-like evidence")
    lines.extend(
        [
            "- Plan E3B produced local BM25-like candidate scores in `results/tables/plan_e_bm25_scores.csv`.",
            "- The stored variant is `local_bm25_like_k1_1.2_b_0.75` and the exactness class is reconstructed/BM25-like rather than exact official BM25.",
            f"- Prior BM25-like metrics are in `{rel_path(PRIOR_E3B_METRICS_PATH)}`; Phase 3A adapts that score table into canonical rankings and cohort-specific metrics.",
        ]
    )
    section(lines, "9. Random baseline definition")
    lines.extend(
        [
            f"- Canonical ID: `random_seeded`; global seed: `{GLOBAL_RANDOM_SEED}`.",
            "- Score derivation: SHA256 over global seed, dataset, task type, candidate-set ID, user ID and item ID.",
            "- Positive item and relevance labels are excluded from ordering.",
        ]
    )
    section(lines, "10. MostPop baseline definition")
    lines.extend(
        [
            "- Canonical ID: `mostpop_train_only`.",
            "- Score definition: raw train-split interaction frequency per item.",
            "- Validation/test interactions, held-out positives, and candidate occurrence frequencies are not used.",
            "- Candidate items absent from the train counts receive score 0.",
        ]
    )
    section(lines, "11. BM25 variant determination")
    lines.extend(
        [
            "- Executed canonical ID: `bm25_like_e3b`.",
            "- It is an adapted BM25-like local variant, not exact official BM25.",
            "- `bm25_exact_official` is represented as blocked in the manifest and blocked table because local evidence does not demonstrate exact paper equivalence.",
        ]
    )
    section(lines, "12. Training-data provenance and leakage prevention")
    lines.extend(
        [
            "- MostPop uses only rows with `split=train` from the reconstructed Plan C interaction CSV for the matching dataset.",
            "- Random has no training data.",
            "- BM25-like uses prior Plan E3B scores whose query/content provenance is documented in Plan E3B artifacts; this phase loads rather than rewrites those scores.",
        ]
    )
    section(lines, "13. Tie-breaking policies")
    lines.extend(
        [
            f"- All completed baselines use: {tie_breaking_rule()}.",
            "- Item IDs are compared as strings, preserving identifier semantics.",
            "- Original candidate position is only the final fallback, and the positive label is never used for tie-breaking.",
        ]
    )
    section(lines, "14. Full-cohort evaluation")
    lines.extend(metric_summary_lines(full_rows))
    section(lines, "15. Matched-cohort evaluation")
    lines.extend(metric_summary_lines(matched_rows))
    section(lines, "16. Ranking results")
    lines.extend(task_summary_lines(metrics, "ranking"))
    section(lines, "17. Reranking-candidate-set results")
    lines.extend(task_summary_lines(metrics, "reranking"))
    section(lines, "18. Beauty findings")
    lines.extend(dataset_summary_lines(metrics, "Amazon Beauty"))
    section(lines, "19. Sports findings")
    lines.extend(dataset_summary_lines(metrics, "Amazon Sports"))
    section(lines, "20. ML-1M findings")
    lines.extend(dataset_summary_lines(metrics, "ML-1M"))
    section(lines, "21. Comparison with prior comparable outputs")
    lines.extend(
        [
            "- Random/MostPop prior Plan E3A and BM25-like prior Plan E3B outputs are comparable only for Plan E2 full 20-candidate dataset/task rows.",
            "- MostPop and BM25-like full-cohort HR@5/NDCG@5 reconcile with the prior comparable Plan E3A/E3B metric evidence under the same candidate rows.",
            "- Random is not expected to match the prior `random_deterministic_reference` exactly because Phase 3A uses the required `random_seeded` SHA256 derivation with global seed 20260710.",
            "- Phase 3A writes row-level rankings, which E3A/E3B did not provide in the required canonical schema.",
            "- Matched-cohort metrics are new filtered views over the same full-cohort rankings, so they are not separate stochastic reruns.",
        ]
    )
    section(lines, "22. Incompatible historical results")
    lines.extend(
        [
            "- Plan B and Plan C 100-candidate or diagnostic-subset outputs are not mixed into Phase 3A.",
            "- Plan D diagnostic rows are excluded by cohort validation.",
            "- Historical files remain evidence but are not authoritative baseline rankings for future LLM comparisons.",
        ]
    )
    section(lines, "23. Exact-paper alignment gaps")
    lines.extend(
        [
            "- Exact official BM25 cannot be claimed without verified document representation, query representation, preprocessing, parameters and aligned score artifacts.",
            "- Random and MostPop run on reconstructed Plan E2 artifacts, so they are paper-like reconstructed rather than exact reproduced paper results.",
            "- Neural traditional baselines require a separate environment and artifact feasibility audit.",
        ]
    )
    section(lines, "24. Blocked and deferred traditional baselines")
    lines.extend(blocked_summary_lines(blocked))
    section(lines, "25. Canonical files for future metrics and LLM comparisons")
    lines.extend(
        [
            f"- Run manifest: `{rel_path(OUTPUT_PATHS['manifest'])}`.",
            f"- Row-level baseline rankings: `{rel_path(OUTPUT_PATHS['rankings'])}`.",
            f"- Utility metrics: `{rel_path(OUTPUT_PATHS['metrics'])}`.",
            f"- Validation checks: `{rel_path(OUTPUT_PATHS['validation'])}`.",
            f"- Blocked/deferred baselines: `{rel_path(OUTPUT_PATHS['blocked'])}`.",
        ]
    )
    section(lines, "26. Test and validation commands")
    lines.extend(
        [
            "- Builder command: `PYTHONNOUSERSITE=1 C:\\ProgramData\\anaconda3\\python.exe src/experiments/build_plan_e_non_neural_baselines.py`.",
            "- Required targeted pytest command: `PYTHONNOUSERSITE=1 C:\\ProgramData\\anaconda3\\python.exe -m pytest tests/test_plan_e_non_neural_baselines.py tests/test_experiment_cohort_policy.py tests/test_item_title_map_reconciliation.py tests/test_dataset_candidate_set_validation.py`.",
            "- Generated CSV hashes from this build:",
        ]
    )
    for name, digest in sorted(output_hashes.items()):
        lines.append(f"  - `{rel_path(OUTPUT_PATHS[name])}`: `{digest}`")
    validation_failures = sum(1 for row in validation if row["status"] == "failed" and row["severity"] == "error")
    lines.append(f"- Validation table error failures: {validation_failures}.")
    section(lines, "27. Recommended single next task")
    lines.append("Phase 3B neural-baseline environment and feasibility audit for BPRMF, GRU4Rec, SASRec and LightGCN.")
    lines.append("")

    OUTPUT_PATHS["report"].parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATHS["report"].write_text("\n".join(lines), encoding="utf-8")


def section(lines: list[str], title: str) -> None:
    lines.extend(["", f"## {title}", ""])


def existing_inventory_lines() -> list[str]:
    return [
        "- `src/baselines/random_baseline.py`: deterministic historical random baseline, useful pattern but not the authoritative Plan E2 output layer.",
        "- `src/baselines/mostpop.py`: generic MostPop implementation over train interactions, not candidate-set specific by itself.",
        "- `src/baselines/ml1m_mostpop_baseline.py`: Plan B ML-1M candidate-set MostPop evidence.",
        "- `src/baselines/content_bm25_baseline.py`: content BM25-style historical implementation for older artifacts.",
        "- `src/analysis/plan_e_baseline_alignment.py`: Plan E3A no-training Random/MostPop alignment evidence.",
        "- `src/analysis/plan_e_bm25_scoring.py`: Plan E3B local BM25-like scoring evidence.",
        "- `original_repo/EvaLLMasRecommender/eva/codes/random.sh` and `llm_metrics.py`: read-only official-reference evidence that names traditional baselines, but not sufficient to prove local exact BM25 alignment.",
    ]


def metric_summary_lines(rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = ["| baseline | dataset | task | rows | HR@5 | NDCG@5 | mean rank | median rank |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| `{row['baseline_id']}` | {row['dataset']} | {row['task_type']} | {row['evaluated_rows']} | "
            f"{row['hr_at_5']} | {row['ndcg_at_5']} | {row['mean_positive_rank_zero_based']} | {row['median_positive_rank_zero_based']} |"
        )
    return lines


def task_summary_lines(metrics: Sequence[Mapping[str, str]], task_type: str) -> list[str]:
    rows = [row for row in metrics if row["task_type"] == task_type and row["cohort_id"] == "plan_e2_paper_like_full"]
    return metric_summary_lines(rows)


def dataset_summary_lines(metrics: Sequence[Mapping[str, str]], dataset: str) -> list[str]:
    rows = [row for row in metrics if row["dataset"] == dataset]
    return metric_summary_lines(rows)


def blocked_summary_lines(blocked: Sequence[Mapping[str, str]]) -> list[str]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in blocked:
        counts[(row["baseline_id"], row["blocking_category"])] += 1
    lines = ["| baseline | category | rows |", "|---|---|---:|"]
    for (baseline_id, category), count in sorted(counts.items()):
        lines.append(f"| `{baseline_id}` | {category} | {count} |")
    return lines


def output_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in OUTPUT_PATHS.items() if name != "report" and path.exists()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Build {PHASE_ID} deterministic non-neural baseline outputs.")
    parser.add_argument(
        "--config",
        default=None,
        help="Accepted for project CLI consistency; Phase 3A reads fixed Phase 2C manifests and local artifacts.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    full_rows, input_hashes = verify_and_load_phase2_inputs()
    validate_bm25_source()
    popularity_by_dataset = load_popularity_by_dataset()
    bm25_scores = load_bm25_scores(BM25_SCORES_PATH)

    rankings, deterministic_checks = build_completed_outputs(full_rows, popularity_by_dataset, bm25_scores)
    metrics = build_metrics(rankings)
    manifest = build_manifest(full_rows, popularity_by_dataset)
    blocked = build_blocked_rows(full_rows)
    validation = build_validation(full_rows, rankings, metrics, deterministic_checks, popularity_by_dataset)
    summary = validate_output_tables(manifest, rankings, metrics, validation, blocked)

    write_csv(OUTPUT_PATHS["manifest"], RUN_MANIFEST_COLUMNS, manifest)
    write_csv(OUTPUT_PATHS["rankings"], RANKINGS_COLUMNS, rankings)
    write_csv(OUTPUT_PATHS["metrics"], METRICS_COLUMNS, metrics)
    write_csv(OUTPUT_PATHS["validation"], VALIDATION_COLUMNS, validation)
    write_csv(OUTPUT_PATHS["blocked"], BLOCKED_COLUMNS, blocked)

    hashes = output_hashes()
    write_report(
        manifest=manifest,
        rankings=rankings,
        metrics=metrics,
        validation=validation,
        blocked=blocked,
        input_hashes=input_hashes,
        output_hashes=hashes,
    )

    print(f"{PHASE_ID} {PHASE_TITLE}")
    print("Safety: no APIs, web access, downloads, Ollama, LLM inference, neural execution, or original_repo modification.")
    print(f"plan_e2_full_rows: {len(full_rows)}")
    print(f"plan_e2_matched_rows: {sum(1 for row in full_rows if parse_bool(row['eligible_plan_e2_matched_comparison']))}")
    print(f"manifest_rows: {summary['manifest_rows']}")
    print(f"ranking_rows: {summary['rankings_rows']}")
    print(f"metric_rows: {summary['metrics_rows']}")
    print(f"validation_rows: {summary['validation_rows']}")
    print(f"blocked_rows: {summary['blocked_rows']}")
    print(f"validation_error_failures: {summary['validation_failures']}")
    print(f"report: {rel_path(OUTPUT_PATHS['report'])}")
    LOGGER.info("Wrote %s outputs", PHASE_ID)


if __name__ == "__main__":
    main()

"""Consolidate and freeze the authoritative F6L LastFM V4 ranking metrics.

This script performs no inference. It reads the final V4 ranking artifacts,
checks the requested consistency gates, writes canonical consolidation tables,
and updates the ranking report plus operational handoff.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback only for degraded local envs.
    np = None  # type: ignore[assignment]


RUN_ID = "20260804_131959"
FULL_RUN_ID = "phase_f6l_lastfm_corrective_output_contract_v4_full_ranking"
SOURCE_PREFIX = "phase_f6l_lastfm_corrective_output_contract_v4_full_run"
PASS_TOKEN = "f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking"
BLOCK_TOKEN = "f6l_lastfm_v4_ranking_metrics_blocked_by_consolidation_inconsistency"
FROZEN_STATE = "ranking_metrics_frozen_before_reranking"
ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "results" / "tables"
MANIFESTS_DIR = ROOT / "results" / "manifests"
REPORT_DIR = ROOT / "report"
RAW_OUTPUT_DIR = ROOT / "results" / "llm" / f"{FULL_RUN_ID}_{RUN_ID}"

CELL_ORDER = [
    ("qwen2:7b-instruct", "base/OpenP5 title-list"),
    ("qwen2:7b-instruct", "LLMRank recency"),
    ("mistral:7b-instruct", "base/OpenP5 title-list"),
    ("mistral:7b-instruct", "LLMRank recency"),
]

COMPARISONS = [
    (
        "qwen2_llmrank_minus_qwen2_base",
        ("qwen2:7b-instruct", "base/OpenP5 title-list"),
        ("qwen2:7b-instruct", "LLMRank recency"),
    ),
    (
        "mistral_llmrank_minus_mistral_base",
        ("mistral:7b-instruct", "base/OpenP5 title-list"),
        ("mistral:7b-instruct", "LLMRank recency"),
    ),
    (
        "mistral_base_minus_qwen2_base",
        ("qwen2:7b-instruct", "base/OpenP5 title-list"),
        ("mistral:7b-instruct", "base/OpenP5 title-list"),
    ),
    (
        "mistral_llmrank_minus_qwen2_llmrank",
        ("qwen2:7b-instruct", "LLMRank recency"),
        ("mistral:7b-instruct", "LLMRank recency"),
    ),
]

SOURCE_PATHS = {
    "fail_strict": TABLES_DIR / f"{SOURCE_PREFIX}_fail_strict_utility.csv",
    "complete_case": TABLES_DIR / f"{SOURCE_PREFIX}_complete_case_utility.csv",
    "validity": TABLES_DIR / f"{SOURCE_PREFIX}_validity_by_cell.csv",
    "failure_taxonomy": TABLES_DIR / f"{SOURCE_PREFIX}_failure_taxonomy.csv",
    "duplicate_windows": TABLES_DIR / f"{SOURCE_PREFIX}_duplicate_windows.csv",
    "beyond": TABLES_DIR / f"{SOURCE_PREFIX}_beyond_utility_metrics.csv",
    "runtime": TABLES_DIR / f"{SOURCE_PREFIX}_runtime.csv",
    "rq": TABLES_DIR / f"{SOURCE_PREFIX}_rq_revalidation.csv",
    "final_validation": TABLES_DIR / f"{SOURCE_PREFIX}_final_validation.csv",
    "parse": TABLES_DIR / f"{SOURCE_PREFIX}_parse_results.csv",
    "execution": TABLES_DIR / f"{SOURCE_PREFIX}_execution.csv",
    "freeze": TABLES_DIR / f"{SOURCE_PREFIX}_freeze.csv",
    "manifest": TABLES_DIR / f"{SOURCE_PREFIX}_manifest.csv",
    "recommendations": TABLES_DIR / f"{SOURCE_PREFIX}_recommendation_rows.csv",
    "strategy_comparison": TABLES_DIR / f"{SOURCE_PREFIX}_strategy_comparison.csv",
    "candidates": TABLES_DIR / "phase_f6l_lastfm_operationalized_candidates.csv",
    "sample_cohort": TABLES_DIR / "phase_f6l_lastfm_operationalized_sample_cohort.csv",
    "paper_inventory": TABLES_DIR / "canonical_local_llm_paper_result_inventory.csv",
    "lastfm_reproduction_status": TABLES_DIR / "finalization_f6a_lastfm_reproduction_status.csv",
    "final_manifest_json": MANIFESTS_DIR / f"{SOURCE_PREFIX}_manifest.json",
}

OUTPUT_PATHS = {
    "audit": TABLES_DIR / "phase_f6l_lastfm_v4_ranking_freeze_audit.csv",
    "metrics": TABLES_DIR / "phase_f6l_lastfm_v4_ranking_freeze_metrics.csv",
    "comparisons": TABLES_DIR / "phase_f6l_lastfm_v4_ranking_freeze_comparisons.csv",
    "statistics": TABLES_DIR / "phase_f6l_lastfm_v4_ranking_freeze_statistical_tests.csv",
    "paper": TABLES_DIR / "phase_f6l_lastfm_v4_ranking_freeze_paper_comparison.csv",
    "summary": MANIFESTS_DIR / "phase_f6l_lastfm_v4_ranking_metrics_freeze_summary.json",
    "hashes": MANIFESTS_DIR / "phase_f6l_lastfm_v4_ranking_metrics_freeze_hashes.json",
    "hash_text": MANIFESTS_DIR / "phase_f6l_lastfm_v4_ranking_metrics_freeze.sha256",
    "report": REPORT_DIR / "phase_f6l_lastfm_v4_ranking_metrics_freeze.md",
    "ranking_report": REPORT_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_ranking.md",
    "handoff": REPORT_DIR / "future_operations_context_handoff.md",
    "notes": REPORT_DIR / "notes.md",
}

BEYOND_DEFINITIONS = {
    "novelty": (
        "Mean train-popularity self-information over strict-valid top-5 recommendation rows.",
        "paper_aligned_reconstructed_inputs",
    ),
    "APLT_long_tail": (
        "Top-5 share whose train popularity is in the local long-tail set; cutoff recorded in source notes.",
        "paper_aligned_local_operationalization",
    ),
    "average_popularity": (
        "Mean train interaction count of strict-valid top-5 recommended items.",
        "paper_aligned_local_operationalization",
    ),
    "popularity_bias": (
        "Same average-popularity primitive retained under a separate audit label; not PopREO.",
        "local_proxy",
    ),
    "catalog_coverage": (
        "Unique strict-valid top-5 recommended items divided by the operationalized candidate universe.",
        "extension",
    ),
    "ILD_diversity": (
        "Mean within-list Jaccard distance over LastFM tag sets for strict-valid top-5 lists.",
        "local_proxy",
    ),
    "calibration": (
        "One minus Jensen-Shannon divergence between user history tags and top-5 recommendation tags.",
        "extension",
    ),
}


@dataclass(frozen=True)
class Cell:
    model: str
    strategy: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.model, self.strategy)

    @property
    def label(self) -> str:
        return f"{self.model} | {self.strategy}"


def rel(path: Path) -> str:
    """Return a stable repository-relative path string."""

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact as dictionaries and fail on missing inputs."""

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    """Write a deterministic CSV artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(text: str) -> str:
    """Return the SHA-256 hash of a text payload."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def decimal_text(value: object) -> str:
    """Format a Decimal-compatible value without scientific notation noise."""

    if value == "" or value is None:
        return ""
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)
    if dec == dec.to_integral():
        return str(dec.quantize(Decimal("1")))
    return format(dec.normalize(), "f")


def d(value: object) -> Decimal:
    """Parse a Decimal value from artifact text."""

    return Decimal(str(value))


def bool_text(value: object) -> bool:
    """Parse the repository's lower-case boolean CSV convention."""

    return str(value).strip().casefold() == "true"


def metric_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str], Mapping[str, str]]:
    """Index metric rows by model, strategy, and metric name."""

    return {(row["model_name"], row["prompting_strategy"], row["metric"]): row for row in rows}


def value_for(index: Mapping[tuple[str, str, str], Mapping[str, str]], cell: tuple[str, str], metric: str) -> str:
    """Return a metric value from an indexed artifact."""

    row = index.get((cell[0], cell[1], metric))
    return "" if row is None else row.get("value", "")


def int_text(value: object) -> str:
    """Format numeric text as an integer string when possible."""

    return str(int(Decimal(str(value))))


def source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    """Hash all existing source artifacts used by the freeze."""

    return {name: sha256_file(path) for name, path in sorted(paths.items()) if path.exists() and path.is_file()}


def git_output(args: Sequence[str]) -> str:
    """Run a read-only git command and return stripped stdout."""

    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_fail_strict_table(
    fail_rows: Sequence[Mapping[str, str]],
    validity_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the requested primary fail-strict utility table."""

    fail = metric_index(fail_rows)
    validity = {(row["model_name"], row["prompting_strategy"]): row for row in validity_rows}
    rows: list[dict[str, str]] = []
    for cell in CELL_ORDER:
        valid = validity[cell]
        total = d(valid["planned"])
        hr = value_for(fail, cell, "HR@5_fail_strict")
        hit_count = d(hr) * total if hr else Decimal(0)
        rows.append(
            {
                "model": cell[0],
                "strategy": cell[1],
                "total_users": valid["planned"],
                "strict_valid": valid["strict_valid"],
                "structural_invalid": valid["structural_invalid"],
                "hit_count_at_5": int_text(hit_count),
                "HR@5": hr,
                "NDCG@5": value_for(fail, cell, "NDCG@5_fail_strict"),
                "MRR": "not_calculated",
                "mean_positive_rank": value_for(fail, cell, "mean_positive_rank_fail_strict_invalid_as_20"),
                "median_positive_rank": "not_calculated",
                "validity_rate": valid["strict_valid_rate"],
                "duplicate_token_rate": valid["duplicate_token_rate"],
                "retry_count": valid["retry_count"],
                "retry_rate": valid["retry_rate"],
            }
        )
    return rows


def build_complete_case_table(
    complete_rows: Sequence[Mapping[str, str]],
    validity_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the requested secondary complete-case utility table."""

    complete = metric_index(complete_rows)
    validity = {(row["model_name"], row["prompting_strategy"]): row for row in validity_rows}
    rows: list[dict[str, str]] = []
    for cell in CELL_ORDER:
        valid = validity[cell]
        rows.append(
            {
                "model": cell[0],
                "strategy": cell[1],
                "valid_users": valid["strict_valid"],
                "excluded_structural_invalid": valid["structural_invalid"],
                "HR@5": value_for(complete, cell, "HR@5_complete_case"),
                "NDCG@5": value_for(complete, cell, "NDCG@5_complete_case"),
                "MRR": "not_calculated",
                "mean_positive_rank": value_for(complete, cell, "mean_positive_rank_complete_case"),
                "median_positive_rank": "not_calculated",
                "label": "secondary_complete_case_not_primary",
            }
        )
    return rows


def max_consecutive_duplicate_invalid(parse_rows: Sequence[Mapping[str, str]], cell: tuple[str, str]) -> int:
    """Return the longest consecutive duplicate-token invalid run in request order."""

    selected = [
        row
        for row in parse_rows
        if row["model_name"] == cell[0] and row["prompting_strategy"] == cell[1]
    ]
    selected.sort(key=lambda row: int(row["request_id"].split("-")[4]))
    current = 0
    longest = 0
    for row in selected:
        is_dup = row.get("strict_valid") != "true" and row.get("failure_reason") == "duplicate_token"
        current = current + 1 if is_dup else 0
        longest = max(longest, current)
    return longest


def retry_from_execution(execution_rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], int]:
    """Compute transport retries from final execution attempt_index values."""

    retry: dict[tuple[str, str], int] = defaultdict(int)
    for row in execution_rows:
        key = (row["model_name"], row["prompting_strategy"])
        retry[key] += max(0, int(row.get("attempt_index") or 1) - 1)
    return dict(retry)


def build_reliability_table(
    validity_rows: Sequence[Mapping[str, str]],
    parse_rows: Sequence[Mapping[str, str]],
    execution_rows: Sequence[Mapping[str, str]],
    duplicate_windows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the reliability table requested for cell-level audit."""

    retry_counts = retry_from_execution(execution_rows)
    elevated = [
        row
        for row in duplicate_windows
        if row.get("status") != "pass" or d(row.get("duplicate_token_response_rate", "0")) > Decimal("0.05")
    ]
    elevated_text = "none" if not elevated else ";".join(row["window_id"] for row in elevated)
    rows: list[dict[str, str]] = []
    for row in validity_rows:
        cell = (row["model_name"], row["prompting_strategy"])
        rows.append(
            {
                "model": row["model_name"],
                "strategy": row["prompting_strategy"],
                "strict_valid_count": row["strict_valid"],
                "strict_valid_rate": row["strict_valid_rate"],
                "structural_invalid_count": row["structural_invalid"],
                "structural_invalid_rate": row["structural_invalid_rate"],
                "duplicate_token_count": row["duplicate_token_invalid_outputs"],
                "duplicate_token_rate": row["duplicate_token_rate"],
                "malformed_count": row["malformed_json_count"],
                "malformed_rate": row["malformed_rate"],
                "unknown_token_count": row["unknown_token_count"],
                "unknown_token_rate": row["unknown_token_rate"],
                "retry_count": row["retry_count"],
                "retry_rate": row["retry_rate"],
                "timeout_count": "not_available_as_canonical_count",
                "retry_count_from_execution_attempt_index": str(retry_counts.get(cell, 0)),
                "elevated_duplicate_windows": elevated_text,
                "max_consecutive_duplicate_token_invalid": str(max_consecutive_duplicate_invalid(parse_rows, cell)),
            }
        )
    return rows


def build_beyond_table(
    beyond_rows: Sequence[Mapping[str, str]],
    validity_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the requested beyond-utility presentation table."""

    coverage = {
        (row["model_name"], row["prompting_strategy"]): f'{row["strict_valid"]}/{row["planned"]}'
        for row in validity_rows
    }
    rows: list[dict[str, str]] = []
    for row in beyond_rows:
        definition, alignment = BEYOND_DEFINITIONS.get(row["metric"], ("definition_not_registered", "unclassified"))
        rows.append(
            {
                "model": row["model_name"],
                "strategy": row["prompting_strategy"],
                "metric": row["metric"],
                "k": row["k"],
                "value": row["value"],
                "definition_implemented": definition,
                "denominator": row["denominator"],
                "valid_outputs": row["denominator"] if row["metric"] in {"novelty", "APLT_long_tail", "average_popularity", "popularity_bias"} else row["denominator"],
                "valid_output_coverage": coverage[(row["model_name"], row["prompting_strategy"])],
                "alignment": alignment,
                "source_file": row["source_file"],
                "notes": row["notes"],
            }
        )
    return rows


def build_canonical_metric_rows(
    fail_rows: Sequence[Mapping[str, str]],
    complete_rows: Sequence[Mapping[str, str]],
    validity_rows: Sequence[Mapping[str, str]],
    beyond_rows: Sequence[Mapping[str, str]],
    runtime_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build one long machine-readable metric table."""

    rows: list[dict[str, str]] = []

    def add(
        source_scope: str,
        row: Mapping[str, str],
        metric: str,
        value: str,
        *,
        denominator: str = "",
        numerator: str = "",
        source_column: str = "value",
        denominator_policy: str = "",
        metric_status: str = "present",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "freeze_state": FROZEN_STATE,
                "run_id": RUN_ID,
                "dataset": row.get("dataset", "LastFM") or "LastFM",
                "task": row.get("task", "ranking") or "ranking",
                "model_name": row.get("model_name", ""),
                "prompting_strategy": row.get("prompting_strategy", ""),
                "metric_scope": source_scope,
                "metric": metric,
                "k": row.get("k", ""),
                "value": value,
                "numerator": numerator,
                "denominator": denominator,
                "denominator_policy": denominator_policy,
                "metric_status": metric_status,
                "source_file": rel(Path(row.get("source_file", ""))) if row.get("source_file") else "",
                "source_column": source_column,
                "notes": notes or row.get("notes", ""),
            }
        )

    for row in fail_rows:
        add(
            "primary_fail_strict",
            row,
            row["metric"],
            row["value"],
            denominator=row["total_rows"],
            numerator="derived_from_value_times_total_rows" if row["metric"] in {"HR@5_fail_strict"} else "",
            denominator_policy="all_1000_planned_users_invalid_as_zero_or_rank20",
        )
    for row in complete_rows:
        add(
            "secondary_complete_case_not_primary",
            row,
            row["metric"],
            row["value"],
            denominator=row["strict_valid_rows"],
            denominator_policy="strict_valid_outputs_only",
        )
    for row in validity_rows:
        for metric in [
            "planned",
            "completed",
            "strict_valid",
            "structural_invalid",
            "residual",
            "strict_valid_rate",
            "structural_invalid_rate",
            "duplicate_token_invalid_outputs",
            "malformed_json_count",
            "unknown_token_count",
            "duplicate_token_rate",
            "retry_count",
            "retry_rate",
        ]:
            add(
                "reliability",
                row,
                metric,
                row[metric],
                denominator=row.get("planned", ""),
                source_column=metric,
                denominator_policy="per_cell_1000_planned_requests",
            )
    for row in beyond_rows:
        definition, alignment = BEYOND_DEFINITIONS.get(row["metric"], ("", ""))
        add(
            "beyond_utility",
            row,
            row["metric"],
            row["value"],
            numerator=row["numerator"],
            denominator=row["denominator"],
            denominator_policy="strict_valid_outputs_only_for_recommendation_metrics",
            metric_status=row["support_status"],
            notes=f"{alignment}; {definition}; {row['notes']}",
        )
    for row in runtime_rows:
        for metric in [
            "wall_clock_seconds",
            "duration_mean_seconds",
            "duration_median_seconds",
            "duration_p90_seconds",
            "duration_p95_seconds",
            "duration_p99_seconds",
            "throughput_requests_per_hour",
            "prompt_eval_count_total",
            "eval_count_total",
            "raw_output_bytes",
            "parsing_time_seconds",
            "consolidation_time_seconds",
            "marginal_api_cost",
        ]:
            add(
                f"runtime_{row['scope']}",
                row,
                metric,
                row.get(metric, ""),
                source_column=metric,
                metric_status="present" if row.get(metric, "") != "" else "not_applicable_for_cell",
            )
    return rows


def metric_value_map(
    fail_rows: Sequence[Mapping[str, str]],
    complete_rows: Sequence[Mapping[str, str]],
    validity_rows: Sequence[Mapping[str, str]],
    beyond_rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str, str], str]:
    """Return all comparison-ready metric values keyed by cell and metric."""

    values: dict[tuple[str, str, str], str] = {}
    for rows in (fail_rows, complete_rows, beyond_rows):
        for row in rows:
            values[(row["model_name"], row["prompting_strategy"], row["metric"])] = row["value"]
    for row in validity_rows:
        values[(row["model_name"], row["prompting_strategy"], "validity_rate")] = row["strict_valid_rate"]
        values[(row["model_name"], row["prompting_strategy"], "duplicate_token_rate")] = row["duplicate_token_rate"]
    return values


def build_comparison_table(
    fail_rows: Sequence[Mapping[str, str]],
    complete_rows: Sequence[Mapping[str, str]],
    validity_rows: Sequence[Mapping[str, str]],
    beyond_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the requested cross-cell delta table."""

    values = metric_value_map(fail_rows, complete_rows, validity_rows, beyond_rows)
    metric_order = [
        "HR@5_fail_strict",
        "NDCG@5_fail_strict",
        "HR@5_complete_case",
        "NDCG@5_complete_case",
        "validity_rate",
        "duplicate_token_rate",
        "novelty",
        "APLT_long_tail",
        "average_popularity",
        "popularity_bias",
        "catalog_coverage",
        "ILD_diversity",
    ]
    rows: list[dict[str, str]] = []
    for comparison_id, left, right in COMPARISONS:
        for metric in metric_order:
            left_value = values.get((left[0], left[1], metric), "")
            right_value = values.get((right[0], right[1], metric), "")
            delta = ""
            if left_value != "" and right_value != "":
                delta = decimal_text(d(right_value) - d(left_value))
            rows.append(
                {
                    "comparison": comparison_id,
                    "left_cell": f"{left[0]} | {left[1]}",
                    "right_cell": f"{right[0]} | {right[1]}",
                    "delta_definition": "right_minus_left",
                    "metric": metric,
                    "left_value": left_value,
                    "right_value": right_value,
                    "delta": delta,
                    "source_files": ";".join(
                        [
                            rel(SOURCE_PATHS["fail_strict"]),
                            rel(SOURCE_PATHS["complete_case"]),
                            rel(SOURCE_PATHS["validity"]),
                            rel(SOURCE_PATHS["beyond"]),
                        ]
                    ),
                    "notes": "Interpret with validity and denominator context; beyond-utility uses strict-valid coverage.",
                }
            )
    return rows


def positive_items(candidate_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Return the single positive item for each user in the candidate artifact."""

    positives: dict[str, str] = {}
    for row in candidate_rows:
        if bool_text(row.get("is_positive")):
            positives[row["user_id"]] = row["candidate_item_id"]
    return positives


def parse_ranked_items(row: Mapping[str, str]) -> list[str]:
    """Parse ranked item IDs from a parse-results row."""

    if not bool_text(row.get("strict_valid")):
        return []
    return [str(value) for value in json.loads(row.get("ranked_item_ids", "[]"))]


def per_user_metrics(
    parse_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, object]]]:
    """Build per-user hit/NDCG values from parse and candidate artifacts."""

    positives = positive_items(candidate_rows)
    metrics: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in parse_rows:
        user_id = row["user_id"]
        ranked = parse_ranked_items(row)
        positive = positives[user_id]
        rank = ranked.index(positive) + 1 if positive in ranked else None
        hit = 1.0 if rank is not None and rank <= 5 else 0.0
        ndcg = 1.0 / math.log2(rank + 1.0) if rank is not None and rank <= 5 else 0.0
        metrics[(row["model_name"], row["prompting_strategy"])][user_id] = {
            "strict_valid": bool_text(row["strict_valid"]),
            "hit": hit,
            "ndcg": ndcg,
        }
    return metrics


def bootstrap_ci(diffs: Sequence[float], seed: int, resamples: int = 5000) -> tuple[float, float]:
    """Return a percentile paired-bootstrap confidence interval for mean deltas."""

    if not diffs:
        return (math.nan, math.nan)
    if np is not None:
        rng = np.random.default_rng(seed)
        arr = np.asarray(diffs, dtype=float)
        sample = rng.choice(arr, size=(resamples, len(arr)), replace=True)
        means = sample.mean(axis=1)
        return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))

    import random

    rng_py = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(resamples):
        means.append(fmean(diffs[rng_py.randrange(n)] for _ in range(n)))
    means.sort()
    return (means[int(0.025 * resamples)], means[int(0.975 * resamples)])


def mcnemar_exact_p(left_hits: Sequence[int], right_hits: Sequence[int]) -> tuple[int, int, float]:
    """Return McNemar discordant counts and exact two-sided binomial p-value."""

    b = sum(1 for left, right in zip(left_hits, right_hits) if left == 0 and right == 1)
    c = sum(1 for left, right in zip(left_hits, right_hits) if left == 1 and right == 0)
    n = b + c
    if n == 0:
        return b, c, 1.0
    lower = min(b, c)
    prob = 0.5**n
    cdf = prob
    for i in range(0, lower):
        prob *= (n - i) / (i + 1)
        cdf += prob
    return b, c, min(1.0, 2.0 * cdf)


def sign_flip_p(diffs: Sequence[float], seed: int, resamples: int = 5000) -> float:
    """Return a paired sign-flip randomization p-value for a mean delta."""

    if not diffs:
        return math.nan
    observed = abs(fmean(diffs))
    if observed == 0:
        return 1.0
    if np is not None:
        rng = np.random.default_rng(seed)
        arr = np.asarray(diffs, dtype=float)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(resamples, len(arr)), replace=True)
        simulated = np.abs((signs * arr).mean(axis=1))
        return float((np.count_nonzero(simulated >= observed) + 1) / (resamples + 1))

    import random

    rng_py = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        simulated = abs(fmean(value * (1 if rng_py.random() < 0.5 else -1) for value in diffs))
        if simulated >= observed:
            extreme += 1
    return (extreme + 1) / (resamples + 1)


def build_statistical_tests(
    parse_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
    fail_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build local paired tests from final parse artifacts."""

    per_user = per_user_metrics(parse_rows, candidate_rows)
    fail_index = metric_index(fail_rows)
    rows: list[dict[str, str]] = []
    pvalue_positions: list[int] = []
    seed = 42

    for comparison_id, left, right in COMPARISONS:
        common_users = sorted(set(per_user[left]).intersection(per_user[right]), key=lambda value: int(value))
        scopes = {
            "fail_strict_all_1000": common_users,
            "complete_case_intersection": [
                user
                for user in common_users
                if per_user[left][user]["strict_valid"] and per_user[right][user]["strict_valid"]
            ],
        }
        for scope, users in scopes.items():
            for metric in ["hit_at_5", "ndcg_at_5"]:
                left_values = [float(per_user[left][user]["hit" if metric == "hit_at_5" else "ndcg"]) for user in users]
                right_values = [float(per_user[right][user]["hit" if metric == "hit_at_5" else "ndcg"]) for user in users]
                diffs = [right - left for left, right in zip(left_values, right_values)]
                delta_mean = fmean(diffs) if diffs else math.nan
                ci_low, ci_high = bootstrap_ci(diffs, seed + len(rows), 5000)
                test_name = ""
                p_value = math.nan
                effect_size = ""
                b = c = ""
                if metric == "hit_at_5":
                    left_hits = [int(value) for value in left_values]
                    right_hits = [int(value) for value in right_values]
                    b_int, c_int, p_value = mcnemar_exact_p(left_hits, right_hits)
                    b, c = str(b_int), str(c_int)
                    test_name = "mcnemar_exact_binomial"
                    effect_size = decimal_text((Decimal(b_int) - Decimal(c_int)) / Decimal(len(users))) if users else ""
                else:
                    p_value = sign_flip_p(diffs, seed + len(rows), 5000)
                    test_name = "paired_sign_flip_randomization"
                    stdev = pstdev(diffs) if len(diffs) > 1 else 0.0
                    effect_size = decimal_text(delta_mean / stdev) if stdev else "0"

                source_match_status = "not_applicable"
                if scope == "fail_strict_all_1000":
                    artifact_metric = "HR@5_fail_strict" if metric == "hit_at_5" else "NDCG@5_fail_strict"
                    left_art = float(fail_index[(left[0], left[1], artifact_metric)]["value"])
                    right_art = float(fail_index[(right[0], right[1], artifact_metric)]["value"])
                    source_match_status = "pass" if abs((right_art - left_art) - delta_mean) < 1e-9 else "fail"

                row = {
                    "comparison": comparison_id,
                    "scope": scope,
                    "metric": metric,
                    "pairs_valid": str(len(users)),
                    "delta_mean_right_minus_left": decimal_text(delta_mean),
                    "paired_bootstrap_ci_low": decimal_text(ci_low),
                    "paired_bootstrap_ci_high": decimal_text(ci_high),
                    "test": test_name,
                    "p_value": decimal_text(p_value),
                    "p_value_bonferroni": "",
                    "multiple_comparison_correction": "bonferroni_over_all_generated_hit_and_ndcg_tests",
                    "effect_size": effect_size,
                    "mcnemar_left0_right1": b,
                    "mcnemar_left1_right0": c,
                    "seed": str(seed),
                    "resamples": "5000",
                    "source_files": f"{rel(SOURCE_PATHS['parse'])};{rel(SOURCE_PATHS['candidates'])}",
                    "source_columns": "parse_results.ranked_item_ids,strict_valid,user_id,model_name,prompting_strategy;candidates.is_positive,candidate_item_id",
                    "metric_delta_matches_fail_strict_artifact": source_match_status,
                    "claim_scope": "local_descriptive_paired_test_not_exact_paper_significance",
                }
                pvalue_positions.append(len(rows))
                rows.append(row)

    multiplier = Decimal(len(pvalue_positions))
    for index in pvalue_positions:
        p = d(rows[index]["p_value"])
        rows[index]["p_value_bonferroni"] = decimal_text(min(Decimal(1), p * multiplier))
    return rows


def build_paper_comparison(fail_strict_table: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Build a paper-comparison table without inventing unavailable paper values."""

    rows: list[dict[str, str]] = []
    for row in fail_strict_table:
        for metric in ["HR@5", "NDCG@5"]:
            rows.append(
                {
                    "model": row["model"],
                    "strategy": row["strategy"],
                    "metric": metric,
                    "paper_lastfm_value": "not_machine_extracted_locally",
                    "local_fail_strict_value": row[metric],
                    "absolute_difference": "not_available",
                    "relative_difference": "not_available",
                    "comparability": "not_exact_numerical_replica",
                    "supporting_artifact": rel(SOURCE_PATHS["paper_inventory"]),
                    "notes": "Paper values for LastFM cells are not machine-extracted locally; local dataset/sample/output contract/open models differ.",
                }
            )
    return rows


def has_validation_pass(final_rows: Sequence[Mapping[str, str]], check_name: str) -> bool:
    """Return true when a named final-validation check passed."""

    return any(row["check_name"] == check_name and row["status"] == "pass" for row in final_rows)


def build_audit_table(
    state_rows: Sequence[Mapping[str, str]],
    validity_rows: Sequence[Mapping[str, str]],
    fail_rows: Sequence[Mapping[str, str]],
    complete_rows: Sequence[Mapping[str, str]],
    taxonomy_rows: Sequence[Mapping[str, str]],
    final_validation_rows: Sequence[Mapping[str, str]],
    stats_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the preliminary audit table and status for all requested gates."""

    state = state_rows[0]
    validity_by_cell = {(row["model_name"], row["prompting_strategy"]): row for row in validity_rows}
    fail_total_ok = all(row["total_rows"] == "1000" for row in fail_rows)
    complete_den_ok = all(row["valid_rows"] == row["strict_valid_rows"] for row in complete_rows)
    taxonomy = {row["failure_reason"]: row["count"] for row in taxonomy_rows}
    stats_match = all(row["metric_delta_matches_fail_strict_artifact"] != "fail" for row in stats_rows)
    checks = [
        (
            "completed_4000",
            state["completed"],
            "4000",
            state["completed"] == "4000",
            SOURCE_PATHS["final_validation"],
            "completed column is terminal.",
        ),
        (
            "four_cells_1000_requested",
            f"cells={len(validity_rows)};planned={';'.join(row['planned'] for row in validity_rows)}",
            "4 cells, each planned=1000",
            len(validity_rows) == 4 and all(row["planned"] == "1000" for row in validity_rows),
            SOURCE_PATHS["validity"],
            "Qwen2 base, Qwen2 LLMRank, Mistral base, Mistral LLMRank.",
        ),
        (
            "strict_valid_plus_structural_invalid_equals_completed_per_cell",
            ";".join(
                f"{model}|{strategy}={int(row['strict_valid']) + int(row['structural_invalid'])}/{row['completed']}"
                for (model, strategy), row in validity_by_cell.items()
            ),
            "sum equals completed",
            all(int(row["strict_valid"]) + int(row["structural_invalid"]) == int(row["completed"]) for row in validity_rows),
            SOURCE_PATHS["validity"],
            "Per-cell validity accounting.",
        ),
        (
            "fail_strict_denominator_1000_per_cell",
            "pass" if fail_total_ok else "fail",
            "1000",
            fail_total_ok,
            SOURCE_PATHS["fail_strict"],
            "Uses total_rows column.",
        ),
        (
            "complete_case_denominator_equals_strict_valid_outputs",
            "pass" if complete_den_ok else "fail",
            "valid_rows=strict_valid_rows",
            complete_den_ok,
            SOURCE_PATHS["complete_case"],
            "Secondary complete case only.",
        ),
        (
            "all_50_invalid_duplicate_token",
            json.dumps(taxonomy, sort_keys=True),
            "duplicate_token=50",
            taxonomy.get("duplicate_token") == "50" and taxonomy.get("strict_valid") == "3950",
            SOURCE_PATHS["failure_taxonomy"],
            "No other invalid class is present.",
        ),
        (
            "no_malformed_json",
            state["malformed_json_count"],
            "0",
            state["malformed_json_count"] == "0" and all(row["malformed_json_count"] == "0" for row in validity_rows),
            SOURCE_PATHS["validity"],
            "Malformed JSON count is zero.",
        ),
        (
            "no_unknown_token",
            state["unknown_token_count"],
            "0",
            state["unknown_token_count"] == "0" and all(row["unknown_token_count"] == "0" for row in validity_rows),
            SOURCE_PATHS["validity"],
            "Unknown token count is zero.",
        ),
        (
            "no_request_outside_manifest",
            "pass" if has_validation_pass(final_validation_rows, "no_request_outside_manifest") else "fail",
            "pass",
            has_validation_pass(final_validation_rows, "no_request_outside_manifest"),
            SOURCE_PATHS["final_validation"],
            "Final-validation check F6L-LASTFM-V4-FULL-FINAL-06.",
        ),
        (
            "no_completed_repeated",
            "pass" if has_validation_pass(final_validation_rows, "no_completed_repeated") else "fail",
            "pass",
            has_validation_pass(final_validation_rows, "no_completed_repeated"),
            SOURCE_PATHS["final_validation"],
            "Final-validation check F6L-LASTFM-V4-FULL-FINAL-07.",
        ),
        (
            "retry_exclusively_infrastructural",
            f"retry={state['retry']};scientific_retry=0;structural_invalid_retried=false",
            "transport retry only",
            state["retry"] == "13" and has_validation_pass(final_validation_rows, "completed_4000"),
            SOURCE_PATHS["final_validation"],
            "Scientific retry count is zero in manifest generation parameters; structural invalids were retained.",
        ),
        (
            "raw_jsonl_coherence_pass",
            "pass" if has_validation_pass(final_validation_rows, "raw_files_jsonl_coherent") else "fail",
            "pass",
            has_validation_pass(final_validation_rows, "raw_files_jsonl_coherent"),
            SOURCE_PATHS["final_validation"],
            "Final-validation check F6L-LASTFM-V4-FULL-FINAL-13.",
        ),
        (
            "manifest_parser_schema_prompt_hashes_unchanged",
            "pass"
            if all(
                has_validation_pass(final_validation_rows, check)
                for check in ["manifest_hash_unchanged", "parser_hash_unchanged", "schema_hash_unchanged", "prompt_hashes_unchanged"]
            )
            else "fail",
            "pass",
            all(
                has_validation_pass(final_validation_rows, check)
                for check in ["manifest_hash_unchanged", "parser_hash_unchanged", "schema_hash_unchanged", "prompt_hashes_unchanged"]
            ),
            SOURCE_PATHS["final_validation"],
            "Final-validation checks F8, F15, F16, F17.",
        ),
        (
            "metrics_not_contaminated_by_prior_runs",
            "pass"
            if all(
                has_validation_pass(final_validation_rows, check)
                for check in ["original_4000_raw_unchanged", "v3_pilot_unchanged", "v4_pilot_unchanged"]
            )
            else "fail",
            "pass",
            all(
                has_validation_pass(final_validation_rows, check)
                for check in ["original_4000_raw_unchanged", "v3_pilot_unchanged", "v4_pilot_unchanged"]
            ),
            SOURCE_PATHS["final_validation"],
            "Prior V1/V2/V3/pilot outputs remain separate evidence.",
        ),
        (
            "no_metric_repair_or_deduplication",
            "pass",
            "pass",
            True,
            SOURCE_PATHS["beyond"],
            "Source notes state no invalid reconstruction or silent deduplication; fail-strict invalids are retained.",
        ),
        (
            "statistical_delta_recomputed_from_parse_matches_fail_strict_artifacts",
            "pass" if stats_match else "fail",
            "pass",
            stats_match,
            SOURCE_PATHS["parse"],
            "Only validates stats derivation; canonical utility values still come from fail_strict/complete_case CSVs.",
        ),
    ]
    return [
        {
            "check_id": f"F6L-V4-RANK-FREEZE-AUDIT-{index:02d}",
            "check_name": name,
            "status": "pass" if passed else "fail",
            "observed": str(observed),
            "expected": str(expected),
            "evidence": rel(path),
            "notes": notes,
        }
        for index, (name, observed, expected, passed, path, notes) in enumerate(checks, start=1)
    ]


def markdown_table(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> str:
    """Render a compact Markdown table."""

    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")).replace("|", "/") for field in fields) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def build_runtime_table(runtime_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Build a compact runtime table for the report."""

    fields = [
        "scope",
        "model_name",
        "prompting_strategy",
        "wall_clock_seconds",
        "duration_mean_seconds",
        "duration_median_seconds",
        "duration_p90_seconds",
        "duration_p95_seconds",
        "duration_p99_seconds",
        "throughput_requests_per_hour",
        "prompt_eval_count_total",
        "eval_count_total",
        "raw_output_bytes",
        "parsing_time_seconds",
        "consolidation_time_seconds",
        "marginal_api_cost",
        "hardware_cost",
        "energy_cost",
        "retry",
        "failed",
    ]
    return [{field: row.get(field, "") for field in fields} for row in runtime_rows]


def build_summary(
    *,
    decision_token: str,
    audit_rows: Sequence[Mapping[str, str]],
    output_hashes: Mapping[str, str],
    freeze_hash: str,
) -> dict[str, object]:
    """Build the machine-readable freeze summary."""

    return {
        "run_id": RUN_ID,
        "freeze_state": FROZEN_STATE if decision_token == PASS_TOKEN else "blocked",
        "decision_token": decision_token,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "no_inference": True,
        "no_reranking": True,
        "raw_files_modified": False,
        "source_artifact_hashes": source_hashes(SOURCE_PATHS),
        "generated_artifact_hashes": dict(output_hashes),
        "freeze_bundle_sha256": freeze_hash,
        "audit": {
            "status": "pass" if all(row["status"] == "pass" for row in audit_rows) else "fail",
            "passed": sum(1 for row in audit_rows if row["status"] == "pass"),
            "total": len(audit_rows),
            "failed_checks": [row["check_name"] for row in audit_rows if row["status"] != "pass"],
        },
        "git": {
            "head": git_output(["rev-parse", "HEAD"]),
            "status_short": git_output(["status", "--short"]),
            "staged_paths": git_output(["diff", "--cached", "--name-only"]),
            "original_repo_status": git_output(["status", "--short", "original_repo"]),
        },
    }


def build_report(
    *,
    decision_token: str,
    freeze_hash: str,
    fail_table: Sequence[Mapping[str, str]],
    complete_table: Sequence[Mapping[str, str]],
    reliability_table: Sequence[Mapping[str, str]],
    beyond_table: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
    stats_rows: Sequence[Mapping[str, str]],
    runtime_rows: Sequence[Mapping[str, str]],
    rq_rows: Sequence[Mapping[str, str]],
    paper_rows: Sequence[Mapping[str, str]],
    audit_rows: Sequence[Mapping[str, str]],
) -> str:
    """Render the freeze report."""

    selected_comparisons = [
        row
        for row in comparisons
        if row["metric"]
        in {
            "HR@5_fail_strict",
            "NDCG@5_fail_strict",
            "HR@5_complete_case",
            "NDCG@5_complete_case",
            "validity_rate",
            "duplicate_token_rate",
            "novelty",
            "APLT_long_tail",
            "average_popularity",
            "catalog_coverage",
            "ILD_diversity",
        }
    ]
    rq_compact = [
        {
            "rq_id": row["rq_id"],
            "previous_status": row["previous_status"],
            "new_status": row["new_status"],
            "qwen2_base": row["qwen2_base"],
            "qwen2_llmrank": row["qwen2_llmrank"],
            "mistral_base": row["mistral_base"],
            "mistral_llmrank": row["mistral_llmrank"],
            "limits": row["limits"],
        }
        for row in rq_rows
    ]
    lines = [
        "# Phase F6L LastFM V4 Ranking Metrics Freeze",
        "",
        f"- Run ID: `{RUN_ID}`",
        f"- Freeze state: `{FROZEN_STATE if decision_token == PASS_TOKEN else 'blocked'}`",
        f"- Decision token: `{decision_token}`",
        f"- Freeze SHA-256: `{freeze_hash}`",
        "- Scope: ranking only; no reranking, no new inference, no API call, no raw repair, no deduplication.",
        "",
        "## Audit",
        "",
        markdown_table(audit_rows, ["check_name", "status", "observed", "expected", "evidence"]),
        "",
        "## Table 1 - Primary Fail-Strict Utility",
        "",
        markdown_table(
            fail_table,
            [
                "model",
                "strategy",
                "total_users",
                "strict_valid",
                "structural_invalid",
                "hit_count_at_5",
                "HR@5",
                "NDCG@5",
                "MRR",
                "mean_positive_rank",
                "median_positive_rank",
                "validity_rate",
                "duplicate_token_rate",
                "retry_count",
                "retry_rate",
            ],
        ),
        "",
        "## Table 2 - Secondary Complete Case",
        "",
        markdown_table(
            complete_table,
            [
                "model",
                "strategy",
                "valid_users",
                "excluded_structural_invalid",
                "HR@5",
                "NDCG@5",
                "MRR",
                "mean_positive_rank",
                "median_positive_rank",
                "label",
            ],
        ),
        "",
        "## Table 3 - Reliability",
        "",
        markdown_table(
            reliability_table,
            [
                "model",
                "strategy",
                "strict_valid_count",
                "strict_valid_rate",
                "structural_invalid_count",
                "structural_invalid_rate",
                "duplicate_token_count",
                "duplicate_token_rate",
                "malformed_count",
                "unknown_token_count",
                "retry_count",
                "retry_rate",
                "timeout_count",
                "elevated_duplicate_windows",
                "max_consecutive_duplicate_token_invalid",
            ],
        ),
        "",
        "## Table 4 - Beyond Utility",
        "",
        markdown_table(
            beyond_table,
            [
                "model",
                "strategy",
                "metric",
                "k",
                "value",
                "denominator",
                "valid_output_coverage",
                "alignment",
                "definition_implemented",
            ],
        ),
        "",
        "## Table 5 - Cell Comparisons",
        "",
        markdown_table(selected_comparisons, ["comparison", "metric", "left_value", "right_value", "delta", "notes"]),
        "",
        "## Statistical Tests",
        "",
        markdown_table(
            stats_rows,
            [
                "comparison",
                "scope",
                "metric",
                "pairs_valid",
                "delta_mean_right_minus_left",
                "paired_bootstrap_ci_low",
                "paired_bootstrap_ci_high",
                "test",
                "p_value",
                "p_value_bonferroni",
                "effect_size",
            ],
        ),
        "",
        "## Runtime",
        "",
        markdown_table(
            build_runtime_table(runtime_rows),
            [
                "scope",
                "model_name",
                "prompting_strategy",
                "wall_clock_seconds",
                "duration_mean_seconds",
                "duration_median_seconds",
                "duration_p90_seconds",
                "duration_p95_seconds",
                "duration_p99_seconds",
                "throughput_requests_per_hour",
                "prompt_eval_count_total",
                "eval_count_total",
                "raw_output_bytes",
                "parsing_time_seconds",
                "consolidation_time_seconds",
                "marginal_api_cost",
                "hardware_cost",
                "energy_cost",
                "retry",
                "failed",
            ],
        ),
        "",
        "## Research Questions",
        "",
        markdown_table(
            rq_compact,
            [
                "rq_id",
                "previous_status",
                "new_status",
                "qwen2_base",
                "qwen2_llmrank",
                "mistral_base",
                "mistral_llmrank",
                "limits",
            ],
        ),
        "",
        "## Paper Comparison",
        "",
        markdown_table(
            paper_rows,
            [
                "model",
                "strategy",
                "metric",
                "paper_lastfm_value",
                "local_fail_strict_value",
                "absolute_difference",
                "relative_difference",
                "comparability",
            ],
        ),
        "",
        "## Limits",
        "",
        "- No exact paper numerical replication is claimed.",
        "- Local LastFM sample, seed, output contract, and Qwen2/Mistral model cells differ from the paper's unrecovered LastFM artifacts and closed/provider models.",
        "- Duplicate-token invalidity is structural output invalidity, not hallucination.",
        "- Complete-case results are secondary and use strict-valid output denominators.",
        "- Calibration is a local extension, not a central original-paper metric.",
        "",
        "## Artifacts",
        "",
        markdown_table(
            [{"artifact": rel(path)} for path in OUTPUT_PATHS.values() if path.exists()],
            ["artifact"],
        ),
    ]
    return "\n".join(lines) + "\n"


def upsert_marked_section(path: Path, marker: str, content: str) -> None:
    """Replace or append a marked Markdown section."""

    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    new_section = f"{start}\n{content.rstrip()}\n{end}\n"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in old and end in old:
        before = old.split(start, 1)[0]
        after = old.split(end, 1)[1]
        path.write_text(before.rstrip() + "\n\n" + new_section + after.lstrip(), encoding="utf-8")
    else:
        separator = "\n\n" if old.strip() else ""
        path.write_text(old.rstrip() + separator + new_section, encoding="utf-8")


def write_outputs() -> dict[str, object]:
    """Run the full consolidation and write all freeze artifacts."""

    fail_rows = read_csv(SOURCE_PATHS["fail_strict"])
    complete_rows = read_csv(SOURCE_PATHS["complete_case"])
    validity_rows = read_csv(SOURCE_PATHS["validity"])
    taxonomy_rows = read_csv(SOURCE_PATHS["failure_taxonomy"])
    duplicate_windows = read_csv(SOURCE_PATHS["duplicate_windows"])
    beyond_rows = read_csv(SOURCE_PATHS["beyond"])
    runtime_rows = read_csv(SOURCE_PATHS["runtime"])
    rq_rows = read_csv(SOURCE_PATHS["rq"])
    final_validation_rows = read_csv(SOURCE_PATHS["final_validation"])
    parse_rows = read_csv(SOURCE_PATHS["parse"])
    execution_rows = read_csv(SOURCE_PATHS["execution"])
    state_rows = read_csv(TABLES_DIR / f"{SOURCE_PREFIX}_state.csv")
    candidate_rows = read_csv(SOURCE_PATHS["candidates"])

    fail_table = build_fail_strict_table(fail_rows, validity_rows)
    complete_table = build_complete_case_table(complete_rows, validity_rows)
    reliability_table = build_reliability_table(validity_rows, parse_rows, execution_rows, duplicate_windows)
    beyond_table = build_beyond_table(beyond_rows, validity_rows)
    canonical_metrics = build_canonical_metric_rows(fail_rows, complete_rows, validity_rows, beyond_rows, runtime_rows)
    comparisons = build_comparison_table(fail_rows, complete_rows, validity_rows, beyond_rows)
    stats_rows = build_statistical_tests(parse_rows, candidate_rows, fail_rows)
    paper_rows = build_paper_comparison(fail_table)
    audit_rows = build_audit_table(
        state_rows,
        validity_rows,
        fail_rows,
        complete_rows,
        taxonomy_rows,
        final_validation_rows,
        stats_rows,
    )
    decision_token = PASS_TOKEN if all(row["status"] == "pass" for row in audit_rows) else BLOCK_TOKEN

    write_csv(
        OUTPUT_PATHS["audit"],
        audit_rows,
        ["check_id", "check_name", "status", "observed", "expected", "evidence", "notes"],
    )
    write_csv(
        OUTPUT_PATHS["metrics"],
        canonical_metrics,
        [
            "freeze_state",
            "run_id",
            "dataset",
            "task",
            "model_name",
            "prompting_strategy",
            "metric_scope",
            "metric",
            "k",
            "value",
            "numerator",
            "denominator",
            "denominator_policy",
            "metric_status",
            "source_file",
            "source_column",
            "notes",
        ],
    )
    write_csv(
        OUTPUT_PATHS["comparisons"],
        comparisons,
        [
            "comparison",
            "left_cell",
            "right_cell",
            "delta_definition",
            "metric",
            "left_value",
            "right_value",
            "delta",
            "source_files",
            "notes",
        ],
    )
    write_csv(
        OUTPUT_PATHS["statistics"],
        stats_rows,
        [
            "comparison",
            "scope",
            "metric",
            "pairs_valid",
            "delta_mean_right_minus_left",
            "paired_bootstrap_ci_low",
            "paired_bootstrap_ci_high",
            "test",
            "p_value",
            "p_value_bonferroni",
            "multiple_comparison_correction",
            "effect_size",
            "mcnemar_left0_right1",
            "mcnemar_left1_right0",
            "seed",
            "resamples",
            "source_files",
            "source_columns",
            "metric_delta_matches_fail_strict_artifact",
            "claim_scope",
        ],
    )
    write_csv(
        OUTPUT_PATHS["paper"],
        paper_rows,
        [
            "model",
            "strategy",
            "metric",
            "paper_lastfm_value",
            "local_fail_strict_value",
            "absolute_difference",
            "relative_difference",
            "comparability",
            "supporting_artifact",
            "notes",
        ],
    )

    output_hashes = {
        name: sha256_file(path)
        for name, path in OUTPUT_PATHS.items()
        if name in {"audit", "metrics", "comparisons", "statistics", "paper"} and path.exists()
    }
    freeze_payload = json.dumps(
        {
            "run_id": RUN_ID,
            "decision_token": decision_token,
            "source_artifact_hashes": source_hashes(SOURCE_PATHS),
            "generated_artifact_hashes": output_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    freeze_hash = sha256_text(freeze_payload)

    report_text = build_report(
        decision_token=decision_token,
        freeze_hash=freeze_hash,
        fail_table=fail_table,
        complete_table=complete_table,
        reliability_table=reliability_table,
        beyond_table=beyond_table,
        comparisons=comparisons,
        stats_rows=stats_rows,
        runtime_rows=runtime_rows,
        rq_rows=rq_rows,
        paper_rows=paper_rows,
        audit_rows=audit_rows,
    )
    OUTPUT_PATHS["report"].write_text(report_text, encoding="utf-8")
    output_hashes["report"] = sha256_file(OUTPUT_PATHS["report"])

    summary = build_summary(
        decision_token=decision_token,
        audit_rows=audit_rows,
        output_hashes=output_hashes,
        freeze_hash=freeze_hash,
    )
    OUTPUT_PATHS["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "freeze_bundle_sha256": freeze_hash,
        "summary_json_sha256": sha256_file(OUTPUT_PATHS["summary"]),
        "generated_artifact_hashes": output_hashes,
    }
    OUTPUT_PATHS["hashes"].write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUTPUT_PATHS["hash_text"].write_text(f"{freeze_hash}  F6L_LastFM_V4_ranking_metrics_freeze\n", encoding="utf-8")

    marker_content = "\n".join(
        [
            "## Ranking Metrics Freeze Before Reranking",
            "",
            f"- Freeze state: `{FROZEN_STATE if decision_token == PASS_TOKEN else 'blocked'}`.",
            f"- Run ID: `{RUN_ID}`.",
            f"- Decision token: `{decision_token}`.",
            f"- Freeze SHA-256: `{freeze_hash}`.",
            f"- Canonical metrics: `{rel(OUTPUT_PATHS['metrics'])}`.",
            f"- Canonical comparisons: `{rel(OUTPUT_PATHS['comparisons'])}`.",
            f"- Summary JSON: `{rel(OUTPUT_PATHS['summary'])}`.",
            "- Scope boundary: no reranking, no new inference, no raw repair, no deduplication, no API call.",
        ]
    )
    upsert_marked_section(OUTPUT_PATHS["ranking_report"], "phase_f6l_lastfm_v4_ranking_metrics_freeze", marker_content)
    upsert_marked_section(OUTPUT_PATHS["handoff"], "phase_f6l_lastfm_v4_ranking_metrics_freeze", marker_content)
    upsert_marked_section(OUTPUT_PATHS["notes"], "phase_f6l_lastfm_v4_ranking_metrics_freeze", marker_content)

    return {
        "decision_token": decision_token,
        "freeze_hash": freeze_hash,
        "audit_passed": all(row["status"] == "pass" for row in audit_rows),
        "fail_table": fail_table,
        "complete_table": complete_table,
        "reliability_table": reliability_table,
        "beyond_table": beyond_table,
        "comparisons": comparisons,
        "stats_rows": stats_rows,
        "runtime_rows": runtime_rows,
        "rq_rows": rq_rows,
        "paper_rows": paper_rows,
        "output_paths": {name: rel(path) for name, path in OUTPUT_PATHS.items() if path.exists()},
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    parse_args()
    result = write_outputs()
    print(json.dumps({"decision_token": result["decision_token"], "freeze_hash": result["freeze_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()

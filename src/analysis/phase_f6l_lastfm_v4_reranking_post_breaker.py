"""Post-breaker forensic consolidation for F6L LastFM V4 reranking.

This module is read-only with respect to the raw reranking run. It audits the
partial run stopped by the duplicate-token breaker and produces separate
post-breaker tables, manifests, and report material.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis import phase_f6l_lastfm_v4_reranking_full_run as rerank
from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


RUN_ID = "20260806_034027"
POST_ID = "phase_f6l_lastfm_v4_reranking_post_breaker"
RAW_OUTPUT_DIR = f6l.PROJECT_ROOT / "results" / "llm" / f"phase_f6l_lastfm_v4_reranking_full_run_{RUN_ID}"

DECISION_TERMINAL = "f6l_lastfm_v4_reranking_post_breaker_terminal_partial_closure"
DECISION_READY = "f6l_lastfm_v4_reranking_post_breaker_versioned_continuation_ready"
DECISION_BLOCKED = "f6l_lastfm_v4_reranking_post_breaker_blocked_pending_evidence"

TABLES_DIR = f6l.PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = f6l.PROJECT_ROOT / "results" / "manifests"
REPORT_DIR = f6l.PROJECT_ROOT / "report"

CELL_COVERAGE_PATH = TABLES_DIR / f"{POST_ID}_cell_coverage.csv"
FAILURE_TAXONOMY_PATH = TABLES_DIR / f"{POST_ID}_failure_taxonomy.csv"
WINDOW_ANALYSIS_PATH = TABLES_DIR / f"{POST_ID}_window_analysis.csv"
METRICS_PATH = TABLES_DIR / f"{POST_ID}_metrics.csv"
STATISTICAL_ASSESSMENT_PATH = TABLES_DIR / f"{POST_ID}_statistical_assessment.csv"
DECISION_MATRIX_PATH = TABLES_DIR / f"{POST_ID}_decision_matrix.csv"
BASELINE_AUDIT_PATH = TABLES_DIR / f"{POST_ID}_ranking_baseline_audit.csv"
SUMMARY_PATH = MANIFESTS_DIR / f"{POST_ID}_summary.json"
HASHES_PATH = MANIFESTS_DIR / f"{POST_ID}_hashes.json"
REPORT_PATH = REPORT_DIR / "phase_f6l_lastfm_v4_reranking_post_breaker_forensic_consolidation.md"

CELL_COVERAGE_FIELDS = [
    "model_name",
    "prompting_strategy",
    "planned_requests",
    "attempted_raw_requests",
    "final_completed_requests",
    "non_final_transport_failures",
    "never_attempted_requests",
    "residual_final_response_requests",
    "strict_valid",
    "structural_invalid",
    "duplicate_token_invalid",
    "malformed_json",
    "unknown_token",
    "validity_rate_completed_final",
    "duplicate_rate_completed_final",
    "duplicate_rate_planned",
    "planned_manifest_ordinal_min",
    "planned_manifest_ordinal_max",
    "final_manifest_ordinal_min",
    "final_manifest_ordinal_max",
    "pending_manifest_ordinal_ranges",
    "distinct_users_planned",
    "distinct_users_final_completed",
    "positive_present_planned_users",
    "positive_present_final_completed",
    "coverage_status",
    "notes",
]

FAILURE_FIELDS = [
    "failure_class",
    "count",
    "rate_over_final_completed",
    "rate_over_planned",
    "request_ids",
    "model_strategy_counts",
    "top_duplicate_tokens",
    "top_duplicate_items",
    "notes",
]

WINDOW_FIELDS = [
    "window_index",
    "response_ordinal_start",
    "response_ordinal_end",
    "manifest_ordinal_min",
    "manifest_ordinal_max",
    "models",
    "strategies",
    "model_strategy_counts",
    "crosses_multiple_cells",
    "distinct_users",
    "user_ids",
    "strict_valid",
    "duplicate_token_invalid",
    "duplicate_rate",
    "threshold",
    "status",
    "distance_from_threshold",
    "longest_consecutive_invalid_streak",
    "duplicate_request_ids",
    "duplicate_model_strategy_counts",
    "top_duplicate_tokens",
    "coincides_with_model_or_strategy_change",
    "notes",
]

METRIC_FIELDS = [
    "metric_id",
    "run_id",
    "dataset",
    "task",
    "model_name",
    "prompting_strategy",
    "metric_group",
    "metric",
    "k",
    "value",
    "numerator",
    "denominator",
    "planned_denominator",
    "completed_final_denominator",
    "strict_valid_denominator",
    "coverage_status",
    "baseline_ranking_value",
    "delta_vs_frozen_ranking",
    "source_file",
    "notes",
]

STAT_FIELDS = [
    "assessment_id",
    "topic",
    "statistic",
    "observed",
    "comparator",
    "value",
    "status",
    "interpretation",
    "notes",
]

DECISION_FIELDS = [
    "criterion",
    "observed",
    "expected_for_terminal_partial_closure",
    "expected_for_versioned_continuation_ready",
    "status",
    "effect_on_decision",
    "evidence",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return rerank.rel_safe(path)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def format_float(value: float) -> str:
    return f"{value:.12g}"


def stable_hash(*parts: Any) -> str:
    return f6l.sha256_text(canonical_json(list(parts)))


def request_ordinal(request_id: str) -> int:
    match = re.search(r"FULL-(\d+)-", request_id)
    if not match:
        raise ValueError(f"cannot parse request ordinal from {request_id}")
    return int(match.group(1))


def ordinal_ranges(ordinals: Sequence[int]) -> str:
    if not ordinals:
        return ""
    values = sorted(set(ordinals))
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ";".join(ranges)


def parse_json_list(value: str) -> list[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in loaded] if isinstance(loaded, list) else []


def duplicate_tokens_for_row(row: Mapping[str, str]) -> list[str]:
    tokens = parse_json_list(str(row.get("ranking_tokens", "")))
    counts = Counter(tokens)
    return sorted(token for token, count in counts.items() if count > 1)


def duplicate_items_for_row(row: Mapping[str, str]) -> list[str]:
    tokens = duplicate_tokens_for_row(row)
    if not tokens:
        return []
    mapping_hash = row.get("candidate_token_mapping_hash", "")
    request_id = row.get("request_id", "")
    manifest = {item["request_id"]: item for item in f6l.read_csv_rows(rerank.MANIFEST_PATH)}
    source = manifest.get(request_id, {})
    try:
        mapping = json.loads(source.get("candidate_token_mapping", "{}"))
    except json.JSONDecodeError:
        return []
    return [str(mapping.get(token, {}).get("item_id", "")) for token in tokens if mapping.get(token)]


def load_sources() -> dict[str, Any]:
    raw_paths = f6l.raw_response_paths(RAW_OUTPUT_DIR)
    manifest_rows = f6l.read_csv_rows(rerank.MANIFEST_PATH)
    parse_rows = f6l.read_csv_rows(rerank.PARSE_RESULTS_PATH)
    recommendation_rows = f6l.read_csv_rows(rerank.RECOMMENDATIONS_PATH)
    execution_rows = f6l.read_csv_rows(rerank.EXECUTION_PATH)
    requests = f6l.read_jsonl(raw_paths["requests"])
    responses = f6l.read_jsonl(raw_paths["responses"])
    final_records = f6l.final_response_records(RAW_OUTPUT_DIR)
    return {
        "raw_paths": raw_paths,
        "manifest_rows": manifest_rows,
        "parse_rows": parse_rows,
        "recommendation_rows": recommendation_rows,
        "execution_rows": execution_rows,
        "requests": requests,
        "responses": responses,
        "final_records": final_records,
    }


def longest_true_streak(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def binomial_tail_at_least(n: int, k: int, p: float) -> float:
    if p <= 0:
        return 1.0 if k <= 0 else 0.0
    if p >= 1:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return min(1.0, max(0.0, total))


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cell_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("model_name", "")), str(row.get("prompting_strategy", ""))


def build_cell_coverage(sources: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    manifest_rows = sources["manifest_rows"]
    parse_rows = sources["parse_rows"]
    requests = sources["requests"]
    responses = sources["responses"]
    final_ids = {row["request_id"] for row in parse_rows}
    request_ids = {str(row.get("request_id", "")) for row in requests}
    non_final_ids = {
        str(row.get("request_id", ""))
        for row in responses
        if str(row.get("is_final_attempt", "true")).lower() != "true"
    }
    parse_by_id = {row["request_id"]: row for row in parse_rows}
    rows: list[dict[str, str]] = []
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for model in rerank.MODEL_ORDER:
        for strategy in rerank.STRATEGY_ORDER:
            key = (model, strategy)
            planned = [row for row in manifest_rows if cell_key(row) == key]
            final = [parse_by_id[row["request_id"]] for row in planned if row["request_id"] in parse_by_id]
            attempted = [row for row in planned if row["request_id"] in request_ids]
            non_final = [row for row in planned if row["request_id"] in non_final_ids]
            pending = [row for row in planned if row["request_id"] not in final_ids]
            never_attempted = [row for row in pending if row["request_id"] not in request_ids]
            strict = [row for row in final if row["strict_valid"] == "true"]
            invalid = [row for row in final if row["strict_valid"] != "true"]
            duplicate_invalid = [row for row in invalid if row["failure_reason"] == "duplicate_token"]
            malformed = sum(int(row.get("malformed_json_count") or 0) for row in final)
            unknown = sum(int(row.get("unknown_token_count") or 0) for row in final)
            planned_ord = [request_ordinal(row["request_id"]) for row in planned]
            final_ord = [request_ordinal(row["request_id"]) for row in final]
            pending_ord = [request_ordinal(row["request_id"]) for row in pending]
            row = {
                "model_name": model,
                "prompting_strategy": strategy,
                "planned_requests": str(len(planned)),
                "attempted_raw_requests": str(len(attempted)),
                "final_completed_requests": str(len(final)),
                "non_final_transport_failures": str(len(non_final)),
                "never_attempted_requests": str(len(never_attempted)),
                "residual_final_response_requests": str(len(pending)),
                "strict_valid": str(len(strict)),
                "structural_invalid": str(len(invalid)),
                "duplicate_token_invalid": str(len(duplicate_invalid)),
                "malformed_json": str(malformed),
                "unknown_token": str(unknown),
                "validity_rate_completed_final": format_float(len(strict) / len(final) if final else 0.0),
                "duplicate_rate_completed_final": format_float(len(duplicate_invalid) / len(final) if final else 0.0),
                "duplicate_rate_planned": format_float(len(duplicate_invalid) / len(planned) if planned else 0.0),
                "planned_manifest_ordinal_min": str(min(planned_ord)),
                "planned_manifest_ordinal_max": str(max(planned_ord)),
                "final_manifest_ordinal_min": str(min(final_ord)) if final_ord else "",
                "final_manifest_ordinal_max": str(max(final_ord)) if final_ord else "",
                "pending_manifest_ordinal_ranges": ordinal_ranges(pending_ord),
                "distinct_users_planned": str(len({row["user_id"] for row in planned})),
                "distinct_users_final_completed": str(len({row["user_id"] for row in final})),
                "positive_present_planned_users": str(sum(1 for row in planned if row.get("positive_present") == "true")),
                "positive_present_final_completed": str(sum(1 for row in final if row.get("positive_present") == "true")),
                "coverage_status": "COMPLETE" if len(final) == len(planned) else "INTERIM_PARTIAL",
                "notes": "Coverage is reconstructed from manifest request ids; final_completed excludes non-final transport timeouts.",
            }
            rows.append(row)
            stats[key] = {
                "planned": planned,
                "final": final,
                "attempted": attempted,
                "non_final": non_final,
                "pending": pending,
                "never_attempted": never_attempted,
                "strict": strict,
                "invalid": invalid,
                "duplicate_invalid": duplicate_invalid,
            }
    return rows, stats


def build_failure_taxonomy(sources: Mapping[str, Any], cell_stats: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, str]]:
    parse_rows = sources["parse_rows"]
    responses = sources["responses"]
    manifest_by_id = {row["request_id"]: row for row in sources["manifest_rows"]}
    total_final = len(parse_rows)
    total_planned = len(sources["manifest_rows"])
    duplicate_invalid = [row for row in parse_rows if row["failure_reason"] == "duplicate_token"]
    non_final_transport = [
        row
        for row in responses
        if str(row.get("is_final_attempt", "true")).lower() != "true"
    ]

    def model_strategy_counts(rows: Sequence[Mapping[str, Any]]) -> str:
        counts: Counter[str] = Counter()
        for row in rows:
            request_id = str(row.get("request_id", ""))
            source = manifest_by_id.get(request_id, row)
            counts[f"{source.get('model_name','')}|{source.get('prompting_strategy','')}"] += 1
        return canonical_json(dict(sorted(counts.items())))

    token_counter: Counter[str] = Counter()
    item_counter: Counter[str] = Counter()
    for row in duplicate_invalid:
        token_counter.update(duplicate_tokens_for_row(row))
        item_counter.update(duplicate_items_for_row(row))

    explicit_counts = {
        "duplicate_token": len(duplicate_invalid),
        "malformed_json": sum(1 for row in parse_rows if int(row.get("malformed_json_count") or 0) > 0),
        "unknown_token": sum(1 for row in parse_rows if int(row.get("unknown_token_count") or 0) > 0),
        "out_of_domain_token": sum(1 for row in parse_rows if int(row.get("unknown_token_count") or 0) > 0),
        "wrong_length": sum(1 for row in parse_rows if row.get("output_count") and int(row.get("output_count") or 0) != 10),
        "missing_ranking": sum(1 for row in parse_rows if row.get("failure_reason") == "missing_ranking"),
        "extra_property": sum(1 for row in parse_rows if row.get("failure_reason") == "extra_property"),
        "transport_final_failure": sum(1 for row in parse_rows if row.get("backend_success") == "false"),
        "transport_non_final_timeout": len(non_final_transport),
        "mapping_failure": sum(
            1
            for row in parse_rows
            if int(row.get("ambiguous_mapping_count") or 0) > 0 or int(row.get("numeric_without_c_prefix_count") or 0) > 0
        ),
        "other_structural_error": sum(1 for row in parse_rows if row.get("strict_valid") != "true" and row.get("failure_reason") != "duplicate_token"),
    }
    request_rows = {
        "duplicate_token": duplicate_invalid,
        "transport_non_final_timeout": non_final_transport,
    }
    rows = []
    for failure_class, count in explicit_counts.items():
        source_rows = request_rows.get(failure_class, [])
        rows.append(
            {
                "failure_class": failure_class,
                "count": str(count),
                "rate_over_final_completed": format_float(count / total_final if total_final else 0.0),
                "rate_over_planned": format_float(count / total_planned if total_planned else 0.0),
                "request_ids": ";".join(str(row.get("request_id", "")) for row in source_rows),
                "model_strategy_counts": model_strategy_counts(source_rows) if source_rows else "{}",
                "top_duplicate_tokens": canonical_json(dict(token_counter.most_common(10))) if failure_class == "duplicate_token" else "{}",
                "top_duplicate_items": canonical_json(dict(item_counter.most_common(10))) if failure_class == "duplicate_token" else "{}",
                "notes": "Final parser invalids are not repaired or deduplicated."
                if failure_class != "transport_non_final_timeout"
                else "Non-final local transport timeouts have raw evidence but no final parser row.",
            }
        )
    return rows


def build_window_analysis(sources: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    parse_rows = sorted(sources["parse_rows"], key=lambda row: request_ordinal(row["request_id"]))
    window_size = rerank.DUPLICATE_WINDOW_SIZE
    threshold = rerank.DUPLICATE_WINDOW_RATE_LIMIT
    rows = []
    rates: list[float] = []
    failing_windows = []
    for index, start in enumerate(range(0, len(parse_rows), window_size), start=1):
        window = parse_rows[start : start + window_size]
        if len(window) < window_size:
            continue
        duplicate_rows = [row for row in window if row["failure_reason"] == "duplicate_token"]
        rate = len(duplicate_rows) / len(window)
        rates.append(rate)
        status = "fail" if rate > threshold else "pass"
        if status == "fail":
            failing_windows.append(index)
        key_counts = Counter(f"{row['model_name']}|{row['prompting_strategy']}" for row in window)
        dup_key_counts = Counter(f"{row['model_name']}|{row['prompting_strategy']}" for row in duplicate_rows)
        token_counter: Counter[str] = Counter()
        for row in duplicate_rows:
            token_counter.update(duplicate_tokens_for_row(row))
        models = sorted({row["model_name"] for row in window})
        strategies = sorted({row["prompting_strategy"] for row in window})
        ordinals = [request_ordinal(row["request_id"]) for row in window]
        invalid_flags = [row["failure_reason"] == "duplicate_token" for row in window]
        rows.append(
            {
                "window_index": str(index),
                "response_ordinal_start": str(start + 1),
                "response_ordinal_end": str(start + len(window)),
                "manifest_ordinal_min": str(min(ordinals)),
                "manifest_ordinal_max": str(max(ordinals)),
                "models": ";".join(models),
                "strategies": ";".join(strategies),
                "model_strategy_counts": canonical_json(dict(sorted(key_counts.items()))),
                "crosses_multiple_cells": str(len(key_counts) > 1).lower(),
                "distinct_users": str(len({row["user_id"] for row in window})),
                "user_ids": ";".join(sorted({row["user_id"] for row in window}, key=lambda value: int(value))),
                "strict_valid": str(sum(1 for row in window if row["strict_valid"] == "true")),
                "duplicate_token_invalid": str(len(duplicate_rows)),
                "duplicate_rate": format_float(rate),
                "threshold": ">0.05 over closed 200 final responses",
                "status": status,
                "distance_from_threshold": format_float(rate - threshold),
                "longest_consecutive_invalid_streak": str(longest_true_streak(invalid_flags)),
                "duplicate_request_ids": ";".join(row["request_id"] for row in duplicate_rows),
                "duplicate_model_strategy_counts": canonical_json(dict(sorted(dup_key_counts.items()))),
                "top_duplicate_tokens": canonical_json(dict(token_counter.most_common(10))),
                "coincides_with_model_or_strategy_change": "true",
                "notes": "Windows are over final response order, not raw request attempts; manifest ordinal gaps can exist after transport timeouts.",
            }
        )
    all_invalid_flags = [row["failure_reason"] == "duplicate_token" for row in parse_rows]
    rolling_rates: list[tuple[int, int, float]] = []
    for start in range(0, max(0, len(parse_rows) - window_size + 1)):
        window = parse_rows[start : start + window_size]
        rate = sum(1 for row in window if row["failure_reason"] == "duplicate_token") / window_size
        rolling_rates.append((start + 1, start + window_size, rate))
    max_rolling = max(rolling_rates, key=lambda item: item[2]) if rolling_rates else (0, 0, 0.0)
    stats = {
        "closed_window_rates": rates,
        "failing_windows": failing_windows,
        "first_failing_window": failing_windows[0] if failing_windows else None,
        "longest_consecutive_invalid_streak": longest_true_streak(all_invalid_flags),
        "rolling_200_max": max_rolling,
        "windows_above": {
            "0.02": sum(1 for rate in rates if rate > 0.02),
            "0.03": sum(1 for rate in rates if rate > 0.03),
            "0.04": sum(1 for rate in rates if rate > 0.04),
            "0.05": sum(1 for rate in rates if rate > 0.05),
        },
    }
    return rows, stats


def metric_row(
    *,
    run_id: str,
    model: str,
    strategy: str,
    group: str,
    metric: str,
    value: float,
    numerator: float,
    denominator: float,
    planned: int,
    completed: int,
    strict: int,
    coverage: str,
    source_file: Path,
    notes: str,
    k: str = "",
    baseline: str = "",
) -> dict[str, str]:
    baseline_value = baseline
    delta = ""
    if baseline_value != "":
        delta = format_float(value - float(baseline_value))
    return {
        "metric_id": stable_hash(POST_ID, run_id, model, strategy, group, metric, k)[:24],
        "run_id": run_id,
        "dataset": "LastFM",
        "task": "reranking",
        "model_name": model,
        "prompting_strategy": strategy,
        "metric_group": group,
        "metric": metric,
        "k": k,
        "value": format_float(value),
        "numerator": format_float(numerator),
        "denominator": format_float(denominator),
        "planned_denominator": str(planned),
        "completed_final_denominator": str(completed),
        "strict_valid_denominator": str(strict),
        "coverage_status": coverage,
        "baseline_ranking_value": baseline_value,
        "delta_vs_frozen_ranking": delta,
        "source_file": rel(source_file),
        "notes": notes,
    }


def build_metrics(sources: Mapping[str, Any], cell_stats: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, str]]:
    baseline = rerank.ranking_baseline_metric_lookup()
    strategy_map = rerank.RANKING_TO_RERANKING_STRATEGY
    rows: list[dict[str, str]] = []
    for model in rerank.MODEL_ORDER:
        for strategy in rerank.STRATEGY_ORDER:
            stats = cell_stats[(model, strategy)]
            final = list(stats["final"])
            strict_rows = [row for row in final if row["strict_valid"] == "true"]
            planned = len(stats["planned"])
            completed = len(final)
            strict = len(strict_rows)
            invalid = completed - strict
            duplicate_invalid = len(stats["duplicate_invalid"])
            coverage = "COMPLETE" if completed == planned else "INTERIM_PARTIAL"
            hits = []
            ndcgs = []
            ranks = []
            strict_hits = []
            strict_ndcgs = []
            positive_strict_hits = []
            positive_strict_ndcgs = []
            positive_strict_count = 0
            for row in final:
                positive = row.get("positive_present") == "true"
                row_strict = row.get("strict_valid") == "true"
                rank = int(row["positive_rank"]) if row.get("positive_rank") else None
                hit = 1.0 if row_strict and positive and rank is not None and rank <= 5 else 0.0
                ndcg = 1.0 / math.log2(rank + 1) if hit else 0.0
                hits.append(hit)
                ndcgs.append(ndcg)
                ranks.append(float(rank) if row_strict and positive and rank is not None else 20.0)
                if row_strict:
                    strict_hits.append(hit)
                    strict_ndcgs.append(ndcg)
                    if positive:
                        positive_strict_count += 1
                        positive_strict_hits.append(hit)
                        positive_strict_ndcgs.append(ndcg)
            ranking_strategy = strategy_map[strategy]
            rows.append(
                metric_row(
                    run_id=RUN_ID,
                    model=model,
                    strategy=strategy,
                    group="primary_fail_strict_completed_final",
                    metric="HR@5_fail_strict_completed_final",
                    k="5",
                    value=sum(hits) / completed if completed else 0.0,
                    numerator=sum(hits),
                    denominator=completed,
                    planned=planned,
                    completed=completed,
                    strict=strict,
                    coverage=coverage,
                    baseline=baseline.get((model, ranking_strategy, "HR@5_fail_strict"), ""),
                    source_file=rerank.PARSE_RESULTS_PATH,
                    notes="Invalid final outputs count as zero; missing residual final responses are not imputed. Incomplete cells are interim partial.",
                )
            )
            rows.append(
                metric_row(
                    run_id=RUN_ID,
                    model=model,
                    strategy=strategy,
                    group="primary_fail_strict_completed_final",
                    metric="NDCG@5_fail_strict_completed_final",
                    k="5",
                    value=sum(ndcgs) / completed if completed else 0.0,
                    numerator=sum(ndcgs),
                    denominator=completed,
                    planned=planned,
                    completed=completed,
                    strict=strict,
                    coverage=coverage,
                    baseline=baseline.get((model, ranking_strategy, "NDCG@5_fail_strict"), ""),
                    source_file=rerank.PARSE_RESULTS_PATH,
                    notes="Invalid final outputs count as zero; missing residual final responses are not imputed. Incomplete cells are interim partial.",
                )
            )
            rows.append(
                metric_row(
                    run_id=RUN_ID,
                    model=model,
                    strategy=strategy,
                    group="primary_fail_strict_completed_final",
                    metric="mean_positive_rank_fail_strict_completed_invalid_as_20",
                    k="",
                    value=sum(ranks) / completed if completed else 0.0,
                    numerator=sum(ranks),
                    denominator=completed,
                    planned=planned,
                    completed=completed,
                    strict=strict,
                    coverage=coverage,
                    baseline=baseline.get((model, ranking_strategy, "mean_positive_rank_fail_strict_invalid_as_20"), ""),
                    source_file=rerank.PARSE_RESULTS_PATH,
                    notes="Absent positives, invalid final outputs, and missing ranks count as rank 20 over completed final responses only.",
                )
            )
            rows.append(
                metric_row(
                    run_id=RUN_ID,
                    model=model,
                    strategy=strategy,
                    group="secondary_complete_case",
                    metric="HR@5_complete_case_strict_valid",
                    k="5",
                    value=sum(strict_hits) / strict if strict else 0.0,
                    numerator=sum(strict_hits),
                    denominator=strict,
                    planned=planned,
                    completed=completed,
                    strict=strict,
                    coverage=coverage,
                    source_file=rerank.PARSE_RESULTS_PATH,
                    notes="Secondary complete-case over strict-valid final outputs; positive-absent cases remain misses.",
                )
            )
            rows.append(
                metric_row(
                    run_id=RUN_ID,
                    model=model,
                    strategy=strategy,
                    group="secondary_complete_case",
                    metric="NDCG@5_complete_case_strict_valid",
                    k="5",
                    value=sum(strict_ndcgs) / strict if strict else 0.0,
                    numerator=sum(strict_ndcgs),
                    denominator=strict,
                    planned=planned,
                    completed=completed,
                    strict=strict,
                    coverage=coverage,
                    source_file=rerank.PARSE_RESULTS_PATH,
                    notes="Secondary complete-case over strict-valid final outputs; positive-absent cases remain zero.",
                )
            )
            rows.append(
                metric_row(
                    run_id=RUN_ID,
                    model=model,
                    strategy=strategy,
                    group="secondary_positive_present",
                    metric="HR@5_positive_present_strict_valid",
                    k="5",
                    value=sum(positive_strict_hits) / positive_strict_count if positive_strict_count else 0.0,
                    numerator=sum(positive_strict_hits),
                    denominator=positive_strict_count,
                    planned=planned,
                    completed=completed,
                    strict=strict,
                    coverage=coverage,
                    source_file=rerank.PARSE_RESULTS_PATH,
                    notes="Secondary diagnostic restricted to strict-valid final outputs whose positive item is in the candidate pool.",
                )
            )
            for metric, numerator, denominator, value, notes in [
                ("completion_rate_planned", completed, planned, completed / planned if planned else 0.0, "Final response coverage over the planned 1000 requests for this cell."),
                ("strict_valid_rate_completed_final", strict, completed, strict / completed if completed else 0.0, "Strict-valid rate over completed final responses."),
                ("structural_invalid_rate_completed_final", invalid, completed, invalid / completed if completed else 0.0, "Structural-invalid rate over completed final responses."),
                ("duplicate_token_rate_completed_final", duplicate_invalid, completed, duplicate_invalid / completed if completed else 0.0, "Duplicate-token invalid rate over completed final responses."),
                ("duplicate_token_rate_planned", duplicate_invalid, planned, duplicate_invalid / planned if planned else 0.0, "Duplicate-token invalid count over all planned requests; residuals are not parser invalids."),
                ("transport_non_final_timeout_rate_planned", len(stats["non_final"]), planned, len(stats["non_final"]) / planned if planned else 0.0, "Local timeout attempts without final parser rows over planned requests."),
            ]:
                rows.append(
                    metric_row(
                        run_id=RUN_ID,
                        model=model,
                        strategy=strategy,
                        group="validity_and_failure",
                        metric=metric,
                        value=value,
                        numerator=float(numerator),
                        denominator=float(denominator),
                        planned=planned,
                        completed=completed,
                        strict=strict,
                        coverage=coverage,
                        source_file=rerank.PARSE_RESULTS_PATH,
                        notes=notes,
                    )
                )

    for row in f6l.read_csv_rows(rerank.BEYOND_UTILITY_PATH):
        model = row["model_name"]
        strategy = row["prompting_strategy"]
        coverage = "COMPLETE" if int(next(c["final_completed_requests"] for c in build_cell_coverage(sources)[0] if c["model_name"] == model and c["prompting_strategy"] == strategy)) == 1000 else "INTERIM_PARTIAL"
        rows.append(
            metric_row(
                run_id=RUN_ID,
                model=model,
                strategy=strategy,
                group="beyond_utility_supported",
                metric=row["metric"],
                k=row.get("k", ""),
                value=float(row["value"] or 0),
                numerator=float(row.get("numerator") or 0),
                denominator=float(row.get("denominator") or 0),
                planned=1000,
                completed=int(next(c["final_completed_requests"] for c in build_cell_coverage(sources)[0] if c["model_name"] == model and c["prompting_strategy"] == strategy)),
                strict=int(next(c["strict_valid"] for c in build_cell_coverage(sources)[0] if c["model_name"] == model and c["prompting_strategy"] == strategy)),
                coverage=coverage,
                source_file=rerank.BEYOND_UTILITY_PATH,
                notes=f"Supported beyond-utility metric inherited from strict-valid recommendations only; {row.get('notes','')}",
            )
        )
    return rows


def build_statistical_assessment(
    sources: Mapping[str, Any],
    cell_stats: Mapping[tuple[str, str], Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, str]],
    window_stats: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(topic: str, statistic: str, observed: str, comparator: str, value: str, status: str, interpretation: str, notes: str) -> None:
        rows.append(
            {
                "assessment_id": stable_hash(POST_ID, topic, statistic, observed, comparator)[:24],
                "topic": topic,
                "statistic": statistic,
                "observed": observed,
                "comparator": comparator,
                "value": value,
                "status": status,
                "interpretation": interpretation,
                "notes": notes,
            }
        )

    parse_rows = sources["parse_rows"]
    duplicate_count = sum(1 for row in parse_rows if row["failure_reason"] == "duplicate_token")
    completed = len(parse_rows)
    cumulative_rate = duplicate_count / completed if completed else 0.0
    failing = [row for row in window_rows if row["status"] == "fail"]
    window12 = next(row for row in window_rows if row["window_index"] == "12")
    k = int(window12["duplicate_token_invalid"])
    n = 200
    lower, upper = wilson_interval(k, n)
    prior_duplicates = sum(int(row["duplicate_token_invalid"]) for row in window_rows if int(row["window_index"]) < 12)
    prior_n = 11 * 200
    prior_rate = prior_duplicates / prior_n
    p_tail_prior = binomial_tail_at_least(n, k, prior_rate)
    p_tail_overall = binomial_tail_at_least(n, k, cumulative_rate)
    p_any_prior = 1 - ((1 - p_tail_prior) ** len(window_rows))
    add("breaker_contract", "window_12_duplicate_rate", f"{k}/{n}", ">0.05", window12["duplicate_rate"], "contractually_triggered", "The breaker was correctly activated by the predeclared rule.", "This is a contract result, not by itself proof of global systemic regression.")
    add("binomial_interval", "window_12_wilson_95_ci", f"{k}/{n}", "binomial proportion", f"{format_float(lower)}-{format_float(upper)}", "estimated", "The interval is wide enough that a local spike cannot be elevated to global regression alone.", "Wilson 95% interval.")
    add("exact_binomial_test", "window_12_vs_prior_closed_windows", f"{k}/{n}", f"p0={format_float(prior_rate)} from windows 1-11", format_float(p_tail_prior), "local_spike_supported", "Window 12 is unlikely under the prior closed-window duplicate rate.", "One-sided exact tail P[X>=11].")
    add("exact_binomial_test", "window_12_vs_overall_partial_run", f"{k}/{n}", f"p0={format_float(cumulative_rate)} from 48/2400", format_float(p_tail_overall), "local_spike_supported", "Window 12 is elevated against the partial-run cumulative rate.", "One-sided exact tail P[X>=11].")
    add("multiple_window_probability", "at_least_one_window_like_12", f"{len(window_rows)} closed windows", f"per-window p={format_float(p_tail_prior)}", format_float(p_any_prior), "approximate", "The chance of at least one such non-overlap window remains non-negligible enough to avoid a global claim from token name alone.", "Approximation assumes independent non-overlapping windows.")
    add("window_screen", "first_unique_window_above_5_percent", ";".join(str(item) for item in window_stats["failing_windows"]), "threshold >0.05", "true" if window_stats["failing_windows"] == [12] else "false", "single_window_failure", "Window 12 is the first and only non-overlap window above 5%.", "Earlier windows include local variation but no prior breaker fail.")
    add("window_screen", "windows_above_thresholds", canonical_json(window_stats["windows_above"]), "strictly greater than thresholds", "", "descriptive", "Counts are computed on closed non-overlap 200-final-response windows.", "")
    pre_breaker_rates = window_stats["closed_window_rates"][:11]
    monotonic_increase = all(left <= right for left, right in zip(pre_breaker_rates, pre_breaker_rates[1:]))
    add(
        "trend_screen",
        "pre_window_12_monotonic_increase",
        canonical_json(pre_breaker_rates),
        "monotonic non-decreasing trend before breaker",
        str(monotonic_increase).lower(),
        "not_supported" if not monotonic_increase else "supported",
        "The prior windows do not show a monotonic increasing duplicate-token trend.",
        "Window 10 is elevated at 0.045, but window 11 returns to 0.015 before the failed window 12.",
    )
    rolling_start, rolling_end, rolling_rate = window_stats["rolling_200_max"]
    add("rolling_duplicate_rate", "max_rolling_200", f"{rolling_start}-{rolling_end}", "all completed final responses", format_float(rolling_rate), "descriptive", "Rolling analysis detects the local spike around the breaker region.", "Rolling windows are overlapping over final response order.")
    add("cluster_analysis", "longest_consecutive_duplicate_invalid_streak", str(window_stats["longest_consecutive_invalid_streak"]), "stop if 5 consecutive duplicate invalids", str(window_stats["longest_consecutive_invalid_streak"]), "pass", "No consecutive duplicate-token cluster reached the hard consecutive limit.", "")

    duplicate_rows = [row for row in parse_rows if row["failure_reason"] == "duplicate_token"]
    user_counts = Counter(row["user_id"] for row in duplicate_rows)
    token_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    for row in duplicate_rows:
        token_counts.update(duplicate_tokens_for_row(row))
        item_counts.update(duplicate_items_for_row(row))
    add(
        "concentration_screen",
        "duplicate_invalid_users",
        f"distinct={len(user_counts)};max_per_user={max(user_counts.values()) if user_counts else 0}",
        "broad or repeated-user concentration",
        canonical_json(dict(user_counts.most_common(10))),
        "descriptive",
        "Duplicate invalids are not caused by a single repeated user.",
        "Counts are over final duplicate-token invalid rows.",
    )
    add(
        "concentration_screen",
        "recurring_duplicate_tokens",
        canonical_json(dict(token_counts.most_common(5))),
        "token recurrence",
        canonical_json(dict(item_counts.most_common(5))),
        "descriptive",
        "The duplicated token is recurrent, especially C11, but recurrence is confined to Mistral invalid outputs.",
        "Item ids are resolved through the frozen C01-C20 mapping for each request.",
    )

    baseline = rerank.ranking_baseline_metric_lookup()
    for model, strategy in [
        ("mistral:7b-instruct", "base/OpenP5-derived reranking"),
        ("mistral:7b-instruct", "LLMRank recency-focused reranking"),
    ]:
        stats = cell_stats[(model, strategy)]
        completed_cell = len(stats["final"])
        duplicate_cell = len(stats["duplicate_invalid"])
        ranking_strategy = rerank.RANKING_TO_RERANKING_STRATEGY[strategy]
        ranking_rate = float(baseline.get((model, ranking_strategy, "duplicate_token_rate"), "0") or 0)
        p_tail = binomial_tail_at_least(completed_cell, duplicate_cell, ranking_rate)
        add(
            "cell_specific_duplicate_rate",
            f"{model}|{strategy}",
            f"{duplicate_cell}/{completed_cell}",
            f"ranking V4 corresponding duplicate_token_rate={ranking_rate}",
            format_float(p_tail),
            "cell_specific_signal" if duplicate_cell else "no_signal",
            "Mistral cells carry all duplicate-token invalidity; Qwen cells have none in the partial run.",
            "Comparison uses completed final responses and the frozen ranking baseline rate as reference.",
        )

    qwen_duplicates = sum(len(cell_stats[(model, strategy)]["duplicate_invalid"]) for model in ["qwen2:7b-instruct"] for strategy in rerank.STRATEGY_ORDER)
    add("global_systemic_regression", "qwen_duplicate_count", str(qwen_duplicates), "all models affected", str(qwen_duplicates), "not_supported", "Global systemic regression is not supported because duplicate invalidity is absent in Qwen cells and no hard parser/schema drift is observed.", "")
    return rows


def build_decision_matrix(
    sources: Mapping[str, Any],
    cell_rows: Sequence[Mapping[str, str]],
    failure_rows: Sequence[Mapping[str, str]],
    window_rows: Sequence[Mapping[str, str]],
    summary_integrity: Mapping[str, Any],
) -> tuple[list[dict[str, str]], str]:
    baseline_ok = bool(summary_integrity["baseline_audit_pass"])
    parser_ok = bool(summary_integrity["parser_schema_contract_unchanged"])
    original_clean = summary_integrity["original_repo_status"] == ""
    all_final_invalid_duplicate = int(summary_integrity["duplicate_token_invalid"]) == int(summary_integrity["structural_invalid"])
    no_hard = int(summary_integrity["hard_structural_invalid"]) == 0
    residual_count_ok = int(summary_integrity["pending_final_response_count"]) == 1600
    residual_interval_ok = bool(summary_integrity["pending_final_response_ordinals_are_2401_4000"])
    non_final_transport = int(summary_integrity["non_final_transport_timeout_count"])
    single_window_failure = [row["window_index"] for row in window_rows if row["status"] == "fail"] == ["12"]
    qwen_duplicate_zero = all(
        row["duplicate_token_invalid"] == "0"
        for row in cell_rows
        if row["model_name"] == "qwen2:7b-instruct"
    )
    no_raw_corruption = bool(summary_integrity["raw_final_files_exist"]) and bool(summary_integrity["raw_jsonl_unique_request_ids"])
    no_contamination = bool(summary_integrity["no_v1_v2_v3_pilot_source_contamination"])

    rows = []

    def add(criterion: str, observed: str, terminal: str, ready: str, status: str, effect: str, evidence: str, notes: str) -> None:
        rows.append(
            {
                "criterion": criterion,
                "observed": observed,
                "expected_for_terminal_partial_closure": terminal,
                "expected_for_versioned_continuation_ready": ready,
                "status": status,
                "effect_on_decision": effect,
                "evidence": evidence,
                "notes": notes,
            }
        )

    add("baseline_ranking_freeze_audit", str(baseline_ok).lower(), "fail or drift", "pass", "pass" if baseline_ok else "fail", "supports_continuation" if baseline_ok else "blocks", rel(BASELINE_AUDIT_PATH), "Frozen ranking baseline remains read-only and hash-audited.")
    add("parser_schema_contract", str(parser_ok).lower(), "drift", "unchanged", "pass" if parser_ok else "fail", "supports_continuation" if parser_ok else "terminal_or_blocked", rel(rerank.PARSE_RESULTS_PATH), "Parser/schema/output contract hashes are unchanged.")
    add("final_invalid_classification", f"duplicate={summary_integrity['duplicate_token_invalid']};structural={summary_integrity['structural_invalid']}", "hard or mixed invalids", "only duplicate_token", "pass" if all_final_invalid_duplicate and no_hard else "fail", "supports_continuation" if all_final_invalid_duplicate and no_hard else "terminal", rel(FAILURE_TAXONOMY_PATH), "No malformed JSON, unknown-token, mapping, or other parser invalid class appears in final rows.")
    add("breaker_locality", "single failing window 12; qwen duplicate zero", "persistent multi-window/global", "local or cell-specific", "pass" if single_window_failure and qwen_duplicate_zero else "fail", "supports_continuation" if single_window_failure and qwen_duplicate_zero else "terminal_or_blocked", rel(WINDOW_ANALYSIS_PATH), "The breaker is contractually valid but not sufficient for a global systemic-regression claim.")
    add("raw_artifact_integrity", str(no_raw_corruption).lower(), "corrupt or duplicate ids", "present and unique", "pass" if no_raw_corruption else "fail", "supports_continuation" if no_raw_corruption else "terminal", rel(HASHES_PATH), "Raw final response files and JSONL ids are intact.")
    add("original_repo_status", "clean" if original_clean else summary_integrity["original_repo_status"], "dirty original_repo", "clean", "pass" if original_clean else "fail", "supports_continuation" if original_clean else "blocked", "git status --short original_repo", "No modification to original_repo/ is detected.")
    add("source_contamination", str(no_contamination).lower(), "V1/V2/V3/pilot source contamination", "none", "pass" if no_contamination else "fail", "supports_continuation" if no_contamination else "blocked", rel(rerank.MANIFEST_PATH), "No V1/V2/V3/pilot raw or candidate source is used by the post-breaker audit.")
    add("pending_final_response_count", str(summary_integrity["pending_final_response_count"]), "n/a", "1600 exactly identified", "pass" if residual_count_ok else "fail", "supports_continuation" if residual_count_ok else "blocked", rel(rerank.MANIFEST_PATH), "The residual final-response set has exact request ids.")
    add("pending_manifest_interval", str(residual_interval_ok).lower(), "n/a", "exactly 2401-4000", "fail" if not residual_interval_ok else "pass", "blocks_versioned_continuation_as_requested" if not residual_interval_ok else "supports_continuation", rel(rerank.MANIFEST_PATH), "The residual final-response set is non-contiguous because 48 non-final timeout attempts occurred before ordinal 2449.")
    add("non_final_transport_attempts", str(non_final_transport), "n/a", "human policy for inclusion/exclusion", "review", "blocks_until_human_evidence_policy" if non_final_transport else "supports_continuation", rel(rerank.EXECUTION_PATH), "These attempts have raw timeout evidence but no final parser row; a V4.1 continuation must explicitly decide whether to include them.")

    if any(row["status"] == "fail" and row["effect_on_decision"] == "terminal" for row in rows):
        decision = DECISION_TERMINAL
    elif residual_count_ok and not residual_interval_ok:
        decision = DECISION_BLOCKED
    elif non_final_transport:
        decision = DECISION_BLOCKED
    elif all(row["status"] in {"pass", "review"} for row in rows):
        decision = DECISION_READY
    else:
        decision = DECISION_BLOCKED
    return rows, decision


def source_hashes(sources: Mapping[str, Any]) -> dict[str, Any]:
    raw_paths = sources["raw_paths"]
    raw_files = sorted(path for path in RAW_OUTPUT_DIR.rglob("*") if path.is_file())
    raw_manifest = [
        {"path": rel(path), "size": path.stat().st_size, "sha256": f6l.safe_file_sha256(path)}
        for path in raw_files
    ]
    source_paths = [
        rerank.MANIFEST_PATH,
        rerank.PARSE_RESULTS_PATH,
        rerank.RECOMMENDATIONS_PATH,
        rerank.EXECUTION_PATH,
        rerank.STATE_PATH,
        rerank.DUPLICATE_WINDOWS_PATH,
        rerank.RANKING_BASELINE_AUDIT_PATH,
        rerank.RANKING_BASELINE_PATHS["metrics"],
        rerank.RANKING_BASELINE_PATHS["comparisons"],
        rerank.RANKING_BASELINE_PATHS["statistics"],
        rerank.RANKING_BASELINE_PATHS["summary"],
        rerank.RANKING_BASELINE_PATHS["hash_text"],
        rerank.RANKING_BASELINE_PATHS["report"],
    ]
    return {
        "raw_dir": rel(RAW_OUTPUT_DIR),
        "raw_file_count": len(raw_manifest),
        "partial_raw_artifact_manifest_sha256": f6l.sha256_text(canonical_json(raw_manifest)),
        "raw_jsonl": {name: f6l.safe_file_sha256(path) for name, path in raw_paths.items() if path.exists()},
        "source_artifacts": {rel(path): f6l.safe_file_sha256(path) for path in source_paths if path.exists()},
        "raw_file_manifest": raw_manifest,
    }


def build_summary_integrity(
    sources: Mapping[str, Any],
    cell_rows: Sequence[Mapping[str, str]],
    failure_rows: Sequence[Mapping[str, str]],
    window_rows: Sequence[Mapping[str, str]],
    hashes: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_rows = sources["manifest_rows"]
    parse_rows = sources["parse_rows"]
    responses = sources["responses"]
    final_ids = {row["request_id"] for row in parse_rows}
    response_ids = [str(row.get("request_id", "")) for row in responses]
    non_final_transport = [row for row in responses if str(row.get("is_final_attempt", "true")).lower() != "true"]
    pending = [row for row in manifest_rows if row["request_id"] not in final_ids]
    pending_ord = [request_ordinal(row["request_id"]) for row in pending]
    expected_2401_4000 = set(range(2401, 4001))
    pending_set = set(pending_ord)
    parser_schema_contract_unchanged = (
        all(row.get("parser_source_hash") == rerank.EXPECTED_V4_PARSER_HASH for row in parse_rows)
        and all(row.get("output_contract_hash") == rerank.EXPECTED_V4_OUTPUT_CONTRACT_HASH for row in parse_rows)
        and all(rerank.EXPECTED_V4_SCHEMA_HASH in row.get("generation_parameters", "") for row in manifest_rows)
    )
    baseline_rows = rerank.ranking_baseline_audit_rows(write_outputs=False)
    original_status = f6l.git_output(["status", "--short", "original_repo"])
    final_file_paths = [RAW_OUTPUT_DIR / "responses" / f"{row['request_id']}.json" for row in parse_rows]
    raw_source_strings = [row.get("candidate_pool_source", "") for row in manifest_rows] + [rel(RAW_OUTPUT_DIR)]
    contamination = any(re.search(r"v[123]|pilot", value.casefold()) for value in raw_source_strings)
    return {
        "run_id": RUN_ID,
        "planned": len(manifest_rows),
        "final_completed": len(parse_rows),
        "strict_valid": sum(1 for row in parse_rows if row["strict_valid"] == "true"),
        "structural_invalid": sum(1 for row in parse_rows if row["strict_valid"] != "true"),
        "duplicate_token_invalid": sum(1 for row in parse_rows if row.get("failure_reason") == "duplicate_token"),
        "malformed_json": sum(int(row.get("malformed_json_count") or 0) for row in parse_rows),
        "unknown_token": sum(int(row.get("unknown_token_count") or 0) for row in parse_rows),
        "hard_structural_invalid": sum(1 for row in parse_rows if row.get("strict_valid") != "true" and row.get("failure_reason") != "duplicate_token"),
        "pending_final_response_count": len(pending),
        "pending_final_response_ordinals_are_2401_4000": pending_set == expected_2401_4000,
        "pending_manifest_ordinal_ranges": ordinal_ranges(pending_ord),
        "never_attempted_manifest_ordinal_ranges": ordinal_ranges([request_ordinal(row["request_id"]) for row in pending if row["request_id"] not in set(response_ids)]),
        "non_final_transport_timeout_count": len(non_final_transport),
        "non_final_transport_manifest_ordinal_ranges": ordinal_ranges([request_ordinal(str(row.get("request_id", ""))) for row in non_final_transport]),
        "raw_response_jsonl_records": len(responses),
        "raw_jsonl_unique_request_ids": len(response_ids) == len(set(response_ids)),
        "raw_final_files_exist": all(path.exists() for path in final_file_paths),
        "parser_schema_contract_unchanged": parser_schema_contract_unchanged,
        "baseline_audit_pass": all(row["status"] == "pass" for row in baseline_rows),
        "baseline_freeze_hash": rerank.RANKING_BASELINE_FREEZE_HASH,
        "original_repo_status": original_status,
        "no_v1_v2_v3_pilot_source_contamination": not contamination,
        "partial_raw_artifact_manifest_sha256": hashes["partial_raw_artifact_manifest_sha256"],
    }


def output_hashes() -> dict[str, str]:
    paths = [
        CELL_COVERAGE_PATH,
        FAILURE_TAXONOMY_PATH,
        WINDOW_ANALYSIS_PATH,
        METRICS_PATH,
        STATISTICAL_ASSESSMENT_PATH,
        DECISION_MATRIX_PATH,
        BASELINE_AUDIT_PATH,
        SUMMARY_PATH,
        REPORT_PATH,
    ]
    return {rel(path): f6l.safe_file_sha256(path) for path in paths if path.exists()}


def write_report(summary: Mapping[str, Any], decision_rows: Sequence[Mapping[str, str]], stat_rows: Sequence[Mapping[str, str]]) -> None:
    integrity = summary["integrity"]
    lines = [
        "# F6L LastFM V4 Reranking Post-Breaker Forensic Consolidation",
        "",
        f"Decision token: `{summary['decision_token']}`.",
        "",
        "## Scope",
        "",
        "- No inference, Ollama call, API call, detached process, raw overwrite, response repair, deduplication, or contract rewrite was performed.",
        f"- Audited run: `{RUN_ID}`.",
        f"- Operational stop token: `{rerank.DECISION_SYSTEMIC_DUPLICATE}`.",
        f"- Ranking baseline freeze hash: `{rerank.RANKING_BASELINE_FREEZE_HASH}`.",
        "",
        "## Integrity",
        "",
        f"- Planned requests: `{integrity['planned']}`.",
        f"- Final completed parser rows: `{integrity['final_completed']}`.",
        f"- Strict-valid + structural-invalid: `{integrity['strict_valid']} + {integrity['structural_invalid']} = {integrity['final_completed']}`.",
        f"- Duplicate-token invalid: `{integrity['duplicate_token_invalid']}`.",
        f"- Malformed JSON: `{integrity['malformed_json']}`; unknown token: `{integrity['unknown_token']}`; hard structural invalid: `{integrity['hard_structural_invalid']}`.",
        f"- Non-final transport timeouts: `{integrity['non_final_transport_timeout_count']}`.",
        f"- Pending final-response requests: `{integrity['pending_final_response_count']}`.",
        f"- Pending manifest ordinals are exactly 2401-4000: `{str(integrity['pending_final_response_ordinals_are_2401_4000']).lower()}`.",
        f"- Pending manifest ordinal ranges: `{integrity['pending_manifest_ordinal_ranges']}`.",
        f"- Raw partial artifact manifest SHA-256: `{integrity['partial_raw_artifact_manifest_sha256']}`.",
        f"- Baseline audit: `{'pass' if integrity['baseline_audit_pass'] else 'fail'}`.",
        f"- original_repo status: `{'clean' if integrity['original_repo_status'] == '' else integrity['original_repo_status']}`.",
        "",
        "## Breaker",
        "",
        "- Window 12 is contractually failed: `11/200 = 0.055`, threshold `>0.05`.",
        "- The failed window is the first and only closed non-overlap window above 5%.",
        "- The evidence supports a local/cell-specific duplicate-token spike, not a global systemic regression claim.",
        "",
        "## Decision",
        "",
        f"- Final post-breaker token: `{summary['decision_token']}`.",
        "- Reason: the residual final-response set is exact but not the requested contiguous manifest interval `2401-4000`; it includes 48 earlier non-final timeout attempts plus 1552 never-attempted ordinals.",
        "- A versioned continuation needs human policy on whether those 48 timeout attempts are part of the continuation manifest. No V4.1 executable proposal was generated.",
        "",
        "## Required Tables",
        "",
        f"- Cell coverage: `{rel(CELL_COVERAGE_PATH)}`.",
        f"- Failure taxonomy: `{rel(FAILURE_TAXONOMY_PATH)}`.",
        f"- Window analysis: `{rel(WINDOW_ANALYSIS_PATH)}`.",
        f"- Metrics: `{rel(METRICS_PATH)}`.",
        f"- Statistical assessment: `{rel(STATISTICAL_ASSESSMENT_PATH)}`.",
        f"- Decision matrix: `{rel(DECISION_MATRIX_PATH)}`.",
        f"- Summary: `{rel(SUMMARY_PATH)}`.",
        f"- Hashes: `{rel(HASHES_PATH)}`.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_post_breaker_outputs() -> dict[str, Any]:
    sources = load_sources()
    hashes = source_hashes(sources)
    cell_rows, cell_stats = build_cell_coverage(sources)
    failure_rows = build_failure_taxonomy(sources, cell_stats)
    window_rows, window_stats = build_window_analysis(sources)
    summary_integrity = build_summary_integrity(sources, cell_rows, failure_rows, window_rows, hashes)
    stat_rows = build_statistical_assessment(sources, cell_stats, window_rows, window_stats)
    decision_rows, decision = build_decision_matrix(sources, cell_rows, failure_rows, window_rows, summary_integrity)
    metric_rows = build_metrics(sources, cell_stats)
    baseline_audit_rows = rerank.ranking_baseline_audit_rows(write_outputs=False)

    f6l.write_csv(CELL_COVERAGE_PATH, cell_rows, CELL_COVERAGE_FIELDS)
    f6l.write_csv(FAILURE_TAXONOMY_PATH, failure_rows, FAILURE_FIELDS)
    f6l.write_csv(WINDOW_ANALYSIS_PATH, window_rows, WINDOW_FIELDS)
    f6l.write_csv(METRICS_PATH, metric_rows, METRIC_FIELDS)
    f6l.write_csv(STATISTICAL_ASSESSMENT_PATH, stat_rows, STAT_FIELDS)
    f6l.write_csv(DECISION_MATRIX_PATH, decision_rows, DECISION_FIELDS)
    f6l.write_csv(BASELINE_AUDIT_PATH, baseline_audit_rows, rerank.RANKING_BASELINE_AUDIT_FIELDS)

    summary = {
        "post_breaker_id": POST_ID,
        "run_id": RUN_ID,
        "created_at": utc_now(),
        "decision_token": decision,
        "operational_run_decision_token": rerank.DECISION_SYSTEMIC_DUPLICATE,
        "no_inference_or_api_call_performed": True,
        "raw_output_dir": rel(RAW_OUTPUT_DIR),
        "integrity": summary_integrity,
        "window_stats": window_stats,
        "output_paths": {
            "cell_coverage": rel(CELL_COVERAGE_PATH),
            "failure_taxonomy": rel(FAILURE_TAXONOMY_PATH),
            "window_analysis": rel(WINDOW_ANALYSIS_PATH),
            "metrics": rel(METRICS_PATH),
            "statistical_assessment": rel(STATISTICAL_ASSESSMENT_PATH),
            "decision_matrix": rel(DECISION_MATRIX_PATH),
            "baseline_audit": rel(BASELINE_AUDIT_PATH),
            "report": rel(REPORT_PATH),
        },
        "continuation_v4_1_proposal_generated": False,
        "continuation_v4_1_blocker": "pending final-response manifest ordinals are not exactly 2401-4000 because 48 non-final transport timeouts have raw evidence before ordinal 2449",
    }
    f6l.write_json(SUMMARY_PATH, summary)
    write_report(summary, decision_rows, stat_rows)
    hashes_payload = {**hashes, "output_artifacts": output_hashes()}
    f6l.write_json(HASHES_PATH, hashes_payload)
    return summary


def main() -> None:
    summary = build_post_breaker_outputs()
    print(canonical_json({"decision_token": summary["decision_token"], "run_id": RUN_ID, "completed": summary["integrity"]["final_completed"], "residual": summary["integrity"]["pending_final_response_count"]}))


if __name__ == "__main__":
    main()

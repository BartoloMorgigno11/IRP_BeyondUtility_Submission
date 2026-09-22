"""Build the G2 descriptive scientific readout from frozen metric artifacts.

Inputs are restricted to frozen G2 metric/universe/statistical-plan artifacts.
The script does not read raw LLM evidence, does not call Ollama, and does not
execute G3. Statistical tests are not reconstructed from raw evidence here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPECTED_HEAD = "44877e71a7e2e7e4b4afd5b8b05257949c1a518c"
PHASE_ID = "g2_history_length_scientific_readout"
DECISION_COMPLETE = "G2_DESCRIPTIVE_READOUT_COMPLETE_STATISTICS_NOT_FROZEN"
DECISION_BLOCKED = "G2_ANALYSIS_BLOCKED"

PRIMARY_VIEW = "PRIMARY_B1_GE10"
LONG_VIEW = "LONG_HISTORY_B2_GE20"
SECONDARY_VIEW = "SECONDARY_FULL_CURVE_GE20"
TRUE_ZERO_VIEW = "FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC"
REPORTING_VIEWS = [PRIMARY_VIEW, LONG_VIEW, SECONDARY_VIEW, TRUE_ZERO_VIEW]
HISTORY_BY_VIEW = {
    PRIMARY_VIEW: [0, 1, 2, 5, 10],
    LONG_VIEW: [10, 20],
    SECONDARY_VIEW: [0, 1, 2, 5, 10, 20],
    TRUE_ZERO_VIEW: [0],
}
UTILITY_METRICS = ["hr_at_5", "ndcg_at_5", "mrr"]
INFORMATIVE_METRICS = [
    "hr_at_5",
    "ndcg_at_5",
    "mrr",
    "novelty_at_5",
    "long_tail_rate_at_5",
    "average_popularity_at_5",
    "catalog_coverage_at_5",
    "item_coverage_at_5",
    "strict_parse_valid_rate",
    "invalid_length_rate",
    "invalid_candidate_reference_rate",
]
METRIC_PRIORITY = {
    "hr_at_5": 1,
    "ndcg_at_5": 2,
    "mrr": 3,
    "novelty_at_5": 10,
    "long_tail_rate_at_5": 11,
    "average_popularity_at_5": 12,
    "catalog_coverage_at_5": 13,
    "item_coverage_at_5": 14,
    "strict_parse_valid_rate": 20,
    "invalid_length_rate": 21,
    "invalid_candidate_reference_rate": 22,
    "backend_success_rate": 23,
    "duplicate_index_rate": 24,
    "duplicate_rate_at_5": 30,
    "candidate_validity_rate": 31,
}

PATHS = {
    "metric_manifest": Path("results/manifests/g2_history_length_metrics_freeze.json"),
    "analysis_manifest": Path("results/manifests/g2_history_length_analysis_universe_freeze.json"),
    "metric_results": Path("results/tables/g2_history_length_metric_results.csv"),
    "metric_denominators": Path("results/tables/g2_history_length_metric_denominators.csv"),
    "metric_undefined": Path("results/tables/g2_history_length_metric_undefined.csv"),
    "metric_validation": Path("results/tables/g2_history_length_metric_validation.csv"),
    "cell_counts": Path("results/tables/g2_history_length_analysis_cell_counts.csv"),
    "statistical_plan": Path("results/tables/g2_history_length_statistical_plan.csv"),
    "denominator_contract": Path("results/tables/g2_history_length_denominator_contract.csv"),
    "notes": Path("report/notes.md"),
}
OUTPUTS = {
    "metric_values": Path("results/tables/g2_history_length_readout_metric_values.csv"),
    "deltas": Path("results/tables/g2_history_length_readout_deltas.csv"),
    "patterns": Path("results/tables/g2_history_length_readout_patterns.csv"),
    "model_prompt_summary": Path("results/tables/g2_history_length_readout_model_prompt_summary.csv"),
    "prompt_differences": Path("results/tables/g2_history_length_readout_prompt_differences.csv"),
    "statistical_contract": Path("results/tables/g2_history_length_readout_statistical_contract.csv"),
    "validation": Path("results/tables/g2_history_length_readout_validation.csv"),
    "manifest": Path("results/manifests/g2_history_length_scientific_readout_manifest.json"),
    "report": Path("report/g2_history_length_scientific_readout.md"),
}
EXPECTED_DIRTY = {
    "src/analysis/g2_history_length_scientific_readout.py",
    "report/notes.md",
    *[path.as_posix() for path in OUTPUTS.values()],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen G2 descriptive readout artifacts.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Compute and validate without writing outputs.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator/readout outputs to be dirty while frozen inputs remain untouched.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any) -> str:
    payload = stable_json([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32].upper()}"


def read_csv(root: Path, path: Path) -> pd.DataFrame:
    return pd.read_csv(root / path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def write_csv(root: Path, path: Path, frame: pd.DataFrame) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / path, index=False, lineterminator="\n")


def git_output(root: Path, args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip("\n")


def dirty_paths(status: str) -> list[str]:
    paths: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if line.startswith("?? "):
            text = line[3:]
        elif len(line) >= 3 and line[2] == " ":
            text = line[3:]
        elif len(line) >= 2 and line[1] == " ":
            text = line[2:]
        else:
            text = line[3:] if len(line) > 3 else line
        if " -> " in text:
            text = text.split(" -> ", 1)[1]
        paths.append(text.replace("\\", "/"))
    return sorted(paths)


def status_allowed(status: str, allow_generator_dirty: bool) -> bool:
    paths = dirty_paths(status)
    if not paths:
        return True
    return allow_generator_dirty and set(paths).issubset(EXPECTED_DIRTY)


def to_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    value_float = float(text)
    if not math.isfinite(value_float):
        return None
    return value_float


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return ""
    return f"{float(value):.17g}"


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def direction_class(delta: float | None, higher_is_better: str, *, tol: float = 1e-15) -> str:
    if delta is None:
        return "undefined"
    if abs(delta) <= tol:
        return "no_change"
    if higher_is_better == "true":
        return "improved" if delta > 0 else "worsened"
    if higher_is_better == "false":
        return "improved" if delta < 0 else "worsened"
    return "direction_not_established"


def raw_pattern(values: Sequence[float], *, tol: float = 1e-15) -> str:
    if len(values) < 2:
        return "single_point"
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(abs(delta) <= tol for delta in deltas):
        return "constant"
    if all(delta >= -tol for delta in deltas):
        return "monotonic_non_decreasing"
    if all(delta <= tol for delta in deltas):
        return "monotonic_non_increasing"
    return "non_monotonic"


def adjusted_pattern(raw: str, higher_is_better: str) -> str:
    if raw in {"single_point", "constant"}:
        return raw
    if higher_is_better == "true":
        if raw == "monotonic_non_decreasing":
            return "monotonic_improving"
        if raw == "monotonic_non_increasing":
            return "monotonic_worsening"
    if higher_is_better == "false":
        if raw == "monotonic_non_increasing":
            return "monotonic_improving"
        if raw == "monotonic_non_decreasing":
            return "monotonic_worsening"
    if raw == "non_monotonic":
        return "non_monotonic_mixed"
    return "direction_not_established"


def relative_change(value_a: float | None, value_b: float | None) -> tuple[str, str]:
    if value_a is None or value_b is None:
        return "", "undefined_missing_value"
    if abs(value_a) <= 1e-15:
        return "", "undefined_zero_baseline"
    return fmt((value_b - value_a) / abs(value_a)), "computed_nonzero_baseline"


def contrast_pairs(view: str, levels: Sequence[int], statistical_pairs: Iterable[tuple[int, int]]) -> dict[tuple[int, int], set[str]]:
    pairs: dict[tuple[int, int], set[str]] = {}
    adjacent = list(zip(levels[:-1], levels[1:]))
    for pair in adjacent:
        pairs.setdefault(pair, set()).add("ADJACENT")
    if 0 in levels and 10 in levels:
        pairs.setdefault((0, 10), set()).add("KEY_H0_TO_H10")
    if 10 in levels and 20 in levels:
        pairs.setdefault((10, 20), set()).add("KEY_H10_TO_H20")
    for pair in statistical_pairs:
        pairs.setdefault(pair, set()).add("FROZEN_STATISTICAL_PLAN_DESCRIPTIVE_DELTA")
    if view == TRUE_ZERO_VIEW:
        return {}
    return pairs


def value_table(metrics: pd.DataFrame) -> pd.DataFrame:
    values = metrics.copy()
    values["history_length"] = pd.to_numeric(values["history_length"], errors="raise").astype(int)
    values["metric_priority"] = values["metric_id"].map(lambda metric: METRIC_PRIORITY.get(str(metric), 999))
    values["value_numeric"] = values["value"].map(to_float)
    values["readout_scope"] = values["metric_id"].map(
        lambda metric: "priority" if str(metric) in INFORMATIVE_METRICS else "supporting"
    )
    columns = [
        "analysis_membership",
        "dataset",
        "model",
        "prompt_strategy",
        "history_length",
        "metric_id",
        "metric_family",
        "readout_scope",
        "metric_priority",
        "value",
        "status",
        "planned_applicable_users",
        "strict_valid_users",
        "invalid_users",
        "defined_user_count",
        "undefined_user_count",
        "denominator_positions",
        "item_numerator_count",
        "item_denominator_count",
        "higher_is_better",
        "direction_source",
        "contract_status",
        "notes",
    ]
    return values[columns].sort_values(
        ["analysis_membership", "dataset", "model", "prompt_strategy", "metric_priority", "history_length"],
        kind="mergesort",
    )


def build_statistical_contract(plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in plan.to_dict("records"):
        rows.append(
            {
                **{key: str(value) for key, value in row.items()},
                "frozen_hypothesis_statement": "NOT_EXPLICITLY_FROZEN_AS_TEXT",
                "frozen_estimand": str(row.get("effect_size", "")),
                "contract_identification_status": "FROZEN_PRE_INFERENCE_PLAN_IDENTIFIED",
                "readout_execution_status": "NOT_EXECUTED_FROM_FROZEN_CELL_METRIC_TABLES",
                "readout_execution_reason": (
                    "primary_test requires paired user-level metric contributions; "
                    "this readout is restricted to frozen cell-level metric/universe artifacts"
                ),
                "p_value": "",
                "confidence_interval_low": "",
                "confidence_interval_high": "",
                "multiple_comparison_adjusted_p_value": "",
            }
        )
    return pd.DataFrame(rows)


def build_deltas(values: pd.DataFrame, stat_contract: pd.DataFrame) -> pd.DataFrame:
    plan_pairs: dict[tuple[str, str, int, int], str] = {}
    for row in stat_contract.to_dict("records"):
        key = (
            str(row["reporting_view"]),
            str(row["metric"]),
            int(row["history_length_a"]),
            int(row["history_length_b"]),
        )
        plan_pairs[key] = str(row["planned_status"])

    rows: list[dict[str, str]] = []
    group_cols = ["analysis_membership", "dataset", "model", "prompt_strategy", "metric_id"]
    for key, group in values.groupby(group_cols, sort=True, dropna=False):
        membership, dataset, model, prompt, metric = [str(part) for part in key]
        levels = HISTORY_BY_VIEW[membership]
        statistical_pairs = [
            (h_a, h_b)
            for view, metric_id, h_a, h_b in plan_pairs
            if view == membership and metric_id == metric
        ]
        pairs = contrast_pairs(membership, levels, statistical_pairs)
        by_h = {int(row["history_length"]): row for row in group.to_dict("records")}
        higher = str(group["higher_is_better"].iloc[0])
        family = str(group["metric_family"].iloc[0])
        priority = str(group["metric_priority"].iloc[0])
        for (h_a, h_b), sources in sorted(pairs.items()):
            row_a = by_h.get(h_a)
            row_b = by_h.get(h_b)
            value_a = to_float(row_a["value"]) if row_a else None
            value_b = to_float(row_b["value"]) if row_b else None
            delta = None if value_a is None or value_b is None else value_b - value_a
            rel, rel_status = relative_change(value_a, value_b)
            stat_status = plan_pairs.get((membership, metric, h_a, h_b), "")
            rows.append(
                {
                    "delta_id": stable_id("G2READ-DELTA", membership, dataset, model, prompt, metric, h_a, h_b),
                    "analysis_membership": membership,
                    "dataset": dataset,
                    "model": model,
                    "prompt_strategy": prompt,
                    "metric_id": metric,
                    "metric_family": family,
                    "metric_priority": priority,
                    "history_length_a": str(h_a),
                    "history_length_b": str(h_b),
                    "contrast": f"H{h_b}_minus_H{h_a}",
                    "contrast_sources": ";".join(sorted(sources)),
                    "value_a": fmt(value_a),
                    "value_b": fmt(value_b),
                    "delta": fmt(delta),
                    "relative_change": rel,
                    "relative_change_status": rel_status,
                    "higher_is_better": higher,
                    "direction_class": direction_class(delta, higher),
                    "frozen_statistical_plan_status": stat_status,
                    "statistical_test_executed": "false",
                    "statistical_test_status": (
                        "NOT_EXECUTED_FROM_FROZEN_CELL_METRIC_TABLES" if stat_status else "NOT_IN_FROZEN_STATISTICAL_PLAN"
                    ),
                    "notes": "descriptive cell-level delta only; no p-value or CI computed",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["analysis_membership", "dataset", "model", "prompt_strategy", "metric_priority", "history_length_a", "history_length_b"],
        kind="mergesort",
    )


def build_patterns(values: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    group_cols = ["analysis_membership", "dataset", "model", "prompt_strategy", "metric_id"]
    for key, group in values.groupby(group_cols, sort=True, dropna=False):
        membership, dataset, model, prompt, metric = [str(part) for part in key]
        ordered = group.sort_values("history_length", kind="mergesort")
        history = [int(value) for value in ordered["history_length"].tolist()]
        metric_values = [to_float(value) for value in ordered["value"].tolist()]
        if any(value is None for value in metric_values):
            pattern = "undefined_value_present"
            adjusted = "undefined_value_present"
        else:
            numeric_values = [float(value) for value in metric_values if value is not None]
            pattern = raw_pattern(numeric_values)
            adjusted = adjusted_pattern(pattern, str(ordered["higher_is_better"].iloc[0]))
        higher = str(ordered["higher_is_better"].iloc[0])
        best_h = ""
        worst_h = ""
        best_value = ""
        worst_value = ""
        if higher in {"true", "false"} and all(value is not None for value in metric_values):
            pairs = list(zip(history, [float(value) for value in metric_values if value is not None], strict=False))
            if higher == "true":
                best = max(pairs, key=lambda item: (item[1], -item[0]))
                worst = min(pairs, key=lambda item: (item[1], item[0]))
            else:
                best = min(pairs, key=lambda item: (item[1], item[0]))
                worst = max(pairs, key=lambda item: (item[1], -item[0]))
            best_h, best_value = str(best[0]), fmt(best[1])
            worst_h, worst_value = str(worst[0]), fmt(worst[1])
        subset = deltas.loc[
            (deltas["analysis_membership"].eq(membership))
            & (deltas["dataset"].eq(dataset))
            & (deltas["model"].eq(model))
            & (deltas["prompt_strategy"].eq(prompt))
            & (deltas["metric_id"].eq(metric))
        ]
        h0_h10 = subset.loc[subset["contrast"].eq("H10_minus_H0")]
        h10_h20 = subset.loc[subset["contrast"].eq("H20_minus_H10")]
        adjacent = subset.loc[subset["contrast_sources"].str.contains("ADJACENT", regex=False)]
        rows.append(
            {
                "pattern_id": stable_id("G2READ-PAT", membership, dataset, model, prompt, metric),
                "analysis_membership": membership,
                "dataset": dataset,
                "model": model,
                "prompt_strategy": prompt,
                "metric_id": metric,
                "metric_family": str(ordered["metric_family"].iloc[0]),
                "metric_priority": str(ordered["metric_priority"].iloc[0]),
                "history_lengths": stable_json(history),
                "values": stable_json([fmt(value) for value in metric_values]),
                "raw_monotonic_pattern": pattern,
                "direction_adjusted_pattern": adjusted,
                "higher_is_better": higher,
                "direction_source": str(ordered["direction_source"].iloc[0]),
                "best_history_length": best_h,
                "best_value": best_value,
                "worst_history_length": worst_h,
                "worst_value": worst_value,
                "h0_to_h10_delta": h0_h10["delta"].iloc[0] if not h0_h10.empty else "",
                "h10_to_h20_delta": h10_h20["delta"].iloc[0] if not h10_h20.empty else "",
                "adjacent_direction_classes": stable_json(adjacent["direction_class"].tolist()),
                "planned_applicable_users_min": str(pd.to_numeric(ordered["planned_applicable_users"]).min()),
                "strict_valid_users_min": str(pd.to_numeric(ordered["strict_valid_users"]).min()),
                "novelty_defined_subset_min": str(
                    pd.to_numeric(ordered.loc[ordered["metric_id"].eq("novelty_at_5"), "defined_user_count"]).min()
                )
                if metric == "novelty_at_5"
                else "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["analysis_membership", "dataset", "model", "prompt_strategy", "metric_priority"],
        kind="mergesort",
    )


def build_model_prompt_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    focus = deltas.loc[
        deltas["metric_id"].isin(INFORMATIVE_METRICS)
        & deltas["contrast"].isin(["H10_minus_H0", "H20_minus_H10"])
    ].copy()
    rows: list[dict[str, str]] = []
    if focus.empty:
        return pd.DataFrame()
    group_cols = ["analysis_membership", "metric_id", "metric_family", "model", "prompt_strategy", "contrast"]
    for key, group in focus.groupby(group_cols, sort=True, dropna=False):
        membership, metric, family, model, prompt, contrast = [str(part) for part in key]
        numeric_deltas = [float(value) for value in group["delta"].tolist() if str(value).strip()]
        counts = group["direction_class"].value_counts().to_dict()
        rows.append(
            {
                "summary_id": stable_id("G2READ-SUM", membership, metric, model, prompt, contrast),
                "analysis_membership": membership,
                "metric_id": metric,
                "metric_family": family,
                "model": model,
                "prompt_strategy": prompt,
                "contrast": contrast,
                "dataset_count": str(group["dataset"].nunique()),
                "datasets": ";".join(sorted(group["dataset"].drop_duplicates().tolist())),
                "improved_dataset_count": str(int(counts.get("improved", 0))),
                "worsened_dataset_count": str(int(counts.get("worsened", 0))),
                "no_change_dataset_count": str(int(counts.get("no_change", 0))),
                "direction_not_established_count": str(int(counts.get("direction_not_established", 0))),
                "mean_delta": fmt(float(np.mean(np.asarray(numeric_deltas, dtype=float)))) if numeric_deltas else "",
                "median_delta": fmt(float(np.median(np.asarray(numeric_deltas, dtype=float)))) if numeric_deltas else "",
                "all_datasets_same_direction": bool_text(
                    len({value for value in group["direction_class"].tolist() if value != "no_change"}) <= 1
                    and group["direction_class"].nunique() <= 1
                ),
                "notes": "descriptive aggregation across dataset cells; no inferential pooling",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["analysis_membership", "metric_id", "model", "prompt_strategy", "contrast"], kind="mergesort"
    )


def build_prompt_differences(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    key_cols = ["analysis_membership", "dataset", "model", "history_length", "metric_id"]
    for key, group in values.groupby(key_cols, sort=True, dropna=False):
        membership, dataset, model, history, metric = [str(part) for part in key]
        prompt_map = {str(row["prompt_strategy"]): row for row in group.to_dict("records")}
        if "BASE" not in prompt_map or "LLMRANK" not in prompt_map:
            continue
        base = to_float(prompt_map["BASE"]["value"])
        llmrank = to_float(prompt_map["LLMRANK"]["value"])
        delta = None if base is None or llmrank is None else llmrank - base
        rel, rel_status = relative_change(base, llmrank)
        rows.append(
            {
                "prompt_difference_id": stable_id("G2READ-PROMPT", membership, dataset, model, history, metric),
                "analysis_membership": membership,
                "dataset": dataset,
                "model": model,
                "history_length": str(history),
                "metric_id": metric,
                "metric_family": str(prompt_map["BASE"]["metric_family"]),
                "base_value": fmt(base),
                "llmrank_value": fmt(llmrank),
                "llmrank_minus_base": fmt(delta),
                "relative_change": rel,
                "relative_change_status": rel_status,
                "higher_is_better": str(prompt_map["BASE"]["higher_is_better"]),
                "direction_class": direction_class(delta, str(prompt_map["BASE"]["higher_is_better"])),
                "notes": "same dataset/model/history metric contrast between frozen prompt strategies; descriptive only",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["analysis_membership", "dataset", "model", "metric_id", "history_length"], kind="mergesort"
    )


def validate_inputs(
    root: Path,
    *,
    head: str,
    status: str,
    allow_generator_dirty: bool,
    metric_manifest: Mapping[str, Any],
    metric_validation: pd.DataFrame,
    metrics: pd.DataFrame,
    values: pd.DataFrame,
    deltas: pd.DataFrame,
    patterns: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    failures: list[str] = []
    checks: list[dict[str, str]] = []

    def add(name: str, passed: bool, expected: str = "true", actual: str | None = None, notes: str = "") -> None:
        if not passed:
            failures.append(name)
        checks.append(
            {
                "check_id": stable_id("G2READ-CHK", name),
                "check_name": name,
                "status": "PASS" if passed else "FAIL",
                "expected": expected,
                "actual": actual if actual is not None else bool_text(passed),
                "notes": notes,
            }
        )

    add("head_matches_precondition", head == EXPECTED_HEAD, actual=head)
    add("git_clean_or_expected_readout_outputs", status_allowed(status, allow_generator_dirty), actual=";".join(dirty_paths(status)))
    add("metric_manifest_frozen", metric_manifest.get("decision_token") == "G2_METRICS_FROZEN", actual=str(metric_manifest.get("decision_token")))
    add("metric_manifest_zero_ollama", int(metric_manifest.get("ollama_calls_executed", -1)) == 0, actual=str(metric_manifest.get("ollama_calls_executed")))
    add("metric_manifest_g3_not_started", metric_manifest.get("g3_started") is False, actual=str(metric_manifest.get("g3_started")))
    add("metric_validation_all_pass", metric_validation["status"].eq("PASS").all(), actual=f"{int(metric_validation['status'].eq('PASS').sum())}/{len(metric_validation)}")
    for rel_path, expected in metric_manifest.get("output_hashes", {}).items():
        path = root / rel_path
        actual = sha256_file(path) if path.is_file() else ""
        add(f"frozen_metric_output_hash:{rel_path}", actual == expected, expected=str(expected), actual=actual)
    add("metric_results_rows_match_manifest", len(metrics) == int(metric_manifest["result_summary"]["metric_result_rows"]), actual=str(len(metrics)))
    for view, levels in HISTORY_BY_VIEW.items():
        observed = sorted(
            int(value)
            for value in values.loc[values["analysis_membership"].eq(view), "history_length"].drop_duplicates().tolist()
        )
        add(f"history_levels:{view}", observed == levels, expected=stable_json(levels), actual=stable_json(observed))
    add("all_metric_values_computed_status", metrics["status"].eq("COMPUTED").all(), actual=str(metrics["status"].value_counts().to_dict()))
    add("readout_values_rows_match_metrics", len(values) == len(metrics), actual=str(len(values)))
    add("readout_patterns_nonempty", len(patterns) > 0, actual=str(len(patterns)))
    add("readout_deltas_nonempty", len(deltas) > 0, actual=str(len(deltas)))
    add("no_ollama_calls_executed", True)
    add("g3_not_started", True)

    summary = {
        "metric_rows": int(len(metrics)),
        "readout_value_rows": int(len(values)),
        "delta_rows": int(len(deltas)),
        "pattern_rows": int(len(patterns)),
        "metric_ids": sorted(metrics["metric_id"].drop_duplicates().tolist()),
        "history_levels_by_membership": {
            view: sorted(
                int(value)
                for value in values.loc[values["analysis_membership"].eq(view), "history_length"].drop_duplicates().tolist()
            )
            for view in REPORTING_VIEWS
        },
    }
    return pd.DataFrame(checks), failures, summary


def compact_utility_table(values: pd.DataFrame, view: str, contrast: str) -> pd.DataFrame:
    utility = values.loc[
        values["analysis_membership"].eq(view) & values["metric_id"].isin(UTILITY_METRICS)
    ].copy()
    pivot = utility.pivot_table(
        index=["dataset", "model", "prompt_strategy", "metric_id"],
        columns="history_length",
        values="value",
        aggfunc="first",
    ).reset_index()
    for history in HISTORY_BY_VIEW[view]:
        if history not in pivot.columns:
            pivot[history] = ""
    rename = {history: f"H{history}" for history in HISTORY_BY_VIEW[view]}
    pivot = pivot.rename(columns=rename)
    if contrast == "H10_minus_H0" and "H0" in pivot and "H10" in pivot:
        pivot["H10_minus_H0"] = pivot.apply(lambda row: fmt(to_float(row["H10"]) - to_float(row["H0"])), axis=1)
    if contrast == "H20_minus_H10" and "H10" in pivot and "H20" in pivot:
        pivot["H20_minus_H10"] = pivot.apply(lambda row: fmt(to_float(row["H20"]) - to_float(row["H10"])), axis=1)
    return pivot.sort_values(["dataset", "model", "prompt_strategy", "metric_id"], kind="mergesort")


def build_report(parts: Mapping[str, Any], output_hashes: Mapping[str, str]) -> str:
    values = parts["tables"]["metric_values"]
    deltas = parts["tables"]["deltas"]
    patterns = parts["tables"]["patterns"]
    prompt_differences = parts["tables"]["prompt_differences"]
    summary = parts["summary"]
    stat_contract = parts["tables"]["statistical_contract"]
    validation = parts["tables"]["validation"]
    utility_primary = compact_utility_table(values, PRIMARY_VIEW, "H10_minus_H0")
    utility_long = compact_utility_table(values, LONG_VIEW, "H20_minus_H10")

    plan_status_counts = stat_contract["planned_status"].value_counts().sort_index().to_dict()
    stat_exec_counts = stat_contract["readout_execution_status"].value_counts().sort_index().to_dict()
    primary_key = deltas.loc[
        deltas["analysis_membership"].eq(PRIMARY_VIEW)
        & deltas["metric_id"].isin(UTILITY_METRICS)
        & deltas["contrast"].eq("H10_minus_H0")
    ]
    long_key = deltas.loc[
        deltas["analysis_membership"].eq(LONG_VIEW)
        & deltas["metric_id"].isin(UTILITY_METRICS)
        & deltas["contrast"].eq("H20_minus_H10")
    ]
    primary_counts = primary_key["direction_class"].value_counts().sort_index().to_dict()
    long_counts = long_key["direction_class"].value_counts().sort_index().to_dict()
    utility_key = deltas.loc[
        deltas["metric_id"].isin(UTILITY_METRICS)
        & deltas["contrast"].isin(["H10_minus_H0", "H20_minus_H10"])
    ].copy()
    model_prompt_counts = (
        utility_key.groupby(["analysis_membership", "model", "prompt_strategy", "contrast", "direction_class"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["improved", "no_change", "worsened"]:
        if column not in model_prompt_counts:
            model_prompt_counts[column] = 0
    pattern_counts = (
        patterns.loc[patterns["metric_id"].isin(UTILITY_METRICS)]
        .groupby(["analysis_membership", "direction_adjusted_pattern"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["single_point", "constant", "monotonic_improving", "monotonic_worsening", "non_monotonic_mixed"]:
        if column not in pattern_counts:
            pattern_counts[column] = 0
    prompt_counts = (
        prompt_differences.loc[prompt_differences["metric_id"].isin(UTILITY_METRICS)]
        .groupby(["analysis_membership", "model", "direction_class"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["improved", "no_change", "worsened"]:
        if column not in prompt_counts:
            prompt_counts[column] = 0

    lines = [
        "# G2 Scientific Readout",
        "",
        f"- Phase: `{PHASE_ID}`",
        f"- Decision: `{parts['decision_token']}`",
        f"- HEAD before readout: `{parts['head_before']}`",
        "- Scope: frozen G2 metric/universe artifacts only; zero Ollama calls; no G3; no protocol changes.",
        "",
        "## Frozen Statistical Contract",
        "",
        f"- Frozen plan rows identified: `{len(stat_contract)}`.",
        f"- Planned-status counts: `{stable_json(plan_status_counts)}`.",
        "- Frozen estimand: `paired_mean_delta` for HR@5, NDCG@5, and MRR within dataset x model x prompt.",
        "- Frozen primary test: `paired_user_bootstrap_mean_delta` with 10000 resamples and seed 20260901.",
        "- Frozen correction: `holm_bonferroni_within_metric_dataset_model_prompt_family`.",
        "- Explicit hypothesis text: `NOT_EXPLICITLY_FROZEN_AS_TEXT`; no new hypotheses are invented here.",
        f"- Statistical execution status: `{stable_json(stat_exec_counts)}`.",
        "- Reason: paired-user metric contributions are not present in the frozen cell-level metric readout artifacts, and this phase is restricted to frozen metrics only.",
        "",
        "## Descriptive Coverage",
        "",
        f"- Metric value rows: `{summary['readout_value_rows']}`.",
        f"- Delta rows: `{summary['delta_rows']}`.",
        f"- Pattern rows: `{summary['pattern_rows']}`.",
        f"- Validation: `{int(validation['status'].eq('PASS').sum())}` / `{len(validation)}` checks PASS.",
        f"- History levels: `{stable_json(summary['history_levels_by_membership'])}`.",
        "",
        "## Utility Direction Counts",
        "",
        f"- Primary B1 H10-H0 utility direction counts: `{stable_json(primary_counts)}`.",
        f"- Long B2 H20-H10 utility direction counts: `{stable_json(long_counts)}`.",
        "- These are descriptive dataset-metric cell counts only; they are not significance tests.",
        "",
        "## Model And Prompt Interaction Counts",
        "",
        "| view | model | prompt | contrast | improved | no change | worsened |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in model_prompt_counts.to_dict("records"):
        if row["analysis_membership"] == TRUE_ZERO_VIEW:
            continue
        lines.append(
            "| {analysis_membership} | {model} | {prompt_strategy} | {contrast} | {improved} | {no_change} | {worsened} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Utility Pattern Counts",
            "",
            "| view | single point | constant | monotonic improving | monotonic worsening | non-monotonic mixed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pattern_counts.to_dict("records"):
        lines.append(
            "| {analysis_membership} | {single_point} | {constant} | {monotonic_improving} | {monotonic_worsening} | {non_monotonic_mixed} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## BASE vs LLMRANK Utility Counts",
            "",
            "| view | model | LLMRANK better | no change | BASE better |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in prompt_counts.to_dict("records"):
        lines.append(
            "| {analysis_membership} | {model} | {improved} | {no_change} | {worsened} |".format(
                **row
            )
        )
    lines.extend(
        [
        "",
        "## Primary B1 Utility Snapshot",
        "",
        "| dataset | model | prompt | metric | H0 | H1 | H2 | H5 | H10 | H10-H0 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in utility_primary.head(90).to_dict("records"):
        lines.append(
            "| {dataset} | {model} | {prompt_strategy} | {metric_id} | {H0} | {H1} | {H2} | {H5} | {H10} | {H10_minus_H0} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Long-History Utility Snapshot",
            "",
            "| dataset | model | prompt | metric | H10 | H20 | H20-H10 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in utility_long.head(90).to_dict("records"):
        lines.append(
            "| {dataset} | {model} | {prompt_strategy} | {metric_id} | {H10} | {H20} | {H20_minus_H10} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Denominator Notes",
            "",
            "- Utility values use planned/applicable users; invalid terminal responses count as zero.",
            "- Beyond-utility values use strict-valid users.",
            "- Novelty values use the frozen no-smoothing defined subset where every top-5 item has positive train support; defined-subset rows are listed in `g2_history_length_metric_undefined.csv` and summarized in the readout tables.",
            "- Blocked-by-contract metrics remain NA: `popularity_bias_at_5`, `intra_list_diversity_at_5`.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for rel_path, digest in sorted(output_hashes.items()):
        lines.append(f"- `{rel_path}` SHA256 `{digest}`")
    if parts["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in parts["failures"])
    lines.append("")
    return "\n".join(lines)


def build_readout(root: Path, *, allow_generator_dirty: bool) -> dict[str, Any]:
    head = git_output(root, ["rev-parse", "HEAD"])
    status = git_output(root, ["status", "--short", "--untracked-files=all"], check=False)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True)

    metric_manifest = read_json(root, PATHS["metric_manifest"])
    analysis_manifest = read_json(root, PATHS["analysis_manifest"])
    metrics = read_csv(root, PATHS["metric_results"])
    denominators = read_csv(root, PATHS["metric_denominators"])
    undefined = read_csv(root, PATHS["metric_undefined"])
    metric_validation = read_csv(root, PATHS["metric_validation"])
    cell_counts = read_csv(root, PATHS["cell_counts"])
    statistical_plan = read_csv(root, PATHS["statistical_plan"])
    denominator_contract = read_csv(root, PATHS["denominator_contract"])

    values = value_table(metrics)
    statistical_contract = build_statistical_contract(statistical_plan)
    deltas = build_deltas(values, statistical_contract)
    patterns = build_patterns(values, deltas)
    model_prompt_summary = build_model_prompt_summary(deltas)
    prompt_differences = build_prompt_differences(values)
    validation, failures, summary = validate_inputs(
        root,
        head=head,
        status=status,
        allow_generator_dirty=allow_generator_dirty,
        metric_manifest=metric_manifest,
        metric_validation=metric_validation,
        metrics=metrics,
        values=values,
        deltas=deltas,
        patterns=patterns,
    )
    if diff_check.returncode != 0:
        failures.append("git_diff_check_failed")
        validation = pd.concat(
            [
                validation,
                pd.DataFrame(
                    [
                        {
                            "check_id": stable_id("G2READ-CHK", "git_diff_check_clean"),
                            "check_name": "git_diff_check_clean",
                            "status": "FAIL",
                            "expected": "true",
                            "actual": (diff_check.stdout + diff_check.stderr).strip(),
                            "notes": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        validation = pd.concat(
            [
                validation,
                pd.DataFrame(
                    [
                        {
                            "check_id": stable_id("G2READ-CHK", "git_diff_check_clean"),
                            "check_name": "git_diff_check_clean",
                            "status": "PASS",
                            "expected": "true",
                            "actual": "true",
                            "notes": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    tables = {
        "metric_values": values,
        "deltas": deltas,
        "patterns": patterns,
        "model_prompt_summary": model_prompt_summary,
        "prompt_differences": prompt_differences,
        "statistical_contract": statistical_contract,
        "validation": validation,
    }
    input_hashes = {
        rel.as_posix(): sha256_file(root / rel)
        for rel in [
            PATHS["metric_manifest"],
            PATHS["analysis_manifest"],
            PATHS["metric_results"],
            PATHS["metric_denominators"],
            PATHS["metric_undefined"],
            PATHS["metric_validation"],
            PATHS["cell_counts"],
            PATHS["statistical_plan"],
            PATHS["denominator_contract"],
        ]
    }
    summary.update(
        {
            "statistical_plan_rows": int(len(statistical_contract)),
            "statistical_execution_status_counts": statistical_contract["readout_execution_status"]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict(),
            "blocked_metric_rows": int(undefined["status"].eq("BLOCKED_BY_FROZEN_CONTRACT").sum()),
            "novelty_defined_subset_rows": int(undefined["status"].eq("DEFINED_SUBSET_USED").sum()),
            "model_prompt_summary_rows": int(len(model_prompt_summary)),
            "prompt_difference_rows": int(len(prompt_differences)),
            "metric_manifest_decision": metric_manifest.get("decision_token"),
            "analysis_manifest_decision": analysis_manifest.get("decision_token"),
            "denominator_contract_rows": int(len(denominator_contract)),
            "cell_count_rows": int(len(cell_counts)),
        }
    )
    return {
        "decision_token": DECISION_COMPLETE if not failures else DECISION_BLOCKED,
        "failures": failures,
        "head_before": head,
        "status_before": status,
        "tables": tables,
        "summary": summary,
        "input_hashes": input_hashes,
    }


def append_notes(root: Path, parts: Mapping[str, Any]) -> None:
    text = (
        f"\n## {utc_now()} - G2 scientific readout\n\n"
        f"- Decision: `{parts['decision_token']}`.\n"
        "- Scope: descriptive readout from frozen G2 metric/universe artifacts only; no Ollama calls, no G3, no protocol changes.\n"
        f"- Metric values `{parts['summary']['readout_value_rows']}`; deltas `{parts['summary']['delta_rows']}`; patterns `{parts['summary']['pattern_rows']}`.\n"
        f"- Frozen statistical plan rows `{parts['summary']['statistical_plan_rows']}` identified but not executed from cell-level frozen metric tables.\n"
        "- Outputs: `results/tables/g2_history_length_readout_*.csv`, `results/manifests/g2_history_length_scientific_readout_manifest.json`, `report/g2_history_length_scientific_readout.md`.\n"
        "- Next safe step: decide whether to freeze paired-user metric contributions/statistical execution, or move to report drafting from descriptive tables; do not start G3 yet.\n"
    )
    with (root / PATHS["notes"]).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_outputs(root: Path, parts: dict[str, Any]) -> dict[str, str]:
    for name, frame in parts["tables"].items():
        write_csv(root, OUTPUTS[name], frame)
    output_hashes = {
        OUTPUTS[name].as_posix(): sha256_file(root / OUTPUTS[name])
        for name in [
            "metric_values",
            "deltas",
            "patterns",
            "model_prompt_summary",
            "prompt_differences",
            "statistical_contract",
            "validation",
        ]
    }
    report = build_report(parts, output_hashes)
    (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUTS["report"]).write_text(report, encoding="utf-8", newline="\n")
    output_hashes[OUTPUTS["report"].as_posix()] = sha256_file(root / OUTPUTS["report"])
    manifest = {
        "phase_id": PHASE_ID,
        "decision_token": parts["decision_token"],
        "created_at_utc": utc_now(),
        "precondition_head": EXPECTED_HEAD,
        "head_before_generation": parts["head_before"],
        "zero_inference": True,
        "ollama_calls_executed": 0,
        "g3_started": False,
        "protocol_changes": False,
        "frozen_metric_inputs_only": True,
        "statistical_tests_executed": False,
        "statistical_execution_status": "NOT_EXECUTED_FROM_FROZEN_CELL_METRIC_TABLES",
        "failures": parts["failures"],
        "input_hashes": parts["input_hashes"],
        "output_hashes": output_hashes,
        "summary": parts["summary"],
    }
    (root / OUTPUTS["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUTS["manifest"]).write_text(stable_json(manifest) + "\n", encoding="utf-8")
    output_hashes[OUTPUTS["manifest"].as_posix()] = sha256_file(root / OUTPUTS["manifest"])
    append_notes(root, parts)
    return output_hashes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    parts = build_readout(root, allow_generator_dirty=args.allow_generator_dirty)
    if args.no_write:
        print(stable_json({"decision_token": parts["decision_token"], "failures": parts["failures"], "summary": parts["summary"]}))
        return 0 if not parts["failures"] else 1
    output_hashes = write_outputs(root, parts)
    print(stable_json({"decision_token": parts["decision_token"], "failures": parts["failures"], "output_hashes": output_hashes, "summary": parts["summary"]}))
    return 0 if not parts["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

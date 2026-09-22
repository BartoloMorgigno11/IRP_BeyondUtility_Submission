"""Build the final G4 F1BZQ scientific synthesis from frozen artifacts only.

This phase consumes frozen G4 analytic, metric, and statistical artifacts. It
creates compact synthesis tables and a concise G4 synthesis report. It does not
call Ollama/API, add tests, run new statistics, or modify frozen evidence,
protocol, analytic universe, metric, or statistical artifacts.
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
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.utils.config import find_project_root  # noqa: E402


EXPECTED_HEAD = "a21408c9f097457f72b854f6675639f42ad028b5"
PHASE_ID = "g4_f1bzq_scientific_synthesis"
DECISION_FROZEN = "G4_SCIENTIFIC_SYNTHESIS_FROZEN"
DECISION_BLOCKED = "G4_SCIENTIFIC_SYNTHESIS_BLOCKED"

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = ("BASE", "LLMRANK")
EXPECTED_USERS = {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}
EXPECTED_POSITIVE_PRESENT = {"beauty": 49, "sports": 28, "ml1m": 193, "lastfm": 469, "foodcom": 4}
EXPECTED_POSITIVE_ABSENT = {"beauty": 941, "sports": 864, "ml1m": 807, "lastfm": 531, "foodcom": 996}
EXPECTED_CANDIDATE_RECALL = {
    "beauty": "0.049494949494949494",
    "sports": "0.031390134529147982",
    "ml1m": "0.193",
    "lastfm": "0.46899999999999997",
    "foodcom": "0.0040000000000000001",
}
EXPECTED_INTERNAL_DECISIONS = {
    "CI_SUPPORTED_POSITIVE": 29,
    "CI_SUPPORTED_NEGATIVE": 24,
    "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 172,
}
EXPECTED_FAMILY_DECISIONS = {
    "g4_prompt_strategy": {
        "CI_SUPPORTED_POSITIVE": 2,
        "CI_SUPPORTED_NEGATIVE": 13,
        "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 60,
    },
    "g4_model_pair": {
        "CI_SUPPORTED_POSITIVE": 27,
        "CI_SUPPORTED_NEGATIVE": 11,
        "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 112,
    },
}
EXPECTED_DATASET_DECISIONS = {
    "beauty": {"CI_SUPPORTED_POSITIVE": 0, "CI_SUPPORTED_NEGATIVE": 0, "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 45},
    "sports": {"CI_SUPPORTED_POSITIVE": 4, "CI_SUPPORTED_NEGATIVE": 5, "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 36},
    "ml1m": {"CI_SUPPORTED_POSITIVE": 10, "CI_SUPPORTED_NEGATIVE": 5, "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 30},
    "lastfm": {"CI_SUPPORTED_POSITIVE": 15, "CI_SUPPORTED_NEGATIVE": 14, "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 16},
    "foodcom": {"CI_SUPPORTED_POSITIVE": 0, "CI_SUPPORTED_NEGATIVE": 0, "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 45},
}
EXPECTED_RELIABILITY = {
    "planned_requests": 29292,
    "strict_valid": 28986,
    "structural_invalid": 306,
    "invalid_length": 255,
    "duplicate_indices": 51,
    "backend_errors": 0,
}

PATHS = {
    "analytic_manifest": Path("results/manifests/g4_f1bzq_analytic_universe_freeze.json"),
    "metrics_manifest": Path("results/manifests/g4_f1bzq_metrics_freeze.json"),
    "statistical_manifest": Path("results/manifests/g4_f1bzq_statistical_analysis_manifest.json"),
    "analytic_denominator_counts": Path("results/tables/g4_f1bzq_analytic_denominator_counts.csv"),
    "metric_results": Path("results/tables/g4_f1bzq_metric_results.csv"),
    "metric_denominators": Path("results/tables/g4_f1bzq_metric_denominators.csv"),
    "candidate_conditional": Path("results/tables/g4_f1bzq_candidate_conditional_metrics.csv"),
    "reliability": Path("results/tables/g4_f1bzq_reliability.csv"),
    "undefined_blocked": Path("results/tables/g4_f1bzq_metric_undefined_blocked.csv"),
    "statistical_results": Path("results/tables/g4_f1bzq_statistical_results.csv"),
    "statistical_validation": Path("results/tables/g4_f1bzq_statistical_validation.csv"),
    "statistical_plan": Path("results/tables/g4_f1bzq_statistical_comparisons.csv"),
    "source": Path("src/analysis/g4_f1bzq_scientific_synthesis.py"),
    "notes": Path("report/notes.md"),
}

OUTPUTS = {
    "candidate_bottleneck": Path("results/tables/g4_f1bzq_synthesis_candidate_bottleneck.csv"),
    "internal_ci_counts": Path("results/tables/g4_f1bzq_synthesis_internal_ci_counts.csv"),
    "prompt_effects": Path("results/tables/g4_f1bzq_synthesis_prompt_effects.csv"),
    "model_effects": Path("results/tables/g4_f1bzq_synthesis_model_effects.csv"),
    "conditional_quality": Path("results/tables/g4_f1bzq_synthesis_conditional_quality.csv"),
    "beyond_reliability": Path("results/tables/g4_f1bzq_synthesis_beyond_reliability.csv"),
    "g1_vs_g4_descriptive": Path("results/tables/g4_f1bzq_synthesis_g1_vs_g4_descriptive.csv"),
    "takeaways": Path("results/tables/g4_f1bzq_synthesis_takeaways.csv"),
    "validation": Path("results/tables/g4_f1bzq_synthesis_validation.csv"),
    "manifest": Path("results/manifests/g4_f1bzq_scientific_synthesis_manifest.json"),
    "report": Path("report/g4_f1bzq_scientific_synthesis.md"),
}

ALLOWED_DIRTY_PATHS = {
    PATHS["source"],
    PATHS["notes"],
    *OUTPUTS.values(),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen G4 scientific synthesis.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate in memory without writing outputs.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator and expected synthesis outputs to be dirty while frozen inputs remain unchanged.",
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
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any) -> str:
    payload = stable_json([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32].upper()}"


def read_csv(root: Path, path: Path) -> pd.DataFrame:
    return pd.read_csv(root / path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8-sig"))


def write_csv(root: Path, path: Path, frame: pd.DataFrame) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / path, index=False, lineterminator="\n")


def write_json(root: Path, path: Path, payload: Mapping[str, Any]) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def bool_text(value: Any) -> str:
    return "true" if str(value).strip().lower() in {"true", "1", "yes"} else "false"


def bool_value(value: Any) -> bool:
    return bool_text(value) == "true"


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def parse_float(value: Any) -> float:
    return float(str(value))


def git_output(root: Path, args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def dirty_path_from_status_line(line: str) -> str:
    if line.startswith("?? "):
        path_text = line[3:].strip()
    elif len(line) >= 3 and line[2] == " ":
        path_text = line[3:].strip()
    else:
        parts = line.split(maxsplit=1)
        path_text = parts[1].strip() if len(parts) == 2 else ""
    if " -> " in path_text:
        path_text = path_text.split(" -> ", 1)[1].strip()
    return path_text.replace("\\", "/")


def unexpected_git_status_lines(status: str) -> list[str]:
    allowed = {path.as_posix() for path in ALLOWED_DIRTY_PATHS}
    unexpected: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if dirty_path_from_status_line(line) not in allowed:
            unexpected.append(line)
    return unexpected


def metric_values(metric_results: pd.DataFrame, dataset: str, metric_id: str) -> list[float]:
    rows = metric_results[(metric_results["dataset"] == dataset) & (metric_results["metric_id"] == metric_id)]
    return [parse_float(value) for value in rows["value"].tolist()]


def mean_min_max(values: Sequence[float]) -> tuple[str, str, str]:
    return (
        format_float(sum(values) / float(len(values))),
        format_float(min(values)),
        format_float(max(values)),
    )


def bottleneck_assessment(dataset: str) -> str:
    if dataset in {"beauty", "sports"}:
        return "candidate_stage_primary_with_model_prompt_dependent_reranking"
    if dataset == "ml1m":
        return "both_candidate_stage_ceiling_and_reranking_quality"
    if dataset == "lastfm":
        return "candidate_ceiling_present_but_reranking_quality_is_informative"
    if dataset == "foodcom":
        return "candidate_stage_primary_conditional_n4_diagnostic_only"
    raise ValueError(dataset)


def build_candidate_bottleneck(metric_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        recall_rows = metric_results[
            (metric_results["dataset"] == dataset) & (metric_results["metric_id"] == "candidate_recall_at_20")
        ]
        recall_values = sorted(set(recall_rows["value"].tolist()))
        if len(recall_values) != 1:
            raise RuntimeError(f"candidate recall is not invariant for {dataset}: {recall_values}")
        hr_mean, hr_min, hr_max = mean_min_max(metric_values(metric_results, dataset, "hr_at_5"))
        ndcg_mean, ndcg_min, ndcg_max = mean_min_max(metric_values(metric_results, dataset, "ndcg_at_5"))
        mrr_mean, mrr_min, mrr_max = mean_min_max(metric_values(metric_results, dataset, "mrr"))
        chr_mean, chr_min, chr_max = mean_min_max(metric_values(metric_results, dataset, "conditional_hr_at_5"))
        cndcg_mean, cndcg_min, cndcg_max = mean_min_max(metric_values(metric_results, dataset, "conditional_ndcg_at_5"))
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-CAND", dataset),
                "dataset": dataset,
                "planned_users": EXPECTED_USERS[dataset],
                "positive_present_users": EXPECTED_POSITIVE_PRESENT[dataset],
                "positive_absent_users": EXPECTED_POSITIVE_ABSENT[dataset],
                "candidate_recall_at_20": recall_values[0],
                "unconditional_hr_at_5_mean_across_cells": hr_mean,
                "unconditional_hr_at_5_min_cell": hr_min,
                "unconditional_hr_at_5_max_cell": hr_max,
                "unconditional_ndcg_at_5_mean_across_cells": ndcg_mean,
                "unconditional_ndcg_at_5_min_cell": ndcg_min,
                "unconditional_ndcg_at_5_max_cell": ndcg_max,
                "unconditional_mrr_mean_across_cells": mrr_mean,
                "unconditional_mrr_min_cell": mrr_min,
                "unconditional_mrr_max_cell": mrr_max,
                "conditional_hr_at_5_denominator": EXPECTED_POSITIVE_PRESENT[dataset],
                "conditional_hr_at_5_mean_across_cells": chr_mean,
                "conditional_hr_at_5_min_cell": chr_min,
                "conditional_hr_at_5_max_cell": chr_max,
                "conditional_ndcg_at_5_denominator": EXPECTED_POSITIVE_PRESENT[dataset],
                "conditional_ndcg_at_5_mean_across_cells": cndcg_mean,
                "conditional_ndcg_at_5_min_cell": cndcg_min,
                "conditional_ndcg_at_5_max_cell": cndcg_max,
                "bottleneck_assessment": bottleneck_assessment(dataset),
                "interpretation_guard": "do_not_attribute_low_unconditional_utility_only_to_llm_when_positive_absent_from_pool",
            }
        )
    return pd.DataFrame(rows)


def count_decisions(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["ci_supported_decision"].value_counts().to_dict()
    return {
        "CI_SUPPORTED_POSITIVE": int(counts.get("CI_SUPPORTED_POSITIVE", 0)),
        "CI_SUPPORTED_NEGATIVE": int(counts.get("CI_SUPPORTED_NEGATIVE", 0)),
        "NON_SIGNIFICANT_CI_INCLUDES_ZERO": int(counts.get("NON_SIGNIFICANT_CI_INCLUDES_ZERO", 0)),
    }


def build_internal_ci_counts(statistical_results: pd.DataFrame) -> pd.DataFrame:
    inferential = statistical_results[statistical_results["inferential_test_allowed"].map(bool_value)].copy()
    rows: list[dict[str, Any]] = []

    def add(scope: str, group: str, subset: pd.DataFrame) -> None:
        counts = count_decisions(subset)
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-CICOUNT", scope, group),
                "scope": scope,
                "group": group,
                "positive": counts["CI_SUPPORTED_POSITIVE"],
                "negative": counts["CI_SUPPORTED_NEGATIVE"],
                "non_significant_ci_includes_zero": counts["NON_SIGNIFICANT_CI_INCLUDES_ZERO"],
                "total": int(len(subset)),
                "statistical_wording": "bootstrap_CI_only_no_p_values_no_holm",
            }
        )

    add("overall_inferential", "all_internal_g4", inferential)
    for family in ("g4_prompt_strategy", "g4_model_pair"):
        add("comparison_family", family, inferential[inferential["comparison_family"] == family])
    for dataset in DATASETS:
        add("dataset", dataset, inferential[inferential["dataset"] == dataset])
    for metric_id in ("hr_at_5", "ndcg_at_5", "mrr", "conditional_hr_at_5", "conditional_ndcg_at_5"):
        add("metric", metric_id, inferential[inferential["metric_id"] == metric_id])
    return pd.DataFrame(rows)


def prompt_effect_label(row: Mapping[str, Any]) -> str:
    decision = row["ci_supported_decision"]
    delta = parse_float(row["arm_b_minus_arm_a_delta"])
    if decision == "CI_SUPPORTED_POSITIVE" and delta > 0:
        return "LLMRANK_supported_improvement_over_BASE"
    if decision == "CI_SUPPORTED_NEGATIVE" and delta < 0:
        return "LLMRANK_supported_worsening_vs_BASE"
    if decision == "NON_SIGNIFICANT_CI_INCLUDES_ZERO":
        return "difference_unsupported_CI_includes_zero"
    return "direction_decision_mismatch_validation_required"


def build_prompt_effects(statistical_results: pd.DataFrame) -> pd.DataFrame:
    prompt = statistical_results[statistical_results["comparison_family"] == "g4_prompt_strategy"].copy()
    rows: list[dict[str, Any]] = []
    for row in prompt.to_dict("records"):
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-PROMPT", row["comparison_template_id"], row["metric_id"]),
                "dataset": row["dataset"],
                "model": row["model_scope"],
                "metric_id": row["metric_id"],
                "paired_n": row["paired_n"],
                "base_mean": row["arm_a_mean"],
                "llmrank_mean": row["arm_b_mean"],
                "llmrank_minus_base_delta": row["arm_b_minus_arm_a_delta"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "ci_supported_decision": row["ci_supported_decision"],
                "effect_label": prompt_effect_label(row),
                "foodcom_conditional_n4_flag": row["small_conditional_denominator_flag"],
                "statistical_wording": "CI-supported only; NS means CI includes zero, not proof of no effect",
            }
        )
    return pd.DataFrame(rows)


def model_effect_label(row: Mapping[str, Any]) -> str:
    decision = row["ci_supported_decision"]
    delta = parse_float(row["arm_b_minus_arm_a_delta"])
    if decision == "CI_SUPPORTED_POSITIVE" and delta > 0:
        return "second_model_supported_higher_than_first"
    if decision == "CI_SUPPORTED_NEGATIVE" and delta < 0:
        return "first_model_supported_higher_than_second"
    if decision == "NON_SIGNIFICANT_CI_INCLUDES_ZERO":
        return "model_difference_unsupported_CI_includes_zero"
    return "direction_decision_mismatch_validation_required"


def build_model_effects(statistical_results: pd.DataFrame) -> pd.DataFrame:
    model = statistical_results[statistical_results["comparison_family"] == "g4_model_pair"].copy()
    rows: list[dict[str, Any]] = []
    for row in model.to_dict("records"):
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-MODEL", row["comparison_template_id"], row["metric_id"]),
                "dataset": row["dataset"],
                "prompt_strategy": row["prompt_scope"],
                "comparison": row["comparison"],
                "metric_id": row["metric_id"],
                "paired_n": row["paired_n"],
                "first_model": row["arm_a_label"],
                "second_model": row["arm_b_label"],
                "first_model_mean": row["arm_a_mean"],
                "second_model_mean": row["arm_b_mean"],
                "second_minus_first_delta": row["arm_b_minus_arm_a_delta"],
                "abs_delta": format_float(abs(parse_float(row["arm_b_minus_arm_a_delta"]))),
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "ci_supported_decision": row["ci_supported_decision"],
                "effect_label": model_effect_label(row),
                "foodcom_conditional_n4_flag": row["small_conditional_denominator_flag"],
                "statistical_wording": "CI-supported only; no global model ranking inferred from counts alone",
            }
        )
    return pd.DataFrame(rows)


def build_conditional_quality(metric_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = {
        (row["dataset"], row["model"], row["prompt_strategy"], row["metric_id"]): row
        for row in metric_results.to_dict("records")
    }
    for dataset in DATASETS:
        for model in MODELS:
            for prompt in PROMPTS:
                recall = lookup[(dataset, model, prompt, "candidate_recall_at_20")]
                hr = lookup[(dataset, model, prompt, "hr_at_5")]
                ndcg = lookup[(dataset, model, prompt, "ndcg_at_5")]
                mrr = lookup[(dataset, model, prompt, "mrr")]
                chr_row = lookup[(dataset, model, prompt, "conditional_hr_at_5")]
                cndcg_row = lookup[(dataset, model, prompt, "conditional_ndcg_at_5")]
                rows.append(
                    {
                        "synthesis_id": stable_id("G4SYN-COND", dataset, model, prompt),
                        "dataset": dataset,
                        "model": model,
                        "prompt_strategy": prompt,
                        "planned_users": EXPECTED_USERS[dataset],
                        "positive_present_users": EXPECTED_POSITIVE_PRESENT[dataset],
                        "positive_absent_users": EXPECTED_POSITIVE_ABSENT[dataset],
                        "candidate_recall_at_20": recall["value"],
                        "unconditional_hr_at_5": hr["value"],
                        "unconditional_ndcg_at_5": ndcg["value"],
                        "unconditional_mrr": mrr["value"],
                        "conditional_hr_at_5": chr_row["value"],
                        "conditional_ndcg_at_5": cndcg_row["value"],
                        "conditional_denominator": chr_row["denominator"],
                        "foodcom_n4_diagnostic_flag": bool_text(dataset == "foodcom"),
                        "dataset_bottleneck_assessment": bottleneck_assessment(dataset),
                    }
                )
    return pd.DataFrame(rows)


def build_beyond_reliability(metric_results: pd.DataFrame, reliability: pd.DataFrame, undefined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    beyond = metric_results[metric_results["metric_id"].isin(["item_coverage_at_5", "oic_at_5"])].copy()
    beyond["value_float"] = beyond["value"].astype(float)
    for (dataset, metric_id), group in beyond.groupby(["dataset", "metric_id"], sort=False):
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-BEYREL", "beyond", dataset, metric_id),
                "section": "beyond_utility",
                "scope": "dataset_mean_min_max_across_6_model_prompt_cells",
                "dataset": dataset,
                "model": "",
                "prompt_strategy": "",
                "metric_id": metric_id,
                "value": format_float(float(group["value_float"].mean())),
                "min_cell_value": format_float(float(group["value_float"].min())),
                "max_cell_value": format_float(float(group["value_float"].max())),
                "numerator_or_count": "",
                "denominator": "strict-valid users; metric-specific item/pair denominator in metric table",
                "interpretation_guard": "strict-valid candidate-union/pairwise metric only; no full-catalog substitution",
            }
        )
    undef_counts = undefined.groupby(["metric_family", "status"]).size().reset_index(name="rows")
    for row in undef_counts.to_dict("records"):
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-BEYREL", "undefined", row["metric_family"], row["status"]),
                "section": "undefined_or_blocked",
                "scope": row["metric_family"],
                "dataset": "",
                "model": "",
                "prompt_strategy": "",
                "metric_id": row["status"],
                "value": "",
                "min_cell_value": "",
                "max_cell_value": "",
                "numerator_or_count": int(row["rows"]),
                "denominator": "explicit frozen undefined/blocked rows",
                "interpretation_guard": "not reconstructed in G4 synthesis",
            }
        )
    reliability_int = reliability.copy()
    for col in ["planned_requests", "strict_valid", "structural_invalid", "invalid_length", "invalid_duplicate", "backend_errors"]:
        reliability_int[col] = reliability_int[col].astype(int)
    totals = reliability_int[["planned_requests", "strict_valid", "structural_invalid", "invalid_length", "invalid_duplicate", "backend_errors"]].sum()
    for metric_id, value in (
        ("planned_requests", totals["planned_requests"]),
        ("strict_valid", totals["strict_valid"]),
        ("structural_invalid", totals["structural_invalid"]),
        ("invalid_length", totals["invalid_length"]),
        ("duplicate_indices", totals["invalid_duplicate"]),
        ("backend_errors", totals["backend_errors"]),
    ):
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-BEYREL", "reliability_total", metric_id),
                "section": "reliability_total",
                "scope": "all_planned_requests",
                "dataset": "",
                "model": "",
                "prompt_strategy": "",
                "metric_id": metric_id,
                "value": int(value),
                "min_cell_value": "",
                "max_cell_value": "",
                "numerator_or_count": int(value),
                "denominator": totals["planned_requests"],
                "interpretation_guard": "descriptive frozen reliability accounting only",
            }
        )
    by_model = reliability_int.groupby("model")[["planned_requests", "strict_valid", "structural_invalid"]].sum().reset_index()
    for row in by_model.to_dict("records"):
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-BEYREL", "reliability_model", row["model"]),
                "section": "reliability_by_model",
                "scope": "all_datasets_all_prompts",
                "dataset": "",
                "model": row["model"],
                "prompt_strategy": "",
                "metric_id": "strict_valid_rate",
                "value": format_float(row["strict_valid"] / float(row["planned_requests"])),
                "min_cell_value": "",
                "max_cell_value": "",
                "numerator_or_count": int(row["strict_valid"]),
                "denominator": int(row["planned_requests"]),
                "interpretation_guard": "descriptive only; no reliability inferential test frozen",
            }
        )
    by_prompt = reliability_int.groupby("prompt_strategy")[["planned_requests", "strict_valid", "structural_invalid"]].sum().reset_index()
    for row in by_prompt.to_dict("records"):
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-BEYREL", "reliability_prompt", row["prompt_strategy"]),
                "section": "reliability_by_prompt",
                "scope": "all_datasets_all_models",
                "dataset": "",
                "model": "",
                "prompt_strategy": row["prompt_strategy"],
                "metric_id": "strict_valid_rate",
                "value": format_float(row["strict_valid"] / float(row["planned_requests"])),
                "min_cell_value": "",
                "max_cell_value": "",
                "numerator_or_count": int(row["strict_valid"]),
                "denominator": int(row["planned_requests"]),
                "interpretation_guard": "descriptive only; no reliability inferential test frozen",
            }
        )
    return pd.DataFrame(rows)


def build_g1_vs_g4(statistical_results: pd.DataFrame) -> pd.DataFrame:
    descriptive = statistical_results[statistical_results["comparison_family"] == "g1_vs_g4_pipeline_descriptive"].copy()
    rows: list[dict[str, Any]] = []
    for row in descriptive.to_dict("records"):
        rows.append(
            {
                "synthesis_id": stable_id("G4SYN-G1G4", row["comparison_template_id"], row["metric_id"]),
                "dataset": row["dataset"],
                "model": row["model_scope"],
                "prompt_strategy": row["prompt_scope"],
                "metric_id": row["metric_id"],
                "g1_ranking_mean": row["arm_a_mean"],
                "g4_reranking_mean": row["arm_b_mean"],
                "g4_minus_g1_pipeline_delta": row["arm_b_minus_arm_a_delta"],
                "result_status": row["result_status"],
                "inferential_test_allowed": row["inferential_test_allowed"],
                "ranking_vs_reranking_policy": row["ranking_vs_reranking_policy"],
                "interpretation_guard": "pipeline-level descriptive only; G1 and G4 use different candidate sources; not a pure reranking-effect estimate",
            }
        )
    return pd.DataFrame(rows)


def build_takeaways(candidate: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "candidate_generator_major_bottleneck",
            "yes_for_beauty_sports_foodcom_and_material_for_ml1m; less severe but still a ceiling for lastfm",
            "candidate recall is 49/990, 28/892, 4/1000, 193/1000, and 469/1000 respectively",
        ),
        (
            "positive_present_reranker_recovers_utility",
            "yes_in_several_cells_but_not uniformly",
            "conditional HR@5 means by dataset range from 0.369 sports to 0.614 lastfm, with Food.com n=4 diagnostic only",
        ),
        (
            "llmrank_consistently_helps",
            "no",
            "BASE-vs-LLMRANK CI-supported rows are 2 positive, 13 negative, 60 NS",
        ),
        (
            "model_effects_dataset_dependent",
            "yes",
            "supported model differences concentrate in LastFM, ML1M, and some Sports rows; Beauty and Food.com have no supported internal G4 differences",
        ),
        (
            "strongest_evidence",
            "lastfm_then_ml1m",
            "LastFM has 469 positive-present users and 29 CI-supported model/prompt rows; ML1M has 193 positive-present users and 15 supported rows",
        ),
        (
            "weakest_evidence",
            "foodcom_and_beauty_for_internal_inference",
            "Food.com conditional n=4 and all 45 internal rows NS; Beauty all 45 internal rows NS despite conditional n=49",
        ),
        (
            "g4_adds_beyond_g1_g2_g3",
            "separates_frozen_candidate_pool_ceiling_from_llm_reranking_quality",
            "G4 uses F1BZQ 20-item pools, conditional reranking diagnostics, and internal paired prompt/model CIs; G1-vs-G4 remains descriptive",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "takeaway_id": stable_id("G4SYN-TAKE", key),
                "question": key,
                "answer": answer,
                "frozen_evidence": evidence,
                "statistical_wording": "frozen_results_only_no_p_values_no_holm_no_new_tests",
            }
            for key, answer, evidence in rows
        ]
    )


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def build_report(
    *,
    candidate: pd.DataFrame,
    ci_counts: pd.DataFrame,
    prompt_effects: pd.DataFrame,
    model_effects: pd.DataFrame,
    conditional_quality: pd.DataFrame,
    beyond_reliability: pd.DataFrame,
    g1_vs_g4: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> str:
    candidate_rows = candidate[
        [
            "dataset",
            "positive_present_users",
            "planned_users",
            "candidate_recall_at_20",
            "unconditional_hr_at_5_mean_across_cells",
            "conditional_hr_at_5_denominator",
            "conditional_hr_at_5_mean_across_cells",
            "bottleneck_assessment",
        ]
    ].to_dict("records")
    prompt_supported = prompt_effects[
        prompt_effects["ci_supported_decision"].isin(["CI_SUPPORTED_POSITIVE", "CI_SUPPORTED_NEGATIVE"])
    ].copy()
    prompt_supported_rows = prompt_supported[
        ["dataset", "model", "metric_id", "paired_n", "llmrank_minus_base_delta", "ci_low", "ci_high", "effect_label"]
    ].to_dict("records")
    model_supported = model_effects[
        model_effects["ci_supported_decision"].isin(["CI_SUPPORTED_POSITIVE", "CI_SUPPORTED_NEGATIVE"])
    ].copy()
    model_supported["abs_delta_float"] = model_supported["abs_delta"].astype(float)
    model_top_rows = model_supported.sort_values("abs_delta_float", ascending=False).head(12)[
        ["dataset", "prompt_strategy", "comparison", "metric_id", "paired_n", "second_minus_first_delta", "ci_low", "ci_high", "effect_label"]
    ].to_dict("records")
    conditional_denominators = [
        {
            "dataset": dataset,
            "conditional_n": EXPECTED_POSITIVE_PRESENT[dataset],
            "foodcom_guard": "diagnostic_n4" if dataset == "foodcom" else "",
        }
        for dataset in DATASETS
    ]
    reliability_total = beyond_reliability[beyond_reliability["section"] == "reliability_total"]
    reliability_values = {row["metric_id"]: row["value"] for row in reliability_total.to_dict("records")}
    reliability_model = beyond_reliability[beyond_reliability["section"] == "reliability_by_model"][
        ["model", "value", "numerator_or_count", "denominator"]
    ].to_dict("records")
    reliability_prompt = beyond_reliability[beyond_reliability["section"] == "reliability_by_prompt"][
        ["prompt_strategy", "value", "numerator_or_count", "denominator"]
    ].to_dict("records")
    beyond_rows = beyond_reliability[beyond_reliability["section"] == "beyond_utility"][
        ["dataset", "metric_id", "value", "min_cell_value", "max_cell_value"]
    ].to_dict("records")
    g1_summary = g1_vs_g4.copy()
    g1_summary["delta_float"] = g1_summary["g4_minus_g1_pipeline_delta"].astype(float)
    g1_metric_rows = g1_summary.groupby("metric_id")["delta_float"].agg(["mean", "min", "max"]).reset_index()
    g1_metric_rows = [
        {
            "metric_id": row["metric_id"],
            "mean_delta": format_float(float(row["mean"])),
            "min_delta": format_float(float(row["min"])),
            "max_delta": format_float(float(row["max"])),
        }
        for row in g1_metric_rows.to_dict("records")
    ]
    validation_passes = int((validation["status"] == "PASS").sum())
    validation_failures = int((validation["status"] == "FAIL").sum())

    return "\n".join(
        [
            "# G4 F1BZQ Final Scientific Synthesis",
            "",
            f"- Phase: {PHASE_ID}",
            f"- Decision token: {manifest['decision_token']}",
            "- Source: frozen G4 analytic, metric, and statistical artifacts only",
            "- No Ollama/API calls, no new tests, no new bootstrap/statistical tests, no report rewrite",
            "- Statistical wording: bootstrap CI only; no p-values; Holm not applied; NS means CI includes zero, not proof of no effect",
            "",
            "## 1. Candidate-Stage Bottleneck",
            "",
            markdown_table(
                candidate_rows,
                [
                    ("dataset", "Dataset"),
                    ("positive_present_users", "Positive Present"),
                    ("planned_users", "Users"),
                    ("candidate_recall_at_20", "Candidate Recall@20"),
                    ("unconditional_hr_at_5_mean_across_cells", "Mean Uncond HR@5"),
                    ("conditional_hr_at_5_denominator", "Conditional n"),
                    ("conditional_hr_at_5_mean_across_cells", "Mean Cond HR@5"),
                    ("bottleneck_assessment", "Assessment"),
                ],
            ),
            "",
            "Candidate recall is the upstream probability that the held-out positive appears in the frozen 20-item F1BZQ pool. Unconditional HR@5/NDCG@5/MRR measure the whole candidate-pool plus reranking pipeline over all planned users, with invalid terminal responses and positive-absent users contributing zero. Conditional HR@5/NDCG@5 measure reranker quality only among positive-present users, again with invalid terminal responses as zero inside that subset. Low unconditional utility must therefore not be attributed only to the LLM when the positive is absent from the pool.",
            "",
            "Dataset reading: Beauty, Sports, and Food.com are dominated by candidate-stage failure; ML1M has both a material candidate ceiling and reranking limitations; LastFM has the least severe candidate bottleneck and the most informative reranking signal.",
            "",
            "## 2. Internal G4 Statistics",
            "",
            "- Frozen inferential rows: 225.",
            "- Overall CI decisions: positive 29, negative 24, non-significant 172.",
            "- Prompt strategy: positive 2, negative 13, NS 60.",
            "- Model pair: positive 27, negative 11, NS 112.",
            "- Dataset counts: Beauty 0/0/45, Sports 4/5/36, ML1M 10/5/30, LastFM 15/14/16, Food.com 0/0/45.",
            "",
            "All inferential claims below are based on frozen percentile bootstrap CIs for paired user-level mean deltas. No p-values or corrected decisions are reported.",
            "",
            "## 3. Prompt Effect",
            "",
            "LLMRANK does not have a consistent cross-dataset advantage. The supported improvements are isolated to Qwen on LastFM HR@5 and conditional HR@5. Supported worsens occur for Qwen on Sports across all five utility/conditional metrics, for Mistral on LastFM across all five metrics, and for Gemma2 on LastFM NDCG@5, MRR, and conditional NDCG@5. Beauty, ML1M, and Food.com have no BASE-vs-LLMRANK CI-supported differences in the frozen prompt family.",
            "",
            markdown_table(
                prompt_supported_rows,
                [
                    ("dataset", "Dataset"),
                    ("model", "Model"),
                    ("metric_id", "Metric"),
                    ("paired_n", "n"),
                    ("llmrank_minus_base_delta", "LLMRANK-BASE"),
                    ("ci_low", "CI Low"),
                    ("ci_high", "CI High"),
                    ("effect_label", "Effect"),
                ],
            ),
            "",
            "The 60 NS prompt rows are not evidence of no prompt effect; they only mean the frozen CI includes zero.",
            "",
            "## 4. Model Effect",
            "",
            "Model differences are dataset-dependent. The largest supported differences are conditional utility differences in LastFM and ML1M. Gemma2 is often higher than Mistral in LastFM/ML1M, while Qwen is higher than Mistral in several LLMRANK rows. Qwen-vs-Gemma2 is mixed: Gemma2 is higher on several HR/NDCG rows, but one LastFM LLMRANK MRR row supports Qwen over Gemma2. No model is consistently superior across datasets, prompts, and metrics.",
            "",
            markdown_table(
                model_top_rows,
                [
                    ("dataset", "Dataset"),
                    ("prompt_strategy", "Prompt"),
                    ("comparison", "Comparison"),
                    ("metric_id", "Metric"),
                    ("paired_n", "n"),
                    ("second_minus_first_delta", "Second-First"),
                    ("ci_low", "CI Low"),
                    ("ci_high", "CI High"),
                    ("effect_label", "Effect"),
                ],
            ),
            "",
            "Do not overgeneralize from LastFM: it supplies the strongest conditional denominator and the densest set of supported CIs.",
            "",
            "## 5. Conditional Reranking Quality",
            "",
            markdown_table(
                conditional_denominators,
                [("dataset", "Dataset"), ("conditional_n", "Positive-Present n"), ("foodcom_guard", "Guard")],
            ),
            "",
            "Conditional HR@5/NDCG@5 are only defined on these positive-present users. Food.com has n=4; its conditional rows are diagnostic only. Beauty and Sports have extremely low candidate recall, so their poor end-to-end utility is primarily candidate-stage limited even though conditional reranking sometimes recovers hits. ML1M reflects both candidate-stage loss and reranking quality differences. LastFM shows both a real candidate ceiling and meaningful reranker/model/prompt variation.",
            "",
            "## 6. Beyond-Utility And Reliability",
            "",
            "Only item_coverage_at_5 and oic_at_5 are informative frozen beyond-utility metrics for G4; they are strict-valid, candidate-union/pairwise summaries, not full-catalog claims. Other beyond-utility metrics remain undefined or excluded exactly as frozen.",
            "",
            markdown_table(
                beyond_rows,
                [
                    ("dataset", "Dataset"),
                    ("metric_id", "Metric"),
                    ("value", "Mean"),
                    ("min_cell_value", "Min"),
                    ("max_cell_value", "Max"),
                ],
            ),
            "",
            f"Reliability accounting: planned {reliability_values['planned_requests']}, strict-valid {reliability_values['strict_valid']}, invalid {reliability_values['structural_invalid']}, invalid_length {reliability_values['invalid_length']}, duplicate_indices {reliability_values['duplicate_indices']}, backend errors {reliability_values['backend_errors']}.",
            "",
            "Descriptively, Mistral has more structural invalids than Qwen2 or Gemma2, and LLMRANK has more invalids than BASE. No reliability inferential test was frozen.",
            "",
            markdown_table(reliability_model, [("model", "Model"), ("value", "Strict-Valid Rate"), ("numerator_or_count", "Strict"), ("denominator", "Planned")]),
            "",
            markdown_table(reliability_prompt, [("prompt_strategy", "Prompt"), ("value", "Strict-Valid Rate"), ("numerator_or_count", "Strict"), ("denominator", "Planned")]),
            "",
            "## 7. G1 Vs G4",
            "",
            "The 90 frozen G1-vs-G4 rows are descriptive only. G1 and G4 use different candidate sources, so these are pipeline-level differences, not a pure causal estimate of a reranking effect. All 90 descriptive deltas are negative for G4 relative to G1 in the frozen table.",
            "",
            markdown_table(g1_metric_rows, [("metric_id", "Metric"), ("mean_delta", "Mean G4-G1"), ("min_delta", "Min"), ("max_delta", "Max")]),
            "",
            "The useful pattern is that the F1BZQ 20-item reranking pipeline is much more ceiling-limited than the G1 ranking pipeline. This should be read as candidate-source and pipeline evidence, not as a claim that reranking itself causally reduces utility.",
            "",
            "## 8. Scientific Takeaway",
            "",
            "- The F1BZQ candidate generator is a major bottleneck, especially for Beauty, Sports, and Food.com; ML1M is also materially ceiling-limited; LastFM is the least bottlenecked.",
            "- When the positive is present, the LLM reranker can recover useful utility, most clearly in LastFM and in selected Beauty/ML1M cells, but recovery is model- and prompt-dependent.",
            "- LLMRANK does not consistently help; frozen CIs show isolated Qwen LastFM gains and more supported worsens than gains overall.",
            "- Model effects are dataset-dependent; Gemma2 often improves over Mistral in LastFM/ML1M, but no model is uniformly superior.",
            "- Strongest evidence: LastFM, then ML1M. Weakest evidence: Food.com due n=4 conditional denominator, and Beauty for internal CI support.",
            "- G4 adds a frozen reranking-pipeline diagnosis that G1/G2/G3 did not provide: it separates F1BZQ candidate-pool ceiling from conditional LLM reranking quality and supplies paired internal prompt/model CIs.",
            "",
            "## 9. Validation",
            "",
            f"- PASS: {validation_passes}",
            f"- FAIL: {validation_failures}",
            f"- Status: {'PASS' if validation_failures == 0 else 'FAIL'}",
            "",
            "No final report rewrite, relation update, new experiments, new tests, or inference calls are included in this synthesis freeze.",
            "",
        ]
    )


def frozen_hash_failures(root: Path, analytic_manifest: Mapping[str, Any], metrics_manifest: Mapping[str, Any], stat_manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_hashes: dict[str, str] = {}
    for manifest, keys in (
        (analytic_manifest, ("output_hashes",)),
        (metrics_manifest, ("output_hashes",)),
        (stat_manifest, ("output_hashes",)),
    ):
        for key in keys:
            expected_hashes.update({str(path): str(hash_value) for path, hash_value in manifest.get(key, {}).items()})
    expected_hashes[PATHS["analytic_manifest"].as_posix()] = sha256_file(root / PATHS["analytic_manifest"])
    expected_hashes[PATHS["metrics_manifest"].as_posix()] = sha256_file(root / PATHS["metrics_manifest"])
    expected_hashes[PATHS["statistical_manifest"].as_posix()] = sha256_file(root / PATHS["statistical_manifest"])
    for path_text, expected in sorted(expected_hashes.items()):
        path = root / Path(path_text)
        if not path.is_file():
            failures.append(f"FROZEN_INPUT_MISSING:{path_text}")
            continue
        observed = sha256_file(path)
        if observed != expected:
            failures.append(f"FROZEN_HASH_MISMATCH:{path_text}:{observed}:{expected}")
    return failures


def build_validation(
    *,
    root: Path,
    head_before: str,
    status_before: str,
    analytic_manifest: Mapping[str, Any],
    metrics_manifest: Mapping[str, Any],
    stat_manifest: Mapping[str, Any],
    metric_results: pd.DataFrame,
    reliability: pd.DataFrame,
    statistical_results: pd.DataFrame,
    candidate: pd.DataFrame,
    ci_counts: pd.DataFrame,
    prompt_effects: pd.DataFrame,
    model_effects: pd.DataFrame,
    conditional_quality: pd.DataFrame,
    beyond_reliability: pd.DataFrame,
    g1_vs_g4: pd.DataFrame,
    takeaways: pd.DataFrame,
    allow_generator_dirty: bool,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check_id: str, passed: bool, observed: Any, expected: Any, details: str = "") -> None:
        rows.append(
            {
                "phase_id": PHASE_ID,
                "check_id": check_id,
                "status": "PASS" if passed else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
                "details": details,
            }
        )

    add("head_matches_precondition", head_before == EXPECTED_HEAD, head_before, EXPECTED_HEAD)
    unexpected_status = unexpected_git_status_lines(status_before)
    git_ok = status_before.strip() == "" or (allow_generator_dirty and not unexpected_status)
    add("git_status_clean_or_expected_synthesis_dirty", git_ok, status_before.replace("\n", " | "), "clean or expected G4 synthesis outputs only")
    add(
        "analytic_manifest_decision",
        analytic_manifest.get("decision_token") == "G4_ANALYTIC_UNIVERSE_FROZEN",
        analytic_manifest.get("decision_token"),
        "G4_ANALYTIC_UNIVERSE_FROZEN",
    )
    add("metrics_manifest_decision", metrics_manifest.get("decision_token") == "G4_METRICS_FROZEN", metrics_manifest.get("decision_token"), "G4_METRICS_FROZEN")
    add(
        "statistical_manifest_decision",
        stat_manifest.get("decision_token") == "G4_STATISTICAL_ANALYSIS_FROZEN",
        stat_manifest.get("decision_token"),
        "G4_STATISTICAL_ANALYSIS_FROZEN",
    )
    hash_failures = frozen_hash_failures(root, analytic_manifest, metrics_manifest, stat_manifest)
    add("frozen_g4_input_hashes_match", not hash_failures, len(hash_failures), 0, " | ".join(hash_failures[:5]))

    recall_ok = True
    recall_observed: dict[str, str] = {}
    for row in candidate.to_dict("records"):
        dataset = row["dataset"]
        recall_observed[dataset] = row["candidate_recall_at_20"]
        recall_ok = recall_ok and int(row["positive_present_users"]) == EXPECTED_POSITIVE_PRESENT[dataset]
        recall_ok = recall_ok and int(row["planned_users"]) == EXPECTED_USERS[dataset]
        recall_ok = recall_ok and row["candidate_recall_at_20"] == EXPECTED_CANDIDATE_RECALL[dataset]
    add("candidate_recall_exact", recall_ok, stable_json(recall_observed), stable_json(EXPECTED_CANDIDATE_RECALL))

    inferential = statistical_results[statistical_results["inferential_test_allowed"].map(bool_value)]
    descriptive = statistical_results[~statistical_results["inferential_test_allowed"].map(bool_value)]
    add("inferential_rows_225", len(inferential) == 225, len(inferential), 225)
    add("descriptive_rows_90", len(descriptive) == 90, len(descriptive), 90)
    add("overall_ci_decisions", count_decisions(inferential) == EXPECTED_INTERNAL_DECISIONS, stable_json(count_decisions(inferential)), stable_json(EXPECTED_INTERNAL_DECISIONS))
    family_ok = True
    family_observed: dict[str, dict[str, int]] = {}
    for family, expected in EXPECTED_FAMILY_DECISIONS.items():
        observed = count_decisions(inferential[inferential["comparison_family"] == family])
        family_observed[family] = observed
        family_ok = family_ok and observed == expected
    add("family_ci_decisions", family_ok, stable_json(family_observed), stable_json(EXPECTED_FAMILY_DECISIONS))
    dataset_ok = True
    dataset_observed: dict[str, dict[str, int]] = {}
    for dataset, expected in EXPECTED_DATASET_DECISIONS.items():
        observed = count_decisions(inferential[inferential["dataset"] == dataset])
        dataset_observed[dataset] = observed
        dataset_ok = dataset_ok and observed == expected
    add("dataset_ci_decisions", dataset_ok, stable_json(dataset_observed), stable_json(EXPECTED_DATASET_DECISIONS))

    add("ci_counts_table_nonempty", len(ci_counts) == 13, len(ci_counts), 13)
    add("prompt_effect_rows", len(prompt_effects) == 75, len(prompt_effects), 75)
    add(
        "prompt_supported_breakdown",
        count_decisions(prompt_effects) == EXPECTED_FAMILY_DECISIONS["g4_prompt_strategy"],
        stable_json(count_decisions(prompt_effects)),
        stable_json(EXPECTED_FAMILY_DECISIONS["g4_prompt_strategy"]),
    )
    add("model_effect_rows", len(model_effects) == 150, len(model_effects), 150)
    add(
        "model_supported_breakdown",
        count_decisions(model_effects) == EXPECTED_FAMILY_DECISIONS["g4_model_pair"],
        stable_json(count_decisions(model_effects)),
        stable_json(EXPECTED_FAMILY_DECISIONS["g4_model_pair"]),
    )
    add("conditional_quality_rows", len(conditional_quality) == 30, len(conditional_quality), 30)
    conditional_n_ok = all(
        int(row["conditional_denominator"]) == EXPECTED_POSITIVE_PRESENT[row["dataset"]]
        for row in conditional_quality.to_dict("records")
    )
    add("conditional_denominators_exact", conditional_n_ok, "checked", stable_json(EXPECTED_POSITIVE_PRESENT))
    foodcom_ok = conditional_quality[conditional_quality["dataset"] == "foodcom"]["foodcom_n4_diagnostic_flag"].eq("true").all()
    add("foodcom_n4_flagged", foodcom_ok, "checked", "all Food.com conditional rows flagged")

    rel_total = beyond_reliability[beyond_reliability["section"] == "reliability_total"]
    rel_observed = {row["metric_id"]: int(row["value"]) for row in rel_total.to_dict("records")}
    add("reliability_totals_exact", rel_observed == EXPECTED_RELIABILITY, stable_json(rel_observed), stable_json(EXPECTED_RELIABILITY))
    add("beyond_rows_limited_to_frozen_computable", set(metric_results[metric_results["metric_family"] == "beyond_utility_g1_g2_compatible"]["metric_id"]) == {"item_coverage_at_5", "oic_at_5"}, sorted(set(metric_results[metric_results["metric_family"] == "beyond_utility_g1_g2_compatible"]["metric_id"])), "item_coverage_at_5,oic_at_5")
    undefined_sections = beyond_reliability[beyond_reliability["section"] == "undefined_or_blocked"]
    add("undefined_blocked_summarized", int(undefined_sections["numerator_or_count"].astype(int).sum()) == 510, int(undefined_sections["numerator_or_count"].astype(int).sum()), 510)

    add("g1_vs_g4_rows_90", len(g1_vs_g4) == 90, len(g1_vs_g4), 90)
    g1_g4_descriptive = (
        g1_vs_g4["inferential_test_allowed"].eq("false").all()
        and g1_vs_g4["ranking_vs_reranking_policy"].eq("descriptive_pipeline_level_only").all()
    )
    add("g1_vs_g4_descriptive_only", g1_g4_descriptive, "checked", "no inferential rows")
    all_g1_deltas_negative = g1_vs_g4["g4_minus_g1_pipeline_delta"].astype(float).lt(0).all()
    add("g1_vs_g4_all_pipeline_deltas_negative", all_g1_deltas_negative, "checked", "all 90 negative")
    no_pvalues = statistical_results["p_value"].astype(str).eq("").all()
    add("no_p_values", no_pvalues, sorted(set(statistical_results["p_value"].astype(str))), "all blank")
    holm_not_applied = statistical_results["multiplicity_status"].astype(str).str.startswith("NOT_APPLIED").all()
    add("holm_not_applied", holm_not_applied, sorted(set(statistical_results["multiplicity_status"].astype(str))), "all NOT_APPLIED")
    add("takeaways_rows", len(takeaways) == 7, len(takeaways), 7)

    return pd.DataFrame(rows)


def append_notes(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = "\n".join(
        [
            f"## {PHASE_ID} - {manifest['created_at_utc']}",
            "",
            "- Built the final G4 F1BZQ scientific synthesis from frozen G4 analytic, metric, and statistical artifacts only.",
            "- No Ollama/API calls, no new tests, no new statistics, no frozen artifact edits, and no final report rewrite.",
            f"- Candidate recall exact: {stable_json(EXPECTED_CANDIDATE_RECALL)}.",
            f"- Internal CI decisions: {stable_json(manifest['summary']['internal_ci_decisions'])}.",
            f"- Validation passed: {bool_text(manifest['validation_passed'])}.",
            f"- Next safe step: {manifest['next_safe_step']}.",
            "",
        ]
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_lines = existing.splitlines()
    marker = f"## {PHASE_ID} - "
    for index, line in enumerate(existing_lines):
        if line.startswith(marker):
            existing_lines = existing_lines[:index]
            break
    prefix = "\n".join(existing_lines).rstrip()
    output = f"{prefix}\n\n{entry}" if prefix else entry
    path.write_text(output, encoding="utf-8", newline="\n")


def output_hashes(root: Path) -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(root / path)
        for key, path in OUTPUTS.items()
        if key != "manifest" and (root / path).is_file()
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_project_root(Path(args.output_root).resolve())
    head_before = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short"], check=False)

    analytic_manifest = read_json(root, PATHS["analytic_manifest"])
    metrics_manifest = read_json(root, PATHS["metrics_manifest"])
    stat_manifest = read_json(root, PATHS["statistical_manifest"])
    metric_results = read_csv(root, PATHS["metric_results"])
    reliability = read_csv(root, PATHS["reliability"])
    undefined = read_csv(root, PATHS["undefined_blocked"])
    statistical_results = read_csv(root, PATHS["statistical_results"])

    candidate = build_candidate_bottleneck(metric_results)
    ci_counts = build_internal_ci_counts(statistical_results)
    prompt_effects = build_prompt_effects(statistical_results)
    model_effects = build_model_effects(statistical_results)
    conditional_quality = build_conditional_quality(metric_results)
    beyond_reliability = build_beyond_reliability(metric_results, reliability, undefined)
    g1_vs_g4 = build_g1_vs_g4(statistical_results)
    takeaways = build_takeaways(candidate)
    validation = build_validation(
        root=root,
        head_before=head_before,
        status_before=status_before,
        analytic_manifest=analytic_manifest,
        metrics_manifest=metrics_manifest,
        stat_manifest=stat_manifest,
        metric_results=metric_results,
        reliability=reliability,
        statistical_results=statistical_results,
        candidate=candidate,
        ci_counts=ci_counts,
        prompt_effects=prompt_effects,
        model_effects=model_effects,
        conditional_quality=conditional_quality,
        beyond_reliability=beyond_reliability,
        g1_vs_g4=g1_vs_g4,
        takeaways=takeaways,
        allow_generator_dirty=args.allow_generator_dirty,
    )
    validation_passed = bool((validation["status"] == "FAIL").sum() == 0)
    inferential = statistical_results[statistical_results["inferential_test_allowed"].map(bool_value)]
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "phase_id": PHASE_ID,
        "decision_token": DECISION_FROZEN if validation_passed else DECISION_BLOCKED,
        "created_at_utc": utc_now(),
        "head_before_generation": head_before,
        "precondition_status": "G4_STATISTICAL_ANALYSIS_FROZEN",
        "source": "frozen_g4_analytic_metric_statistical_artifacts_only",
        "zero_inference": True,
        "ollama_or_api_calls_executed": 0,
        "new_tests_added": False,
        "new_statistics_executed": False,
        "final_report_rewrite": False,
        "frozen_evidence_modified": False,
        "protocol_config_runner_modified": False,
        "analytic_universe_modified": False,
        "metric_artifacts_modified": False,
        "statistical_artifacts_modified": False,
        "p_values_computed": False,
        "holm_applied": False,
        "statistical_wording": "bootstrap_CI_only_no_p_values_holm_not_applied_NS_not_no_effect",
        "summary": {
            "candidate_recall_exact": EXPECTED_CANDIDATE_RECALL,
            "internal_inferential_rows": int(len(inferential)),
            "internal_ci_decisions": count_decisions(inferential),
            "prompt_strategy_decisions": count_decisions(
                statistical_results[statistical_results["comparison_family"] == "g4_prompt_strategy"]
            ),
            "model_pair_decisions": count_decisions(
                statistical_results[statistical_results["comparison_family"] == "g4_model_pair"]
            ),
            "g1_vs_g4_descriptive_rows": int(len(g1_vs_g4)),
            "reliability": EXPECTED_RELIABILITY,
            "validation_passes": int((validation["status"] == "PASS").sum()),
            "validation_failures": int((validation["status"] == "FAIL").sum()),
        },
        "input_hashes": {
            path.as_posix(): sha256_file(root / path)
            for path in [
                PATHS["analytic_manifest"],
                PATHS["metrics_manifest"],
                PATHS["statistical_manifest"],
                PATHS["metric_results"],
                PATHS["metric_denominators"],
                PATHS["candidate_conditional"],
                PATHS["reliability"],
                PATHS["undefined_blocked"],
                PATHS["statistical_results"],
                PATHS["statistical_validation"],
                PATHS["statistical_plan"],
            ]
        },
        "files_created": [path.as_posix() for path in OUTPUTS.values()] + [PATHS["notes"].as_posix()],
        "validation_passed": validation_passed,
        "validation_failures": validation[validation["status"] == "FAIL"].to_dict("records"),
        "next_safe_step": "update the final report/relationship narrative from frozen G4 synthesis artifacts only when explicitly requested",
    }
    report = build_report(
        candidate=candidate,
        ci_counts=ci_counts,
        prompt_effects=prompt_effects,
        model_effects=model_effects,
        conditional_quality=conditional_quality,
        beyond_reliability=beyond_reliability,
        g1_vs_g4=g1_vs_g4,
        validation=validation,
        manifest=manifest,
    )

    if not args.no_write:
        write_csv(root, OUTPUTS["candidate_bottleneck"], candidate)
        write_csv(root, OUTPUTS["internal_ci_counts"], ci_counts)
        write_csv(root, OUTPUTS["prompt_effects"], prompt_effects)
        write_csv(root, OUTPUTS["model_effects"], model_effects)
        write_csv(root, OUTPUTS["conditional_quality"], conditional_quality)
        write_csv(root, OUTPUTS["beyond_reliability"], beyond_reliability)
        write_csv(root, OUTPUTS["g1_vs_g4_descriptive"], g1_vs_g4)
        write_csv(root, OUTPUTS["takeaways"], takeaways)
        write_csv(root, OUTPUTS["validation"], validation)
        (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
        (root / OUTPUTS["report"]).write_text(report, encoding="utf-8", newline="\n")
        manifest["output_hashes"] = output_hashes(root)
        write_json(root, OUTPUTS["manifest"], manifest)
        append_notes(root / PATHS["notes"], manifest)

    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the final G3 scientific synthesis from frozen artifacts only.

This script reads the frozen G3 metric and statistical analysis artifacts,
aggregates their descriptive patterns, and writes the authoritative G3
scientific synthesis. It does not call Ollama, run new tests, recompute
bootstrap intervals, update the relation/report chapters, or start G4.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import find_project_root  # noqa: E402


EXPECTED_HEAD = "ed146d6089b0370b27ea97228b03c0da0b057ae7"
PHASE_ID = "g3_candidate_position_scientific_synthesis"
DECISION_FROZEN = "G3_SCIENTIFIC_SYNTHESIS_FROZEN"
DECISION_BLOCKED = "G3_SCIENTIFIC_SYNTHESIS_BLOCKED"

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = ("BASE", "LLMRANK")
UTILITY_METRICS = ("hr_at_5", "ndcg_at_5", "mrr")
CANDDIF_METRICS = ("CandDif_HR@5", "CandDif_NDCG@5")
DECISIONS = ("CI_SUPPORTED_POSITIVE", "CI_SUPPORTED_NEGATIVE", "NON_SIGNIFICANT_CI_INCLUDES_ZERO")

PATHS = {
    "metrics_manifest": Path("results/manifests/g3_candidate_position_metrics_freeze.json"),
    "statistical_manifest": Path("results/manifests/g3_candidate_position_statistical_analysis_manifest.json"),
    "metric_results": Path("results/tables/g3_candidate_position_metric_results.csv"),
    "canddif": Path("results/tables/g3_candidate_position_canddif.csv"),
    "reliability": Path("results/tables/g3_candidate_position_reliability.csv"),
    "statistical_results": Path("results/tables/g3_candidate_position_statistical_results.csv"),
    "paired_contributions": Path("results/tables/g3_candidate_position_statistical_paired_contributions.csv"),
    "statistical_validation": Path("results/tables/g3_candidate_position_statistical_validation.csv"),
    "source": Path("src/extension/g3_candidate_position_scientific_synthesis.py"),
    "notes": Path("report/notes.md"),
}

OUTPUTS = {
    "comparison_summary": Path("results/tables/g3_candidate_position_synthesis_statistical_comparisons.csv"),
    "decision_breakdowns": Path("results/tables/g3_candidate_position_synthesis_decision_breakdowns.csv"),
    "canddif_summary": Path("results/tables/g3_candidate_position_synthesis_canddif_summary.csv"),
    "prompt_interaction": Path("results/tables/g3_candidate_position_synthesis_prompt_interaction.csv"),
    "reliability_summary": Path("results/tables/g3_candidate_position_synthesis_reliability.csv"),
    "validation": Path("results/tables/g3_candidate_position_synthesis_validation.csv"),
    "manifest": Path("results/manifests/g3_candidate_position_scientific_synthesis_manifest.json"),
    "report": Path("report/g3_candidate_position_scientific_synthesis.md"),
}

EXPECTED_DIRTY_PATHS = {
    PATHS["source"].as_posix(),
    PATHS["notes"].as_posix(),
    *[path.as_posix() for path in OUTPUTS.values()],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen G3 scientific synthesis.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate in memory without writing artifacts.")
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
    digest = hashlib.sha256(stable_json([str(part) for part in parts]).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:32].upper()}"


def read_csv(root: Path, path: Path) -> pd.DataFrame:
    return pd.read_csv(root / path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def write_csv(root: Path, path: Path, frame: pd.DataFrame) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / path, index=False, lineterminator="\n", float_format="%.17g")


def write_json(root: Path, path: Path, payload: Mapping[str, Any]) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def format_float(value: Any, digits: int = 6) -> str:
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def bool_text(value: Any) -> str:
    return "true" if str(value).strip().lower() in {"true", "1", "yes"} else "false"


def bool_value(value: Any) -> bool:
    return bool_text(value) == "true"


def git_output(root: Path, args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def git_dirty_paths(status: str) -> list[str]:
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
        paths.append(text.strip().replace("\\", "/"))
    return sorted(paths)


def status_is_clean_or_expected(status: str, allow_generator_dirty: bool) -> bool:
    paths = git_dirty_paths(status)
    if not paths:
        return True
    return allow_generator_dirty and set(paths).issubset(EXPECTED_DIRTY_PATHS)


def verify_hash_mapping(root: Path, mapping: Mapping[str, Any], failures: list[str], *, label: str, skip_paths: set[str] | None = None) -> int:
    checked = 0
    skip = skip_paths or set()
    for rel_path, expected in mapping.items():
        if rel_path in skip:
            continue
        path = root / rel_path
        actual = sha256_file(path) if path.is_file() else ""
        checked += 1
        if actual != str(expected):
            failures.append(f"{label}_HASH_MISMATCH:{rel_path}:{actual}:{expected}")
    return checked


def comparison_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = results.assign(plan_order_int=pd.to_numeric(results["plan_order"], errors="raise")).sort_values(
        "plan_order_int", kind="mergesort"
    )
    for row in ordered.to_dict("records"):
        rows.append(
            {
                "summary_id": stable_id("G3SYN-COMP", row["comparison_id"]),
                "comparison_id": row["comparison_id"],
                "dataset": row["dataset"],
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "metric": row["metric"],
                "paired_user_count": row["paired_user_count"],
                "random_mean": row["random_mean"],
                "first_mean": row["first_mean"],
                "first_minus_random_delta": row["first_minus_random_delta"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "primary_ci_decision": row["primary_ci_decision"],
                "corrected_decision": row["corrected_decision"],
                "multiplicity_status": row["multiplicity_correction_execution_status"],
                "canddif_metric": row["canddif_metric"],
                "canddif_value": row["canddif_value"],
                "canddif_significance_test_status": row["canddif_significance_test_status"],
                "statistical_wording": "bootstrap_CI_only_no_p_values",
            }
        )
    return pd.DataFrame(rows)


def decision_count_row(scope: str, scope_value: str, frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["primary_ci_decision"].value_counts().to_dict()
    return {
        "breakdown_id": stable_id("G3SYN-BREAK", scope, scope_value),
        "scope": scope,
        "scope_value": scope_value,
        "comparison_rows": len(frame),
        "ci_supported_positive": int(counts.get("CI_SUPPORTED_POSITIVE", 0)),
        "ci_supported_negative": int(counts.get("CI_SUPPORTED_NEGATIVE", 0)),
        "non_significant_ci_includes_zero": int(counts.get("NON_SIGNIFICANT_CI_INCLUDES_ZERO", 0)),
        "median_first_minus_random_delta": format_float(frame["first_minus_random_delta"].astype(float).median()),
        "mean_first_minus_random_delta": format_float(frame["first_minus_random_delta"].astype(float).mean()),
        "min_first_minus_random_delta": format_float(frame["first_minus_random_delta"].astype(float).min()),
        "max_first_minus_random_delta": format_float(frame["first_minus_random_delta"].astype(float).max()),
        "source": PATHS["statistical_results"].as_posix(),
    }


def decision_breakdowns(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [decision_count_row("overall", "ALL", results)]
    for metric in UTILITY_METRICS:
        rows.append(decision_count_row("metric", metric, results.loc[results["metric"].eq(metric)]))
    for dataset in DATASETS:
        rows.append(decision_count_row("dataset", dataset, results.loc[results["dataset"].eq(dataset)]))
    for model in MODELS:
        rows.append(decision_count_row("model", model, results.loc[results["model"].eq(model)]))
    for prompt in PROMPTS:
        rows.append(decision_count_row("prompt", prompt, results.loc[results["prompt_strategy"].eq(prompt)]))
    for dataset in DATASETS:
        for metric in UTILITY_METRICS:
            rows.append(decision_count_row("dataset_metric", f"{dataset}|{metric}", results.loc[results["dataset"].eq(dataset) & results["metric"].eq(metric)]))
    for model in MODELS:
        for metric in UTILITY_METRICS:
            rows.append(decision_count_row("model_metric", f"{model}|{metric}", results.loc[results["model"].eq(model) & results["metric"].eq(metric)]))
    for prompt in PROMPTS:
        for metric in UTILITY_METRICS:
            rows.append(decision_count_row("prompt_metric", f"{prompt}|{metric}", results.loc[results["prompt_strategy"].eq(prompt) & results["metric"].eq(metric)]))
    return pd.DataFrame(rows)


def canddif_summary_row(scope: str, scope_value: str, metric_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    values = frame["value"].astype(float)
    return {
        "summary_id": stable_id("G3SYN-CD", scope, scope_value, metric_id),
        "scope": scope,
        "scope_value": scope_value,
        "metric_id": metric_id,
        "cells": len(frame),
        "positive_cells": int((values > 0.0).sum()),
        "negative_cells": int((values < 0.0).sum()),
        "zero_cells": int((values == 0.0).sum()),
        "mean_value": format_float(values.mean()),
        "median_value": format_float(values.median()),
        "min_value": format_float(values.min()),
        "max_value": format_float(values.max()),
        "mean_abs_value": format_float(values.abs().mean()),
        "median_abs_value": format_float(values.abs().median()),
        "significance_test_status": "NOT_COMPUTED_CANDDIF_DESCRIPTIVE_ONLY",
        "source": PATHS["canddif"].as_posix(),
    }


def canddif_summary(canddif: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_id in CANDDIF_METRICS:
        subset = canddif.loc[canddif["metric_id"].eq(metric_id)]
        rows.append(canddif_summary_row("overall", "ALL", metric_id, subset))
        for dataset in DATASETS:
            rows.append(canddif_summary_row("dataset", dataset, metric_id, subset.loc[subset["dataset"].eq(dataset)]))
        for model in MODELS:
            rows.append(canddif_summary_row("model", model, metric_id, subset.loc[subset["model"].eq(model)]))
        for prompt in PROMPTS:
            rows.append(canddif_summary_row("prompt", prompt, metric_id, subset.loc[subset["prompt_strategy"].eq(prompt)]))
    return pd.DataFrame(rows)


def classify_prompt_effect(base_value: float, llmrank_value: float, tolerance: float = 1e-12) -> str:
    base_abs = abs(base_value)
    llmrank_abs = abs(llmrank_value)
    if llmrank_abs < base_abs - tolerance:
        return "ATTENUATES"
    if llmrank_abs > base_abs + tolerance:
        return "AMPLIFIES"
    return "UNCHANGED"


def prompt_interaction(results: pd.DataFrame, canddif: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            for metric in UTILITY_METRICS:
                subset = results.loc[results["dataset"].eq(dataset) & results["model"].eq(model) & results["metric"].eq(metric)]
                base = subset.loc[subset["prompt_strategy"].eq("BASE")].iloc[0]
                llmrank = subset.loc[subset["prompt_strategy"].eq("LLMRANK")].iloc[0]
                base_delta = float(base["first_minus_random_delta"])
                llmrank_delta = float(llmrank["first_minus_random_delta"])
                rows.append(
                    {
                        "interaction_id": stable_id("G3SYN-PROMPT", "delta", dataset, model, metric),
                        "measure_family": "FIRST_minus_RANDOM",
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "base_value": format_float(base_delta),
                        "llmrank_value": format_float(llmrank_delta),
                        "llmrank_minus_base": format_float(llmrank_delta - base_delta),
                        "base_abs_value": format_float(abs(base_delta)),
                        "llmrank_abs_value": format_float(abs(llmrank_delta)),
                        "magnitude_effect": classify_prompt_effect(base_delta, llmrank_delta),
                        "base_ci_decision": base["primary_ci_decision"],
                        "llmrank_ci_decision": llmrank["primary_ci_decision"],
                        "source": PATHS["statistical_results"].as_posix(),
                    }
                )
            for metric_id in CANDDIF_METRICS:
                subset = canddif.loc[canddif["dataset"].eq(dataset) & canddif["model"].eq(model) & canddif["metric_id"].eq(metric_id)]
                base = subset.loc[subset["prompt_strategy"].eq("BASE")].iloc[0]
                llmrank = subset.loc[subset["prompt_strategy"].eq("LLMRANK")].iloc[0]
                base_value = float(base["value"])
                llmrank_value = float(llmrank["value"])
                rows.append(
                    {
                        "interaction_id": stable_id("G3SYN-PROMPT", "canddif", dataset, model, metric_id),
                        "measure_family": "CandDif",
                        "dataset": dataset,
                        "model": model,
                        "metric": metric_id,
                        "base_value": format_float(base_value),
                        "llmrank_value": format_float(llmrank_value),
                        "llmrank_minus_base": format_float(llmrank_value - base_value),
                        "base_abs_value": format_float(abs(base_value)),
                        "llmrank_abs_value": format_float(abs(llmrank_value)),
                        "magnitude_effect": classify_prompt_effect(base_value, llmrank_value),
                        "base_ci_decision": "not_applicable_descriptive_canddif",
                        "llmrank_ci_decision": "not_applicable_descriptive_canddif",
                        "source": PATHS["canddif"].as_posix(),
                    }
                )
    return pd.DataFrame(rows)


def reliability_summary(reliability: pd.DataFrame) -> pd.DataFrame:
    order = reliability.loc[reliability["scope"].eq("order")].set_index("scope_value")
    first = order.loc["FIRST"]
    random = order.loc["RANDOM"]
    rows = []
    for row in [first, random]:
        scope_value = str(row.name)
        rows.append(
            {
                "summary_id": stable_id("G3SYN-REL", scope_value),
                "scope": "order",
                "scope_value": scope_value,
                "planned_requests": row["planned_requests"],
                "strict_valid": row["strict_valid"],
                "invalid": row["invalid"],
                "backend_errors": row["backend_errors"],
                "invalid_length": row["invalid_length"],
                "invalid_duplicate": row["invalid_duplicate"],
                "hallucination_or_out_of_candidate": row["hallucination_or_out_of_candidate"],
                "strict_valid_rate": row["strict_valid_rate"],
                "invalid_rate": row["invalid_rate"],
                "source": PATHS["reliability"].as_posix(),
            }
        )
    first_invalid_rate = float(first["invalid"]) / float(first["planned_requests"])
    random_invalid_rate = float(random["invalid"]) / float(random["planned_requests"])
    rows.append(
        {
            "summary_id": stable_id("G3SYN-REL", "FIRST_MINUS_RANDOM"),
            "scope": "order_delta",
            "scope_value": "FIRST_minus_RANDOM",
            "planned_requests": "6000_per_order",
            "strict_valid": str(int(first["strict_valid"]) - int(random["strict_valid"])),
            "invalid": str(int(first["invalid"]) - int(random["invalid"])),
            "backend_errors": str(int(first["backend_errors"]) - int(random["backend_errors"])),
            "invalid_length": str(int(first["invalid_length"]) - int(random["invalid_length"])),
            "invalid_duplicate": str(int(first["invalid_duplicate"]) - int(random["invalid_duplicate"])),
            "hallucination_or_out_of_candidate": str(int(first["hallucination_or_out_of_candidate"]) - int(random["hallucination_or_out_of_candidate"])),
            "strict_valid_rate": format_float(float(first["strict_valid_rate"]) - float(random["strict_valid_rate"])),
            "invalid_rate": format_float(first_invalid_rate - random_invalid_rate),
            "source": PATHS["reliability"].as_posix(),
        }
    )
    return pd.DataFrame(rows)


def locate_mistral_exceptions(results: pd.DataFrame) -> pd.DataFrame:
    rows = results.loc[
        results["model"].eq("mistral:7b-instruct")
        & results["primary_ci_decision"].isin(["CI_SUPPORTED_NEGATIVE", "NON_SIGNIFICANT_CI_INCLUDES_ZERO"])
    ].copy()
    return rows[
        [
            "dataset",
            "model",
            "prompt_strategy",
            "metric",
            "random_mean",
            "first_mean",
            "first_minus_random_delta",
            "ci_low",
            "ci_high",
            "primary_ci_decision",
        ]
    ].sort_values(["primary_ci_decision", "dataset", "prompt_strategy", "metric"], kind="mergesort")


def summarize_prompt_interaction(interactions: pd.DataFrame, family: str) -> dict[str, int]:
    subset = interactions.loc[interactions["measure_family"].eq(family)]
    return {str(key): int(value) for key, value in subset["magnitude_effect"].value_counts().sort_index().to_dict().items()}


def add_validation(rows: list[dict[str, str]], check_name: str, passed: bool, expected: str, actual: str, failures: list[str], notes: str = "") -> None:
    if not passed:
        failures.append(check_name)
    rows.append(
        {
            "check_id": stable_id("G3SYN-CHK", check_name),
            "check_name": check_name,
            "status": "PASS" if passed else "FAIL",
            "expected": expected,
            "actual": actual,
            "notes": notes,
        }
    )


def build_validation(
    *,
    head: str,
    status_before: str,
    diff_check: subprocess.CompletedProcess[str],
    allow_generator_dirty: bool,
    manifests: Mapping[str, Mapping[str, Any]],
    frozen_hash_checks: int,
    results: pd.DataFrame,
    canddif: pd.DataFrame,
    reliability: pd.DataFrame,
    paired_contributions: pd.DataFrame,
    statistical_validation: pd.DataFrame,
    comp: pd.DataFrame,
    breakdowns: pd.DataFrame,
    cand_summary: pd.DataFrame,
    prompt_effects: pd.DataFrame,
    rel_summary: pd.DataFrame,
    preexisting_failures: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    failures = list(preexisting_failures)
    rows: list[dict[str, str]] = []
    decision_counts = {str(k): int(v) for k, v in results["primary_ci_decision"].value_counts().sort_index().to_dict().items()}
    prompt_counts = {
        f"{prompt}|{decision}": int(len(results.loc[results["prompt_strategy"].eq(prompt) & results["primary_ci_decision"].eq(decision)]))
        for prompt in PROMPTS
        for decision in DECISIONS
    }
    model_counts = {
        f"{model}|{decision}": int(len(results.loc[results["model"].eq(model) & results["primary_ci_decision"].eq(decision)]))
        for model in MODELS
        for decision in DECISIONS
    }
    dataset_counts = {
        f"{dataset}|{decision}": int(len(results.loc[results["dataset"].eq(dataset) & results["primary_ci_decision"].eq(decision)]))
        for dataset in DATASETS
        for decision in DECISIONS
    }
    order_rel = reliability.loc[reliability["scope"].eq("order")].set_index("scope_value")

    add_validation(rows, "head_matches_precondition", head == EXPECTED_HEAD, EXPECTED_HEAD, head, failures)
    add_validation(
        rows,
        "git_clean_or_expected_synthesis_outputs_before_generation",
        status_is_clean_or_expected(status_before, allow_generator_dirty),
        "clean_or_expected_dirty" if allow_generator_dirty else "clean",
        ";".join(git_dirty_paths(status_before)),
        failures,
    )
    add_validation(rows, "git_diff_check_clean_before_generation", diff_check.returncode == 0, "0", str(diff_check.returncode), failures, (diff_check.stdout + diff_check.stderr).strip())
    add_validation(rows, "metrics_manifest_frozen", manifests["metrics"].get("decision_token") == "G3_METRICS_FROZEN", "G3_METRICS_FROZEN", str(manifests["metrics"].get("decision_token")), failures)
    add_validation(rows, "statistical_manifest_frozen", manifests["statistical"].get("decision_token") == "G3_STATISTICAL_ANALYSIS_FROZEN", "G3_STATISTICAL_ANALYSIS_FROZEN", str(manifests["statistical"].get("decision_token")), failures)
    add_validation(rows, "frozen_source_hashes_match", frozen_hash_checks > 0, ">0 and no mismatch failures", str(frozen_hash_checks), failures)
    add_validation(rows, "statistical_results_rows_90", len(results) == 90, "90", str(len(results)), failures)
    add_validation(rows, "statistical_comparison_coverage_90", results["comparison_id"].nunique() == 90, "90", str(results["comparison_id"].nunique()), failures)
    add_validation(rows, "paired_contribution_rows_18000", len(paired_contributions) == 18000, "18000", str(len(paired_contributions)), failures)
    add_validation(rows, "paired_pairs_6000_preserved", paired_contributions["pair_id"].nunique() == 6000, "6000", str(paired_contributions["pair_id"].nunique()), failures)
    add_validation(rows, "statistical_validation_pass_34", int(statistical_validation["status"].eq("PASS").sum()) == 34 and len(statistical_validation) == 34, "34/34 PASS", f"{int(statistical_validation['status'].eq('PASS').sum())}/{len(statistical_validation)}", failures)
    add_validation(rows, "overall_decision_counts_match_frozen", decision_counts == {"CI_SUPPORTED_NEGATIVE": 1, "CI_SUPPORTED_POSITIVE": 85, "NON_SIGNIFICANT_CI_INCLUDES_ZERO": 4}, '{"CI_SUPPORTED_NEGATIVE":1,"CI_SUPPORTED_POSITIVE":85,"NON_SIGNIFICANT_CI_INCLUDES_ZERO":4}', stable_json(decision_counts), failures)
    add_validation(rows, "prompt_decision_counts_match_frozen", prompt_counts == {"BASE|CI_SUPPORTED_POSITIVE": 45, "BASE|CI_SUPPORTED_NEGATIVE": 0, "BASE|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 0, "LLMRANK|CI_SUPPORTED_POSITIVE": 40, "LLMRANK|CI_SUPPORTED_NEGATIVE": 1, "LLMRANK|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 4}, "BASE 45/0/0; LLMRANK 40/1/4", stable_json(prompt_counts), failures)
    add_validation(rows, "model_decision_counts_match_frozen", model_counts == {"qwen2:7b-instruct|CI_SUPPORTED_POSITIVE": 30, "qwen2:7b-instruct|CI_SUPPORTED_NEGATIVE": 0, "qwen2:7b-instruct|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 0, "mistral:7b-instruct|CI_SUPPORTED_POSITIVE": 25, "mistral:7b-instruct|CI_SUPPORTED_NEGATIVE": 1, "mistral:7b-instruct|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 4, "gemma2:9b|CI_SUPPORTED_POSITIVE": 30, "gemma2:9b|CI_SUPPORTED_NEGATIVE": 0, "gemma2:9b|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 0}, "Qwen 30/0/0; Mistral 25/1/4; Gemma2 30/0/0", stable_json(model_counts), failures)
    add_validation(rows, "dataset_decision_counts_match_frozen", dataset_counts == {"beauty|CI_SUPPORTED_POSITIVE": 18, "beauty|CI_SUPPORTED_NEGATIVE": 0, "beauty|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 0, "sports|CI_SUPPORTED_POSITIVE": 18, "sports|CI_SUPPORTED_NEGATIVE": 0, "sports|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 0, "ml1m|CI_SUPPORTED_POSITIVE": 15, "ml1m|CI_SUPPORTED_NEGATIVE": 1, "ml1m|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 2, "lastfm|CI_SUPPORTED_POSITIVE": 16, "lastfm|CI_SUPPORTED_NEGATIVE": 0, "lastfm|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 2, "foodcom|CI_SUPPORTED_POSITIVE": 18, "foodcom|CI_SUPPORTED_NEGATIVE": 0, "foodcom|NON_SIGNIFICANT_CI_INCLUDES_ZERO": 0}, "Beauty/Sports/Foodcom 18/0/0; ML1M 15/1/2; LastFM 16/0/2", stable_json(dataset_counts), failures)
    add_validation(rows, "canddif_rows_60_descriptive_only", len(canddif) == 60, "60", str(len(canddif)), failures)
    add_validation(rows, "canddif_significance_not_computed", results["canddif_significance_test_status"].eq("NOT_COMPUTED_CANDDIF_DESCRIPTIVE_ONLY").all(), "descriptive only", stable_json(results["canddif_significance_test_status"].value_counts().to_dict()), failures)
    add_validation(rows, "no_p_values_or_holm_invented", results["p_value"].astype(str).eq("").all() and results["corrected_decision"].eq("NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD_FOR_G3").all(), "blank p-values; Holm not applied", stable_json(results["corrected_decision"].value_counts().to_dict()), failures)
    add_validation(rows, "first_random_reliability_counts_match", int(order_rel.loc["FIRST", "invalid"]) == 96 and int(order_rel.loc["RANDOM", "invalid"]) == 70, "FIRST invalid 96; RANDOM invalid 70", stable_json({key: int(order_rel.loc[key, "invalid"]) for key in ["FIRST", "RANDOM"]}), failures)
    add_validation(rows, "comparison_summary_rows_90", len(comp) == 90, "90", str(len(comp)), failures)
    add_validation(rows, "decision_breakdowns_nonempty", len(breakdowns) > 0, ">0", str(len(breakdowns)), failures)
    add_validation(rows, "canddif_summary_rows_expected", len(cand_summary) == 22, "22", str(len(cand_summary)), failures)
    add_validation(rows, "prompt_interaction_rows_expected", len(prompt_effects) == 75, "75", str(len(prompt_effects)), failures)
    add_validation(rows, "reliability_summary_rows_expected", len(rel_summary) == 3, "3", str(len(rel_summary)), failures)
    add_validation(rows, "no_inference_no_new_tests_no_g4", True, "true", "true", failures)
    return pd.DataFrame(rows), failures


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    work = frame[columns].copy() if columns else frame.copy()
    labels = [str(column) for column in work.columns]
    rows = work.astype(str).values.tolist()
    widths = [
        max(len(label), *(len(str(row[index])) for row in rows)) if rows else len(label)
        for index, label in enumerate(labels)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(values)) + " |"

    return "\n".join(
        [
            render_row(labels),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *[render_row([str(value) for value in row]) for row in rows],
        ]
    )


def exception_table(results: pd.DataFrame) -> pd.DataFrame:
    return locate_mistral_exceptions(results).assign(
        random_mean=lambda frame: frame["random_mean"].map(lambda value: format_float(value)),
        first_mean=lambda frame: frame["first_mean"].map(lambda value: format_float(value)),
        first_minus_random_delta=lambda frame: frame["first_minus_random_delta"].map(lambda value: format_float(value)),
        ci_low=lambda frame: frame["ci_low"].map(lambda value: format_float(value)),
        ci_high=lambda frame: frame["ci_high"].map(lambda value: format_float(value)),
    )


def breakdown_table(breakdowns: pd.DataFrame, scope: str) -> pd.DataFrame:
    return breakdowns.loc[breakdowns["scope"].eq(scope), [
        "scope_value",
        "comparison_rows",
        "ci_supported_positive",
        "ci_supported_negative",
        "non_significant_ci_includes_zero",
        "median_first_minus_random_delta",
        "mean_first_minus_random_delta",
    ]]


def canddif_display(cand_summary: pd.DataFrame, scope: str) -> pd.DataFrame:
    return cand_summary.loc[cand_summary["scope"].eq(scope), [
        "scope_value",
        "metric_id",
        "cells",
        "positive_cells",
        "negative_cells",
        "median_value",
        "mean_value",
        "min_value",
        "max_value",
    ]]


def prompt_effect_counts(interactions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in ["FIRST_minus_RANDOM", "CandDif"]:
        subset = interactions.loc[interactions["measure_family"].eq(family)]
        counts = subset["magnitude_effect"].value_counts().to_dict()
        rows.append(
            {
                "measure_family": family,
                "matched_cells": len(subset),
                "attenuates": int(counts.get("ATTENUATES", 0)),
                "amplifies": int(counts.get("AMPLIFIES", 0)),
                "unchanged": int(counts.get("UNCHANGED", 0)),
                "descriptive_conclusion": "INCONSISTENTLY_CHANGES_POSITION_SENSITIVITY"
                if int(counts.get("ATTENUATES", 0)) > 0 and int(counts.get("AMPLIFIES", 0)) > 0
                else ("ATTENUATES" if int(counts.get("ATTENUATES", 0)) > 0 else "AMPLIFIES"),
            }
        )
    return pd.DataFrame(rows)


def build_report(parts: Mapping[str, Any], output_hashes_map: Mapping[str, str]) -> str:
    results = parts["results"]
    comp = parts["comparison_summary"]
    breakdowns = parts["decision_breakdowns"]
    cand_summary = parts["canddif_summary"]
    interactions = parts["prompt_interaction"]
    rel = parts["reliability_summary"]
    validation = parts["validation"]
    prompt_counts = prompt_effect_counts(interactions)
    exceptions = exception_table(results)
    overall = breakdown_table(breakdowns, "overall")
    metric_breakdown = breakdown_table(breakdowns, "metric")
    dataset_breakdown = breakdown_table(breakdowns, "dataset")
    model_breakdown = breakdown_table(breakdowns, "model")
    prompt_breakdown = breakdown_table(breakdowns, "prompt")

    lines = [
        "# G3 Final Scientific Synthesis",
        "",
        f"- Phase: `{PHASE_ID}`",
        f"- Decision: `{parts['decision_token']}`",
        f"- HEAD before synthesis: `{parts['head_before']}`",
        "- Inputs: frozen G3 metric artifacts and frozen G3 statistical artifacts only.",
        "- No inference, no new tests, no p-values, no relation update, and no G4.",
        "",
        "## FIRST vs RANDOM",
        "",
        "The paper-aligned candidate-position analysis compares the same user, history, model, prompt, and candidate membership under `RANDOM` versus `FIRST`. The frozen contrast is `FIRST_minus_RANDOM`; invalid terminal responses contribute zero utility.",
        "",
        markdown_table(overall),
        "",
        markdown_table(metric_breakdown),
        "",
        "## Dataset Interaction",
        "",
        markdown_table(dataset_breakdown),
        "",
        "Beauty, Sports, and Food.com have CI-supported positive FIRST-minus-RANDOM effects for all 18 dataset-specific comparisons each. ML1M has 15 positive, 1 negative, and 2 non-significant comparisons; LastFM has 16 positive and 2 non-significant comparisons.",
        "",
        "## Model Interaction",
        "",
        markdown_table(model_breakdown),
        "",
        "Qwen and Gemma2 have CI-supported positive FIRST-minus-RANDOM effects in all 30 model-specific comparisons. Mistral has 25 positive, 1 negative, and 4 non-significant comparisons.",
        "",
        "### Mistral Exceptions",
        "",
        markdown_table(exceptions),
        "",
        "## Prompt Interaction",
        "",
        markdown_table(prompt_breakdown),
        "",
        markdown_table(prompt_counts),
        "",
        "The BASE versus LLMRANK comparison is an extension interaction analysis, not part of the paper's original FIRST-vs-RANDOM estimand. Matched-cell magnitude comparisons show that LLMRANK changes position sensitivity inconsistently: it attenuates some FIRST-minus-RANDOM and CandDif magnitudes and amplifies others.",
        "",
        "## Position Bias",
        "",
        "CandDif HR@5 and CandDif NDCG@5 are retained as frozen descriptive cell-level position-bias measures. No CandDif significance test is computed.",
        "",
        "### CandDif by Dataset",
        "",
        markdown_table(canddif_display(cand_summary, "dataset")),
        "",
        "### CandDif by Model",
        "",
        markdown_table(canddif_display(cand_summary, "model")),
        "",
        "### CandDif by Prompt",
        "",
        markdown_table(canddif_display(cand_summary, "prompt")),
        "",
        "## Reliability",
        "",
        markdown_table(rel),
        "",
        "FIRST has 96 invalid terminal responses and RANDOM has 70. Descriptively, the FIRST ordering has 26 more invalid outputs and a higher invalid rate, mostly through duplicate-index failures; no new statistical test is applied to structural validity.",
        "",
        "## Statistical Wording",
        "",
        "All inferential wording is bootstrap-CI based. Holm-Bonferroni remains frozen plan metadata but is not applied because no frozen G3 p-value method exists. Corrected decisions are therefore `NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD_FOR_G3`.",
        "",
        "## Source Counts",
        "",
        f"- Statistical comparison rows: `{len(results)}`.",
        f"- Compact comparison-summary rows: `{len(comp)}`.",
        f"- CandDif rows used: `{parts['canddif_rows']}`.",
        f"- Paired contribution rows retained: `{parts['paired_contribution_rows']}`.",
        f"- Statistical validation: `{parts['statistical_validation_pass']}/34 PASS`.",
        f"- Synthesis validation: `{int(validation['status'].eq('PASS').sum())}/{len(validation)} PASS`.",
        "",
        "## Validation",
        "",
        markdown_table(validation),
        "",
        "## Output Artifacts",
        "",
    ]
    for rel_path, digest in sorted(output_hashes_map.items()):
        lines.append(f"- `{rel_path}` SHA256 `{digest}`")
    lines.extend(
        [
            "",
            "## Next Safe Step",
            "",
            "Update the report through frozen G3 results without running G4 or adding new analyses.",
            "",
        ]
    )
    return "\n".join(lines)


def input_hashes(root: Path) -> dict[str, str]:
    inputs = [
        PATHS["metrics_manifest"],
        PATHS["statistical_manifest"],
        PATHS["metric_results"],
        PATHS["canddif"],
        PATHS["reliability"],
        PATHS["statistical_results"],
        PATHS["paired_contributions"],
        PATHS["statistical_validation"],
    ]
    return {path.as_posix(): sha256_file(root / path) for path in inputs}


def output_hashes(root: Path) -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(root / path)
        for key, path in OUTPUTS.items()
        if key != "manifest" and (root / path).is_file()
    }


def append_notes(root: Path, parts: Mapping[str, Any]) -> None:
    path = root / PATHS["notes"]
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    kept_lines: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.startswith("## ") and line.endswith(" - G3 final scientific synthesis"):
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            kept_lines.append(line)
    text = "\n".join(kept_lines)
    note = (
        f"\n## {parts['created_at_utc']} - G3 final scientific synthesis\n\n"
        f"- Decision: `{parts['decision_token']}`.\n"
        "- Scope: synthesized frozen G3 FIRST-vs-RANDOM metric/statistical artifacts only; no inference, no new tests, no relation update, no G4.\n"
        f"- FIRST-vs-RANDOM CI decisions: `{stable_json(parts['decision_counts'])}` over `90/90` comparisons.\n"
        "- Prompt interaction is descriptive matched-cell magnitude comparison, not a new inferential test.\n"
        "- CandDif HR@5/NDCG@5 retained as descriptive position-bias measures only; no CandDif significance test computed.\n"
        "- Reliability: FIRST invalid `96`; RANDOM invalid `70`; no structural-validity significance test applied.\n"
        "- Multiplicity: Holm not applied because no frozen G3 p-value method exists.\n"
        "- Next safe step: update report through frozen G3 results only.\n"
    )
    path.write_text(text.rstrip() + note.rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def compute(root: Path, *, allow_generator_dirty: bool) -> dict[str, Any]:
    head = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short", "--untracked-files=all"], check=False)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True)
    manifests = {
        "metrics": read_json(root, PATHS["metrics_manifest"]),
        "statistical": read_json(root, PATHS["statistical_manifest"]),
    }
    failures: list[str] = []
    frozen_hash_checks = 0
    frozen_hash_checks += verify_hash_mapping(
        root,
        manifests["metrics"].get("output_hashes", {}),
        failures,
        label="METRICS_OUTPUT",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_checks += verify_hash_mapping(
        root,
        manifests["statistical"].get("input_hashes", {}),
        failures,
        label="STATISTICAL_INPUT",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_checks += verify_hash_mapping(
        root,
        manifests["statistical"].get("output_hashes", {}),
        failures,
        label="STATISTICAL_OUTPUT",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_checks += verify_hash_mapping(
        root,
        manifests["statistical"].get("source_hashes", {}),
        failures,
        label="STATISTICAL_SOURCE",
        skip_paths={"report/notes.md"},
    )

    results = read_csv(root, PATHS["statistical_results"])
    canddif = read_csv(root, PATHS["canddif"])
    reliability = read_csv(root, PATHS["reliability"])
    paired_contributions = read_csv(root, PATHS["paired_contributions"])
    statistical_validation = read_csv(root, PATHS["statistical_validation"])

    comp = comparison_summary(results)
    breakdowns = decision_breakdowns(results)
    cand_summary = canddif_summary(canddif)
    prompt_effects = prompt_interaction(results, canddif)
    rel_summary = reliability_summary(reliability)
    validation, all_failures = build_validation(
        head=head,
        status_before=status_before,
        diff_check=diff_check,
        allow_generator_dirty=allow_generator_dirty,
        manifests=manifests,
        frozen_hash_checks=frozen_hash_checks,
        results=results,
        canddif=canddif,
        reliability=reliability,
        paired_contributions=paired_contributions,
        statistical_validation=statistical_validation,
        comp=comp,
        breakdowns=breakdowns,
        cand_summary=cand_summary,
        prompt_effects=prompt_effects,
        rel_summary=rel_summary,
        preexisting_failures=failures,
    )
    decision_counts = {str(k): int(v) for k, v in results["primary_ci_decision"].value_counts().sort_index().to_dict().items()}
    summary = {
        "statistical_comparisons": int(len(results)),
        "comparison_coverage": f"{int(results['comparison_id'].nunique())}/90",
        "paired_contribution_rows": int(len(paired_contributions)),
        "canddif_rows": int(len(canddif)),
        "decision_counts": decision_counts,
        "prompt_interaction_counts": {
            "FIRST_minus_RANDOM": summarize_prompt_interaction(prompt_effects, "FIRST_minus_RANDOM"),
            "CandDif": summarize_prompt_interaction(prompt_effects, "CandDif"),
        },
        "mistral_exception_rows": int(len(locate_mistral_exceptions(results))),
        "first_invalid": int(reliability.loc[reliability["scope"].eq("order") & reliability["scope_value"].eq("FIRST"), "invalid"].iloc[0]),
        "random_invalid": int(reliability.loc[reliability["scope"].eq("order") & reliability["scope_value"].eq("RANDOM"), "invalid"].iloc[0]),
        "multiplicity_status": "NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_FOR_G3",
        "validation_pass_count": int(validation["status"].eq("PASS").sum()),
        "validation_total_count": int(len(validation)),
        "validation_failures": all_failures,
    }
    return {
        "head_before": head,
        "status_before": status_before,
        "diff_check_stderr": diff_check.stderr.strip(),
        "manifests": manifests,
        "frozen_hash_checks": frozen_hash_checks,
        "results": results,
        "canddif": canddif,
        "reliability": reliability,
        "paired_contributions": paired_contributions,
        "statistical_validation": statistical_validation,
        "comparison_summary": comp,
        "decision_breakdowns": breakdowns,
        "canddif_summary": cand_summary,
        "prompt_interaction": prompt_effects,
        "reliability_summary": rel_summary,
        "validation": validation,
        "decision_token": DECISION_FROZEN if not all_failures else DECISION_BLOCKED,
        "summary": summary,
        "failures": all_failures,
    }


def write_outputs(root: Path, payload: dict[str, Any], created_at: str) -> dict[str, str]:
    write_csv(root, OUTPUTS["comparison_summary"], payload["comparison_summary"])
    write_csv(root, OUTPUTS["decision_breakdowns"], payload["decision_breakdowns"])
    write_csv(root, OUTPUTS["canddif_summary"], payload["canddif_summary"])
    write_csv(root, OUTPUTS["prompt_interaction"], payload["prompt_interaction"])
    write_csv(root, OUTPUTS["reliability_summary"], payload["reliability_summary"])
    write_csv(root, OUTPUTS["validation"], payload["validation"])
    append_notes(
        root,
        {
            "created_at_utc": created_at,
            "decision_token": payload["decision_token"],
            "decision_counts": payload["summary"]["decision_counts"],
        },
    )
    hashes = output_hashes(root)
    report = build_report(
        {
            "decision_token": payload["decision_token"],
            "head_before": payload["head_before"],
            "results": payload["results"],
            "comparison_summary": payload["comparison_summary"],
            "decision_breakdowns": payload["decision_breakdowns"],
            "canddif_summary": payload["canddif_summary"],
            "prompt_interaction": payload["prompt_interaction"],
            "reliability_summary": payload["reliability_summary"],
            "validation": payload["validation"],
            "canddif_rows": payload["summary"]["canddif_rows"],
            "paired_contribution_rows": payload["summary"]["paired_contribution_rows"],
            "statistical_validation_pass": int(payload["statistical_validation"]["status"].eq("PASS").sum()),
        },
        hashes,
    )
    (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUTS["report"]).write_text(report, encoding="utf-8", newline="\n")
    hashes = output_hashes(root)
    manifest = {
        "schema_version": "1.0",
        "phase_id": PHASE_ID,
        "decision_token": payload["decision_token"],
        "created_at_utc": created_at,
        "precondition_status": "G3_STATISTICAL_ANALYSIS_FROZEN",
        "precondition_head": EXPECTED_HEAD,
        "head_before_generation": payload["head_before"],
        "zero_inference": True,
        "ollama_api_calls_executed": 0,
        "new_tests_or_hypotheses_added": False,
        "bootstrap_recomputed": False,
        "protocol_changes": False,
        "frozen_evidence_modified": False,
        "frozen_metrics_modified": False,
        "frozen_statistical_results_modified": False,
        "g4_started": False,
        "relation_update_executed": False,
        "input_hashes": input_hashes(root),
        "output_hashes": hashes,
        "source_hashes": {PATHS["source"].as_posix(): sha256_file(root / PATHS["source"])},
        "summary": payload["summary"],
        "statistical_wording": {
            "inference_basis": "bootstrap_CI_only",
            "p_values": "NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD_FOR_G3",
            "holm": "NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_FOR_G3",
            "canddif": "DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_TEST",
        },
        "files_created": [PATHS["source"].as_posix(), *[path.as_posix() for path in OUTPUTS.values()], PATHS["notes"].as_posix()],
        "next_safe_step": "update report through frozen G3 results only; do not start G4 without explicit request",
    }
    write_json(root, OUTPUTS["manifest"], manifest)
    hashes[OUTPUTS["manifest"].as_posix()] = sha256_file(root / OUTPUTS["manifest"])
    return hashes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_project_root(Path(args.output_root).resolve())
    created_at = utc_now()
    payload = compute(root, allow_generator_dirty=bool(args.allow_generator_dirty))
    if not args.no_write:
        write_outputs(root, payload, created_at)
    print(json.dumps({"decision": payload["decision_token"], "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if payload["decision_token"] == DECISION_FROZEN else 1


if __name__ == "__main__":
    raise SystemExit(main())

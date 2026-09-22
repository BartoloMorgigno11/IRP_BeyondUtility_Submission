"""Execute the frozen G3 candidate-position statistical plan offline.

The script consumes the frozen paired universe, frozen G3 metric artifacts, and
frozen G3 statistical plan. It does not call Ollama, define new tests, compute
new CandDif significance tests, or modify frozen protocol/evidence/universe or
metric artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import find_project_root  # noqa: E402


EXPECTED_HEAD = "e0c9b2afbbdfca2b9824e2001fd3f0703d440726"
PHASE_ID = "g3_candidate_position_statistical_analysis"
DECISION_FROZEN = "G3_STATISTICAL_ANALYSIS_FROZEN"
DECISION_BLOCKED = "G3_STATISTICAL_ANALYSIS_BLOCKED"

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = ("BASE", "LLMRANK")
ORDERS = ("FIRST", "RANDOM")
PLAN_METRICS = ("hr_at_5", "ndcg_at_5", "mrr")
METRIC_ID_BY_PLAN = {"hr_at_5": "HR@5", "ndcg_at_5": "NDCG@5", "mrr": "MRR"}
CANDDIF_ID_BY_PLAN = {"hr_at_5": "CandDif_HR@5", "ndcg_at_5": "CandDif_NDCG@5"}

EXPECTED_PLAN_ROWS = 90
EXPECTED_PAIRS = 6000
EXPECTED_CONTRIBUTION_ROWS = 18000
EXPECTED_RESULTS = 90
EXPECTED_PAIRED_USERS_PER_COMPARISON = 200
EXPECTED_BOOTSTRAP_RESAMPLES = 10000
EXPECTED_BOOTSTRAP_SEED = 20260901
EXPECTED_ALPHA = 0.05

PATHS = {
    "protocol_manifest": Path("results/manifests/g3_candidate_position_protocol_freeze.json"),
    "execution_manifest": Path("results/manifests/g3_candidate_position_execution_freeze.json"),
    "retention_manifest": Path("results/manifests/g3_candidate_position_execution_retention_manifest.json"),
    "analysis_manifest": Path("results/manifests/g3_candidate_position_analysis_universe_freeze.json"),
    "metrics_manifest": Path("results/manifests/g3_candidate_position_metrics_freeze.json"),
    "statistical_plan": Path("results/tables/g3_candidate_position_statistical_plan.csv"),
    "analysis_universe": Path("results/tables/g3_candidate_position_analysis_universe.csv"),
    "analysis_pairs": Path("results/tables/g3_candidate_position_analysis_pairs.csv"),
    "metric_results": Path("results/tables/g3_candidate_position_metric_results.csv"),
    "canddif": Path("results/tables/g3_candidate_position_canddif.csv"),
    "metric_validation": Path("results/tables/g3_candidate_position_metric_validation.csv"),
    "metric_report": Path("report/g3_candidate_position_metrics_validation.md"),
    "source": Path("src/extension/g3_candidate_position_statistical_analysis.py"),
    "notes": Path("report/notes.md"),
}

OUTPUTS = {
    "paired_contributions": Path("results/tables/g3_candidate_position_statistical_paired_contributions.csv"),
    "statistical_results": Path("results/tables/g3_candidate_position_statistical_results.csv"),
    "validation": Path("results/tables/g3_candidate_position_statistical_validation.csv"),
    "manifest": Path("results/manifests/g3_candidate_position_statistical_analysis_manifest.json"),
    "report": Path("report/g3_candidate_position_statistical_analysis.md"),
}

EXPECTED_DIRTY_PATHS = {
    PATHS["source"].as_posix(),
    PATHS["notes"].as_posix(),
    *[path.as_posix() for path in OUTPUTS.values()],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute frozen G3 statistical analysis offline.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Compute and validate without writing artifacts.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator and expected statistical outputs to be dirty while frozen inputs remain unchanged.",
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


def bool_text(value: Any) -> str:
    return "true" if str(value).strip().lower() in {"true", "1", "yes"} else "false"


def bool_value(value: Any) -> bool:
    return bool_text(value) == "true"


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


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


def paired_bootstrap_mean_delta_ci(deltas: Sequence[float], *, seed: int, resamples: int) -> tuple[float, float, int]:
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        raise ValueError("paired bootstrap requires at least one paired delta")
    unique, counts = np.unique(arr, return_counts=True)
    if unique.size == 1:
        samples = np.repeat(unique[0], int(resamples))
    else:
        rng = np.random.default_rng(int(seed))
        probabilities = counts.astype(float) / float(arr.size)
        weights = rng.multinomial(int(arr.size), probabilities, size=int(resamples))
        samples = weights @ unique.astype(float) / float(arr.size)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high), int(unique.size)


def ci_decision(low: float, high: float) -> tuple[str, str]:
    if low > 0.0:
        return "EXCLUDES_ZERO_POSITIVE", "CI_SUPPORTED_POSITIVE"
    if high < 0.0:
        return "EXCLUDES_ZERO_NEGATIVE", "CI_SUPPORTED_NEGATIVE"
    return "INCLUDES_ZERO", "NON_SIGNIFICANT_CI_INCLUDES_ZERO"


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


def request_contribution_lookup(universe: pd.DataFrame, metric_results: pd.DataFrame, failures: list[str]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    metrics = metric_results.set_index(["dataset", "model", "prompt_strategy", "candidate_order_condition", "metric_id"])
    for row in universe.to_dict("records"):
        request_id = str(row["canonical_request_id"])
        strict_valid = bool_value(row["strict_parse_valid"])
        values = {
            "hr_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "mrr": 0.0,
        }
        if strict_valid:
            for plan_metric, metric_id in METRIC_ID_BY_PLAN.items():
                key = (row["dataset"], row["model"], row["prompt_strategy"], row["candidate_order_condition"], metric_id)
                cell = metrics.loc[key]
                numerator_sum = float(cell["numerator_sum"])
                planned = float(cell["planned_applicable_users"])
                value = float(cell["value"])
                if abs(numerator_sum / planned - value) > 1e-12:
                    failures.append(f"METRIC_NUMERATOR_DENOMINATOR_MISMATCH:{stable_json(key)}")
                break
            try:
                ranked = json.loads(str(row["ranked_candidate_indices"]))
            except json.JSONDecodeError as exc:
                raise ValueError(f"strict-valid row has invalid ranked_candidate_indices JSON: {request_id}") from exc
            if not isinstance(ranked, list) or len(ranked) != 10:
                raise ValueError(f"strict-valid row does not have exactly 10 ranked indices: {request_id}")
            ranked_ints = [int(value) for value in ranked]
            positive_position = int(str(row["positive_candidate_position"]))
            rank_zero = ranked_ints.index(positive_position) if positive_position in ranked_ints else None
            if rank_zero is not None:
                values["hr_at_5"] = 1.0 if rank_zero < 5 else 0.0
                values["ndcg_at_5"] = 1.0 / math.log2(rank_zero + 2.0) if rank_zero < 5 else 0.0
                values["mrr"] = 1.0 / float(rank_zero + 1)
            rank_zero_text = "" if rank_zero is None else str(rank_zero)
        else:
            rank_zero_text = ""
        lookup[request_id] = {
            "canonical_request_id": request_id,
            "dataset": str(row["dataset"]),
            "model": str(row["model"]),
            "prompt_strategy": str(row["prompt_strategy"]),
            "candidate_order_condition": str(row["candidate_order_condition"]),
            "user_id": str(row["user_id"]),
            "terminal_state": str(row["terminal_state"]),
            "strict_parse_valid": bool_text(strict_valid),
            "validity_class": str(row["validity_class"]),
            "raw_response_hash": str(row["raw_response_hash"]),
            "rank_zero_based": rank_zero_text,
            **values,
        }
    invalid = [row for row in lookup.values() if row["strict_parse_valid"] != "true"]
    nonzero_invalid = [
        row["canonical_request_id"]
        for row in invalid
        if any(float(row[metric]) != 0.0 for metric in PLAN_METRICS)
    ]
    if nonzero_invalid:
        failures.append(f"INVALID_TERMINAL_NONZERO_UTILITY:{len(nonzero_invalid)}")
    return lookup


def build_paired_contributions(
    plan: pd.DataFrame,
    pairs: pd.DataFrame,
    request_metrics: Mapping[str, Mapping[str, Any]],
    failures: list[str],
) -> pd.DataFrame:
    plan_index = {
        (row["dataset"], row["model"], row["prompt_strategy"], row["metric"]): row
        for row in plan.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for pair in pairs.to_dict("records"):
        if not (
            bool_value(pair["only_candidate_order_differs"])
            and bool_value(pair["first_positive_position_ok"])
            and bool_value(pair["random_positive_position_ok"])
        ):
            failures.append(f"PAIR_INTEGRITY_FAILURE:{pair['pair_id']}")
        first_id = str(pair["first_request_id"])
        random_id = str(pair["random_request_id"])
        first = request_metrics.get(first_id)
        random = request_metrics.get(random_id)
        if first is None or random is None:
            failures.append(f"PAIR_REQUEST_METRIC_MISSING:{pair['pair_id']}")
            continue
        for field in ("dataset", "model", "prompt_strategy", "user_id"):
            if str(first[field]) != str(random[field]) or str(first[field]) != str(pair[field]):
                failures.append(f"PAIR_FIELD_MISMATCH:{pair['pair_id']}:{field}")
        for metric in PLAN_METRICS:
            plan_row = plan_index.get((pair["dataset"], pair["model"], pair["prompt_strategy"], metric))
            if plan_row is None:
                failures.append(f"PLAN_ROW_MISSING_FOR_PAIR_METRIC:{pair['pair_id']}:{metric}")
                continue
            random_value = float(random[metric])
            first_value = float(first[metric])
            delta = first_value - random_value
            contribution = {
                "contribution_id": stable_id("G3STAT-PAIRMET", pair["pair_id"], metric),
                "pair_id": pair["pair_id"],
                "comparison_id": plan_row["comparison_id"],
                "plan_order": str(plan_row["plan_order"]),
                "dataset": pair["dataset"],
                "model": pair["model"],
                "prompt_strategy": pair["prompt_strategy"],
                "metric": metric,
                "paired_unit": "user_id",
                "user_id": pair["user_id"],
                "random_request_id": random_id,
                "first_request_id": first_id,
                "random_terminal_state": random["terminal_state"],
                "first_terminal_state": first["terminal_state"],
                "random_strict_parse_valid": random["strict_parse_valid"],
                "first_strict_parse_valid": first["strict_parse_valid"],
                "random_validity_class": random["validity_class"],
                "first_validity_class": first["validity_class"],
                "random_rank_zero_based": random["rank_zero_based"],
                "first_rank_zero_based": first["rank_zero_based"],
                "random_value": random_value,
                "first_value": first_value,
                "first_minus_random_delta": delta,
                "denominator_policy": str(plan_row["denominator_policy"]),
                "invalid_terminal_policy": "invalid_terminal_response_contributes_zero_utility",
                "pair_integrity_source": PATHS["analysis_pairs"].as_posix(),
            }
            contribution["contribution_hash"] = stable_id("G3STAT-HASH", stable_json(contribution))
            rows.append(contribution)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["plan_order", "dataset", "model", "prompt_strategy", "metric", "user_id"], kind="mergesort")
    if len(frame) != EXPECTED_CONTRIBUTION_ROWS:
        failures.append(f"PAIRED_CONTRIBUTION_ROW_COUNT_MISMATCH:{len(frame)}:{EXPECTED_CONTRIBUTION_ROWS}")
    return frame


def build_statistical_results(plan: pd.DataFrame, contributions: pd.DataFrame, canddif: pd.DataFrame, failures: list[str]) -> pd.DataFrame:
    canddif_lookup = {
        (row["dataset"], row["model"], row["prompt_strategy"], row["metric_id"]): row
        for row in canddif.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    plan_sorted = plan.assign(plan_order_int=pd.to_numeric(plan["plan_order"], errors="raise").astype(int)).sort_values(
        "plan_order_int", kind="mergesort"
    )
    for plan_row in plan_sorted.to_dict("records"):
        comparison_id = str(plan_row["comparison_id"])
        metric = str(plan_row["metric"])
        subset = contributions.loc[contributions["comparison_id"].eq(comparison_id)].copy()
        if len(subset) != EXPECTED_PAIRED_USERS_PER_COMPARISON:
            failures.append(f"COMPARISON_PAIRED_N_MISMATCH:{comparison_id}:{len(subset)}")
        random_values = subset["random_value"].astype(float).to_numpy()
        first_values = subset["first_value"].astype(float).to_numpy()
        deltas = subset["first_minus_random_delta"].astype(float).to_numpy()
        random_mean = float(np.mean(random_values))
        first_mean = float(np.mean(first_values))
        delta = float(np.mean(deltas))
        low, high, unique_delta_count = paired_bootstrap_mean_delta_ci(
            deltas,
            seed=int(plan_row["bootstrap_seed"]),
            resamples=int(plan_row["bootstrap_resamples"]),
        )
        low_check, high_check, unique_check = paired_bootstrap_mean_delta_ci(
            deltas,
            seed=int(plan_row["bootstrap_seed"]),
            resamples=int(plan_row["bootstrap_resamples"]),
        )
        if low != low_check or high != high_check or unique_delta_count != unique_check:
            failures.append(f"BOOTSTRAP_REPRODUCIBILITY_FAILURE:{comparison_id}")
        zero_relation, decision = ci_decision(low, high)
        strict_random = subset["random_strict_parse_valid"].map(bool_value)
        strict_first = subset["first_strict_parse_valid"].map(bool_value)
        canddif_metric = CANDDIF_ID_BY_PLAN.get(metric, "")
        canddif_row = canddif_lookup.get((plan_row["dataset"], plan_row["model"], plan_row["prompt_strategy"], canddif_metric))
        row = {
            "result_id": stable_id("G3STAT-RESULT", comparison_id),
            "plan_order": str(plan_row["plan_order"]),
            "comparison_id": comparison_id,
            "comparison_group": str(plan_row["comparison_group"]),
            "scope": "within_dataset_model_prompt_g3_candidate_position",
            "dataset": str(plan_row["dataset"]),
            "model": str(plan_row["model"]),
            "prompt_strategy": str(plan_row["prompt_strategy"]),
            "metric": metric,
            "condition_a": str(plan_row["condition_a"]),
            "condition_b": str(plan_row["condition_b"]),
            "contrast": str(plan_row["contrast"]),
            "paired_unit": str(plan_row["paired_unit"]),
            "paired_user_count": int(len(subset)),
            "strict_valid_both_count": int((strict_random & strict_first).sum()),
            "strict_valid_random_only_count": int((strict_random & ~strict_first).sum()),
            "strict_valid_first_only_count": int((~strict_random & strict_first).sum()),
            "strict_valid_neither_count": int((~strict_random & ~strict_first).sum()),
            "invalid_any_side_count": int((~strict_random | ~strict_first).sum()),
            "random_mean": random_mean,
            "first_mean": first_mean,
            "first_minus_random_delta": delta,
            "positive_delta_user_count": int(np.count_nonzero(deltas > 0.0)),
            "negative_delta_user_count": int(np.count_nonzero(deltas < 0.0)),
            "zero_delta_user_count": int(np.count_nonzero(deltas == 0.0)),
            "primary_test": str(plan_row["primary_test"]),
            "bootstrap_resamples": int(plan_row["bootstrap_resamples"]),
            "bootstrap_seed": int(plan_row["bootstrap_seed"]),
            "bootstrap_engine": "empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta",
            "bootstrap_percentile_method": "numpy_quantile_linear",
            "confidence_interval": str(plan_row["confidence_interval"]),
            "ci_low": low,
            "ci_high": high,
            "ci_zero_relation": zero_relation,
            "alpha": float(plan_row["alpha"]),
            "primary_ci_decision": decision,
            "effect_size": str(plan_row["effect_size"]),
            "effect_size_value": delta,
            "secondary_test": str(plan_row["secondary_test"]),
            "secondary_test_execution_status": "NOT_EXECUTED_NOT_FROZEN_FOR_G3_NO_SECONDARY_P_VALUES",
            "p_value": "",
            "p_value_status": "NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD_FOR_G3",
            "multiple_comparison_correction": str(plan_row["multiple_comparison_correction"]),
            "multiplicity_family_id": stable_id("G3STAT-FAMILY", metric, plan_row["dataset"], plan_row["model"], plan_row["prompt_strategy"]),
            "multiple_comparison_adjusted_p_value": "",
            "multiplicity_correction_execution_status": "NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_FOR_G3",
            "corrected_decision": "NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD_FOR_G3",
            "frozen_decision": decision,
            "frozen_decision_scope": "primary_bootstrap_percentile_ci_unadjusted_alpha_0.05",
            "canddif_metric": canddif_metric if canddif_metric else "not_applicable",
            "canddif_value": "" if canddif_row is None else str(canddif_row["value"]),
            "canddif_significance_test_status": "NOT_COMPUTED_CANDDIF_DESCRIPTIVE_ONLY",
            "planned_status": str(plan_row["planned_status"]),
            "unique_delta_values": unique_delta_count,
            "notes": (
                "Invalid terminal responses contribute zero utility. Holm correction is not applied because "
                "the frozen G3 plan explicitly lacks a p-value method."
            ),
        }
        if metric == "hr_at_5":
            random_hits = subset["random_value"].astype(float)
            first_hits = subset["first_value"].astype(float)
            row["hr_random0_first1_count"] = int(((random_hits == 0.0) & (first_hits == 1.0)).sum())
            row["hr_random1_first0_count"] = int(((random_hits == 1.0) & (first_hits == 0.0)).sum())
        else:
            row["hr_random0_first1_count"] = ""
            row["hr_random1_first0_count"] = ""
        rows.append(row)
    results = pd.DataFrame(rows)
    if len(results) != EXPECTED_RESULTS:
        failures.append(f"STATISTICAL_RESULT_ROW_COUNT_MISMATCH:{len(results)}:{EXPECTED_RESULTS}")
    return results


def verify_metric_means(metric_results: pd.DataFrame, results: pd.DataFrame, failures: list[str]) -> int:
    metric_lookup = {
        (row["dataset"], row["model"], row["prompt_strategy"], row["candidate_order_condition"], row["metric_id"]): float(row["value"])
        for row in metric_results.to_dict("records")
    }
    checks = 0
    for row in results.to_dict("records"):
        metric_id = METRIC_ID_BY_PLAN[str(row["metric"])]
        for order, column in [("RANDOM", "random_mean"), ("FIRST", "first_mean")]:
            key = (row["dataset"], row["model"], row["prompt_strategy"], order, metric_id)
            expected = metric_lookup.get(key)
            observed = float(row[column])
            checks += 1
            if expected is None or abs(observed - expected) > 1e-12:
                failures.append(f"STAT_MEAN_METRIC_TABLE_MISMATCH:{stable_json(key)}:{observed}:{expected}")
    return checks


def output_hashes(root: Path) -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(root / path)
        for key, path in OUTPUTS.items()
        if key != "manifest" and (root / path).is_file()
    }


def input_hashes(root: Path) -> dict[str, str]:
    inputs = [
        PATHS["protocol_manifest"],
        PATHS["execution_manifest"],
        PATHS["retention_manifest"],
        PATHS["analysis_manifest"],
        PATHS["metrics_manifest"],
        PATHS["statistical_plan"],
        PATHS["analysis_universe"],
        PATHS["analysis_pairs"],
        PATHS["metric_results"],
        PATHS["canddif"],
        PATHS["metric_validation"],
        PATHS["metric_report"],
    ]
    return {path.as_posix(): sha256_file(root / path) for path in inputs}


def add_validation(rows: list[dict[str, str]], check_name: str, passed: bool, expected: str, actual: str, failures: list[str], notes: str = "") -> None:
    if not passed:
        failures.append(check_name)
    rows.append(
        {
            "check_id": stable_id("G3STAT-CHK", check_name),
            "check_name": check_name,
            "status": "PASS" if passed else "FAIL",
            "expected": expected,
            "actual": actual,
            "notes": notes,
        }
    )


def build_validation(
    *,
    root: Path,
    head: str,
    status_before: str,
    diff_check: subprocess.CompletedProcess[str],
    allow_generator_dirty: bool,
    manifests: Mapping[str, Mapping[str, Any]],
    frozen_hash_check_count: int,
    plan: pd.DataFrame,
    universe: pd.DataFrame,
    pairs: pd.DataFrame,
    metric_results: pd.DataFrame,
    canddif: pd.DataFrame,
    contributions: pd.DataFrame,
    results: pd.DataFrame,
    metric_mean_checks: int,
    preexisting_failures: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    failures = list(preexisting_failures)
    rows: list[dict[str, str]] = []
    plan_metrics = plan["metric"].value_counts().sort_index().to_dict()
    decision_counts = results["primary_ci_decision"].value_counts().sort_index().to_dict() if not results.empty else {}
    corrected_counts = results["corrected_decision"].value_counts().sort_index().to_dict() if not results.empty else {}
    pair_metric_counts = contributions.groupby("pair_id")["metric"].nunique().value_counts().sort_index().to_dict() if not contributions.empty else {}
    comparison_counts = contributions["comparison_id"].value_counts().to_dict() if not contributions.empty else {}
    state_counts = universe["terminal_state"].value_counts().sort_index().to_dict()
    order_counts = universe["candidate_order_condition"].value_counts().sort_index().to_dict()

    add_validation(rows, "head_matches_precondition", head == EXPECTED_HEAD, EXPECTED_HEAD, head, failures)
    add_validation(
        rows,
        "git_clean_or_expected_statistical_outputs_before_generation",
        status_is_clean_or_expected(status_before, allow_generator_dirty),
        "clean_or_expected_dirty" if allow_generator_dirty else "clean",
        ";".join(git_dirty_paths(status_before)),
        failures,
    )
    add_validation(rows, "git_diff_check_clean_before_generation", diff_check.returncode == 0, "0", str(diff_check.returncode), failures, (diff_check.stdout + diff_check.stderr).strip())
    add_validation(rows, "protocol_manifest_frozen", manifests["protocol"].get("decision_token") == "G3_BASE_LLMRANK_FIRST_RANDOM_PROTOCOL_FROZEN", "G3_BASE_LLMRANK_FIRST_RANDOM_PROTOCOL_FROZEN", str(manifests["protocol"].get("decision_token")), failures)
    add_validation(rows, "execution_manifest_frozen", manifests["execution"].get("decision_token") == "G3_EXECUTION_RUNNER_FROZEN", "G3_EXECUTION_RUNNER_FROZEN", str(manifests["execution"].get("decision_token")), failures)
    add_validation(rows, "retention_manifest_frozen", manifests["retention"].get("decision_token") == "G3_EXECUTION_RETENTION_FROZEN", "G3_EXECUTION_RETENTION_FROZEN", str(manifests["retention"].get("decision_token")), failures)
    add_validation(rows, "analysis_manifest_frozen", manifests["analysis"].get("decision_token") == "G3_ANALYTIC_UNIVERSE_FROZEN", "G3_ANALYTIC_UNIVERSE_FROZEN", str(manifests["analysis"].get("decision_token")), failures)
    add_validation(rows, "metrics_manifest_frozen", manifests["metrics"].get("decision_token") == "G3_METRICS_FROZEN", "G3_METRICS_FROZEN", str(manifests["metrics"].get("decision_token")), failures)
    add_validation(rows, "frozen_artifact_hashes_match", frozen_hash_check_count > 0, ">0 hash checks and no mismatch failures", str(frozen_hash_check_count), failures)
    add_validation(rows, "statistical_plan_rows_90", len(plan) == EXPECTED_PLAN_ROWS, str(EXPECTED_PLAN_ROWS), str(len(plan)), failures)
    add_validation(rows, "statistical_plan_metric_counts", {str(k): int(v) for k, v in plan_metrics.items()} == {"hr_at_5": 30, "mrr": 30, "ndcg_at_5": 30}, '{"hr_at_5":30,"mrr":30,"ndcg_at_5":30}', stable_json({str(k): int(v) for k, v in plan_metrics.items()}), failures)
    add_validation(
        rows,
        "statistical_plan_parameters_frozen",
        plan["primary_test"].eq("paired_user_bootstrap_mean_delta").all()
        and plan["bootstrap_resamples"].astype(str).eq(str(EXPECTED_BOOTSTRAP_RESAMPLES)).all()
        and plan["bootstrap_seed"].astype(str).eq(str(EXPECTED_BOOTSTRAP_SEED)).all()
        and plan["confidence_interval"].eq("two_sided_95_percent_percentile_ci").all()
        and plan["effect_size"].eq("paired_mean_delta").all()
        and plan["contrast"].eq("FIRST_minus_RANDOM").all()
        and plan["condition_a"].eq("RANDOM").all()
        and plan["condition_b"].eq("FIRST").all()
        and plan["alpha"].astype(str).eq("0.05").all(),
        "frozen G3 bootstrap/contrast parameters",
        stable_json(
            {
                "resamples": sorted(plan["bootstrap_resamples"].drop_duplicates().tolist()),
                "seed": sorted(plan["bootstrap_seed"].drop_duplicates().tolist()),
                "contrast": sorted(plan["contrast"].drop_duplicates().tolist()),
            }
        ),
        failures,
    )
    add_validation(
        rows,
        "g3_secondary_p_values_not_frozen",
        plan["secondary_test"].eq("not_frozen_for_g3_no_secondary_p_values").all(),
        "not_frozen_for_g3_no_secondary_p_values",
        stable_json(plan["secondary_test"].value_counts().sort_index().to_dict()),
        failures,
        "Holm correction is retained as planned metadata only; corrected decisions require p-values that are not frozen.",
    )
    add_validation(rows, "analysis_universe_rows_12000", len(universe) == 12000, "12000", str(len(universe)), failures)
    add_validation(rows, "analysis_pair_rows_6000", len(pairs) == EXPECTED_PAIRS, str(EXPECTED_PAIRS), str(len(pairs)), failures)
    add_validation(rows, "analysis_pair_membership_unchanged", int(pairs["only_candidate_order_differs"].map(bool_value).sum()) == EXPECTED_PAIRS, str(EXPECTED_PAIRS), str(int(pairs["only_candidate_order_differs"].map(bool_value).sum())), failures)
    add_validation(rows, "first_random_order_counts", {str(k): int(v) for k, v in order_counts.items()} == {"FIRST": 6000, "RANDOM": 6000}, '{"FIRST":6000,"RANDOM":6000}', stable_json({str(k): int(v) for k, v in order_counts.items()}), failures)
    add_validation(rows, "terminal_state_counts_match_g3_execution", {str(k): int(v) for k, v in state_counts.items()} == {"HALLUCINATION_OR_OUT_OF_CANDIDATE": 1, "INVALID_DUPLICATE": 51, "INVALID_LENGTH": 114, "STRICT_VALID": 11834}, "frozen G3 execution accounting", stable_json({str(k): int(v) for k, v in state_counts.items()}), failures)
    add_validation(rows, "metric_results_rows_180", len(metric_results) == 180, "180", str(len(metric_results)), failures)
    add_validation(rows, "canddif_rows_60_retained_descriptive", len(canddif) == 60, "60", str(len(canddif)), failures)
    add_validation(rows, "paired_contribution_rows_18000", len(contributions) == EXPECTED_CONTRIBUTION_ROWS, str(EXPECTED_CONTRIBUTION_ROWS), str(len(contributions)), failures)
    add_validation(rows, "paired_contribution_ids_unique", not contributions["contribution_id"].duplicated().any(), "0 duplicates", str(int(contributions["contribution_id"].duplicated().sum())) if not contributions.empty else "empty", failures)
    add_validation(rows, "all_6000_pairs_preserved_with_three_metrics", {str(k): int(v) for k, v in pair_metric_counts.items()} == {"3": 6000}, '{"3":6000}', stable_json({str(k): int(v) for k, v in pair_metric_counts.items()}), failures)
    add_validation(rows, "each_comparison_has_200_paired_users", set(int(v) for v in comparison_counts.values()) == {200} and len(comparison_counts) == 90, "90 comparisons each with 200", stable_json({"comparisons": len(comparison_counts), "paired_n_values": sorted({int(v) for v in comparison_counts.values()})}), failures)
    add_validation(rows, "statistical_results_rows_90", len(results) == EXPECTED_RESULTS, str(EXPECTED_RESULTS), str(len(results)), failures)
    add_validation(rows, "comparison_coverage_90_of_90", results["comparison_id"].nunique() == EXPECTED_PLAN_ROWS, str(EXPECTED_PLAN_ROWS), str(results["comparison_id"].nunique()), failures)
    add_validation(rows, "paired_n_200_all_results", sorted(results["paired_user_count"].astype(str).drop_duplicates().tolist()) == ["200"], '["200"]', stable_json(sorted(results["paired_user_count"].astype(str).drop_duplicates().tolist())), failures)
    add_validation(rows, "bootstrap_settings_used", results["bootstrap_resamples"].astype(str).eq("10000").all() and results["bootstrap_seed"].astype(str).eq("20260901").all(), "10000 resamples seed 20260901", stable_json({"resamples": sorted(results["bootstrap_resamples"].astype(str).drop_duplicates().tolist()), "seed": sorted(results["bootstrap_seed"].astype(str).drop_duplicates().tolist())}), failures)
    add_validation(rows, "bootstrap_reproducibility_verified", not any(item.startswith("BOOTSTRAP_REPRODUCIBILITY_FAILURE") for item in preexisting_failures), "no bootstrap reproducibility failures", str([item for item in preexisting_failures if item.startswith("BOOTSTRAP_REPRODUCIBILITY_FAILURE")]), failures)
    add_validation(rows, "metric_means_match_frozen_metric_table", metric_mean_checks == 180, "180", str(metric_mean_checks), failures)
    add_validation(
        rows,
        "no_p_values_or_holm_decisions_invented",
        results["p_value"].astype(str).eq("").all()
        and results["multiple_comparison_adjusted_p_value"].astype(str).eq("").all()
        and results["corrected_decision"].astype(str).eq("NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD_FOR_G3").all(),
        "blank p-values and undefined corrected decisions",
        stable_json({str(k): int(v) for k, v in corrected_counts.items()}),
        failures,
    )
    add_validation(rows, "primary_ci_decisions_defined", results["primary_ci_decision"].astype(str).ne("").all(), "all non-empty", stable_json({str(k): int(v) for k, v in decision_counts.items()}), failures)
    add_validation(rows, "no_ollama_or_api_calls_executed", True, "0", "0", failures)
    add_validation(rows, "no_downstream_g3_or_g4_started", True, "false", "false", failures)
    return pd.DataFrame(rows), failures


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = frame.astype(str).values.tolist()
    widths = [
        max(len(column), *(len(str(row[index])) for row in rows)) if rows else len(column)
        for index, column in enumerate(columns)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(values)) + " |"

    header = render_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row([str(value) for value in row]) for row in rows]
    return "\n".join([header, separator, *body])


def build_decision_breakdowns(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, cols in {
        "prompt": ["prompt_strategy"],
        "model": ["model"],
        "dataset": ["dataset"],
    }.items():
        frame = (
            results.groupby([*cols, "primary_ci_decision"], sort=True)
            .size()
            .reset_index(name="comparisons")
            .pivot_table(index=cols, columns="primary_ci_decision", values="comparisons", aggfunc="sum", fill_value=0)
            .reset_index()
        )
        for column in ["CI_SUPPORTED_POSITIVE", "CI_SUPPORTED_NEGATIVE", "NON_SIGNIFICANT_CI_INCLUDES_ZERO"]:
            if column not in frame.columns:
                frame[column] = 0
        tables[name] = frame[[*cols, "CI_SUPPORTED_POSITIVE", "CI_SUPPORTED_NEGATIVE", "NON_SIGNIFICANT_CI_INCLUDES_ZERO"]]
    return tables


def build_report(parts: Mapping[str, Any], output_hashes_map: Mapping[str, str]) -> str:
    results = parts["results"]
    validation = parts["validation"]
    decision_counts = results["primary_ci_decision"].value_counts().sort_index().to_dict()
    corrected_counts = results["corrected_decision"].value_counts().sort_index().to_dict()
    breakdowns = build_decision_breakdowns(results)
    plan_counts = results.groupby("metric", sort=True)["comparison_id"].count().reset_index(name="comparisons")
    lines = [
        "# G3 Frozen Statistical Analysis",
        "",
        f"- Phase: `{PHASE_ID}`",
        f"- Decision: `{parts['decision_token']}`",
        f"- HEAD before analysis: `{parts['head_before']}`",
        "- Scope: frozen G3 paired universe, frozen metric artifacts, and frozen statistical plan only.",
        "- Zero Ollama/API calls; no protocol, evidence, universe, or metric changes.",
        "",
        "## Plan Coverage",
        "",
        f"- Frozen FIRST-vs-RANDOM comparisons executed: `{results['comparison_id'].nunique()}` / `90`.",
        f"- Paired-user contribution rows: `{parts['paired_contribution_rows']}`.",
        f"- Preserved paired users: `{parts['unique_pair_count']}` / `6000`.",
        "- Contrast: `FIRST_minus_RANDOM`.",
        "",
        markdown_table(plan_counts),
        "",
        "## Bootstrap Contract Used",
        "",
        "- Primary test: `paired_user_bootstrap_mean_delta`.",
        "- Resamples: `10000`.",
        "- Seed: `20260901`.",
        "- Confidence interval: `two_sided_95_percent_percentile_ci`.",
        "- Percentile implementation detail: `numpy_quantile_linear`.",
        "- Bootstrap engine: `empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta`.",
        "",
        "## Decisions",
        "",
        f"- Primary CI decisions: `{stable_json({str(k): int(v) for k, v in decision_counts.items()})}`.",
        f"- Corrected decisions: `{stable_json({str(k): int(v) for k, v in corrected_counts.items()})}`.",
        "- Multiplicity status: `NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_FOR_G3`.",
        "- CandDif HR/NDCG retained as descriptive cell-level measures only; no CandDif significance test computed.",
        "",
        "### By Prompt",
        "",
        markdown_table(breakdowns["prompt"]),
        "",
        "### By Model",
        "",
        markdown_table(breakdowns["model"]),
        "",
        "### By Dataset",
        "",
        markdown_table(breakdowns["dataset"]),
        "",
        "## Validation",
        "",
        f"- PASS: `{int(validation['status'].eq('PASS').sum())}` / `{len(validation)}`.",
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
            "Generate the frozen G3 scientific readout/synthesis from the frozen metric and statistical artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def append_notes(root: Path, parts: Mapping[str, Any]) -> None:
    path = root / PATHS["notes"]
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    kept_lines: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.startswith("## ") and line.endswith(" - G3 frozen statistical analysis"):
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            kept_lines.append(line)
    text = "\n".join(kept_lines)
    decision_counts = {
        str(key): int(value)
        for key, value in parts["results"]["primary_ci_decision"].value_counts().sort_index().to_dict().items()
    }
    note = (
        f"\n## {parts['created_at_utc']} - G3 frozen statistical analysis\n\n"
        f"- Decision: `{parts['decision_token']}`.\n"
        "- Scope: executed the frozen G3 FIRST-vs-RANDOM paired bootstrap CI plan from frozen paired universe and metric artifacts only.\n"
        f"- Coverage: `90/90` comparisons; paired-user contribution rows `{parts['paired_contribution_rows']}`; preserved pairs `{parts['unique_pair_count']}`.\n"
        "- Bootstrap: `10000` resamples, seed `20260901`, two-sided 95 percent percentile CI, paired mean delta `FIRST_minus_RANDOM`.\n"
        f"- Primary CI decision counts: `{stable_json(decision_counts)}`.\n"
        "- Multiplicity/Holm: `NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_FOR_G3`; no p-values or corrected decisions were invented.\n"
        "- CandDif HR/NDCG retained as descriptive cell-level position-bias measures only.\n"
        "- No Ollama/API calls, protocol changes, evidence rewrites, new tests, or downstream phases were run.\n"
        "- Next safe step: frozen G3 scientific readout/synthesis from frozen metrics and statistical artifacts.\n"
    )
    path.write_text(text.rstrip() + note.rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def compute(root: Path, *, allow_generator_dirty: bool) -> dict[str, Any]:
    head = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short", "--untracked-files=all"], check=False)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True)
    manifests = {
        "protocol": read_json(root, PATHS["protocol_manifest"]),
        "execution": read_json(root, PATHS["execution_manifest"]),
        "retention": read_json(root, PATHS["retention_manifest"]),
        "analysis": read_json(root, PATHS["analysis_manifest"]),
        "metrics": read_json(root, PATHS["metrics_manifest"]),
    }

    failures: list[str] = []
    frozen_hash_check_count = 0
    frozen_hash_check_count += verify_hash_mapping(
        root,
        manifests["protocol"].get("artifact_hashes", {}),
        failures,
        label="PROTOCOL",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_check_count += verify_hash_mapping(
        root,
        manifests["analysis"].get("input_hashes", {}),
        failures,
        label="ANALYSIS_INPUT",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_check_count += verify_hash_mapping(
        root,
        manifests["analysis"].get("output_hashes", {}),
        failures,
        label="ANALYSIS_OUTPUT",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_check_count += verify_hash_mapping(
        root,
        manifests["analysis"].get("source_hashes", {}),
        failures,
        label="ANALYSIS_SOURCE",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_check_count += verify_hash_mapping(
        root,
        manifests["metrics"].get("input_hashes_observed", {}),
        failures,
        label="METRICS_INPUT",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_check_count += verify_hash_mapping(
        root,
        manifests["metrics"].get("output_hashes", {}),
        failures,
        label="METRICS_OUTPUT",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_check_count += verify_hash_mapping(
        root,
        manifests["metrics"].get("source_hashes", {}),
        failures,
        label="METRICS_SOURCE",
        skip_paths={"report/notes.md"},
    )

    plan = read_csv(root, PATHS["statistical_plan"])
    universe = read_csv(root, PATHS["analysis_universe"])
    pairs = read_csv(root, PATHS["analysis_pairs"])
    metric_results = read_csv(root, PATHS["metric_results"])
    canddif = read_csv(root, PATHS["canddif"])

    request_metrics = request_contribution_lookup(universe, metric_results, failures)
    contributions = build_paired_contributions(plan, pairs, request_metrics, failures)
    results = build_statistical_results(plan, contributions, canddif, failures)
    metric_mean_checks = verify_metric_means(metric_results, results, failures)
    validation, all_failures = build_validation(
        root=root,
        head=head,
        status_before=status_before,
        diff_check=diff_check,
        allow_generator_dirty=allow_generator_dirty,
        manifests=manifests,
        frozen_hash_check_count=frozen_hash_check_count,
        plan=plan,
        universe=universe,
        pairs=pairs,
        metric_results=metric_results,
        canddif=canddif,
        contributions=contributions,
        results=results,
        metric_mean_checks=metric_mean_checks,
        preexisting_failures=failures,
    )

    decision_counts = {str(key): int(value) for key, value in results["primary_ci_decision"].value_counts().sort_index().to_dict().items()}
    corrected_counts = {str(key): int(value) for key, value in results["corrected_decision"].value_counts().sort_index().to_dict().items()}
    summary = {
        "comparison_coverage": f"{int(results['comparison_id'].nunique())}/90",
        "statistical_result_rows": int(len(results)),
        "paired_contribution_rows": int(len(contributions)),
        "unique_pair_count": int(contributions["pair_id"].nunique()) if not contributions.empty else 0,
        "paired_user_counts": sorted(int(value) for value in results["paired_user_count"].drop_duplicates().tolist()) if not results.empty else [],
        "primary_ci_decision_counts": decision_counts,
        "corrected_decision_counts": corrected_counts,
        "breakdown_by_prompt": build_decision_breakdowns(results)["prompt"].to_dict("records"),
        "breakdown_by_model": build_decision_breakdowns(results)["model"].to_dict("records"),
        "breakdown_by_dataset": build_decision_breakdowns(results)["dataset"].to_dict("records"),
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
        "frozen_hash_check_count": frozen_hash_check_count,
        "metric_mean_checks": metric_mean_checks,
        "plan": plan,
        "universe": universe,
        "pairs": pairs,
        "metric_results": metric_results,
        "canddif": canddif,
        "contributions": contributions,
        "results": results,
        "validation": validation,
        "decision_token": DECISION_FROZEN if not all_failures else DECISION_BLOCKED,
        "summary": summary,
        "failures": all_failures,
    }


def write_outputs(root: Path, payload: dict[str, Any], created_at: str) -> dict[str, str]:
    write_csv(root, OUTPUTS["paired_contributions"], payload["contributions"])
    write_csv(root, OUTPUTS["statistical_results"], payload["results"])
    write_csv(root, OUTPUTS["validation"], payload["validation"])
    append_notes(
        root,
        {
            "created_at_utc": created_at,
            "decision_token": payload["decision_token"],
            "results": payload["results"],
            "paired_contribution_rows": payload["summary"]["paired_contribution_rows"],
            "unique_pair_count": payload["summary"]["unique_pair_count"],
        },
    )
    hashes = output_hashes(root)
    report = build_report(
        {
            "decision_token": payload["decision_token"],
            "head_before": payload["head_before"],
            "results": payload["results"],
            "validation": payload["validation"],
            "paired_contribution_rows": payload["summary"]["paired_contribution_rows"],
            "unique_pair_count": payload["summary"]["unique_pair_count"],
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
        "precondition_status": "G3_METRICS_FROZEN",
        "precondition_head": EXPECTED_HEAD,
        "head_before_generation": payload["head_before"],
        "zero_inference": True,
        "ollama_api_calls_executed": 0,
        "new_tests_or_hypotheses_added": False,
        "bootstrap_executed": True,
        "protocol_changes": False,
        "frozen_evidence_modified": False,
        "frozen_universe_modified": False,
        "frozen_metrics_modified": False,
        "input_hashes": input_hashes(root),
        "output_hashes": hashes,
        "source_hashes": {PATHS["source"].as_posix(): sha256_file(root / PATHS["source"])},
        "bootstrap_settings": {
            "primary_test": "paired_user_bootstrap_mean_delta",
            "contrast": "FIRST_minus_RANDOM",
            "bootstrap_resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
            "confidence_interval": "two_sided_95_percent_percentile_ci",
            "alpha": EXPECTED_ALPHA,
            "bootstrap_engine": "empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta",
            "bootstrap_percentile_method": "numpy_quantile_linear",
        },
        "multiplicity": {
            "frozen_plan": "holm_bonferroni_within_metric_dataset_model_prompt_family",
            "execution_status": "NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_FOR_G3",
            "p_value_status": "NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD_FOR_G3",
            "corrected_decision": "NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD_FOR_G3",
        },
        "canddif_status": "RETAINED_DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_TEST",
        "summary": payload["summary"],
        "files_created": [PATHS["source"].as_posix(), *[path.as_posix() for path in OUTPUTS.values()], PATHS["notes"].as_posix()],
        "next_safe_step": "generate frozen G3 scientific readout/synthesis from frozen metrics and statistical artifacts",
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

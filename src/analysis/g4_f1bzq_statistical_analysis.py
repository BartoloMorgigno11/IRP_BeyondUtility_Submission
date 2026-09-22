"""Execute the frozen G4 F1BZQ statistical comparison plan offline.

This phase consumes the frozen G4 statistical comparison table, frozen G4
metric artifacts, and frozen G4 analytic universe. It reconstructs user-level
paired contributions where the frozen plan requires paired bootstrap and keeps
G1-vs-G4 rows descriptive only. It does not call Ollama/API, add tests, compute
p-values, apply Holm correction, or modify frozen evidence/protocol/universe/
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


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from src.utils.config import find_project_root  # noqa: E402


EXPECTED_HEAD = "41ac7fe548ba68b4a55f002e1ba408677d6fb2a0"
PHASE_ID = "g4_f1bzq_statistical_analysis"
DECISION_FROZEN = "G4_STATISTICAL_ANALYSIS_FROZEN"
DECISION_BLOCKED = "G4_STATISTICAL_ANALYSIS_BLOCKED"

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = ("BASE", "LLMRANK")
MODEL_PAIRS = (
    ("qwen2:7b-instruct", "mistral:7b-instruct"),
    ("qwen2:7b-instruct", "gemma2:9b"),
    ("mistral:7b-instruct", "gemma2:9b"),
)
EXPECTED_USERS = {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}
EXPECTED_POSITIVE_PRESENT = {"beauty": 49, "sports": 28, "ml1m": 193, "lastfm": 469, "foodcom": 4}
EXPECTED_PLAN_ROWS = 75
EXPECTED_FAMILY_COUNTS = {
    "g4_prompt_strategy": 15,
    "g4_model_pair": 30,
    "g1_vs_g4_pipeline_descriptive": 30,
}
EXPECTED_INFERENTIAL_TEMPLATES = 45
EXPECTED_DESCRIPTIVE_TEMPLATES = 30
EXPECTED_LOGICAL_ROWS = 29292
EXPECTED_USERS_TOTAL = 4882
EXPECTED_CELLS = 30
EXPECTED_BOOTSTRAP_RESAMPLES = 10000
EXPECTED_BOOTSTRAP_SEED = 20260901
EXPECTED_ALPHA = 0.05
TOP_K = 5
OUTPUT_RANKING_LENGTH = 10

INFERENTIAL_METRICS = ("hr_at_5", "ndcg_at_5", "mrr", "conditional_hr_at_5", "conditional_ndcg_at_5")
DESCRIPTIVE_METRICS = ("hr_at_5", "ndcg_at_5", "mrr")
UNCONDITIONAL_METRICS = ("hr_at_5", "ndcg_at_5", "mrr")
CONDITIONAL_METRICS = ("conditional_hr_at_5", "conditional_ndcg_at_5")

PATHS = {
    "statistical_plan": Path("results/tables/g4_f1bzq_statistical_comparisons.csv"),
    "metric_results": Path("results/tables/g4_f1bzq_metric_results.csv"),
    "metric_denominators": Path("results/tables/g4_f1bzq_metric_denominators.csv"),
    "candidate_conditional": Path("results/tables/g4_f1bzq_candidate_conditional_metrics.csv"),
    "reliability": Path("results/tables/g4_f1bzq_reliability.csv"),
    "undefined_blocked": Path("results/tables/g4_f1bzq_metric_undefined_blocked.csv"),
    "metric_validation": Path("results/tables/g4_f1bzq_metric_validation.csv"),
    "metrics_manifest": Path("results/manifests/g4_f1bzq_metrics_freeze.json"),
    "analytic_universe": Path("results/tables/g4_f1bzq_analytic_universe.csv"),
    "analytic_cell_counts": Path("results/tables/g4_f1bzq_analytic_cell_counts.csv"),
    "analytic_manifest": Path("results/manifests/g4_f1bzq_analytic_universe_freeze.json"),
    "protocol_manifest": Path("results/manifests/g4_f1bzq_reranking_protocol_freeze.json"),
    "g1_manifest": Path("results/manifests/g1_primary_ranking_execution_manifest.json"),
    "g1_utility": Path("results/tables/g1_primary_ranking_utility.csv"),
    "source": Path("src/analysis/g4_f1bzq_statistical_analysis.py"),
    "notes": Path("report/notes.md"),
}

OUTPUTS = {
    "paired_contributions": Path("results/tables/g4_f1bzq_statistical_paired_contributions.csv"),
    "statistical_results": Path("results/tables/g4_f1bzq_statistical_results.csv"),
    "validation": Path("results/tables/g4_f1bzq_statistical_validation.csv"),
    "manifest": Path("results/manifests/g4_f1bzq_statistical_analysis_manifest.json"),
    "report": Path("report/g4_f1bzq_statistical_analysis.md"),
}

ALLOWED_DIRTY_PATHS = {
    PATHS["source"],
    PATHS["notes"],
    *OUTPUTS.values(),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute frozen G4 statistical analysis offline.")
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


def int_value(value: Any, default: int = 0) -> int:
    text = str(value).strip()
    return default if text == "" else int(text)


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def parse_json_array(value: Any) -> list[str]:
    if str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("expected JSON array")
    return [str(item) for item in parsed]


def parse_int_json_array(value: Any) -> list[int]:
    return [int(item) for item in parse_json_array(value)]


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


def comparison_template_id(row_number: int, row: Mapping[str, Any]) -> str:
    return stable_id(
        "G4STAT-TEMPLATE",
        row_number,
        row["comparison_family"],
        row["dataset"],
        row["model_scope"],
        row["prompt_scope"],
        row["comparison"],
    )


def result_id(template_id: str, metric_id: str) -> str:
    return stable_id("G4STAT-RESULT", template_id, metric_id)


def paired_bootstrap_mean_delta_ci(deltas: Sequence[float], *, seed: int, resamples: int) -> tuple[float, float, int]:
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        raise ValueError("paired bootstrap requires at least one paired delta")
    unique, counts = np.unique(arr, return_counts=True)
    if unique.size == 1:
        samples = np.repeat(float(unique[0]), int(resamples))
    else:
        rng = np.random.default_rng(int(seed))
        probabilities = counts.astype(float) / float(arr.size)
        weights = rng.multinomial(int(arr.size), probabilities, size=int(resamples))
        samples = weights @ unique.astype(float) / float(arr.size)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high), int(unique.size)


def ci_decision(low: float, high: float) -> tuple[str, str]:
    if low > 0.0:
        return "CI_EXCLUDES_ZERO_POSITIVE", "CI_SUPPORTED_POSITIVE"
    if high < 0.0:
        return "CI_EXCLUDES_ZERO_NEGATIVE", "CI_SUPPORTED_NEGATIVE"
    return "CI_INCLUDES_ZERO", "NON_SIGNIFICANT_CI_INCLUDES_ZERO"


def compute_user_contributions(universe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    defects: list[str] = []
    for row in universe.to_dict("records"):
        strict_valid = bool_value(row["strict_parse_valid"])
        positive_present = bool_value(row["positive_present_in_pool"])
        hr = 0.0
        ndcg = 0.0
        mrr = 0.0
        rank_zero_based = ""
        if strict_valid:
            try:
                ranked_indices = parse_int_json_array(row["parsed_top10_candidate_indices_if_strict_valid"])
                ranked_items = parse_json_array(row["parsed_top10_item_ids_if_strict_valid"])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                defects.append(f"{row['canonical_request_id']}: strict-valid output parse failed: {exc}")
                ranked_indices = []
                ranked_items = []
            if len(ranked_indices) != OUTPUT_RANKING_LENGTH or len(set(ranked_indices)) != OUTPUT_RANKING_LENGTH:
                defects.append(f"{row['canonical_request_id']}: strict-valid output does not contain 10 unique candidate indices")
            if len(ranked_items) != OUTPUT_RANKING_LENGTH:
                defects.append(f"{row['canonical_request_id']}: strict-valid output does not contain 10 item ids")
            if positive_present:
                positive_position = int_value(row["positive_input_position"], default=-1)
                if positive_position < 1:
                    defects.append(f"{row['canonical_request_id']}: positive-present row lacks positive input position")
                elif positive_position in ranked_indices:
                    rank = ranked_indices.index(positive_position)
                    rank_zero_based = str(rank)
                    mrr = 1.0 / float(rank + 1)
                    if rank < TOP_K:
                        hr = 1.0
                        ndcg = 1.0 / math.log2(rank + 2)
        rows.append(
            {
                "canonical_request_id": row["canonical_request_id"],
                "dataset": row["dataset"],
                "user_id": row["user_id"],
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "candidate_set_id": row["candidate_set_id"],
                "pool_id": row["pool_id"],
                "pool_hash": row["pool_hash"],
                "candidate_hash": row["candidate_hash"],
                "history_hash": row["history_hash"],
                "positive_item_id": row["positive_item_id"],
                "positive_present_in_pool": bool_text(positive_present),
                "positive_input_position": row["positive_input_position"],
                "terminal_state": row["terminal_state"],
                "strict_parse_valid": bool_text(strict_valid),
                "invalid_class": row["invalid_class"],
                "hr_at_5": hr,
                "ndcg_at_5": ndcg,
                "mrr": mrr,
                "conditional_hr_at_5": hr,
                "conditional_ndcg_at_5": ndcg,
                "positive_rank_zero_based": rank_zero_based,
            }
        )
    frame = pd.DataFrame(rows)
    return frame, defects


def parse_model_pair(model_scope: str) -> tuple[str, str]:
    left, separator, right = str(model_scope).partition("_vs_")
    if separator != "_vs_" or not left or not right:
        raise ValueError(f"cannot parse model pair: {model_scope}")
    return left, right


def arm_frames_for_template(template: Mapping[str, Any], contributions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    family = str(template["comparison_family"])
    dataset = str(template["dataset"])
    if family == "g4_prompt_strategy":
        model = str(template["model_scope"])
        arm_a = contributions[
            (contributions["dataset"] == dataset)
            & (contributions["model"] == model)
            & (contributions["prompt_strategy"] == "BASE")
        ].copy()
        arm_b = contributions[
            (contributions["dataset"] == dataset)
            & (contributions["model"] == model)
            & (contributions["prompt_strategy"] == "LLMRANK")
        ].copy()
        labels = {
            "arm_a_label": "BASE",
            "arm_b_label": "LLMRANK",
            "arm_a_model": model,
            "arm_b_model": model,
            "arm_a_prompt": "BASE",
            "arm_b_prompt": "LLMRANK",
            "delta_direction": "LLMRANK_minus_BASE",
        }
        return arm_a, arm_b, labels
    if family == "g4_model_pair":
        left_model, right_model = parse_model_pair(str(template["model_scope"]))
        prompt = str(template["prompt_scope"])
        arm_a = contributions[
            (contributions["dataset"] == dataset)
            & (contributions["model"] == left_model)
            & (contributions["prompt_strategy"] == prompt)
        ].copy()
        arm_b = contributions[
            (contributions["dataset"] == dataset)
            & (contributions["model"] == right_model)
            & (contributions["prompt_strategy"] == prompt)
        ].copy()
        labels = {
            "arm_a_label": left_model,
            "arm_b_label": right_model,
            "arm_a_model": left_model,
            "arm_b_model": right_model,
            "arm_a_prompt": prompt,
            "arm_b_prompt": prompt,
            "delta_direction": f"{right_model}_minus_{left_model}",
        }
        return arm_a, arm_b, labels
    raise ValueError(f"template is not inferential: {family}")


def merge_arms(arm_a: pd.DataFrame, arm_b: pd.DataFrame) -> pd.DataFrame:
    join_columns = ["dataset", "user_id"]
    columns = [
        "canonical_request_id",
        "dataset",
        "user_id",
        "model",
        "prompt_strategy",
        "candidate_set_id",
        "pool_id",
        "pool_hash",
        "candidate_hash",
        "history_hash",
        "positive_item_id",
        "positive_present_in_pool",
        "positive_input_position",
        "terminal_state",
        "strict_parse_valid",
        "invalid_class",
        "hr_at_5",
        "ndcg_at_5",
        "mrr",
        "conditional_hr_at_5",
        "conditional_ndcg_at_5",
        "positive_rank_zero_based",
    ]
    return arm_a[columns].merge(
        arm_b[columns],
        on=join_columns,
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )


def build_paired_contributions(plan: pd.DataFrame, user_contrib: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    defects: list[str] = []
    plan_with_order = plan.copy()
    plan_with_order["template_row_number"] = np.arange(1, len(plan_with_order) + 1)
    for template in plan_with_order.to_dict("records"):
        if not bool_value(template["inferential_test_allowed"]):
            continue
        template_id = comparison_template_id(int(template["template_row_number"]), template)
        arm_a, arm_b, labels = arm_frames_for_template(template, user_contrib)
        merged = merge_arms(arm_a, arm_b)
        expected_all_n = EXPECTED_USERS[str(template["dataset"])]
        if len(merged) != expected_all_n:
            defects.append(f"{template_id}: paired all-user n {len(merged)} != {expected_all_n}")
        identity_fields = ["candidate_set_id", "pool_id", "pool_hash", "candidate_hash", "history_hash", "positive_item_id"]
        for field in identity_fields:
            if not merged[f"{field}_a"].astype(str).eq(merged[f"{field}_b"].astype(str)).all():
                defects.append(f"{template_id}: arm identity mismatch in {field}")
        if not merged["positive_present_in_pool_a"].astype(str).eq(merged["positive_present_in_pool_b"].astype(str)).all():
            defects.append(f"{template_id}: positive-present mismatch")
        if not merged["positive_input_position_a"].astype(str).eq(merged["positive_input_position_b"].astype(str)).all():
            defects.append(f"{template_id}: positive input position mismatch")

        for metric_id in INFERENTIAL_METRICS:
            metric_subset = merged
            denominator_policy = "all planned users; invalid terminal response contributes zero"
            denominator_source = "planned users"
            if metric_id in CONDITIONAL_METRICS:
                metric_subset = merged[merged["positive_present_in_pool_a"].map(bool_value)].copy()
                denominator_policy = "positive-present users only; invalid terminal response contributes zero within subset"
                denominator_source = "positive-present users"
                expected_conditional_n = EXPECTED_POSITIVE_PRESENT[str(template["dataset"])]
                if len(metric_subset) != expected_conditional_n:
                    defects.append(f"{template_id}:{metric_id}: conditional paired n {len(metric_subset)} != {expected_conditional_n}")
                if not metric_subset["positive_present_in_pool_b"].map(bool_value).all():
                    defects.append(f"{template_id}:{metric_id}: conditional subset contains positive-absent arm-b rows")
            if metric_id in UNCONDITIONAL_METRICS and len(metric_subset) != expected_all_n:
                defects.append(f"{template_id}:{metric_id}: unconditional paired n {len(metric_subset)} != {expected_all_n}")

            for row in metric_subset.to_dict("records"):
                arm_a_value = float(row[f"{metric_id}_a"])
                arm_b_value = float(row[f"{metric_id}_b"])
                delta = arm_b_value - arm_a_value
                pair_identity = stable_id(
                    "G4STAT-PAIR",
                    row["dataset"],
                    row["user_id"],
                    row["candidate_set_id_a"],
                    row["pool_hash_a"],
                    row["candidate_hash_a"],
                )
                contribution = {
                    "contribution_id": stable_id("G4STAT-CONTRIB", template_id, metric_id, row["user_id"]),
                    "comparison_template_id": template_id,
                    "template_row_number": int(template["template_row_number"]),
                    "comparison_family": template["comparison_family"],
                    "comparison": template["comparison"],
                    "dataset": template["dataset"],
                    "model_scope": template["model_scope"],
                    "prompt_scope": template["prompt_scope"],
                    "metric_id": metric_id,
                    "paired_unit": template["paired_unit"],
                    "pair_identity": pair_identity,
                    "user_id": row["user_id"],
                    "candidate_set_id": row["candidate_set_id_a"],
                    "pool_id": row["pool_id_a"],
                    "pool_hash": row["pool_hash_a"],
                    "candidate_hash": row["candidate_hash_a"],
                    "history_hash": row["history_hash_a"],
                    "positive_item_id": row["positive_item_id_a"],
                    "positive_present_in_pool": row["positive_present_in_pool_a"],
                    "positive_input_position": row["positive_input_position_a"],
                    "denominator_policy": denominator_policy,
                    "denominator_source": denominator_source,
                    "arm_a_label": labels["arm_a_label"],
                    "arm_b_label": labels["arm_b_label"],
                    "arm_a_model": labels["arm_a_model"],
                    "arm_b_model": labels["arm_b_model"],
                    "arm_a_prompt": labels["arm_a_prompt"],
                    "arm_b_prompt": labels["arm_b_prompt"],
                    "arm_a_canonical_request_id": row["canonical_request_id_a"],
                    "arm_b_canonical_request_id": row["canonical_request_id_b"],
                    "arm_a_terminal_state": row["terminal_state_a"],
                    "arm_b_terminal_state": row["terminal_state_b"],
                    "arm_a_strict_parse_valid": row["strict_parse_valid_a"],
                    "arm_b_strict_parse_valid": row["strict_parse_valid_b"],
                    "arm_a_positive_rank_zero_based": row["positive_rank_zero_based_a"],
                    "arm_b_positive_rank_zero_based": row["positive_rank_zero_based_b"],
                    "arm_a_value": format_float(arm_a_value),
                    "arm_b_value": format_float(arm_b_value),
                    "arm_b_minus_arm_a_delta": format_float(delta),
                    "delta_direction": labels["delta_direction"],
                    "invalid_terminal_policy": "invalid terminal responses are terminal evidence and contribute zero to these utility metrics",
                    "small_conditional_denominator_flag": bool_text(
                        template["dataset"] == "foodcom" and metric_id in CONDITIONAL_METRICS
                    ),
                }
                contribution["contribution_hash"] = stable_id("G4STAT-CONTRIBHASH", stable_json(contribution))
                rows.append(contribution)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["template_row_number", "metric_id", "user_id"], kind="mergesort")
    return frame, defects


def g4_metric_lookup(metric_results: pd.DataFrame) -> dict[tuple[str, str, str, str], float]:
    lookup: dict[tuple[str, str, str, str], float] = {}
    for row in metric_results.to_dict("records"):
        lookup[(row["dataset"], row["model"], row["prompt_strategy"], row["metric_id"])] = float(row["value"]) if row["value"] != "" else math.nan
    return lookup


def g1_metric_lookup(g1_utility: pd.DataFrame) -> dict[tuple[str, str, str, str], tuple[float, int]]:
    lookup: dict[tuple[str, str, str, str], tuple[float, int]] = {}
    for row in g1_utility.to_dict("records"):
        lookup[(row["dataset"], row["model_name"], row["prompt_variant"], row["metric"])] = (
            float(row["value"]),
            int(row["planned_users"]),
        )
    return lookup


def inferential_result_rows(plan: pd.DataFrame, paired: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    defects: list[str] = []
    plan_with_order = plan.copy()
    plan_with_order["template_row_number"] = np.arange(1, len(plan_with_order) + 1)
    for template in plan_with_order.to_dict("records"):
        if not bool_value(template["inferential_test_allowed"]):
            continue
        template_id = comparison_template_id(int(template["template_row_number"]), template)
        for metric_id in INFERENTIAL_METRICS:
            subset = paired[(paired["comparison_template_id"] == template_id) & (paired["metric_id"] == metric_id)]
            if subset.empty:
                defects.append(f"{template_id}:{metric_id}: no paired contributions")
                continue
            arm_a_values = subset["arm_a_value"].astype(float).to_numpy()
            arm_b_values = subset["arm_b_value"].astype(float).to_numpy()
            deltas = subset["arm_b_minus_arm_a_delta"].astype(float).to_numpy()
            arm_a_mean = float(np.mean(arm_a_values))
            arm_b_mean = float(np.mean(arm_b_values))
            delta = float(np.mean(deltas))
            low, high, unique_delta_count = paired_bootstrap_mean_delta_ci(
                deltas,
                seed=int(template["seed"]),
                resamples=int(template["bootstrap_resamples"]),
            )
            low_check, high_check, unique_check = paired_bootstrap_mean_delta_ci(
                deltas,
                seed=int(template["seed"]),
                resamples=int(template["bootstrap_resamples"]),
            )
            if low != low_check or high != high_check or unique_delta_count != unique_check:
                defects.append(f"{template_id}:{metric_id}: bootstrap not reproducible")
            zero_relation, decision = ci_decision(low, high)
            rows.append(
                {
                    "result_id": result_id(template_id, metric_id),
                    "comparison_template_id": template_id,
                    "template_row_number": int(template["template_row_number"]),
                    "comparison_family": template["comparison_family"],
                    "dataset": template["dataset"],
                    "model_scope": template["model_scope"],
                    "prompt_scope": template["prompt_scope"],
                    "comparison": template["comparison"],
                    "metric_id": metric_id,
                    "arm_a_label": subset.iloc[0]["arm_a_label"],
                    "arm_b_label": subset.iloc[0]["arm_b_label"],
                    "arm_a_model": subset.iloc[0]["arm_a_model"],
                    "arm_b_model": subset.iloc[0]["arm_b_model"],
                    "arm_a_prompt": subset.iloc[0]["arm_a_prompt"],
                    "arm_b_prompt": subset.iloc[0]["arm_b_prompt"],
                    "paired_unit": template["paired_unit"],
                    "paired_n": int(len(subset)),
                    "paired_n_status": "COMPLETE_USER_LEVEL_PAIRING",
                    "arm_a_mean": format_float(arm_a_mean),
                    "arm_b_mean": format_float(arm_b_mean),
                    "arm_b_minus_arm_a_delta": format_float(delta),
                    "delta_direction": subset.iloc[0]["delta_direction"],
                    "positive_delta_user_count": int(np.count_nonzero(deltas > 0.0)),
                    "negative_delta_user_count": int(np.count_nonzero(deltas < 0.0)),
                    "zero_delta_user_count": int(np.count_nonzero(deltas == 0.0)),
                    "method": template["method"],
                    "bootstrap_resamples": int(template["bootstrap_resamples"]),
                    "bootstrap_seed": int(template["seed"]),
                    "bootstrap_engine": "empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta",
                    "bootstrap_percentile_method": "numpy_quantile_linear",
                    "ci": template["ci"],
                    "ci_low": format_float(low),
                    "ci_high": format_float(high),
                    "ci_zero_relation": zero_relation,
                    "ci_supported_decision": decision,
                    "alpha": template["alpha"],
                    "p_value": "",
                    "p_value_method": template["p_value_method"],
                    "p_value_status": "NOT_COMPUTED_NO_FROZEN_P_VALUE_METHOD_FOR_G4",
                    "holm": template["holm"],
                    "multiplicity_status": "NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD_AND_HOLM_FALSE",
                    "corrected_decision": "NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD_FOR_G4",
                    "inferential_test_allowed": template["inferential_test_allowed"],
                    "ranking_vs_reranking_policy": template["ranking_vs_reranking_policy"],
                    "small_conditional_denominator_flag": bool_text(
                        template["dataset"] == "foodcom" and metric_id in CONDITIONAL_METRICS
                    ),
                    "result_status": "COMPUTED_CI_ONLY_NO_P_VALUES",
                    "notes": "CI-only paired user bootstrap; no p-values or Holm correction frozen for G4.",
                }
            )
    return rows, defects


def descriptive_result_rows(
    plan: pd.DataFrame,
    g4_metrics: pd.DataFrame,
    g1_utility: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    defects: list[str] = []
    g4_lookup = g4_metric_lookup(g4_metrics)
    g1_lookup = g1_metric_lookup(g1_utility)
    plan_with_order = plan.copy()
    plan_with_order["template_row_number"] = np.arange(1, len(plan_with_order) + 1)
    for template in plan_with_order.to_dict("records"):
        if template["comparison_family"] != "g1_vs_g4_pipeline_descriptive":
            continue
        template_id = comparison_template_id(int(template["template_row_number"]), template)
        dataset = str(template["dataset"])
        model = str(template["model_scope"])
        prompt = str(template["prompt_scope"])
        for metric_id in DESCRIPTIVE_METRICS:
            g1_key = (dataset, model, prompt, metric_id)
            g4_key = (dataset, model, prompt, metric_id)
            if g1_key not in g1_lookup:
                defects.append(f"{template_id}:{metric_id}: missing G1 utility metric")
                continue
            if g4_key not in g4_lookup:
                defects.append(f"{template_id}:{metric_id}: missing G4 utility metric")
                continue
            g1_mean, g1_planned = g1_lookup[g1_key]
            g4_mean = g4_lookup[g4_key]
            rows.append(
                {
                    "result_id": result_id(template_id, metric_id),
                    "comparison_template_id": template_id,
                    "template_row_number": int(template["template_row_number"]),
                    "comparison_family": template["comparison_family"],
                    "dataset": dataset,
                    "model_scope": model,
                    "prompt_scope": prompt,
                    "comparison": template["comparison"],
                    "metric_id": metric_id,
                    "arm_a_label": "G1_ranking",
                    "arm_b_label": "G4_reranking",
                    "arm_a_model": model,
                    "arm_b_model": model,
                    "arm_a_prompt": prompt,
                    "arm_b_prompt": prompt,
                    "paired_unit": template["paired_unit"],
                    "paired_n": "",
                    "paired_n_status": "NOT_APPLICABLE_PIPELINE_DESCRIPTIVE_ONLY",
                    "arm_a_mean": format_float(g1_mean),
                    "arm_b_mean": format_float(g4_mean),
                    "arm_b_minus_arm_a_delta": format_float(g4_mean - g1_mean),
                    "delta_direction": "G4_reranking_minus_G1_ranking",
                    "positive_delta_user_count": "",
                    "negative_delta_user_count": "",
                    "zero_delta_user_count": "",
                    "method": template["method"],
                    "bootstrap_resamples": 0,
                    "bootstrap_seed": "",
                    "bootstrap_engine": "NOT_APPLICABLE_DESCRIPTIVE_NO_INFERENCE",
                    "bootstrap_percentile_method": "",
                    "ci": "",
                    "ci_low": "",
                    "ci_high": "",
                    "ci_zero_relation": "NOT_APPLICABLE_DESCRIPTIVE_NO_INFERENCE",
                    "ci_supported_decision": "NOT_APPLICABLE_DESCRIPTIVE_NO_INFERENCE",
                    "alpha": "",
                    "p_value": "",
                    "p_value_method": template["p_value_method"],
                    "p_value_status": "NOT_COMPUTED_DESCRIPTIVE_PIPELINE_LEVEL_ONLY",
                    "holm": template["holm"],
                    "multiplicity_status": "NOT_APPLIED_DESCRIPTIVE_PIPELINE_LEVEL_ONLY",
                    "corrected_decision": "NOT_DEFINED_DESCRIPTIVE_PIPELINE_LEVEL_ONLY",
                    "inferential_test_allowed": template["inferential_test_allowed"],
                    "ranking_vs_reranking_policy": template["ranking_vs_reranking_policy"],
                    "small_conditional_denominator_flag": "false",
                    "result_status": "DESCRIPTIVE_ONLY_NO_INFERENCE",
                    "notes": (
                        f"G1 planned users={g1_planned}; G4 planned users={EXPECTED_USERS[dataset]}; "
                        "pipeline-level descriptive comparison only because ranking and reranking candidate sets differ."
                    ),
                }
            )
    return rows, defects


def build_statistical_results(
    plan: pd.DataFrame,
    paired: pd.DataFrame,
    g4_metrics: pd.DataFrame,
    g1_utility: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    inferential_rows, inferential_defects = inferential_result_rows(plan, paired)
    descriptive_rows, descriptive_defects = descriptive_result_rows(plan, g4_metrics, g1_utility)
    frame = pd.DataFrame(inferential_rows + descriptive_rows)
    if not frame.empty:
        frame = frame.sort_values(["template_row_number", "metric_id"], kind="mergesort")
    return frame, inferential_defects + descriptive_defects


def validate_plan(plan: pd.DataFrame, protocol_manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if len(plan) != EXPECTED_PLAN_ROWS:
        failures.append(f"STATISTICAL_PLAN_ROWS:{len(plan)}:{EXPECTED_PLAN_ROWS}")
    family_counts = plan["comparison_family"].value_counts().to_dict()
    if family_counts != EXPECTED_FAMILY_COUNTS:
        failures.append(f"STATISTICAL_PLAN_FAMILY_COUNTS:{stable_json(family_counts)}")
    if plan["status"].ne("FROZEN").any():
        failures.append("STATISTICAL_PLAN_NON_FROZEN_ROWS")
    inferential = plan[plan["inferential_test_allowed"].map(bool_value)]
    descriptive = plan[~plan["inferential_test_allowed"].map(bool_value)]
    if len(inferential) != EXPECTED_INFERENTIAL_TEMPLATES or len(descriptive) != EXPECTED_DESCRIPTIVE_TEMPLATES:
        failures.append(f"STATISTICAL_PLAN_INFERENTIAL_SPLIT:{len(inferential)}:{len(descriptive)}")
    if not inferential["method"].eq("paired_user_bootstrap_mean_delta").all():
        failures.append("STATISTICAL_PLAN_INFERENTIAL_METHOD_MISMATCH")
    if not inferential["bootstrap_resamples"].astype(int).eq(EXPECTED_BOOTSTRAP_RESAMPLES).all():
        failures.append("STATISTICAL_PLAN_BOOTSTRAP_RESAMPLES_MISMATCH")
    if not inferential["seed"].astype(int).eq(EXPECTED_BOOTSTRAP_SEED).all():
        failures.append("STATISTICAL_PLAN_BOOTSTRAP_SEED_MISMATCH")
    if not inferential["ci"].eq("two_sided_95_percentile").all():
        failures.append("STATISTICAL_PLAN_CI_MISMATCH")
    if not inferential["alpha"].astype(float).eq(EXPECTED_ALPHA).all():
        failures.append("STATISTICAL_PLAN_ALPHA_MISMATCH")
    if not plan["p_value_method"].eq("NOT_FROZEN_DO_NOT_COMPUTE").all():
        failures.append("STATISTICAL_PLAN_P_VALUE_METHOD_MISMATCH")
    if not plan["holm"].eq("false").all():
        failures.append("STATISTICAL_PLAN_HOLM_MISMATCH")
    if not descriptive["method"].eq("descriptive_summary_no_inference").all():
        failures.append("STATISTICAL_PLAN_DESCRIPTIVE_METHOD_MISMATCH")
    if not descriptive["ranking_vs_reranking_policy"].eq("descriptive_pipeline_level_only").all():
        failures.append("STATISTICAL_PLAN_G1_G4_POLICY_MISMATCH")
    return failures


def frozen_hash_failures(root: Path, metrics_manifest: Mapping[str, Any], protocol_manifest: Mapping[str, Any], g1_manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_hashes: dict[str, str] = {}
    expected_hashes.update({str(k): str(v) for k, v in metrics_manifest.get("input_hashes", {}).items()})
    expected_hashes.update({str(k): str(v) for k, v in metrics_manifest.get("output_hashes", {}).items()})
    expected_hashes[PATHS["metrics_manifest"].as_posix()] = sha256_file(root / PATHS["metrics_manifest"])
    expected_hashes[PATHS["statistical_plan"].as_posix()] = str(
        protocol_manifest.get("artifact_hashes", {}).get(PATHS["statistical_plan"].as_posix(), "")
    )
    expected_hashes[PATHS["g1_utility"].as_posix()] = str(
        g1_manifest.get("artifact_hashes", {}).get(PATHS["g1_utility"].as_posix(), "")
    )
    for path_text, expected in sorted(expected_hashes.items()):
        if not expected:
            failures.append(f"FROZEN_HASH_EXPECTATION_MISSING:{path_text}")
            continue
        path = root / Path(path_text)
        if not path.is_file():
            failures.append(f"FROZEN_INPUT_MISSING:{path_text}")
            continue
        observed = sha256_file(path)
        if observed != expected:
            failures.append(f"FROZEN_HASH_MISMATCH:{path_text}:{observed}:{expected}")
    return failures


def metric_mean_validation(metric_results: pd.DataFrame, statistical_results: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    g4_lookup = g4_metric_lookup(metric_results)
    inferential = statistical_results[statistical_results["inferential_test_allowed"].map(bool_value)]
    for row in inferential.to_dict("records"):
        metric_id = row["metric_id"]
        dataset = row["dataset"]
        a_key = (dataset, row["arm_a_model"], row["arm_a_prompt"], metric_id)
        b_key = (dataset, row["arm_b_model"], row["arm_b_prompt"], metric_id)
        if a_key not in g4_lookup:
            failures.append(f"G4_ARM_A_METRIC_MISSING:{stable_json(a_key)}")
            continue
        if b_key not in g4_lookup:
            failures.append(f"G4_ARM_B_METRIC_MISSING:{stable_json(b_key)}")
            continue
        if abs(float(row["arm_a_mean"]) - g4_lookup[a_key]) > 1e-12:
            failures.append(f"G4_ARM_A_MEAN_DRIFT:{stable_json(a_key)}")
        if abs(float(row["arm_b_mean"]) - g4_lookup[b_key]) > 1e-12:
            failures.append(f"G4_ARM_B_MEAN_DRIFT:{stable_json(b_key)}")
    return failures


def build_validation(
    *,
    root: Path,
    head_before: str,
    status_before: str,
    plan: pd.DataFrame,
    universe: pd.DataFrame,
    metric_results: pd.DataFrame,
    statistical_results: pd.DataFrame,
    paired_contributions: pd.DataFrame,
    metrics_manifest: Mapping[str, Any],
    analytic_manifest: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
    g1_manifest: Mapping[str, Any],
    contribution_defects: list[str],
    result_defects: list[str],
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
    add("git_status_clean_or_expected_stat_dirty", git_ok, status_before.replace("\n", " | "), "clean or expected G4 statistical outputs only")

    hash_failures = frozen_hash_failures(root, metrics_manifest, protocol_manifest, g1_manifest)
    add("frozen_input_hashes_match", not hash_failures, len(hash_failures), 0, " | ".join(hash_failures[:5]))
    add("metrics_manifest_decision", metrics_manifest.get("decision_token") == "G4_METRICS_FROZEN", metrics_manifest.get("decision_token"), "G4_METRICS_FROZEN")
    add("metrics_manifest_validation", bool(metrics_manifest.get("validation_passed")), metrics_manifest.get("validation_passed"), True)
    add(
        "analytic_manifest_decision",
        analytic_manifest.get("decision_token") == "G4_ANALYTIC_UNIVERSE_FROZEN",
        analytic_manifest.get("decision_token"),
        "G4_ANALYTIC_UNIVERSE_FROZEN",
    )
    add(
        "protocol_manifest_decision",
        protocol_manifest.get("decision_token") == "G4_F1BZQ_RERANKING_PROTOCOL_FROZEN",
        protocol_manifest.get("decision_token"),
        "G4_F1BZQ_RERANKING_PROTOCOL_FROZEN",
    )
    add("g1_manifest_decision", g1_manifest.get("decision_token") == "G1_PRIMARY_RANKING_VALIDATED", g1_manifest.get("decision_token"), "G1_PRIMARY_RANKING_VALIDATED")

    plan_failures = validate_plan(plan, protocol_manifest)
    add("statistical_plan_75_rows_exact", len(plan) == EXPECTED_PLAN_ROWS, len(plan), EXPECTED_PLAN_ROWS)
    add("statistical_plan_contract_valid", not plan_failures, len(plan_failures), 0, " | ".join(plan_failures[:5]))
    add("comparison_family_counts", plan["comparison_family"].value_counts().to_dict() == EXPECTED_FAMILY_COUNTS, stable_json(plan["comparison_family"].value_counts().to_dict()), stable_json(EXPECTED_FAMILY_COUNTS))

    add("analytic_universe_rows", len(universe) == EXPECTED_LOGICAL_ROWS, len(universe), EXPECTED_LOGICAL_ROWS)
    add("analytic_universe_users", universe[["dataset", "user_id"]].drop_duplicates().shape[0] == EXPECTED_USERS_TOTAL, universe[["dataset", "user_id"]].drop_duplicates().shape[0], EXPECTED_USERS_TOTAL)
    add("analytic_universe_cells", universe[["dataset", "model", "prompt_strategy"]].drop_duplicates().shape[0] == EXPECTED_CELLS, universe[["dataset", "model", "prompt_strategy"]].drop_duplicates().shape[0], EXPECTED_CELLS)

    template_ids = [
        comparison_template_id(index + 1, row)
        for index, row in enumerate(plan.to_dict("records"))
    ]
    result_template_ids = set(statistical_results["comparison_template_id"].astype(str))
    add("comparison_template_coverage_75_of_75", set(template_ids) == result_template_ids, f"{len(result_template_ids)}/75", "75/75")
    inferential_results = statistical_results[statistical_results["inferential_test_allowed"].map(bool_value)]
    descriptive_results = statistical_results[~statistical_results["inferential_test_allowed"].map(bool_value)]
    add("inferential_result_rows", len(inferential_results) == 225, len(inferential_results), 225)
    add("descriptive_result_rows", len(descriptive_results) == 90, len(descriptive_results), 90)
    add("statistical_result_rows", len(statistical_results) == 315, len(statistical_results), 315)
    add(
        "statistical_result_keys_unique",
        not statistical_results.duplicated(["comparison_template_id", "metric_id"]).any(),
        int(statistical_results.duplicated(["comparison_template_id", "metric_id"]).sum()),
        0,
    )

    expected_contrib_rows = (
        3 * sum(EXPECTED_USERS.values())
        + 6 * sum(EXPECTED_USERS.values())
    ) * 3 + (3 * sum(EXPECTED_POSITIVE_PRESENT.values()) + 6 * sum(EXPECTED_POSITIVE_PRESENT.values())) * 2
    add("paired_contribution_rows", len(paired_contributions) == expected_contrib_rows, len(paired_contributions), expected_contrib_rows)
    add(
        "paired_contribution_keys_unique",
        not paired_contributions.duplicated(["comparison_template_id", "metric_id", "user_id"]).any(),
        int(paired_contributions.duplicated(["comparison_template_id", "metric_id", "user_id"]).sum()),
        0,
    )
    add("pairing_integrity_defects_absent", not contribution_defects, len(contribution_defects), 0, " | ".join(contribution_defects[:5]))
    add("statistical_result_defects_absent", not result_defects, len(result_defects), 0, " | ".join(result_defects[:5]))

    denominator_drift = []
    for row in inferential_results.to_dict("records"):
        expected_n = EXPECTED_POSITIVE_PRESENT[row["dataset"]] if row["metric_id"] in CONDITIONAL_METRICS else EXPECTED_USERS[row["dataset"]]
        if int(row["paired_n"]) != expected_n:
            denominator_drift.append(f"{row['comparison_template_id']}:{row['metric_id']}:{row['paired_n']}:{expected_n}")
    add("denominator_no_drift", not denominator_drift, len(denominator_drift), 0, " | ".join(denominator_drift[:5]))
    positive_absent_conditional = paired_contributions[
        paired_contributions["metric_id"].isin(CONDITIONAL_METRICS)
        & ~paired_contributions["positive_present_in_pool"].map(bool_value)
    ]
    add("conditional_excludes_positive_absent_users", positive_absent_conditional.empty, len(positive_absent_conditional), 0)
    foodcom_conditional = inferential_results[
        (inferential_results["dataset"] == "foodcom") & (inferential_results["metric_id"].isin(CONDITIONAL_METRICS))
    ]
    add(
        "foodcom_conditional_n4_marked",
        len(foodcom_conditional) == 18 and set(foodcom_conditional["paired_n"].astype(str)) == {"4"} and foodcom_conditional["small_conditional_denominator_flag"].eq("true").all(),
        f"rows={len(foodcom_conditional)} n={sorted(set(foodcom_conditional['paired_n'].astype(str)))}",
        "18 rows n=4 and flagged",
    )

    mean_failures = metric_mean_validation(metric_results, statistical_results)
    add("arm_means_match_g4_metric_artifacts", not mean_failures, len(mean_failures), 0, " | ".join(mean_failures[:5]))
    bootstrap_repro_failures = []
    for row in inferential_results.to_dict("records"):
        subset = paired_contributions[
            (paired_contributions["comparison_template_id"] == row["comparison_template_id"])
            & (paired_contributions["metric_id"] == row["metric_id"])
        ]
        low, high, _ = paired_bootstrap_mean_delta_ci(
            subset["arm_b_minus_arm_a_delta"].astype(float).to_numpy(),
            seed=int(row["bootstrap_seed"]),
            resamples=int(row["bootstrap_resamples"]),
        )
        if abs(float(row["ci_low"]) - low) > 1e-15 or abs(float(row["ci_high"]) - high) > 1e-15:
            bootstrap_repro_failures.append(f"{row['comparison_template_id']}:{row['metric_id']}")
    add("bootstrap_reproducible", not bootstrap_repro_failures, len(bootstrap_repro_failures), 0, " | ".join(bootstrap_repro_failures[:5]))
    add("no_p_values_computed", statistical_results["p_value"].astype(str).eq("").all(), sorted(set(statistical_results["p_value"].astype(str))), "all blank")
    add(
        "holm_not_applied_without_p_values",
        statistical_results["multiplicity_status"].astype(str).str.startswith("NOT_APPLIED").all(),
        sorted(set(statistical_results["multiplicity_status"].astype(str))),
        "all NOT_APPLIED*",
    )
    add("candidate_recall_not_tested", "candidate_recall_at_20" not in set(statistical_results["metric_id"]), sorted(set(statistical_results["metric_id"])), "candidate_recall_at_20 absent")
    add(
        "beyond_utility_not_tested_absent_from_plan",
        not set(statistical_results["metric_id"]).intersection({"item_coverage_at_5", "oic_at_5"}),
        sorted(set(statistical_results["metric_id"])),
        "no beyond-utility metrics in statistical results",
    )
    add(
        "g1_g4_descriptive_no_inference",
        descriptive_results["ci_supported_decision"].eq("NOT_APPLICABLE_DESCRIPTIVE_NO_INFERENCE").all()
        and descriptive_results["paired_n_status"].eq("NOT_APPLICABLE_PIPELINE_DESCRIPTIVE_ONLY").all(),
        "checked",
        "descriptive rows have no CI/pairing",
    )
    return pd.DataFrame(rows)


def build_report(
    *,
    validation: pd.DataFrame,
    statistical_results: pd.DataFrame,
    paired_contributions: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> str:
    family_counts = manifest["summary"]["comparison_family_counts"]
    decision_counts = statistical_results[statistical_results["inferential_test_allowed"].map(bool_value)][
        "ci_supported_decision"
    ].value_counts().to_dict()
    prompt_counts = statistical_results[
        statistical_results["comparison_family"].eq("g4_prompt_strategy")
    ]["ci_supported_decision"].value_counts().to_dict()
    model_counts = statistical_results[
        statistical_results["comparison_family"].eq("g4_model_pair")
    ]["ci_supported_decision"].value_counts().to_dict()
    dataset_counts = (
        statistical_results[statistical_results["inferential_test_allowed"].map(bool_value)]
        .groupby(["dataset", "ci_supported_decision"])
        .size()
        .reset_index(name="rows")
    )
    dataset_lines = [
        f"- {row.dataset} / {row.ci_supported_decision}: {int(row.rows)}"
        for row in dataset_counts.sort_values(["dataset", "ci_supported_decision"]).itertuples()
    ]
    validation_passes = int((validation["status"] == "PASS").sum())
    validation_failures = int((validation["status"] == "FAIL").sum())
    return "\n".join(
        [
            "# G4 F1BZQ Statistical Analysis",
            "",
            f"- Phase: {PHASE_ID}",
            f"- Decision token: {manifest['decision_token']}",
            "- Source: frozen G4 metric artifacts and frozen G4 analytic universe only",
            "- Ollama/API calls: 0",
            "- P-values: not computed",
            "- Holm correction: not applied; metadata only because no frozen p-value method exists",
            "",
            "## Frozen Plan",
            "",
            f"- Template rows: {manifest['summary']['comparison_templates_covered']}/75",
            f"- Family counts: {stable_json(family_counts)}",
            f"- Inferential templates: {manifest['summary']['inferential_templates']}",
            f"- Descriptive G1-vs-G4 templates: {manifest['summary']['descriptive_templates']}",
            "",
            "## Outputs",
            "",
            f"- Paired contribution rows: {len(paired_contributions)}",
            f"- Statistical result rows: {len(statistical_results)}",
            f"- Inferential result rows: {manifest['summary']['inferential_result_rows']}",
            f"- Descriptive result rows: {manifest['summary']['descriptive_result_rows']}",
            "",
            "## CI Decisions",
            "",
            f"- Overall inferential CI decisions: {stable_json(decision_counts)}",
            f"- Prompt-strategy family: {stable_json(prompt_counts)}",
            f"- Model-pair family: {stable_json(model_counts)}",
            *dataset_lines,
            "",
            "## Denominator Notes",
            "",
            "- Unconditional utility comparisons use all planned users with invalid terminal responses set to zero.",
            "- Conditional HR/NDCG comparisons use positive-present users only, with invalid terminal responses set to zero within that subset.",
            "- Food.com conditional comparisons have paired n=4 and are flagged in the results table.",
            "- Candidate recall is upstream dataset-level evidence and is not tested across model/prompt variants.",
            "- G1-vs-G4 rows are descriptive pipeline-level comparisons only; no causal ranking-vs-reranking inference is made.",
            "",
            "## Validation",
            "",
            f"- PASS: {validation_passes}",
            f"- FAIL: {validation_failures}",
            f"- Status: {'PASS' if validation_failures == 0 else 'FAIL'}",
            "",
            "No final scientific synthesis or relation update is included in this freeze.",
            "",
        ]
    )


def append_notes(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = "\n".join(
        [
            f"## {PHASE_ID} - {manifest['created_at_utc']}",
            "",
            "- Executed the frozen G4 F1BZQ statistical comparison plan offline.",
            "- No Ollama/API calls; no p-values; Holm not applied; no frozen evidence/protocol/universe/metric edits.",
            f"- Comparison templates covered: {manifest['summary']['comparison_templates_covered']}/75.",
            f"- Paired contribution rows: {manifest['summary']['paired_contribution_rows']}; statistical result rows: {manifest['summary']['statistical_result_rows']}.",
            f"- CI decision counts: {stable_json(manifest['summary']['ci_decision_counts'])}.",
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

    plan = read_csv(root, PATHS["statistical_plan"])
    universe = read_csv(root, PATHS["analytic_universe"])
    metric_results = read_csv(root, PATHS["metric_results"])
    g1_utility = read_csv(root, PATHS["g1_utility"])
    metrics_manifest = read_json(root, PATHS["metrics_manifest"])
    analytic_manifest = read_json(root, PATHS["analytic_manifest"])
    protocol_manifest = read_json(root, PATHS["protocol_manifest"])
    g1_manifest = read_json(root, PATHS["g1_manifest"])

    user_contrib, user_contrib_defects = compute_user_contributions(universe)
    paired_contributions, pair_defects = build_paired_contributions(plan, user_contrib)
    statistical_results, result_defects = build_statistical_results(plan, paired_contributions, metric_results, g1_utility)
    validation = build_validation(
        root=root,
        head_before=head_before,
        status_before=status_before,
        plan=plan,
        universe=universe,
        metric_results=metric_results,
        statistical_results=statistical_results,
        paired_contributions=paired_contributions,
        metrics_manifest=metrics_manifest,
        analytic_manifest=analytic_manifest,
        protocol_manifest=protocol_manifest,
        g1_manifest=g1_manifest,
        contribution_defects=user_contrib_defects + pair_defects,
        result_defects=result_defects,
        allow_generator_dirty=args.allow_generator_dirty,
    )
    validation_passed = bool((validation["status"] == "FAIL").sum() == 0)
    inferential_results = statistical_results[statistical_results["inferential_test_allowed"].map(bool_value)]
    descriptive_results = statistical_results[~statistical_results["inferential_test_allowed"].map(bool_value)]
    covered_templates = statistical_results["comparison_template_id"].nunique()
    family_counts = plan["comparison_family"].value_counts().to_dict()
    ci_decision_counts = inferential_results["ci_supported_decision"].value_counts().to_dict()
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "phase_id": PHASE_ID,
        "decision_token": DECISION_FROZEN if validation_passed else DECISION_BLOCKED,
        "created_at_utc": utc_now(),
        "head_before_generation": head_before,
        "precondition_status": "G4_METRICS_FROZEN",
        "source": "frozen_g4_metric_artifacts_and_frozen_g4_analytic_universe_only",
        "zero_inference": True,
        "ollama_or_api_calls_executed": 0,
        "new_tests_added": False,
        "statistical_plan_modified": False,
        "frozen_evidence_modified": False,
        "protocol_config_runner_modified": False,
        "analytic_universe_modified": False,
        "metric_artifacts_modified": False,
        "p_values_computed": False,
        "holm_applied": False,
        "holm_status": "metadata_only_not_applied_no_frozen_p_value_method",
        "bootstrap_settings": {
            "method": "paired_user_bootstrap_mean_delta",
            "resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
            "seed": EXPECTED_BOOTSTRAP_SEED,
            "ci": "two_sided_95_percentile",
            "alpha": EXPECTED_ALPHA,
            "engine": "empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta",
            "percentile_method": "numpy_quantile_linear",
        },
        "metric_scope": {
            "inferential_g4_internal": list(INFERENTIAL_METRICS),
            "descriptive_g1_vs_g4_pipeline": list(DESCRIPTIVE_METRICS),
            "candidate_recall": "not_tested_upstream_dataset_level_quantity",
            "beyond_utility": "not_tested_absent_from_frozen_statistical_plan",
        },
        "summary": {
            "comparison_templates_covered": int(covered_templates),
            "comparison_family_counts": {str(k): int(v) for k, v in family_counts.items()},
            "inferential_templates": int(plan["inferential_test_allowed"].map(bool_value).sum()),
            "descriptive_templates": int((~plan["inferential_test_allowed"].map(bool_value)).sum()),
            "paired_contribution_rows": int(len(paired_contributions)),
            "statistical_result_rows": int(len(statistical_results)),
            "inferential_result_rows": int(len(inferential_results)),
            "descriptive_result_rows": int(len(descriptive_results)),
            "ci_decision_counts": {str(k): int(v) for k, v in ci_decision_counts.items()},
            "validation_passes": int((validation["status"] == "PASS").sum()),
            "validation_failures": int((validation["status"] == "FAIL").sum()),
        },
        "input_hashes": {
            path.as_posix(): sha256_file(root / path)
            for path in [
                PATHS["statistical_plan"],
                PATHS["metric_results"],
                PATHS["metric_denominators"],
                PATHS["candidate_conditional"],
                PATHS["reliability"],
                PATHS["undefined_blocked"],
                PATHS["metric_validation"],
                PATHS["metrics_manifest"],
                PATHS["analytic_universe"],
                PATHS["analytic_cell_counts"],
                PATHS["analytic_manifest"],
                PATHS["protocol_manifest"],
                PATHS["g1_manifest"],
                PATHS["g1_utility"],
            ]
        },
        "files_created": [path.as_posix() for path in OUTPUTS.values()] + [PATHS["notes"].as_posix()],
        "validation_passed": validation_passed,
        "validation_failures": validation[validation["status"] == "FAIL"].to_dict("records"),
        "next_safe_step": "perform final G4 scientific synthesis/report update from frozen metric and statistical artifacts only",
    }

    report = build_report(
        validation=validation,
        statistical_results=statistical_results,
        paired_contributions=paired_contributions,
        manifest=manifest,
    )

    if not args.no_write:
        write_csv(root, OUTPUTS["paired_contributions"], paired_contributions)
        write_csv(root, OUTPUTS["statistical_results"], statistical_results)
        write_csv(root, OUTPUTS["validation"], validation)
        (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
        (root / OUTPUTS["report"]).write_text(report, encoding="utf-8", newline="\n")
        manifest["output_hashes"] = output_hashes(root)
        write_json(root, OUTPUTS["manifest"], manifest)
        append_notes(root / PATHS["notes"], manifest)

    if not validation_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

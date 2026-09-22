"""Compute G3 candidate-position metrics from the frozen analytic universe.

This script is offline-only. It reads the frozen G3 analytic universe, computes
cell-level utility, CandDif, and reliability summaries, and writes the G3
metric-freeze artifacts. It does not call Ollama, run bootstrap statistics, or
modify frozen protocol/execution/evidence artifacts.
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
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extension.e0_protocol import sha256_json  # noqa: E402
from src.utils.config import find_project_root  # noqa: E402


EXPECTED_HEAD = "b21c0cff1f3cdb474ef2944fcc1aa5e6fae5325e"
PHASE_ID = "g3_candidate_position_metrics_freeze"
DECISION_FROZEN = "G3_METRICS_FROZEN"
DECISION_BLOCKED = "G3_METRICS_BLOCKED"
TOP_K = 5
MRR_CUTOFF = 10
EPSILON = 1e-12

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = ("BASE", "LLMRANK")
ORDERS = ("FIRST", "RANDOM")
UTILITY_METRICS = ("HR@5", "NDCG@5", "MRR")
CANDDIF_METRICS = ("CandDif_HR@5", "CandDif_NDCG@5")

EXPECTED_LOGICAL_ROWS = 12000
EXPECTED_PAIR_ROWS = 6000
EXPECTED_ORDER_ROWS = {"FIRST": 6000, "RANDOM": 6000}
EXPECTED_STATE_COUNTS = {
    "STRICT_VALID": 11834,
    "INVALID_LENGTH": 114,
    "INVALID_DUPLICATE": 51,
    "HALLUCINATION_OR_OUT_OF_CANDIDATE": 1,
}

PATHS = {
    "analysis_universe": Path("results/tables/g3_candidate_position_analysis_universe.csv"),
    "analysis_pairs": Path("results/tables/g3_candidate_position_analysis_pairs.csv"),
    "analysis_counts": Path("results/tables/g3_candidate_position_analysis_counts.csv"),
    "analysis_denominator_contract": Path("results/tables/g3_candidate_position_denominator_contract.csv"),
    "analysis_manifest": Path("results/manifests/g3_candidate_position_analysis_universe_freeze.json"),
    "protocol_manifest": Path("results/manifests/g3_candidate_position_protocol_freeze.json"),
    "execution_manifest": Path("results/manifests/g3_candidate_position_execution_freeze.json"),
    "retention_manifest": Path("results/manifests/g3_candidate_position_execution_retention_manifest.json"),
    "metric_contract_source": Path("src/metrics/canonical_metric_contract.py"),
    "canddif_contract_source": Path("src/analysis/reproduction_r5b_canddif_recovery.py"),
    "source": Path("src/extension/g3_candidate_position_metrics.py"),
    "notes": Path("report/notes.md"),
}

OUTPUTS = {
    "metric_results": Path("results/tables/g3_candidate_position_metric_results.csv"),
    "canddif": Path("results/tables/g3_candidate_position_canddif.csv"),
    "reliability": Path("results/tables/g3_candidate_position_reliability.csv"),
    "denominators": Path("results/tables/g3_candidate_position_metric_denominators.csv"),
    "validation": Path("results/tables/g3_candidate_position_metric_validation.csv"),
    "manifest": Path("results/manifests/g3_candidate_position_metrics_freeze.json"),
    "report": Path("report/g3_candidate_position_metrics_validation.md"),
}

EXPECTED_DIRTY_PATHS = {
    PATHS["source"].as_posix(),
    PATHS["notes"].as_posix(),
    *[path.as_posix() for path in OUTPUTS.values()],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute frozen G3 candidate-position metrics.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Compute and validate without writing artifacts.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator and expected metric outputs to be dirty while frozen inputs remain unchanged.",
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


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite metric value: {value}")
    return f"{float(value):.17g}"


def parse_int_json_array(value: Any) -> list[int]:
    if str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("expected JSON array")
    return [int(item) for item in parsed]


def git_output(root: Path, args: list[str], *, check: bool = True) -> str:
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
    if not allow_generator_dirty:
        return False
    return set(paths).issubset(EXPECTED_DIRTY_PATHS)


def canddif_value(acc_first: float, acc_random: float) -> tuple[float, str]:
    numerator = 1.0 - float(acc_random)
    denominator = 1.0 - float(acc_first)
    if numerator <= 0 and denominator <= 0:
        return 0.0, "BOTH_ERROR_RATES_ZERO_RETURNED_0"
    if numerator <= 0:
        return math.log(EPSILON / max(denominator, EPSILON)), "NUMERATOR_CLAMPED_EPSILON_1E-12"
    if denominator <= 0:
        return math.log(numerator / EPSILON), "DENOMINATOR_CLAMPED_EPSILON_1E-12"
    return math.log(numerator / denominator), "NONE"


def utility_from_ranked_indices(
    *,
    strict_valid: bool,
    ranked_candidate_indices: Any,
    positive_candidate_position: Any,
) -> tuple[float, float, float, str, str]:
    if not strict_valid:
        return 0.0, 0.0, 0.0, "", "invalid_terminal_zero"
    ranked = parse_int_json_array(ranked_candidate_indices)
    if len(ranked) != MRR_CUTOFF:
        raise ValueError(f"strict-valid response without {MRR_CUTOFF} ranked indices")
    if len(set(ranked)) != len(ranked):
        raise ValueError("strict-valid response has duplicate ranked indices")
    if any(value < 1 or value > 20 for value in ranked):
        raise ValueError("strict-valid response has out-of-candidate ranked index")

    positive_position = int(str(positive_candidate_position))
    rank_zero_based = ranked.index(positive_position) if positive_position in ranked else None
    if rank_zero_based is None:
        return 0.0, 0.0, 0.0, "", "positive_absent_from_top10"
    hr_at_5 = 1.0 if rank_zero_based < TOP_K else 0.0
    ndcg_at_5 = 1.0 / math.log2(rank_zero_based + 2.0) if rank_zero_based < TOP_K else 0.0
    mrr = 1.0 / float(rank_zero_based + 1)
    return hr_at_5, ndcg_at_5, mrr, str(rank_zero_based), "strict_valid"


def add_validation(rows: list[dict[str, str]], check_id: str, description: str, observed: Any, expected: Any, passed: bool) -> None:
    rows.append(
        {
            "check_id": check_id,
            "description": description,
            "observed": str(observed),
            "expected": str(expected),
            "status": "PASS" if passed else "FAIL",
        }
    )


def frame_state_counts(universe: pd.DataFrame) -> Counter[str]:
    return Counter(str(value) for value in universe["terminal_state"].tolist())


def build_request_contributions(universe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    invalid_nonzero = 0
    strict_positive_absent = 0
    for row in universe.to_dict("records"):
        strict_valid = bool_value(row["strict_parse_valid"])
        hr_at_5, ndcg_at_5, mrr, rank_zero, contribution_status = utility_from_ranked_indices(
            strict_valid=strict_valid,
            ranked_candidate_indices=row["ranked_candidate_indices"],
            positive_candidate_position=row["positive_candidate_position"],
        )
        if not strict_valid and any(value != 0.0 for value in (hr_at_5, ndcg_at_5, mrr)):
            invalid_nonzero += 1
        if contribution_status == "positive_absent_from_top10":
            strict_positive_absent += 1
        rows.append(
            {
                "canonical_request_id": row["canonical_request_id"],
                "dataset": row["dataset"],
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "candidate_order_condition": row["candidate_order_condition"],
                "user_id": row["user_id"],
                "terminal_state": row["terminal_state"],
                "strict_parse_valid": bool_text(strict_valid),
                "validity_class": row["validity_class"],
                "rank_zero_based": rank_zero,
                "hr_at_5": hr_at_5,
                "ndcg_at_5": ndcg_at_5,
                "mrr": mrr,
                "utility_contribution_status": contribution_status,
            }
        )
    return pd.DataFrame(rows), {
        "invalid_nonzero_contribution_rows": invalid_nonzero,
        "strict_positive_absent_from_top10_rows": strict_positive_absent,
    }


def utility_metric_rows(contrib: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    grouped = contrib.groupby(["dataset", "model", "prompt_strategy", "candidate_order_condition"], sort=False)
    for dataset in DATASETS:
        for model in MODELS:
            for prompt in PROMPTS:
                for order in ORDERS:
                    group = grouped.get_group((dataset, model, prompt, order))
                    planned = len(group)
                    strict_valid = int(group["strict_parse_valid"].eq("true").sum())
                    invalid = planned - strict_valid
                    backend_errors = int(group["terminal_state"].eq("BACKEND_FAILURE").sum())
                    metrics = {
                        "HR@5": float(group["hr_at_5"].sum() / planned),
                        "NDCG@5": float(group["ndcg_at_5"].sum() / planned),
                        "MRR": float(group["mrr"].sum() / planned),
                    }
                    sums = {
                        "HR@5": float(group["hr_at_5"].sum()),
                        "NDCG@5": float(group["ndcg_at_5"].sum()),
                        "MRR": float(group["mrr"].sum()),
                    }
                    for metric_id in UTILITY_METRICS:
                        rows.append(
                            {
                                "result_id": stable_id("G3MET", dataset, model, prompt, order, metric_id),
                                "phase_id": PHASE_ID,
                                "dataset": dataset,
                                "model": model,
                                "prompt_strategy": prompt,
                                "candidate_order_condition": order,
                                "metric_id": metric_id,
                                "metric_family": "utility",
                                "k": "5" if metric_id in {"HR@5", "NDCG@5"} else "all_top10",
                                "value": format_float(metrics[metric_id]),
                                "numerator_sum": format_float(sums[metric_id]),
                                "planned_applicable_users": str(planned),
                                "strict_valid_users": str(strict_valid),
                                "invalid_terminal_users": str(invalid),
                                "backend_error_users": str(backend_errors),
                                "denominator_policy": "planned_applicable_users_invalid_terminal_zero",
                                "invalid_terminal_policy": "contributes_zero_utility",
                                "directionality_declared": "higher_is_better",
                                "source_table": PATHS["analysis_universe"].as_posix(),
                                "notes": "Frozen G3 utility cell; invalid terminal responses contribute zero.",
                            }
                        )
    return pd.DataFrame(rows)


def build_canddif_table(metric_results: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    utility = metric_results.set_index(["dataset", "model", "prompt_strategy", "candidate_order_condition", "metric_id"])
    pair_groups = pairs.groupby(["dataset", "model", "prompt_strategy"], sort=False)
    for dataset in DATASETS:
        for model in MODELS:
            for prompt in PROMPTS:
                pair_group = pair_groups.get_group((dataset, model, prompt))
                pair_count = len(pair_group)
                for metric_id, first_metric, random_metric in [
                    ("CandDif_HR@5", "HR@5", "HR@5"),
                    ("CandDif_NDCG@5", "NDCG@5", "NDCG@5"),
                ]:
                    first = utility.loc[(dataset, model, prompt, "FIRST", first_metric)]
                    random = utility.loc[(dataset, model, prompt, "RANDOM", random_metric)]
                    acc_first = float(first["value"])
                    acc_random = float(random["value"])
                    value, safety = canddif_value(acc_first, acc_random)
                    rows.append(
                        {
                            "result_id": stable_id("G3CD", dataset, model, prompt, metric_id),
                            "phase_id": PHASE_ID,
                            "dataset": dataset,
                            "model": model,
                            "prompt_strategy": prompt,
                            "metric_id": metric_id,
                            "metric_family": "candidate_position",
                            "source_accuracy_metric": first_metric,
                            "first_accuracy": format_float(acc_first),
                            "random_accuracy": format_float(acc_random),
                            "first_error_rate": format_float(1.0 - acc_first),
                            "random_error_rate": format_float(1.0 - acc_random),
                            "value": format_float(value),
                            "formula": "log((1-Acc_random)/(1-Acc_first))",
                            "epsilon": "1e-12",
                            "numerical_safety_policy": safety,
                            "paired_users": str(pair_count),
                            "planned_first_users": str(first["planned_applicable_users"]),
                            "planned_random_users": str(random["planned_applicable_users"]),
                            "strict_valid_first": str(first["strict_valid_users"]),
                            "strict_valid_random": str(random["strict_valid_users"]),
                            "invalid_first": str(first["invalid_terminal_users"]),
                            "invalid_random": str(random["invalid_terminal_users"]),
                            "denominator_policy": "paired_FIRST_RANDOM_cell_accuracies_invalid_terminal_zero",
                            "source_table": OUTPUTS["metric_results"].as_posix(),
                            "notes": "Positive values mean FIRST has lower error than RANDOM under the frozen CandDif formula.",
                        }
                    )
    return pd.DataFrame(rows)


def reliability_row(
    scope: str,
    scope_value: str,
    frame: pd.DataFrame,
    *,
    dataset: str = "",
    model: str = "",
    prompt: str = "",
    order: str = "",
) -> dict[str, str]:
    planned = len(frame)
    strict_valid = int(frame["terminal_state"].eq("STRICT_VALID").sum())
    invalid = planned - strict_valid
    backend_errors = int(frame["terminal_state"].eq("BACKEND_FAILURE").sum())
    invalid_length = int(frame["terminal_state"].eq("INVALID_LENGTH").sum())
    invalid_duplicate = int(frame["terminal_state"].eq("INVALID_DUPLICATE").sum())
    hallucination = int(frame["terminal_state"].eq("HALLUCINATION_OR_OUT_OF_CANDIDATE").sum())
    return {
        "result_id": stable_id("G3REL", scope, scope_value),
        "phase_id": PHASE_ID,
        "scope": scope,
        "scope_value": scope_value,
        "dataset": dataset or (scope_value if scope == "dataset" else ""),
        "model": model or (scope_value if scope == "model" else ""),
        "prompt_strategy": prompt or (scope_value if scope == "prompt" else ""),
        "candidate_order_condition": order or (scope_value if scope == "order" else ""),
        "planned_requests": str(planned),
        "strict_valid": str(strict_valid),
        "invalid": str(invalid),
        "backend_errors": str(backend_errors),
        "invalid_length": str(invalid_length),
        "invalid_duplicate": str(invalid_duplicate),
        "hallucination_or_out_of_candidate": str(hallucination),
        "strict_valid_rate": format_float(strict_valid / planned if planned else 0.0),
        "invalid_rate": format_float(invalid / planned if planned else 0.0),
        "backend_error_rate": format_float(backend_errors / planned if planned else 0.0),
        "denominator_policy": "planned_requests",
        "source_table": PATHS["analysis_universe"].as_posix(),
    }


def reliability_rows(universe: pd.DataFrame) -> pd.DataFrame:
    rows = [reliability_row("overall", "ALL", universe)]
    for order in ORDERS:
        rows.append(reliability_row("order", order, universe.loc[universe["candidate_order_condition"].eq(order)]))
    for model in MODELS:
        rows.append(reliability_row("model", model, universe.loc[universe["model"].eq(model)]))
    for prompt in PROMPTS:
        rows.append(reliability_row("prompt", prompt, universe.loc[universe["prompt_strategy"].eq(prompt)]))
    for dataset in DATASETS:
        rows.append(reliability_row("dataset", dataset, universe.loc[universe["dataset"].eq(dataset)]))
    for model in MODELS:
        for prompt in PROMPTS:
            for order in ORDERS:
                rows.append(
                    reliability_row(
                        "model_prompt_order",
                        f"{model}|{prompt}|{order}",
                        universe.loc[
                            universe["model"].eq(model)
                            & universe["prompt_strategy"].eq(prompt)
                            & universe["candidate_order_condition"].eq(order)
                        ],
                        model=model,
                        prompt=prompt,
                        order=order,
                    )
                )
    return pd.DataFrame(rows)


def denominator_rows(metric_results: pd.DataFrame, canddif: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    utility_cells = metric_results.drop_duplicates(["dataset", "model", "prompt_strategy", "candidate_order_condition"])
    for row in utility_cells.to_dict("records"):
        rows.append(
            {
                "denominator_id": stable_id("G3DEN", "utility", row["dataset"], row["model"], row["prompt_strategy"], row["candidate_order_condition"]),
                "metric_family": "utility",
                "applies_to": "HR@5;NDCG@5;MRR",
                "dataset": row["dataset"],
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "candidate_order_condition": row["candidate_order_condition"],
                "scope": "dataset_model_prompt_order",
                "denominator_unit": "planned_applicable_users",
                "denominator_value": row["planned_applicable_users"],
                "strict_valid": row["strict_valid_users"],
                "invalid_terminal": row["invalid_terminal_users"],
                "backend_errors": row["backend_error_users"],
                "invalid_terminal_policy": "invalid terminal response contributes zero utility",
                "source": PATHS["analysis_universe"].as_posix(),
                "status": "FROZEN_APPLICABLE",
            }
        )
    canddif_cells = canddif.drop_duplicates(["dataset", "model", "prompt_strategy"])
    for row in canddif_cells.to_dict("records"):
        rows.append(
            {
                "denominator_id": stable_id("G3DEN", "canddif", row["dataset"], row["model"], row["prompt_strategy"]),
                "metric_family": "candidate_position",
                "applies_to": "CandDif_HR@5;CandDif_NDCG@5",
                "dataset": row["dataset"],
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "candidate_order_condition": "FIRST_RANDOM_PAIR",
                "scope": "dataset_model_prompt_pair",
                "denominator_unit": "paired_FIRST_RANDOM_users",
                "denominator_value": row["paired_users"],
                "strict_valid": f"FIRST={row['strict_valid_first']};RANDOM={row['strict_valid_random']}",
                "invalid_terminal": f"FIRST={row['invalid_first']};RANDOM={row['invalid_random']}",
                "backend_errors": "0",
                "invalid_terminal_policy": "invalid terminal response contributes zero to underlying utility accuracy",
                "source": OUTPUTS["canddif"].as_posix(),
                "status": "FROZEN_APPLICABLE",
            }
        )
    for row in reliability.to_dict("records"):
        rows.append(
            {
                "denominator_id": stable_id("G3DEN", "reliability", row["scope"], row["scope_value"]),
                "metric_family": "reliability",
                "applies_to": "strict_valid_rate;invalid_rate;backend_error_rate;invalid_class_counts",
                "dataset": row["dataset"],
                "model": row["model"],
                "prompt_strategy": row["prompt_strategy"],
                "candidate_order_condition": row["candidate_order_condition"],
                "scope": row["scope"],
                "denominator_unit": "planned_requests",
                "denominator_value": row["planned_requests"],
                "strict_valid": row["strict_valid"],
                "invalid_terminal": row["invalid"],
                "backend_errors": row["backend_errors"],
                "invalid_terminal_policy": "invalid terminal response remains terminal scientific evidence",
                "source": OUTPUTS["reliability"].as_posix(),
                "status": "FROZEN_APPLICABLE",
            }
        )
    rows.append(
        {
            "denominator_id": stable_id("G3DEN", "beyond_utility", "not_computed"),
            "metric_family": "beyond_utility",
            "applies_to": "none_in_G3_metrics_freeze",
            "dataset": "",
            "model": "",
            "prompt_strategy": "",
            "candidate_order_condition": "",
            "scope": "policy",
            "denominator_unit": "strict_valid_users_or_metric_specific_defined_subset",
            "denominator_value": "",
            "strict_valid": "",
            "invalid_terminal": "",
            "backend_errors": "",
            "invalid_terminal_policy": "excluded from strict-valid beyond-utility denominator",
            "source": PATHS["analysis_denominator_contract"].as_posix(),
            "status": "CONDITIONAL_NOT_COMPUTED_NO_FROZEN_G3_BEYOND_UTILITY_METRIC_REQUIRED",
        }
    )
    return pd.DataFrame(rows)


def compare_current_hashes(root: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    expected = {}
    expected.update(manifest.get("input_hashes") or {})
    expected.update(manifest.get("output_hashes") or {})
    expected.update(manifest.get("source_hashes") or {})
    for path_text, expected_sha in expected.items():
        path = root / Path(path_text)
        if not path.exists():
            mismatches.append(f"{path_text}:missing")
            continue
        observed_sha = sha256_file(path)
        observed[path_text] = observed_sha
        if str(observed_sha) != str(expected_sha):
            mismatches.append(f"{path_text}:sha256_mismatch")
    return observed, mismatches


def build_validation(
    *,
    head_before: str,
    status_before: str,
    diff_check_returncode: int,
    analysis_manifest: Mapping[str, Any],
    hash_mismatches: list[str],
    universe: pd.DataFrame,
    pairs: pd.DataFrame,
    contrib: pd.DataFrame,
    contribution_summary: Mapping[str, int],
    metric_results: pd.DataFrame,
    canddif: pd.DataFrame,
    reliability: pd.DataFrame,
    denominators: pd.DataFrame,
    allow_generator_dirty: bool,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    state_counts = frame_state_counts(universe)
    order_counts = universe["candidate_order_condition"].value_counts().to_dict()
    utility_cell_counts = metric_results.groupby("metric_id").size().to_dict()
    canddif_metric_counts = canddif.groupby("metric_id").size().to_dict()
    expected_status_clean = "clean_or_expected_dirty" if allow_generator_dirty else "clean"

    add_validation(rows, "STATIC-001", "current HEAD matches G3 analytic-universe freeze precondition", head_before, EXPECTED_HEAD, head_before == EXPECTED_HEAD)
    add_validation(rows, "STATIC-002", "git status contains only expected generator/output paths", git_dirty_paths(status_before), expected_status_clean, status_is_clean_or_expected(status_before, allow_generator_dirty))
    add_validation(rows, "STATIC-003", "git diff --check passes before metric freeze", diff_check_returncode, 0, diff_check_returncode == 0)
    add_validation(rows, "STATIC-004", "analysis manifest decision token is frozen", analysis_manifest.get("decision_token"), "G3_ANALYTIC_UNIVERSE_FROZEN", analysis_manifest.get("decision_token") == "G3_ANALYTIC_UNIVERSE_FROZEN")
    add_validation(rows, "STATIC-005", "analysis manifest validation passed", analysis_manifest.get("validation_passed"), True, bool(analysis_manifest.get("validation_passed")) is True)
    add_validation(rows, "STATIC-006", "frozen analysis inputs and outputs still match manifest hashes", len(hash_mismatches), 0, len(hash_mismatches) == 0)
    add_validation(rows, "UNIVERSE-001", "analysis universe row count", len(universe), EXPECTED_LOGICAL_ROWS, len(universe) == EXPECTED_LOGICAL_ROWS)
    add_validation(rows, "UNIVERSE-002", "analysis universe canonical IDs are unique", universe["canonical_request_id"].nunique(), EXPECTED_LOGICAL_ROWS, universe["canonical_request_id"].nunique() == EXPECTED_LOGICAL_ROWS)
    add_validation(rows, "UNIVERSE-003", "FIRST/RANDOM request counts", {key: int(order_counts.get(key, 0)) for key in ORDERS}, EXPECTED_ORDER_ROWS, all(int(order_counts.get(key, 0)) == value for key, value in EXPECTED_ORDER_ROWS.items()))
    add_validation(rows, "UNIVERSE-004", "terminal-state counts match frozen execution accounting", dict(sorted(state_counts.items())), EXPECTED_STATE_COUNTS, all(int(state_counts.get(key, 0)) == value for key, value in EXPECTED_STATE_COUNTS.items()) and sum(state_counts.values()) == EXPECTED_LOGICAL_ROWS)
    add_validation(rows, "UNIVERSE-005", "backend errors remain zero", int(universe["terminal_state"].eq("BACKEND_FAILURE").sum()), 0, int(universe["terminal_state"].eq("BACKEND_FAILURE").sum()) == 0)
    add_validation(rows, "UNIVERSE-006", "no missing duplicate ambiguous evidence in analysis manifest", analysis_manifest.get("validation_failures"), [], analysis_manifest.get("validation_failures") == [])
    add_validation(rows, "PAIR-001", "frozen pair row count", len(pairs), EXPECTED_PAIR_ROWS, len(pairs) == EXPECTED_PAIR_ROWS)
    add_validation(rows, "PAIR-002", "frozen FIRST/RANDOM pair membership unchanged", int(pairs["only_candidate_order_differs"].map(bool_value).sum()), EXPECTED_PAIR_ROWS, int(pairs["only_candidate_order_differs"].map(bool_value).sum()) == EXPECTED_PAIR_ROWS)
    add_validation(rows, "PAIR-003", "FIRST positive position is 1 and RANDOM positive position is 2..20", int((pairs["first_positive_position_ok"].map(bool_value) & pairs["random_positive_position_ok"].map(bool_value)).sum()), EXPECTED_PAIR_ROWS, int((pairs["first_positive_position_ok"].map(bool_value) & pairs["random_positive_position_ok"].map(bool_value)).sum()) == EXPECTED_PAIR_ROWS)
    add_validation(rows, "UTILITY-001", "request-level utility contributions cover all logical rows", len(contrib), EXPECTED_LOGICAL_ROWS, len(contrib) == EXPECTED_LOGICAL_ROWS)
    add_validation(rows, "UTILITY-002", "invalid terminal utility contributions are zero", contribution_summary["invalid_nonzero_contribution_rows"], 0, contribution_summary["invalid_nonzero_contribution_rows"] == 0)
    add_validation(rows, "UTILITY-003", "utility result row count is complete", len(metric_results), 180, len(metric_results) == 180)
    add_validation(rows, "UTILITY-004", "HR/NDCG/MRR each have all expected cells", utility_cell_counts, {"HR@5": 60, "NDCG@5": 60, "MRR": 60}, all(int(utility_cell_counts.get(metric, 0)) == 60 for metric in UTILITY_METRICS))
    add_validation(rows, "UTILITY-005", "utility denominators are 200 planned users per order cell", sorted(metric_results["planned_applicable_users"].unique().tolist()), ["200"], sorted(metric_results["planned_applicable_users"].unique().tolist()) == ["200"])
    add_validation(rows, "CANDDIF-001", "CandDif result rows cover HR@5 and NDCG@5 cells", len(canddif), 60, len(canddif) == 60)
    add_validation(rows, "CANDDIF-002", "CandDif metric-specific cell counts are complete", canddif_metric_counts, {"CandDif_HR@5": 30, "CandDif_NDCG@5": 30}, all(int(canddif_metric_counts.get(metric, 0)) == 30 for metric in CANDDIF_METRICS))
    add_validation(rows, "CANDDIF-003", "CandDif paired-user denominator is 200 for every dataset/model/prompt", sorted(canddif["paired_users"].unique().tolist()), ["200"], sorted(canddif["paired_users"].unique().tolist()) == ["200"])
    add_validation(rows, "RELIABILITY-001", "reliability summary rows cover required scopes", len(reliability), 25, len(reliability) == 25)
    add_validation(rows, "DENOM-001", "denominator table covers utility CandDif reliability and beyond-utility policy rows", len(denominators), 116, len(denominators) == 116)
    add_validation(rows, "PHASE-001", "zero inference and no bootstrap/statistical tests declared", "zero_inference=true;statistical_analysis_executed=false", "zero_inference=true;statistical_analysis_executed=false", True)
    return pd.DataFrame(rows)


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


def render_report(
    *,
    head_before: str,
    validation: pd.DataFrame,
    metric_results: pd.DataFrame,
    canddif: pd.DataFrame,
    reliability: pd.DataFrame,
    denominators: pd.DataFrame,
    contribution_summary: Mapping[str, int],
) -> str:
    pass_count = int(validation["status"].eq("PASS").sum())
    total_count = len(validation)
    overall = reliability.loc[reliability["scope"].eq("overall")].iloc[0]
    order = reliability.loc[reliability["scope"].eq("order"), ["scope_value", "planned_requests", "strict_valid", "invalid", "backend_errors"]]
    metric_counts = metric_results.groupby("metric_id").size().to_dict()
    canddif_counts = canddif.groupby("metric_id").size().to_dict()
    lines = [
        "# G3 Candidate-Position Metrics Freeze",
        "",
        f"- Phase: `{PHASE_ID}`",
        f"- HEAD before generation: `{head_before}`",
        "- Inputs: frozen G3 analytic universe and pair table only.",
        "- Inference/API calls: `0`.",
        "- Bootstrap/statistical tests: `0`.",
        "",
        "## Metric Completeness",
        "",
        f"- Utility metric rows: `{len(metric_results)}` (`HR@5={metric_counts.get('HR@5', 0)}`, `NDCG@5={metric_counts.get('NDCG@5', 0)}`, `MRR={metric_counts.get('MRR', 0)}`).",
        f"- CandDif rows: `{len(canddif)}` (`CandDif_HR@5={canddif_counts.get('CandDif_HR@5', 0)}`, `CandDif_NDCG@5={canddif_counts.get('CandDif_NDCG@5', 0)}`).",
        f"- Reliability rows: `{len(reliability)}`.",
        f"- Denominator rows: `{len(denominators)}`.",
        "",
        "## Denominator Semantics",
        "",
        "- Utility: planned/applicable users; invalid terminal responses contribute zero.",
        "- Reliability: planned requests.",
        "- CandDif: paired FIRST/RANDOM cell accuracies over the frozen 6000-pair universe.",
        "- Beyond-utility metrics: not computed in this phase unless separately frozen/applicable to G3.",
        "",
        "## Reliability Summary",
        "",
        f"- Overall requests: `{overall['planned_requests']}`.",
        f"- Strict valid: `{overall['strict_valid']}`.",
        f"- Invalid: `{overall['invalid']}`.",
        f"- Backend errors: `{overall['backend_errors']}`.",
        "",
        markdown_table(order),
        "",
        "## Validation",
        "",
        f"- PASS: `{pass_count}/{total_count}`.",
        f"- Invalid nonzero utility contribution rows: `{contribution_summary['invalid_nonzero_contribution_rows']}`.",
        f"- Strict-valid positive absent from top-10 rows: `{contribution_summary['strict_positive_absent_from_top10_rows']}`.",
        "",
        markdown_table(validation),
        "",
        "## Outputs",
        "",
        f"- `{OUTPUTS['metric_results'].as_posix()}`",
        f"- `{OUTPUTS['canddif'].as_posix()}`",
        f"- `{OUTPUTS['reliability'].as_posix()}`",
        f"- `{OUTPUTS['denominators'].as_posix()}`",
        f"- `{OUTPUTS['validation'].as_posix()}`",
        f"- `{OUTPUTS['manifest'].as_posix()}`",
    ]
    return "\n".join(lines) + "\n"


def append_notes(root: Path, created_at: str, summary: Mapping[str, Any]) -> None:
    path = root / PATHS["notes"]
    marker = f"## {created_at} - G3 candidate-position metrics freeze"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    kept_lines: list[str] = []
    skipping = False
    for line in existing.splitlines():
        if line.startswith("## ") and line.endswith(" - G3 candidate-position metrics freeze"):
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            kept_lines.append(line)
    existing = "\n".join(kept_lines)
    block = "\n".join(
        [
            "",
            marker,
            "",
            "- Computed frozen G3 utility metrics from `results/tables/g3_candidate_position_analysis_universe.csv` only.",
            f"- Utility rows: `{summary['metric_rows']}`; CandDif rows: `{summary['canddif_rows']}`; reliability rows: `{summary['reliability_rows']}`.",
            f"- Reliability: strict-valid `{summary['strict_valid']}`, invalid `{summary['invalid']}`, backend errors `{summary['backend_errors']}`.",
            f"- Validation: `{summary['validation_pass_count']}/{summary['validation_total_count']}` PASS.",
            "- No Ollama/API calls, bootstrap tests, G3 statistics, or downstream phases were run.",
            "- Next safe step: frozen G3 statistical analysis over the paired universe and frozen metric artifacts.",
            "",
        ]
    )
    path.write_text(existing.rstrip() + block, encoding="utf-8", newline="\n")


def compute(root: Path, *, allow_generator_dirty: bool) -> dict[str, Any]:
    head_before = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short", "--untracked-files=all"], check=False)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True)

    analysis_manifest = read_json(root, PATHS["analysis_manifest"])
    observed_input_hashes, frozen_hash_mismatches = compare_current_hashes(root, analysis_manifest)
    universe = read_csv(root, PATHS["analysis_universe"])
    pairs = read_csv(root, PATHS["analysis_pairs"])

    contrib, contribution_summary = build_request_contributions(universe)
    metric_results = utility_metric_rows(contrib)
    canddif = build_canddif_table(metric_results, pairs)
    reliability = reliability_rows(universe)
    denominators = denominator_rows(metric_results, canddif, reliability)
    validation = build_validation(
        head_before=head_before,
        status_before=status_before,
        diff_check_returncode=diff_check.returncode,
        analysis_manifest=analysis_manifest,
        hash_mismatches=frozen_hash_mismatches,
        universe=universe,
        pairs=pairs,
        contrib=contrib,
        contribution_summary=contribution_summary,
        metric_results=metric_results,
        canddif=canddif,
        reliability=reliability,
        denominators=denominators,
        allow_generator_dirty=allow_generator_dirty,
    )

    pass_count = int(validation["status"].eq("PASS").sum())
    validation_total = int(len(validation))
    summary = {
        "logical_rows": int(len(universe)),
        "pair_rows": int(len(pairs)),
        "metric_rows": int(len(metric_results)),
        "utility_cells_per_metric": {metric: int((metric_results["metric_id"] == metric).sum()) for metric in UTILITY_METRICS},
        "canddif_rows": int(len(canddif)),
        "canddif_cells_per_metric": {metric: int((canddif["metric_id"] == metric).sum()) for metric in CANDDIF_METRICS},
        "reliability_rows": int(len(reliability)),
        "denominator_rows": int(len(denominators)),
        "strict_valid": int(universe["terminal_state"].eq("STRICT_VALID").sum()),
        "invalid": int(len(universe) - universe["terminal_state"].eq("STRICT_VALID").sum()),
        "backend_errors": int(universe["terminal_state"].eq("BACKEND_FAILURE").sum()),
        "valid_by_order": {
            order: {
                "strict_valid": int(universe.loc[universe["candidate_order_condition"].eq(order), "terminal_state"].eq("STRICT_VALID").sum()),
                "invalid": int(universe["candidate_order_condition"].eq(order).sum() - universe.loc[universe["candidate_order_condition"].eq(order), "terminal_state"].eq("STRICT_VALID").sum()),
            }
            for order in ORDERS
        },
        "invalid_class_counts": {key: int(value) for key, value in sorted(frame_state_counts(universe).items()) if key != "STRICT_VALID"},
        "invalid_nonzero_contribution_rows": contribution_summary["invalid_nonzero_contribution_rows"],
        "strict_positive_absent_from_top10_rows": contribution_summary["strict_positive_absent_from_top10_rows"],
        "validation_pass_count": pass_count,
        "validation_total_count": validation_total,
        "validation_failures": validation.loc[validation["status"].eq("FAIL"), "check_id"].tolist(),
    }
    return {
        "head_before": head_before,
        "status_before": status_before,
        "diff_check_stderr": diff_check.stderr.strip(),
        "analysis_manifest": analysis_manifest,
        "observed_input_hashes": observed_input_hashes,
        "frozen_hash_mismatches": frozen_hash_mismatches,
        "universe": universe,
        "pairs": pairs,
        "metric_results": metric_results,
        "canddif": canddif,
        "reliability": reliability,
        "denominators": denominators,
        "validation": validation,
        "contribution_summary": contribution_summary,
        "summary": summary,
    }


def write_outputs(root: Path, payload: dict[str, Any], created_at: str) -> dict[str, str]:
    write_csv(root / OUTPUTS["metric_results"], payload["metric_results"])
    write_csv(root / OUTPUTS["canddif"], payload["canddif"])
    write_csv(root / OUTPUTS["reliability"], payload["reliability"])
    write_csv(root / OUTPUTS["denominators"], payload["denominators"])
    write_csv(root / OUTPUTS["validation"], payload["validation"])
    report_text = render_report(
        head_before=payload["head_before"],
        validation=payload["validation"],
        metric_results=payload["metric_results"],
        canddif=payload["canddif"],
        reliability=payload["reliability"],
        denominators=payload["denominators"],
        contribution_summary=payload["contribution_summary"],
    )
    (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUTS["report"]).write_text(report_text, encoding="utf-8", newline="\n")
    append_notes(root, created_at, payload["summary"])

    output_hashes = {path.as_posix(): sha256_file(root / path) for path in OUTPUTS.values() if path != OUTPUTS["manifest"]}
    source_hashes = {PATHS["source"].as_posix(): sha256_file(root / PATHS["source"])}
    contract_hashes = {
        PATHS["metric_contract_source"].as_posix(): sha256_file(root / PATHS["metric_contract_source"]),
        PATHS["canddif_contract_source"].as_posix(): sha256_file(root / PATHS["canddif_contract_source"]),
    }
    manifest = {
        "schema_version": "1.0",
        "phase_id": PHASE_ID,
        "decision_token": DECISION_FROZEN if not payload["summary"]["validation_failures"] else DECISION_BLOCKED,
        "created_at_utc": created_at,
        "precondition_status": "G3_ANALYTIC_UNIVERSE_FROZEN",
        "head_before_generation": payload["head_before"],
        "zero_inference": True,
        "statistical_analysis_executed": False,
        "bootstrap_executed": False,
        "protocol_config_runner_modified": False,
        "frozen_evidence_modified": False,
        "analysis_universe_source": PATHS["analysis_universe"].as_posix(),
        "analysis_pair_source": PATHS["analysis_pairs"].as_posix(),
        "input_hashes_observed": payload["observed_input_hashes"],
        "input_hash_mismatches": payload["frozen_hash_mismatches"],
        "metric_contract_hashes": contract_hashes,
        "source_hashes": source_hashes,
        "output_hashes": output_hashes,
        "summary": payload["summary"],
        "denominator_contract": {
            "utility": "planned/applicable users; invalid terminal response contributes zero",
            "reliability": "planned requests",
            "canddif": "paired FIRST/RANDOM cell accuracies over frozen 6000-pair universe",
            "beyond_utility": "not computed in this phase unless separately frozen/applicable to G3",
        },
        "canddif_formula": {
            "formula": "log((1-Acc_random)/(1-Acc_first))",
            "epsilon": "1e-12",
            "source": PATHS["canddif_contract_source"].as_posix(),
        },
        "files_created": [path.as_posix() for path in [PATHS["source"], *OUTPUTS.values(), PATHS["notes"]]],
        "next_safe_step": "run frozen G3 statistical analysis; do not start G4/G3 downstream without explicit request",
    }
    write_json(root / OUTPUTS["manifest"], manifest)
    output_hashes[OUTPUTS["manifest"].as_posix()] = sha256_file(root / OUTPUTS["manifest"])
    return output_hashes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_project_root(Path(args.output_root).resolve())
    created_at = utc_now()
    payload = compute(root, allow_generator_dirty=args.allow_generator_dirty)
    if not args.no_write:
        write_outputs(root, payload, created_at)

    failures = payload["summary"]["validation_failures"]
    print(json.dumps({"decision": DECISION_FROZEN if not failures else DECISION_BLOCKED, "summary": payload["summary"]}, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

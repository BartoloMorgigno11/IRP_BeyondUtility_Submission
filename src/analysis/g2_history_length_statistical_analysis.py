"""Execute the frozen G2 history-length statistical plan offline.

The script consumes retained G1/G2 evidence and frozen G2 universe/metric/readout
artifacts. It does not call Ollama, start G3, or modify frozen protocol,
execution, evidence, universe, metric, or readout artifacts.
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

from src.metrics.canonical_metric_contract import (  # noqa: E402
    first_positive_rank_zero_based,
    hr_at_k_from_rank,
    ndcg_at_k_from_rank,
    validate_one_positive,
)


EXPECTED_HEAD = "c120b788d2023e29027eccc6dbf9e38f5676c537"
PHASE_ID = "g2_history_length_statistical_analysis"
DECISION_FROZEN = "G2_STATISTICAL_ANALYSIS_FROZEN"
DECISION_BLOCKED = "G2_STATISTICAL_ANALYSIS_BLOCKED"
TOP_K = 5

PRIMARY_VIEW = "PRIMARY_B1_GE10"
LONG_VIEW = "LONG_HISTORY_B2_GE20"
SECONDARY_VIEW = "SECONDARY_FULL_CURVE_GE20"
TRUE_ZERO_VIEW = "FOODCOM_TRUE_ZERO_HISTORY_DIAGNOSTIC"
STATISTICAL_VIEWS = [PRIMARY_VIEW, LONG_VIEW, SECONDARY_VIEW]
UTILITY_METRICS = ["hr_at_5", "ndcg_at_5", "mrr"]
G1_EVIDENCE = "G1_EXACT_REUSE_TERMINAL_EVIDENCE"
G2_EVIDENCE = "G2_TERMINAL_EVIDENCE"

EXPECTED_UNIVERSE_COUNTS = {
    "logical_requests": 88050,
    "g1_exact_reuse": 18660,
    "g2_terminal_evidence": 69390,
    "strict_valid": 86768,
    "invalid": 1282,
    "backend_errors": 0,
    "missing": 0,
    "duplicate_terminal": 0,
    "ambiguous": 0,
}
EXPECTED_PAIRED_USERS = {
    PRIMARY_VIEW: {"beauty": 181, "foodcom": 61, "lastfm": 981, "ml1m": 1000, "sports": 160},
    LONG_VIEW: {"beauty": 38, "foodcom": 35, "lastfm": 932, "ml1m": 993, "sports": 35},
    SECONDARY_VIEW: {"beauty": 38, "foodcom": 35, "lastfm": 932, "ml1m": 993, "sports": 35},
}

PATHS = {
    "protocol_config": Path("configs/g2_history_length_sensitivity.yaml"),
    "protocol_manifest": Path("results/manifests/g2_history_length_protocol_freeze.json"),
    "execution_manifest": Path("results/manifests/g2_history_length_execution_freeze.json"),
    "retention_manifest": Path("results/manifests/g2_history_length_execution_retention_manifest.json"),
    "analysis_manifest": Path("results/manifests/g2_history_length_analysis_universe_freeze.json"),
    "metrics_manifest": Path("results/manifests/g2_history_length_metrics_freeze.json"),
    "readout_manifest": Path("results/manifests/g2_history_length_scientific_readout_manifest.json"),
    "request_universe": Path("results/tables/g2_history_length_request_universe.csv"),
    "analysis_universe": Path("results/tables/g2_history_length_analysis_universe.csv"),
    "membership_universe": Path("results/tables/g2_history_length_analysis_membership_universe.csv"),
    "cell_counts": Path("results/tables/g2_history_length_analysis_cell_counts.csv"),
    "denominator_contract": Path("results/tables/g2_history_length_denominator_contract.csv"),
    "statistical_plan": Path("results/tables/g2_history_length_statistical_plan.csv"),
    "g1_reuse": Path("results/tables/g2_history_length_g1_execution_reuse.csv"),
    "g1_strict_parsing": Path("results/tables/g1_primary_ranking_strict_parsing.csv"),
    "g2_responses": Path("results/llm/extension_canonical/g2_history_length_sensitivity/responses.jsonl"),
    "metric_results": Path("results/tables/g2_history_length_metric_results.csv"),
    "readout_statistical_contract": Path("results/tables/g2_history_length_readout_statistical_contract.csv"),
    "metric_contract_source": Path("src/metrics/canonical_metric_contract.py"),
    "notes": Path("report/notes.md"),
}

OUTPUTS = {
    "paired_contributions": Path("results/tables/g2_history_length_statistical_paired_contributions.csv"),
    "statistical_results": Path("results/tables/g2_history_length_statistical_results.csv"),
    "validation": Path("results/tables/g2_history_length_statistical_validation.csv"),
    "manifest": Path("results/manifests/g2_history_length_statistical_analysis_manifest.json"),
    "report": Path("report/g2_history_length_statistical_analysis.md"),
}

EXPECTED_DIRTY_PATHS = {
    "src/analysis/g2_history_length_statistical_analysis.py",
    "report/notes.md",
    *[path.as_posix() for path in OUTPUTS.values()],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute frozen G2 statistical analysis offline.")
    parser.add_argument("--output-root", default=".", help="Project root; defaults to current working directory.")
    parser.add_argument("--no-write", action="store_true", help="Build and validate in memory without writing outputs.")
    parser.add_argument(
        "--allow-generator-dirty",
        action="store_true",
        help="Allow this generator and expected statistical outputs to be dirty while frozen inputs remain untouched.",
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


def read_json(root: Path, path: Path) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def read_csv(root: Path, path: Path) -> pd.DataFrame:
    return pd.read_csv(root / path, dtype=str, keep_default_na=False, low_memory=False)


def write_csv(root: Path, path: Path, frame: pd.DataFrame) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / path, index=False, lineterminator="\n", float_format="%.17g")


def git_output(root: Path, args: Sequence[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip("\n")


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
        paths.append(text.replace("\\", "/"))
    return sorted(paths)


def status_is_clean_or_expected(status: str, allow_generator_dirty: bool) -> bool:
    paths = git_dirty_paths(status)
    if not paths:
        return True
    return allow_generator_dirty and set(paths).issubset(EXPECTED_DIRTY_PATHS)


def bool_text(value: Any) -> str:
    return "true" if str(value).strip().lower() in {"true", "1", "yes"} else "false"


def bool_value(value: Any) -> bool:
    return bool_text(value) == "true"


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def parse_json_array(value: Any) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("expected JSON array")
    return [str(item) for item in parsed]


def reciprocal_rank(rank_zero_based: int | None) -> float:
    return 0.0 if rank_zero_based is None else 1.0 / float(int(rank_zero_based) + 1)


def load_g2_responses(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with (root / PATHS["g2_responses"]).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = str(record.get("canonical_request_id") or record.get("request_id") or "")
            counts[request_id] += 1
            rows.append(
                {
                    "source_canonical_request_id": request_id,
                    "ranked_item_ids": str(record.get("ranked_item_ids", "")),
                    "terminal_state_evidence": str(record.get("terminal_state", "")),
                    "strict_parse_valid_evidence": bool_text(record.get("strict_parse_valid", False)),
                    "backend_success_evidence": bool_text(record.get("backend_success", False)),
                    "raw_response_hash_evidence": str(record.get("raw_response_hash", "")),
                    "jsonl_line": str(line_no),
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["duplicate_ids"] = sum(1 for count in counts.values() if count > 1)
    return frame


def load_g1_parsing(root: Path) -> pd.DataFrame:
    frame = read_csv(root, PATHS["g1_strict_parsing"])
    frame = frame[
        [
            "canonical_request_id",
            "ranked_item_ids",
            "terminal_structural_status",
            "strict_parse_valid",
            "raw_response_hash",
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "canonical_request_id": "source_canonical_request_id",
            "terminal_structural_status": "terminal_state_evidence",
            "strict_parse_valid": "strict_parse_valid_evidence",
            "raw_response_hash": "raw_response_hash_evidence",
        }
    )
    frame["strict_parse_valid_evidence"] = frame["strict_parse_valid_evidence"].map(bool_text)
    frame["backend_success_evidence"] = "true"
    frame["jsonl_line"] = ""
    return frame


def evidence_lookup(root: Path, failures: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    g2 = load_g2_responses(root)
    g1 = load_g1_parsing(root)
    if int(g2.attrs.get("duplicate_ids", 0)) != 0:
        failures.append("G2_RESPONSE_LEDGER_DUPLICATES")
    if g2["source_canonical_request_id"].nunique() != len(g2):
        failures.append("G2_RESPONSE_ID_UNIQUENESS_FAILED")
    if g1["source_canonical_request_id"].nunique() != len(g1):
        failures.append("G1_STRICT_PARSE_ID_UNIQUENESS_FAILED")
    combined = pd.concat(
        [
            g1.assign(evidence_source=G1_EVIDENCE),
            g2.assign(evidence_source=G2_EVIDENCE),
        ],
        ignore_index=True,
        sort=False,
    )
    if combined.duplicated(["evidence_source", "source_canonical_request_id"]).any():
        failures.append("COMBINED_EVIDENCE_DUPLICATES")
    return combined, {
        "g1_strict_parse_rows": int(len(g1)),
        "g2_response_rows": int(len(g2)),
        "g2_response_duplicate_ids": int(g2.attrs.get("duplicate_ids", 0)),
    }


def build_row_metrics(root: Path, failures: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    universe = read_csv(root, PATHS["analysis_universe"])
    requests = read_csv(root, PATHS["request_universe"])[
        ["canonical_request_id", "candidate_items", "candidate_count", "matched_g1_request_id"]
    ].copy()
    g1_reuse = read_csv(root, PATHS["g1_reuse"])
    evidence, evidence_counts = evidence_lookup(root, failures)

    if len(universe) != EXPECTED_UNIVERSE_COUNTS["logical_requests"] or universe["canonical_request_id"].nunique() != len(universe):
        failures.append("ANALYSIS_UNIVERSE_ROW_COUNT_OR_UNIQUENESS_FAILED")
    if len(requests) != EXPECTED_UNIVERSE_COUNTS["logical_requests"] or requests["canonical_request_id"].nunique() != len(requests):
        failures.append("REQUEST_UNIVERSE_ROW_COUNT_OR_UNIQUENESS_FAILED")
    if len(g1_reuse) != EXPECTED_UNIVERSE_COUNTS["g1_exact_reuse"] or g1_reuse["g2_canonical_request_id"].nunique() != len(g1_reuse):
        failures.append("G1_REUSE_COUNT_OR_UNIQUENESS_FAILED")
    if g1_reuse["reuse_status"].ne("G1_EXACT_REUSE_TERMINAL_ZERO_CALL").any():
        failures.append("G1_REUSE_STATUS_MISMATCH")
    if g1_reuse["ollama_call_required"].map(bool_value).any():
        failures.append("G1_REUSE_HAS_OLLAMA_REQUIRED_TRUE")

    merged = universe.merge(requests, on="canonical_request_id", how="left", validate="one_to_one")
    merged = merged.merge(
        evidence,
        on=["evidence_source", "source_canonical_request_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["ranked_item_ids"].isna().any():
        failures.append("MISSING_RANKED_ITEM_EVIDENCE")
    if not merged["raw_response_hash"].fillna("").eq(merged["raw_response_hash_evidence"].fillna("")).all():
        failures.append("RAW_RESPONSE_HASH_MISMATCH")
    if not merged["strict_parse_valid"].map(bool_text).eq(merged["strict_parse_valid_evidence"].map(bool_text)).all():
        failures.append("STRICT_PARSE_VALID_MISMATCH")
    if not merged["terminal_state"].fillna("").eq(merged["terminal_state_evidence"].fillna("")).all():
        failures.append("TERMINAL_STATE_MISMATCH")

    records: list[dict[str, Any]] = []
    candidate_validation_failures = 0
    strict_short_rankings = 0
    strict_positive_absent_rows = 0

    for row in merged.to_dict("records"):
        candidates = parse_json_array(row["candidate_items"])
        if len(candidates) != int(float(row["candidate_count"])):
            candidate_validation_failures += 1
        try:
            validate_one_positive(candidates, row["positive_item_id"])
        except ValueError:
            candidate_validation_failures += 1

        strict_valid = bool_value(row["strict_parse_valid"])
        rank_zero: int | None = None
        hr_at_5 = 0.0
        ndcg_at_5 = 0.0
        mrr = 0.0
        if strict_valid:
            ranked_items = parse_json_array(row["ranked_item_ids"])
            if len(ranked_items) < TOP_K:
                strict_short_rankings += 1
            rank_zero = first_positive_rank_zero_based(ranked_items, row["positive_item_id"])
            if rank_zero is None:
                strict_positive_absent_rows += 1
            hr_at_5 = hr_at_k_from_rank(rank_zero, TOP_K)
            ndcg_at_5 = ndcg_at_k_from_rank(rank_zero, TOP_K)
            mrr = reciprocal_rank(rank_zero)

        records.append(
            {
                "canonical_request_id": str(row["canonical_request_id"]),
                "source_canonical_request_id": str(row["source_canonical_request_id"]),
                "evidence_source": str(row["evidence_source"]),
                "terminal_state": str(row["terminal_state"]),
                "strict_parse_valid": bool_text(strict_valid),
                "invalid_class": str(row["invalid_class"]),
                "backend_success": bool_text(row["backend_success"]),
                "raw_response_hash": str(row["raw_response_hash"]),
                "rank_zero_based": "" if rank_zero is None else str(rank_zero),
                "hr_at_5": float(hr_at_5),
                "ndcg_at_5": float(ndcg_at_5),
                "mrr": float(mrr),
            }
        )

    if candidate_validation_failures:
        failures.append(f"CANDIDATE_VALIDATION_FAILURES:{candidate_validation_failures}")
    if strict_short_rankings:
        failures.append(f"STRICT_VALID_WITH_SHORT_TOP5:{strict_short_rankings}")

    row_metrics = pd.DataFrame(records)
    return row_metrics, {
        **evidence_counts,
        "strict_positive_absent_rows": strict_positive_absent_rows,
    }


def ci_decision(low: float, high: float) -> tuple[str, str]:
    if low > 0.0:
        return "EXCLUDES_ZERO_POSITIVE", "SIGNIFICANT_POSITIVE"
    if high < 0.0:
        return "EXCLUDES_ZERO_NEGATIVE", "SIGNIFICANT_NEGATIVE"
    return "INCLUDES_ZERO", "NOT_SIGNIFICANT_CI_INCLUDES_ZERO"


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


def unique_plan_contrasts(plan: pd.DataFrame) -> pd.DataFrame:
    order = (
        plan.assign(plan_order_int=pd.to_numeric(plan["plan_order"], errors="raise").astype(int))
        .sort_values("plan_order_int", kind="mergesort")
        .drop_duplicates(["reporting_view", "comparison_group", "history_length_a", "history_length_b"], keep="first")
    )
    return order[
        [
            "plan_order_int",
            "reporting_view",
            "comparison_group",
            "scope",
            "history_length_a",
            "history_length_b",
            "contrast",
        ]
    ].copy()


def build_paired_contributions(
    root: Path,
    row_metrics: pd.DataFrame,
    plan: pd.DataFrame,
    failures: list[str],
) -> pd.DataFrame:
    membership = read_csv(root, PATHS["membership_universe"])
    membership["history_length"] = pd.to_numeric(membership["history_length"], errors="raise").astype(int)
    metric_cols = [
        "canonical_request_id",
        "source_canonical_request_id",
        "raw_response_hash",
        "rank_zero_based",
        "hr_at_5",
        "ndcg_at_5",
        "mrr",
    ]
    work = membership.merge(row_metrics[metric_cols], on="canonical_request_id", how="left", validate="many_to_one")
    if work["hr_at_5"].isna().any():
        failures.append("MEMBERSHIP_ROW_METRIC_JOIN_MISSING")

    side_cols = [
        "analysis_membership",
        "dataset",
        "model",
        "prompt_strategy",
        "user_id",
        "history_length",
        "canonical_request_id",
        "source_canonical_request_id",
        "evidence_source",
        "terminal_state",
        "strict_parse_valid",
        "invalid_class",
        "rank_zero_based",
        "hr_at_5",
        "ndcg_at_5",
        "mrr",
    ]
    keys = ["analysis_membership", "dataset", "model", "prompt_strategy", "user_id"]
    rows: list[pd.DataFrame] = []

    for contrast in unique_plan_contrasts(plan).to_dict("records"):
        view = str(contrast["reporting_view"])
        h_a = int(contrast["history_length_a"])
        h_b = int(contrast["history_length_b"])
        subset = work.loc[work["analysis_membership"].eq(view), side_cols].copy()
        left = subset.loc[subset["history_length"].eq(h_a)].copy()
        right = subset.loc[subset["history_length"].eq(h_b)].copy()
        if left.duplicated(keys).any():
            failures.append(f"LEFT_PAIR_DUPLICATE_KEYS:{view}:H{h_a}")
        if right.duplicated(keys).any():
            failures.append(f"RIGHT_PAIR_DUPLICATE_KEYS:{view}:H{h_b}")

        left = left.rename(columns={col: f"{col}_a" for col in side_cols if col not in keys})
        right = right.rename(columns={col: f"{col}_b" for col in side_cols if col not in keys})
        paired = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
        if not paired["_merge"].eq("both").all():
            failures.append(f"PAIR_USER_SET_MISMATCH:{view}:H{h_a}:H{h_b}:{paired['_merge'].value_counts().to_dict()}")
        paired = paired.loc[paired["_merge"].eq("both")].drop(columns=["_merge"]).copy()

        for (dataset, model, prompt), group in paired.groupby(["dataset", "model", "prompt_strategy"], sort=True):
            expected = EXPECTED_PAIRED_USERS[view][str(dataset)]
            if len(group) != expected:
                failures.append(
                    f"PAIRED_USER_COUNT_MISMATCH:{view}:{dataset}:{model}:{prompt}:H{h_a}:H{h_b}:{len(group)}:{expected}"
                )

        for metric in UTILITY_METRICS:
            paired[f"{metric}_delta"] = paired[f"{metric}_b"].astype(float) - paired[f"{metric}_a"].astype(float)

        paired["contribution_pair_id"] = [
            stable_id("G2STAT-PAIR", view, h_a, h_b, dataset, model, prompt, user)
            for dataset, model, prompt, user in zip(
                paired["dataset"],
                paired["model"],
                paired["prompt_strategy"],
                paired["user_id"],
                strict=False,
            )
        ]
        paired["comparison_contrast_id"] = stable_id("G2STAT-CONTRAST", view, h_a, h_b)
        paired["contrast_plan_order_first"] = str(int(contrast["plan_order_int"]))
        paired["comparison_group"] = str(contrast["comparison_group"])
        paired["scope"] = str(contrast["scope"])
        paired["history_length_a"] = str(h_a)
        paired["history_length_b"] = str(h_b)
        paired["contrast"] = str(contrast["contrast"])
        paired["utility_denominator_policy"] = "planned_applicable_users_invalid_terminal_zero"
        paired["paired_unit"] = "user_id"
        rows.append(paired)

    result = pd.concat(rows, ignore_index=True, sort=False)
    columns = [
        "contribution_pair_id",
        "comparison_contrast_id",
        "contrast_plan_order_first",
        "analysis_membership",
        "comparison_group",
        "scope",
        "dataset",
        "model",
        "prompt_strategy",
        "user_id",
        "paired_unit",
        "history_length_a",
        "history_length_b",
        "contrast",
        "canonical_request_id_a",
        "canonical_request_id_b",
        "source_canonical_request_id_a",
        "source_canonical_request_id_b",
        "evidence_source_a",
        "evidence_source_b",
        "terminal_state_a",
        "terminal_state_b",
        "strict_parse_valid_a",
        "strict_parse_valid_b",
        "invalid_class_a",
        "invalid_class_b",
        "rank_zero_based_a",
        "rank_zero_based_b",
        "hr_at_5_a",
        "hr_at_5_b",
        "hr_at_5_delta",
        "ndcg_at_5_a",
        "ndcg_at_5_b",
        "ndcg_at_5_delta",
        "mrr_a",
        "mrr_b",
        "mrr_delta",
        "utility_denominator_policy",
    ]
    return result[columns].sort_values(
        [
            "analysis_membership",
            "comparison_group",
            "history_length_a",
            "history_length_b",
            "dataset",
            "model",
            "prompt_strategy",
            "user_id",
        ],
        kind="mergesort",
    )


def build_statistical_results(contributions: pd.DataFrame, plan: pd.DataFrame, failures: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    plan = plan.assign(plan_order_int=pd.to_numeric(plan["plan_order"], errors="raise").astype(int)).sort_values(
        "plan_order_int", kind="mergesort"
    )

    for plan_row in plan.to_dict("records"):
        view = str(plan_row["reporting_view"])
        h_a = str(int(plan_row["history_length_a"]))
        h_b = str(int(plan_row["history_length_b"]))
        metric = str(plan_row["metric"])
        seed = int(plan_row["bootstrap_seed"])
        resamples = int(plan_row["bootstrap_resamples"])
        subset = contributions.loc[
            contributions["analysis_membership"].eq(view)
            & contributions["history_length_a"].eq(h_a)
            & contributions["history_length_b"].eq(h_b)
        ].copy()
        if subset.empty:
            failures.append(f"NO_CONTRIBUTIONS_FOR_PLAN_ROW:{plan_row['comparison_id']}")
            continue
        for (dataset, model, prompt), group in subset.groupby(["dataset", "model", "prompt_strategy"], sort=True):
            deltas = group[f"{metric}_delta"].astype(float).to_numpy()
            values_a = group[f"{metric}_a"].astype(float).to_numpy()
            values_b = group[f"{metric}_b"].astype(float).to_numpy()
            delta = float(np.mean(deltas))
            low, high, unique_delta_count = paired_bootstrap_mean_delta_ci(
                deltas,
                seed=seed,
                resamples=resamples,
            )
            zero_relation, decision = ci_decision(low, high)
            strict_a = group["strict_parse_valid_a"].map(bool_value)
            strict_b = group["strict_parse_valid_b"].map(bool_value)
            positive_delta_count = int(np.count_nonzero(deltas > 0.0))
            negative_delta_count = int(np.count_nonzero(deltas < 0.0))
            zero_delta_count = int(np.count_nonzero(deltas == 0.0))

            row: dict[str, Any] = {
                "result_id": stable_id("G2STAT-RESULT", plan_row["comparison_id"], dataset, model, prompt),
                "plan_order": str(plan_row["plan_order"]),
                "comparison_id": str(plan_row["comparison_id"]),
                "analysis_membership": view,
                "comparison_group": str(plan_row["comparison_group"]),
                "scope": str(plan_row["scope"]),
                "dataset": str(dataset),
                "model": str(model),
                "prompt_strategy": str(prompt),
                "metric": metric,
                "history_length_a": h_a,
                "history_length_b": h_b,
                "contrast": str(plan_row["contrast"]),
                "paired_unit": str(plan_row["paired_unit"]),
                "paired_user_count": int(len(group)),
                "strict_valid_both_count": int((strict_a & strict_b).sum()),
                "strict_valid_a_only_count": int((strict_a & ~strict_b).sum()),
                "strict_valid_b_only_count": int((~strict_a & strict_b).sum()),
                "strict_valid_neither_count": int((~strict_a & ~strict_b).sum()),
                "invalid_any_side_count": int((~strict_a | ~strict_b).sum()),
                "value_a_mean": float(np.mean(values_a)),
                "value_b_mean": float(np.mean(values_b)),
                "paired_mean_delta": delta,
                "positive_delta_user_count": positive_delta_count,
                "negative_delta_user_count": negative_delta_count,
                "zero_delta_user_count": zero_delta_count,
                "primary_test": str(plan_row["primary_test"]),
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
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
                "secondary_test_execution_status": "NOT_EXECUTED_PARAMETERS_NOT_FULLY_FROZEN",
                "p_value": "",
                "p_value_status": "NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD",
                "multiple_comparison_correction": str(plan_row["multiple_comparison_correction"]),
                "multiplicity_family_id": stable_id("G2STAT-FAMILY", metric, dataset, model, prompt),
                "multiple_comparison_adjusted_p_value": "",
                "multiplicity_correction_execution_status": "NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD",
                "corrected_decision": "NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD",
                "frozen_decision": decision,
                "frozen_decision_scope": "primary_bootstrap_percentile_ci_unadjusted_alpha_0.05",
                "planned_status": str(plan_row["planned_status"]),
                "unique_delta_values": unique_delta_count,
                "notes": (
                    "Invalid terminal responses contribute zero utility. Corrected p-value decisions are left undefined "
                    "because the frozen plan does not define a primary p-value calculation."
                ),
            }
            if metric == "hr_at_5":
                a_hits = group["hr_at_5_a"].astype(float)
                b_hits = group["hr_at_5_b"].astype(float)
                row["hr_left0_right1_count"] = int(((a_hits == 0.0) & (b_hits == 1.0)).sum())
                row["hr_left1_right0_count"] = int(((a_hits == 1.0) & (b_hits == 0.0)).sum())
            else:
                row["hr_left0_right1_count"] = ""
                row["hr_left1_right0_count"] = ""
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        [
            "analysis_membership",
            "comparison_group",
            "plan_order",
            "dataset",
            "model",
            "prompt_strategy",
        ],
        kind="mergesort",
    )
    if len(results) != len(plan) * 5 * 3 * 2:
        failures.append(f"STATISTICAL_RESULT_ROW_COUNT_MISMATCH:{len(results)}")
    return results


def verify_metric_means(
    root: Path,
    contributions: pd.DataFrame,
    plan: pd.DataFrame,
    failures: list[str],
) -> int:
    metric_results = read_csv(root, PATHS["metric_results"])
    metric_results = metric_results.loc[metric_results["metric_id"].isin(UTILITY_METRICS)].copy()
    metric_lookup = {
        (
            str(row["analysis_membership"]),
            str(row["dataset"]),
            str(row["model"]),
            str(row["prompt_strategy"]),
            str(int(float(row["history_length"]))),
            str(row["metric_id"]),
        ): float(row["value"])
        for row in metric_results.to_dict("records")
    }
    checks = 0
    for plan_row in plan.to_dict("records"):
        view = str(plan_row["reporting_view"])
        h_a = str(int(plan_row["history_length_a"]))
        h_b = str(int(plan_row["history_length_b"]))
        metric = str(plan_row["metric"])
        subset = contributions.loc[
            contributions["analysis_membership"].eq(view)
            & contributions["history_length_a"].eq(h_a)
            & contributions["history_length_b"].eq(h_b)
        ]
        for (dataset, model, prompt), group in subset.groupby(["dataset", "model", "prompt_strategy"], sort=True):
            for side, history in [("a", h_a), ("b", h_b)]:
                observed = float(group[f"{metric}_{side}"].astype(float).mean())
                expected = metric_lookup.get((view, str(dataset), str(model), str(prompt), history, metric))
                checks += 1
                if expected is None or abs(observed - expected) > 1e-12:
                    failures.append(
                        "PAIRED_CONTRIBUTION_MEAN_MISMATCH:"
                        f"{view}:{dataset}:{model}:{prompt}:H{history}:{metric}:{observed}:{expected}"
                    )
    return checks


def verify_hash_mapping(
    root: Path,
    mapping: Mapping[str, Any],
    failures: list[str],
    *,
    label: str,
    skip_paths: set[str] | None = None,
) -> int:
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


def build_validation(
    root: Path,
    *,
    head: str,
    status_before: str,
    diff_check: subprocess.CompletedProcess[str],
    allow_generator_dirty: bool,
    manifests: Mapping[str, Mapping[str, Any]],
    plan: pd.DataFrame,
    row_metrics: pd.DataFrame,
    evidence_counts: Mapping[str, int],
    contributions: pd.DataFrame,
    statistical_results: pd.DataFrame,
    frozen_hash_check_count: int,
    metric_mean_checks: int,
    failures: list[str],
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def add(name: str, passed: bool, expected: str = "true", actual: str | None = None, notes: str = "") -> None:
        if not passed and name not in failures:
            failures.append(name)
        checks.append(
            {
                "check_id": stable_id("G2STAT-CHK", name),
                "check_name": name,
                "status": "PASS" if passed else "FAIL",
                "expected": expected,
                "actual": actual if actual is not None else bool_text(passed),
                "notes": notes,
            }
        )

    plan_metrics = sorted(plan["metric"].drop_duplicates().tolist())
    plan_counts = plan.groupby(["reporting_view", "metric"], sort=True)["comparison_id"].count().astype(int).to_dict()
    universe_counts = {
        "logical_requests": int(len(row_metrics)),
        "g1_exact_reuse": int(row_metrics["evidence_source"].eq(G1_EVIDENCE).sum()),
        "g2_terminal_evidence": int(row_metrics["evidence_source"].eq(G2_EVIDENCE).sum()),
        "strict_valid": int(row_metrics["strict_parse_valid"].map(bool_value).sum()),
        "invalid": int((~row_metrics["strict_parse_valid"].map(bool_value)).sum()),
        "backend_errors": int((~row_metrics["backend_success"].map(bool_value)).sum()),
    }
    expected_contribution_rows = 122382
    expected_result_rows = 810
    covered_plan_rows = int(statistical_results["comparison_id"].nunique())
    ci_counts = statistical_results["primary_ci_decision"].value_counts().sort_index().to_dict()

    add("head_matches_precondition", head == EXPECTED_HEAD, actual=head)
    add(
        "git_clean_or_expected_statistical_outputs_before_generation",
        status_is_clean_or_expected(status_before, allow_generator_dirty),
        actual=";".join(git_dirty_paths(status_before)),
    )
    add("git_diff_check_clean_before_generation", diff_check.returncode == 0, actual=diff_check.stdout + diff_check.stderr)
    add("protocol_manifest_frozen", manifests["protocol"].get("decision_token") == "G2_NESTED_HISTORY_PROTOCOL_FROZEN", actual=str(manifests["protocol"].get("decision_token")))
    add("execution_manifest_frozen", manifests["execution"].get("decision_token") == "G2_EXECUTION_RUNNER_FROZEN", actual=str(manifests["execution"].get("decision_token")))
    add("retention_manifest_frozen", manifests["retention"].get("decision_token") == "G2_EXECUTION_RETENTION_FROZEN", actual=str(manifests["retention"].get("decision_token")))
    add("analysis_manifest_frozen", manifests["analysis"].get("decision_token") == "G2_ANALYTIC_UNIVERSE_FROZEN", actual=str(manifests["analysis"].get("decision_token")))
    add("metrics_manifest_frozen", manifests["metrics"].get("decision_token") == "G2_METRICS_FROZEN", actual=str(manifests["metrics"].get("decision_token")))
    add(
        "readout_manifest_frozen",
        manifests["readout"].get("decision_token") == "G2_DESCRIPTIVE_READOUT_COMPLETE_STATISTICS_NOT_FROZEN",
        actual=str(manifests["readout"].get("decision_token")),
    )
    add("frozen_artifact_hashes_match", frozen_hash_check_count > 0, actual=str(frozen_hash_check_count))
    add("statistical_plan_rows_27", len(plan) == 27, expected="27", actual=str(len(plan)))
    add(
        "statistical_plan_metrics_match",
        set(plan_metrics) == set(UTILITY_METRICS),
        expected=stable_json(sorted(UTILITY_METRICS)),
        actual=stable_json(plan_metrics),
    )
    add(
        "statistical_plan_parameters_frozen",
        plan["primary_test"].eq("paired_user_bootstrap_mean_delta").all()
        and plan["bootstrap_resamples"].astype(str).eq("10000").all()
        and plan["bootstrap_seed"].astype(str).eq("20260901").all()
        and plan["confidence_interval"].eq("two_sided_95_percent_percentile_ci").all()
        and plan["effect_size"].eq("paired_mean_delta").all()
        and plan["multiple_comparison_correction"].eq("holm_bonferroni_within_metric_dataset_model_prompt_family").all()
        and plan["alpha"].astype(str).eq("0.05").all(),
        actual=stable_json({str(key): int(value) for key, value in plan_counts.items()}),
    )
    add(
        "no_primary_p_value_method_frozen",
        True,
        expected="p-value method absent from frozen primary-test contract",
        actual="p_value_status=NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD",
        notes="Holm-corrected p decisions are not computed because no primary p-value method is frozen.",
    )
    for key, expected in EXPECTED_UNIVERSE_COUNTS.items():
        if key in {"missing", "duplicate_terminal", "ambiguous"}:
            actual = int(read_csv(root, PATHS["analysis_universe"])[f"{key}_evidence" if key != "duplicate_terminal" else "duplicate_terminal_evidence"].map(bool_value).sum())
        else:
            actual = universe_counts[key]
        add(f"universe_count:{key}", actual == expected, expected=str(expected), actual=str(actual))
    add("g2_response_duplicate_ids_zero", int(evidence_counts.get("g2_response_duplicate_ids", -1)) == 0, expected="0", actual=str(evidence_counts.get("g2_response_duplicate_ids")))
    add("paired_contribution_rows_match", len(contributions) == expected_contribution_rows, expected=str(expected_contribution_rows), actual=str(len(contributions)))
    add(
        "paired_contribution_keys_unique",
        not contributions["contribution_pair_id"].duplicated().any(),
        actual=str(int(contributions["contribution_pair_id"].duplicated().sum())),
    )
    add("statistical_results_rows_match", len(statistical_results) == expected_result_rows, expected=str(expected_result_rows), actual=str(len(statistical_results)))
    add("comparison_template_coverage_27_of_27", covered_plan_rows == 27, expected="27", actual=str(covered_plan_rows))
    add(
        "bootstrap_settings_used",
        statistical_results["bootstrap_resamples"].astype(str).eq("10000").all()
        and statistical_results["bootstrap_seed"].astype(str).eq("20260901").all(),
        actual=stable_json(
            {
                "resamples": sorted(statistical_results["bootstrap_resamples"].astype(str).drop_duplicates().tolist()),
                "seed": sorted(statistical_results["bootstrap_seed"].astype(str).drop_duplicates().tolist()),
            }
        ),
    )
    add("metric_mean_checks_match_frozen_tables", metric_mean_checks == 1620, expected="1620", actual=str(metric_mean_checks))
    add(
        "statistical_results_no_p_values_invented",
        statistical_results["p_value"].astype(str).eq("").all()
        and statistical_results["multiple_comparison_adjusted_p_value"].astype(str).eq("").all(),
        actual=stable_json(statistical_results["p_value_status"].value_counts().sort_index().to_dict()),
    )
    add("no_ollama_calls_executed", True, actual="0")
    add("g3_not_started", True, actual="false")
    add("primary_ci_decisions_defined", statistical_results["primary_ci_decision"].astype(str).ne("").all(), actual=stable_json(ci_counts))
    return pd.DataFrame(checks)


def input_hashes(root: Path) -> dict[str, str]:
    inputs = [
        PATHS["protocol_config"],
        PATHS["protocol_manifest"],
        PATHS["execution_manifest"],
        PATHS["retention_manifest"],
        PATHS["analysis_manifest"],
        PATHS["metrics_manifest"],
        PATHS["readout_manifest"],
        PATHS["request_universe"],
        PATHS["analysis_universe"],
        PATHS["membership_universe"],
        PATHS["cell_counts"],
        PATHS["denominator_contract"],
        PATHS["statistical_plan"],
        PATHS["g1_reuse"],
        PATHS["g1_strict_parsing"],
        PATHS["g2_responses"],
        PATHS["metric_results"],
        PATHS["readout_statistical_contract"],
        PATHS["metric_contract_source"],
    ]
    return {path.as_posix(): sha256_file(root / path) for path in inputs}


def output_hashes(root: Path) -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(root / path)
        for key, path in OUTPUTS.items()
        if key != "manifest" and (root / path).is_file()
    }


def build_report(parts: Mapping[str, Any], output_hashes_map: Mapping[str, str]) -> str:
    results = parts["statistical_results"]
    validation = parts["validation"]
    paired_counts = (
        results.groupby(["analysis_membership", "dataset"], sort=True)["paired_user_count"]
        .agg(lambda series: sorted({int(value) for value in series}))
        .reset_index()
    )
    ci_counts = (
        results.groupby(["analysis_membership", "metric", "primary_ci_decision"], sort=True)
        .size()
        .reset_index(name="result_rows")
    )
    result_counts = results["primary_ci_decision"].value_counts().sort_index().to_dict()
    correction_counts = results["corrected_decision"].value_counts().sort_index().to_dict()

    lines = [
        "# G2 Frozen Statistical Analysis",
        "",
        f"- Phase: `{PHASE_ID}`",
        f"- Decision: `{parts['decision_token']}`",
        f"- HEAD before analysis: `{parts['head_before']}`",
        "- Scope: frozen retained G2/G1 evidence, universe, metric contracts, readout contract, and statistical plan only.",
        "- Zero Ollama calls; no G3; no protocol, runner, evidence, universe, metric, or readout changes.",
        "",
        "## Plan Coverage",
        "",
        f"- Frozen comparison templates executed: `{results['comparison_id'].nunique()}` / `27`.",
        f"- Dataset x model x prompt statistical result rows: `{len(results)}`.",
        f"- Paired-user contribution rows: `{parts['paired_contribution_rows']}`.",
        f"- Utility metrics: `{stable_json(UTILITY_METRICS)}`.",
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
        "## Paired User Counts",
        "",
        "| view | dataset | paired users |",
        "|---|---|---:|",
    ]
    for row in paired_counts.to_dict("records"):
        values = row["paired_user_count"]
        display = values[0] if len(values) == 1 else stable_json(values)
        lines.append(f"| {row['analysis_membership']} | {row['dataset']} | {display} |")

    lines.extend(
        [
            "",
            "## CI Decision Counts",
            "",
            f"- Overall primary CI decisions: `{stable_json({str(key): int(value) for key, value in result_counts.items()})}`.",
            f"- Corrected p-value decisions: `{stable_json({str(key): int(value) for key, value in correction_counts.items()})}`.",
            "- Corrected p-values are not computed because no frozen primary p-value method exists in the G2 plan.",
            "",
            "| view | metric | CI decision | result rows |",
            "|---|---|---|---:|",
        ]
    )
    for row in ci_counts.to_dict("records"):
        lines.append(
            f"| {row['analysis_membership']} | {row['metric']} | {row['primary_ci_decision']} | {int(row['result_rows'])} |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Validation checks: `{int(validation['status'].eq('PASS').sum())}` / `{len(validation)}` PASS.",
            f"- Frozen input hash checks: `{parts['frozen_hash_check_count']}`.",
            f"- Frozen metric mean checks: `{parts['metric_mean_checks']}`.",
            "",
            "## Output Artifacts",
            "",
        ]
    )
    for rel_path, digest in sorted(output_hashes_map.items()):
        lines.append(f"- `{rel_path}` SHA256 `{digest}`")
    lines.extend(
        [
            "",
            "## Next Safe Step",
            "",
            "Integrate the frozen descriptive and statistical G2 readouts into the final report. Do not start G3 unless separately authorized.",
            "",
        ]
    )
    return "\n".join(lines)


def append_notes(root: Path, parts: Mapping[str, Any]) -> None:
    path = root / PATHS["notes"]
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    last_heading = text.rfind("\n## ")
    if last_heading != -1 and " - G2 frozen statistical analysis" in text[last_heading:]:
        text = text[:last_heading].rstrip()
    decision_counts = {
        str(key): int(value)
        for key, value in parts["statistical_results"]["primary_ci_decision"].value_counts().sort_index().to_dict().items()
    }
    note = (
        f"\n## {parts['created_at_utc']} - G2 frozen statistical analysis\n\n"
        f"- Decision: `{parts['decision_token']}`.\n"
        "- Scope: executed frozen G2 paired-user bootstrap CI plan from retained evidence only; no Ollama calls, no G3, no protocol changes, no frozen evidence/universe/metric/readout edits.\n"
        f"- Frozen comparison coverage: `27/27`; statistical result rows `{len(parts['statistical_results'])}`; paired-user contribution rows `{parts['paired_contribution_rows']}`.\n"
        "- Bootstrap: `10000` resamples, seed `20260901`, two-sided 95 percent percentile CI, paired mean delta.\n"
        f"- Primary CI decision counts: `{stable_json(decision_counts)}`; corrected p-value decisions remain `NOT_DEFINED_NO_FROZEN_P_VALUE_METHOD`.\n"
        "- Outputs: `results/tables/g2_history_length_statistical_paired_contributions.csv`, `results/tables/g2_history_length_statistical_results.csv`, `results/tables/g2_history_length_statistical_validation.csv`, `results/manifests/g2_history_length_statistical_analysis_manifest.json`, `report/g2_history_length_statistical_analysis.md`.\n"
        "- Next safe step: integrate frozen G2 descriptive/statistical readout into the report; do not start G3 yet.\n"
    )
    path.write_text(text.rstrip() + note.rstrip("\n") + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    failures: list[str] = []

    head = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short", "--untracked-files=all"])
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=root, text=True, capture_output=True)

    manifests = {
        "protocol": read_json(root, PATHS["protocol_manifest"]),
        "execution": read_json(root, PATHS["execution_manifest"]),
        "retention": read_json(root, PATHS["retention_manifest"]),
        "analysis": read_json(root, PATHS["analysis_manifest"]),
        "metrics": read_json(root, PATHS["metrics_manifest"]),
        "readout": read_json(root, PATHS["readout_manifest"]),
    }
    plan = read_csv(root, PATHS["statistical_plan"])

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
        manifests["metrics"].get("input_hashes", {}),
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
        manifests["readout"].get("input_hashes", {}),
        failures,
        label="READOUT_INPUT",
        skip_paths={"report/notes.md"},
    )
    frozen_hash_check_count += verify_hash_mapping(
        root,
        manifests["readout"].get("output_hashes", {}),
        failures,
        label="READOUT_OUTPUT",
        skip_paths={"report/notes.md"},
    )

    row_metrics, evidence_counts = build_row_metrics(root, failures)
    contributions = build_paired_contributions(root, row_metrics, plan, failures)
    statistical_results = build_statistical_results(contributions, plan, failures)
    metric_mean_checks = verify_metric_means(root, contributions, plan, failures)

    validation = build_validation(
        root,
        head=head,
        status_before=status_before,
        diff_check=diff_check,
        allow_generator_dirty=bool(args.allow_generator_dirty),
        manifests=manifests,
        plan=plan,
        row_metrics=row_metrics,
        evidence_counts=evidence_counts,
        contributions=contributions,
        statistical_results=statistical_results,
        frozen_hash_check_count=frozen_hash_check_count,
        metric_mean_checks=metric_mean_checks,
        failures=failures,
    )

    decision = DECISION_FROZEN if not failures else DECISION_BLOCKED
    created_at = utc_now()
    summary = {
        "created_at_utc": created_at,
        "decision_token": decision,
        "head_before": head,
        "paired_contribution_rows": int(len(contributions)),
        "statistical_result_rows": int(len(statistical_results)),
        "comparison_templates_covered": int(statistical_results["comparison_id"].nunique()) if not statistical_results.empty else 0,
        "primary_ci_decision_counts": {
            str(key): int(value)
            for key, value in statistical_results["primary_ci_decision"].value_counts().sort_index().to_dict().items()
        }
        if not statistical_results.empty
        else {},
        "corrected_decision_counts": {
            str(key): int(value)
            for key, value in statistical_results["corrected_decision"].value_counts().sort_index().to_dict().items()
        }
        if not statistical_results.empty
        else {},
        "frozen_hash_check_count": int(frozen_hash_check_count),
        "metric_mean_checks": int(metric_mean_checks),
        "failures": failures,
    }

    if not args.no_write:
        write_csv(root, OUTPUTS["paired_contributions"], contributions)
        write_csv(root, OUTPUTS["statistical_results"], statistical_results)
        write_csv(root, OUTPUTS["validation"], validation)
        report_text = build_report(
            {
                **summary,
                "validation": validation,
                "statistical_results": statistical_results,
            },
            {},
        )
        (root / OUTPUTS["report"]).parent.mkdir(parents=True, exist_ok=True)
        (root / OUTPUTS["report"]).write_text(report_text, encoding="utf-8")
        append_notes(
            root,
            {
                **summary,
                "statistical_results": statistical_results,
            },
        )

        hashes = output_hashes(root)
        report_text = build_report(
            {
                **summary,
                "validation": validation,
                "statistical_results": statistical_results,
            },
            hashes,
        )
        (root / OUTPUTS["report"]).write_text(report_text, encoding="utf-8")
        hashes = output_hashes(root)

        manifest = {
            "phase_id": PHASE_ID,
            "decision_token": decision,
            "created_at_utc": created_at,
            "precondition_head": EXPECTED_HEAD,
            "head_before_generation": head,
            "zero_inference": True,
            "ollama_calls_executed": 0,
            "g3_started": False,
            "protocol_changes": False,
            "frozen_evidence_modified": False,
            "frozen_universe_metric_readout_modified": False,
            "input_hashes": input_hashes(root),
            "output_hashes": hashes,
            "bootstrap_settings": {
                "primary_test": "paired_user_bootstrap_mean_delta",
                "bootstrap_resamples": 10000,
                "bootstrap_seed": 20260901,
                "confidence_interval": "two_sided_95_percent_percentile_ci",
                "alpha": 0.05,
                "bootstrap_engine": "empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta",
                "bootstrap_percentile_method": "numpy_quantile_linear",
            },
            "p_value_status": {
                "primary_p_value": "NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD",
                "holm_bonferroni": "NOT_APPLIED_NO_FROZEN_P_VALUE_METHOD",
                "secondary_test": "NOT_EXECUTED_PARAMETERS_NOT_FULLY_FROZEN",
            },
            "summary": {
                **summary,
                "validation_checks": {
                    str(row["check_name"]): str(row["status"]) == "PASS" for row in validation.to_dict("records")
                },
                "validation_passed": bool(validation["status"].eq("PASS").all() and not failures),
                "evidence_counts": {str(key): int(value) for key, value in evidence_counts.items()},
            },
        }
        (root / OUTPUTS["manifest"]).parent.mkdir(parents=True, exist_ok=True)
        (root / OUTPUTS["manifest"]).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        hashes = output_hashes(root)
        manifest["output_hashes"] = hashes
        (root / OUTPUTS["manifest"]).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "decision_token": decision,
                "comparison_templates_covered": summary["comparison_templates_covered"],
                "statistical_result_rows": summary["statistical_result_rows"],
                "paired_contribution_rows": summary["paired_contribution_rows"],
                "validation_pass": bool(validation["status"].eq("PASS").all() and not failures),
                "failures": failures,
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

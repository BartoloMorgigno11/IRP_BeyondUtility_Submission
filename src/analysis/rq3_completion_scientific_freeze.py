"""Freeze paired-user inference for the RQ3 H3/H4/H15 completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tarfile

import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path("results/rq3_completion_scientific_freeze")
REQUEST_MANIFEST = Path("artifacts/rq3_llm_completion/request_manifest.csv")
TERMINAL_DIR = Path("results/llm/rq3_llm_completion/terminal")
COHORT = Path("results/tables/g2_history_length_cohort.csv")
CANONICAL_CONTRIBUTIONS = Path("results/tables/g2_history_length_statistical_paired_contributions.csv")
CANONICAL_GRID = Path("results/tables/g2_history_length_readout_metric_values.csv")
FINAL_GRID = Path("results/tables/rq3_llm_history_metric_grid_final.csv")
G2_CONFIG = Path("configs/g2_history_length_sensitivity.yaml")
G2_ANALYSIS_MANIFEST = Path("results/manifests/g2_history_length_analysis_universe_freeze.json")
G2_READOUT_MANIFEST = Path("results/manifests/g2_history_length_scientific_readout_manifest.json")
G2_STATS_MANIFEST = Path("results/manifests/g2_history_length_statistical_analysis_manifest.json")

METRICS = ("hr_at_5", "ndcg_at_5", "mrr")
EXPECTED_BY_H = {3: 14_298, 4: 14_298, 15: 12_198}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_901

CONTRASTS = (
    ("H2_H3", "PRIMARY_B1_GE10", 2, 3, 14_298),
    ("H3_H4", "PRIMARY_B1_GE10", 3, 4, 14_298),
    ("H4_H5", "PRIMARY_B1_GE10", 4, 5, 14_298),
    ("H10_H15", "SECONDARY_FULL_CURVE_GE20", 10, 15, 12_198),
    ("H15_H20", "SECONDARY_FULL_CURVE_GE20", 15, 20, 12_198),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rq3_completion_freeze.yaml"))
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()



def configure(root: Path, config_path: Path) -> dict[str, Any]:
    """Load the packaged offline protocol without changing canonical G2 inputs."""
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    if config["experiment"]["allow_api_calls"]:
        raise ValueError("This producer supports offline saved evidence only")
    allowed = {"OUTPUT_DIR", "REQUEST_MANIFEST", "TERMINAL_ARCHIVE", "COHORT",
               "CANONICAL_CONTRIBUTIONS", "CANONICAL_GRID", "FINAL_GRID", "G2_CONFIG",
               "G2_ANALYSIS_MANIFEST", "G2_READOUT_MANIFEST", "G2_STATS_MANIFEST"}
    if set(config["paths"]) != allowed:
        raise ValueError("Unexpected package path configuration")
    for name, value in config["paths"].items():
        relative = Path(value)
        (root / relative).resolve().relative_to(root.resolve())
        globals()[name] = relative
    experiment = config["experiment"]
    globals()["EXPECTED_BY_H"] = {int(k): int(v) for k, v in experiment["expected_by_history"].items()}
    globals()["CONTRASTS"] = tuple(tuple(row) for row in experiment["contrasts"])
    globals()["BOOTSTRAP_RESAMPLES"] = int(experiment["bootstrap_resamples"])
    globals()["BOOTSTRAP_SEED"] = int(experiment["bootstrap_seed"])
    return config

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32].upper()}"


def read_json(root: Path, relative: Path) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def percentile_ci(deltas: Sequence[float]) -> tuple[float, float, int]:
    """Use the exact frozen G2 multinomial-equivalent paired bootstrap."""
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one paired delta")
    unique, counts = np.unique(values, return_counts=True)
    if unique.size == 1:
        samples = np.repeat(unique[0], BOOTSTRAP_RESAMPLES)
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        probabilities = counts.astype(float) / float(values.size)
        weights = rng.multinomial(int(values.size), probabilities, size=BOOTSTRAP_RESAMPLES)
        samples = weights @ unique.astype(float) / float(values.size)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(low), float(high), int(unique.size)


def decision(low: float, high: float) -> str:
    if low > 0.0:
        return "POSITIVE"
    if high < 0.0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def utility(record: dict[str, Any], positive_item_id: str) -> tuple[bool, int | None, float, float, float]:
    strict_valid = bool(record.get("strict_parse_valid")) and record.get("terminal_state") == "STRICT_VALID"
    if not strict_valid:
        return False, None, 0.0, 0.0, 0.0
    ranked_raw = record.get("ranked_item_ids", "[]")
    ranked = json.loads(ranked_raw) if isinstance(ranked_raw, str) else list(ranked_raw)
    rank = next((index for index, item in enumerate(ranked) if str(item) == str(positive_item_id)), None)
    if rank is None:
        return True, None, 0.0, 0.0, 0.0
    hr = 1.0 if rank < 5 else 0.0
    ndcg = 1.0 / math.log2(rank + 2.0) if rank < 5 else 0.0
    mrr = 1.0 / float(rank + 1)
    return True, rank, hr, ndcg, mrr


def load_completion(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    manifest = pd.read_csv(root / REQUEST_MANIFEST, dtype=str, keep_default_na=False)
    terminal_records = {}
    archive_path = globals().get("TERMINAL_ARCHIVE")
    if archive_path is not None:
        with tarfile.open(root / archive_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError("Unexpected terminal archive member")
                request_key = member.name.removesuffix(".json")
                if request_key in terminal_records:
                    raise ValueError("Duplicate terminal archive member")
                terminal_records[request_key] = json.load(archive.extractfile(member))
    records: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    terminal_digest_rows: list[str] = []

    for history_length, expected in EXPECTED_BY_H.items():
        subset = manifest.loc[manifest["history_length"].astype(int).eq(history_length)].copy()
        accounted = valid = invalid = backend_incomplete = 0
        for row in subset.to_dict("records"):
            terminal_path = root / TERMINAL_DIR / f"{row['request_key']}.json"
            if archive_path is not None:
                terminal = terminal_records.get(row["request_key"])
                if terminal is None:
                    continue
            else:
                if not terminal_path.exists():
                    continue
                terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            is_terminal = bool(terminal.get("terminal"))
            if is_terminal:
                accounted += 1
            backend_ok = bool(terminal.get("backend_success")) and terminal.get("backend_status") == "BACKEND_SUCCESS"
            is_incomplete = not is_terminal or not backend_ok
            if is_incomplete:
                backend_incomplete += 1
            strict_valid, rank, hr, ndcg, mrr = utility(terminal, row["positive_item_id"])
            if strict_valid and not is_incomplete:
                valid += 1
            elif not is_incomplete:
                invalid += 1
            terminal_digest_rows.append(f"{row['request_key']}|{terminal.get('terminal_record_hash', '')}")
            records.append(
                {
                    "dataset": row["dataset"],
                    "user_id": row["user_id"],
                    "model": row["model"],
                    "prompt_strategy": row["prompt_strategy"],
                    "history_length": history_length,
                    "request_id": row["request_key"],
                    "candidate_set_id": row["candidate_set_id"],
                    "candidate_hash": row["candidate_hash"],
                    "candidate_items": stable_json(json.loads(row["candidate_items"])),
                    "positive_item_id": row["positive_item_id"],
                    "terminal_state": str(terminal.get("terminal_state", "")),
                    "strict_parse_valid": strict_valid,
                    "rank_zero_based": "" if rank is None else rank,
                    "hr_at_5": hr,
                    "ndcg_at_5": ndcg,
                    "mrr": mrr,
                    "evidence_source": "RQ3_LLM_COMPLETION_TERMINAL_EVIDENCE",
                }
            )
        accounting.append(
            {
                "history_length": history_length,
                "cohort": "GE10" if history_length in (3, 4) else "GE20",
                "expected": expected,
                "accounted": accounted,
                "valid": valid,
                "invalid": invalid,
                "backend_incomplete": backend_incomplete,
                "missing": expected - accounted,
                "status": "PASS"
                if len(subset) == expected and accounted == expected and valid + invalid == expected and backend_incomplete == 0
                else "FAIL",
            }
        )

    terminal_set_hash = hashlib.sha256("\n".join(sorted(terminal_digest_rows)).encode("utf-8")).hexdigest()
    return pd.DataFrame(records), pd.DataFrame(accounting), terminal_set_hash


def old_endpoint_frames(root: Path, cohort: pd.DataFrame) -> dict[int, pd.DataFrame]:
    contributions = pd.read_csv(root / CANONICAL_CONTRIBUTIONS, dtype=str, keep_default_na=False)
    endpoint_specs = {
        "primary": ("PRIMARY_B1_GE10", 2, 5),
        "long": ("LONG_HISTORY_B2_GE20", 10, 20),
    }
    endpoint_frames: dict[int, pd.DataFrame] = {}
    keys = ["dataset", "model", "prompt_strategy", "user_id"]
    metadata = cohort[
        ["dataset", "user_id", "candidate_set_id", "candidate_hash", "candidate_items", "positive_item_id"]
    ].copy()
    metadata["candidate_items"] = metadata["candidate_items"].map(lambda value: stable_json(json.loads(value)))

    for membership, h_a, h_b in endpoint_specs.values():
        subset = contributions.loc[
            contributions["analysis_membership"].eq(membership)
            & contributions["history_length_a"].astype(int).eq(h_a)
            & contributions["history_length_b"].astype(int).eq(h_b)
        ].copy()
        for history_length, suffix in ((h_a, "a"), (h_b, "b")):
            frame = subset[keys].copy()
            frame["history_length"] = history_length
            frame["request_id"] = subset[f"canonical_request_id_{suffix}"]
            frame["terminal_state"] = subset[f"terminal_state_{suffix}"]
            frame["strict_parse_valid"] = subset[f"strict_parse_valid_{suffix}"].str.lower().eq("true")
            frame["rank_zero_based"] = subset[f"rank_zero_based_{suffix}"]
            for metric in METRICS:
                frame[metric] = pd.to_numeric(subset[f"{metric}_{suffix}"], errors="raise")
            frame["evidence_source"] = "G2_FROZEN_PAIRED_CONTRIBUTIONS"
            frame = frame.merge(metadata, on=["dataset", "user_id"], how="left", validate="many_to_one")
            endpoint_frames[history_length] = frame
    return endpoint_frames


def verify_metric_grid(root: Path, endpoints: dict[int, pd.DataFrame]) -> tuple[bool, list[str]]:
    grid = pd.read_csv(root / FINAL_GRID, dtype=str, keep_default_na=False)
    grid["history_length_numeric"] = pd.to_numeric(grid["history_length"], errors="coerce")
    failures: list[str] = []
    membership_by_h = {
        2: "PRIMARY_B1_GE10",
        3: "PRIMARY_B1_GE10",
        4: "PRIMARY_B1_GE10",
        5: "PRIMARY_B1_GE10",
        10: "SECONDARY_FULL_CURVE_GE20",
        15: "SECONDARY_FULL_CURVE_GE20",
        20: "SECONDARY_FULL_CURVE_GE20",
    }
    for history_length, frame in endpoints.items():
        grouped = frame.groupby(["dataset", "model", "prompt_strategy"], sort=True)
        for (dataset, model, prompt), group in grouped:
            for metric in METRICS:
                expected_row = grid.loc[
                    grid["row_type"].eq("metric")
                    & grid["analysis_membership"].eq(membership_by_h[history_length])
                    & grid["dataset"].eq(dataset)
                    & grid["model"].eq(model)
                    & grid["prompt_strategy"].eq(prompt)
                    & grid["history_length_numeric"].eq(history_length)
                    & grid["metric_id"].eq(metric)
                ]
                if len(expected_row) != 1:
                    failures.append(f"GRID_CELL_COUNT:{history_length}:{dataset}:{model}:{prompt}:{metric}:{len(expected_row)}")
                    continue
                actual = float(group[metric].mean())
                expected = float(expected_row.iloc[0]["value"])
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                    failures.append(f"GRID_MEAN_MISMATCH:{history_length}:{dataset}:{model}:{prompt}:{metric}")
    return not failures, failures


def build_pairs(endpoints: dict[int, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    pair_frames: dict[str, pd.DataFrame] = {}
    validations: list[dict[str, Any]] = []
    keys = ["dataset", "user_id", "model", "prompt_strategy"]
    side_columns = keys + [
        "request_id",
        "candidate_set_id",
        "candidate_hash",
        "candidate_items",
        "positive_item_id",
        "terminal_state",
        "strict_parse_valid",
        "rank_zero_based",
        "hr_at_5",
        "ndcg_at_5",
        "mrr",
        "evidence_source",
    ]

    for contrast, membership, h_a, h_b, expected in CONTRASTS:
        left = endpoints[h_a][side_columns].copy()
        right = endpoints[h_b][side_columns].copy()
        duplicate_a = int(left.duplicated(keys).sum())
        duplicate_b = int(right.duplicated(keys).sum())
        left = left.rename(columns={column: f"{column}_a" for column in side_columns if column not in keys})
        right = right.rename(columns={column: f"{column}_b" for column in side_columns if column not in keys})
        paired = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
        unmatched_a = int(paired["_merge"].eq("left_only").sum())
        unmatched_b = int(paired["_merge"].eq("right_only").sum())
        paired = paired.loc[paired["_merge"].eq("both")].drop(columns="_merge").copy()
        candidate_membership_mismatch = int(
            (~paired["candidate_hash_a"].eq(paired["candidate_hash_b"])
             | ~paired["candidate_set_id_a"].eq(paired["candidate_set_id_b"])).sum()
        )
        candidate_order_mismatch = int((~paired["candidate_items_a"].eq(paired["candidate_items_b"])).sum())
        target_mismatch = int((~paired["positive_item_id_a"].eq(paired["positive_item_id_b"])).sum())
        fail_strict_violations = 0
        for suffix in ("a", "b"):
            invalid = ~paired[f"strict_parse_valid_{suffix}"].astype(bool)
            fail_strict_violations += int(
                (paired.loc[invalid, [f"{metric}_{suffix}" for metric in METRICS]].astype(float) != 0.0).any(axis=1).sum()
            )
        passed = all(
            value == 0
            for value in (
                duplicate_a,
                duplicate_b,
                unmatched_a,
                unmatched_b,
                candidate_membership_mismatch,
                candidate_order_mismatch,
                target_mismatch,
                fail_strict_violations,
            )
        ) and len(paired) == expected
        validations.append(
            {
                "contrast": contrast,
                "analysis_membership": membership,
                "history_length_a": h_a,
                "history_length_b": h_b,
                "expected_pairs": expected,
                "paired_rows": len(paired),
                "duplicate_keys_a": duplicate_a,
                "duplicate_keys_b": duplicate_b,
                "unmatched_a": unmatched_a,
                "unmatched_b": unmatched_b,
                "candidate_membership_mismatch": candidate_membership_mismatch,
                "candidate_order_mismatch": candidate_order_mismatch,
                "target_mismatch": target_mismatch,
                "metric_implementation": "G2_FROZEN_HR5_NDCG5_MRR",
                "validity_policy": "PLANNED_USERS_INVALID_TERMINAL_ZERO",
                "fail_strict_policy_violations": fail_strict_violations,
                "status": "PASS" if passed else "FAIL",
            }
        )
        paired["contrast"] = contrast
        paired["analysis_membership"] = membership
        paired["history_length_a"] = h_a
        paired["history_length_b"] = h_b
        for metric in METRICS:
            paired[f"{metric}_delta"] = paired[f"{metric}_b"].astype(float) - paired[f"{metric}_a"].astype(float)
        pair_frames[contrast] = paired
    return pair_frames, pd.DataFrame(validations)


def statistical_results(pair_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contrast, membership, h_a, h_b, _ in CONTRASTS:
        paired = pair_frames[contrast]
        for metric in METRICS:
            for (dataset, model, prompt), group in paired.groupby(["dataset", "model", "prompt_strategy"], sort=True):
                deltas = group[f"{metric}_delta"].astype(float).to_numpy()
                values_a = group[f"{metric}_a"].astype(float).to_numpy()
                values_b = group[f"{metric}_b"].astype(float).to_numpy()
                low, high, unique_count = percentile_ci(deltas)
                strict_a = group["strict_parse_valid_a"].astype(bool)
                strict_b = group["strict_parse_valid_b"].astype(bool)
                rows.append(
                    {
                        "result_id": stable_id("RQ3COMP-STAT", contrast, metric, dataset, model, prompt),
                        "contrast": contrast,
                        "analysis_membership": membership,
                        "dataset": dataset,
                        "model": model,
                        "prompt_strategy": prompt,
                        "metric": metric,
                        "history_length_a": h_a,
                        "history_length_b": h_b,
                        "n_pairs": len(group),
                        "strict_valid_both": int((strict_a & strict_b).sum()),
                        "strict_valid_a_only": int((strict_a & ~strict_b).sum()),
                        "strict_valid_b_only": int((~strict_a & strict_b).sum()),
                        "strict_valid_neither": int((~strict_a & ~strict_b).sum()),
                        "value_a_mean": float(np.mean(values_a)),
                        "value_b_mean": float(np.mean(values_b)),
                        "mean_delta": float(np.mean(deltas)),
                        "positive_delta_users": int(np.count_nonzero(deltas > 0.0)),
                        "zero_delta_users": int(np.count_nonzero(deltas == 0.0)),
                        "negative_delta_users": int(np.count_nonzero(deltas < 0.0)),
                        "ci_low": low,
                        "ci_high": high,
                        "decision": decision(low, high),
                        "unique_delta_values": unique_count,
                        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_engine": "empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta",
                        "bootstrap_percentile_method": "numpy_quantile_linear",
                        "confidence_interval": "two_sided_95_percent_percentile_ci",
                        "p_value": "",
                        "p_value_status": "NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD",
                    }
                )
    return pd.DataFrame(rows)


def synthesis(root: Path, stats: pd.DataFrame) -> pd.DataFrame:
    grid = pd.read_csv(root / FINAL_GRID, dtype=str, keep_default_na=False)
    descriptive = grid.loc[grid["row_type"].eq("delta_summary")].copy()
    descriptive["contrast"] = descriptive["comparison"].str.replace("_to_", "_", regex=False)
    rows: list[dict[str, Any]] = []
    for contrast, _, _, _, _ in CONTRASTS:
        for metric in METRICS:
            group = stats.loc[stats["contrast"].eq(contrast) & stats["metric"].eq(metric)]
            desc = descriptive.loc[descriptive["contrast"].eq(contrast) & descriptive["metric_id"].eq(metric)]
            if len(group) != 30 or len(desc) != 1:
                raise ValueError(f"incomplete synthesis input: {contrast} {metric}")
            desc_row = desc.iloc[0]
            decisions = group["decision"].value_counts().to_dict()
            rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "descriptive_positive_configurations": int(desc_row["positive"]),
                    "descriptive_zero_configurations": int(desc_row["zero"]),
                    "descriptive_negative_configurations": int(desc_row["negative"]),
                    "descriptive_mean_delta_across_configurations": float(desc_row["mean_delta"]),
                    "bootstrap_result_rows": len(group),
                    "n_pairs_total_across_configurations": int(group["n_pairs"].sum()),
                    "n_pairs_min": int(group["n_pairs"].min()),
                    "n_pairs_max": int(group["n_pairs"].max()),
                    "ci_positive_configurations": int(decisions.get("POSITIVE", 0)),
                    "ci_negative_configurations": int(decisions.get("NEGATIVE", 0)),
                    "ci_inconclusive_configurations": int(decisions.get("INCONCLUSIVE", 0)),
                    "inferential_scope": "WITHIN_DATASET_MODEL_PROMPT_NO_POOLED_CI",
                }
            )
    return pd.DataFrame(rows)


def old_g2_hash_checks(root: Path) -> dict[str, Any]:
    analysis = read_json(root, G2_ANALYSIS_MANIFEST)
    readout = read_json(root, G2_READOUT_MANIFEST)
    stats = read_json(root, G2_STATS_MANIFEST)
    expected = {
        str(G2_CONFIG): stats["input_hashes"][str(G2_CONFIG).replace("\\", "/")],
        str(COHORT): analysis["frozen_artifact_hash_checks"]["protocol"][str(COHORT).replace("\\", "/")]["expected"],
        str(CANONICAL_GRID): readout["output_hashes"][str(CANONICAL_GRID).replace("\\", "/")],
        str(CANONICAL_CONTRIBUTIONS): stats["output_hashes"][str(CANONICAL_CONTRIBUTIONS).replace("\\", "/")],
    }
    checks = {}
    for relative, expected_hash in expected.items():
        actual_hash = sha256_file(root / Path(relative))
        checks[relative.replace("\\", "/")] = {
            "expected": expected_hash,
            "actual": actual_hash,
            "match": actual_hash == expected_hash,
        }
    return checks


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    configure(root, args.config)
    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    completion, accounting, terminal_set_hash = load_completion(root)
    cohort = pd.read_csv(root / COHORT, dtype=str, keep_default_na=False)

    cohort_lookup = cohort[
        [
            "dataset",
            "user_id",
            "candidate_set_id",
            "candidate_hash",
            "candidate_items",
            "positive_item_id",
            "b1_primary_ge10_eligible",
            "secondary_full_curve_ge20_eligible",
        ]
    ].copy()
    cohort_lookup["candidate_items"] = cohort_lookup["candidate_items"].map(lambda value: stable_json(json.loads(value)))
    completion_check = completion.merge(
        cohort_lookup,
        on=["dataset", "user_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_cohort"),
    )
    completion_candidate_mismatches = int(
        (~completion_check["candidate_set_id"].eq(completion_check["candidate_set_id_cohort"])
         | ~completion_check["candidate_hash"].eq(completion_check["candidate_hash_cohort"])
         | ~completion_check["candidate_items"].eq(completion_check["candidate_items_cohort"])
         | ~completion_check["positive_item_id"].eq(completion_check["positive_item_id_cohort"])).sum()
    )
    completion_cohort_mismatches = int(
        (
            completion_check["history_length"].isin([3, 4])
            & ~completion_check["b1_primary_ge10_eligible"].str.lower().eq("true")
        ).sum()
        + (
            completion_check["history_length"].eq(15)
            & ~completion_check["secondary_full_curve_ge20_eligible"].str.lower().eq("true")
        ).sum()
    )

    endpoints = old_endpoint_frames(root, cohort)
    for history_length in (3, 4, 15):
        endpoints[history_length] = completion.loc[completion["history_length"].eq(history_length)].copy()

    grid_ok, grid_failures = verify_metric_grid(root, endpoints)
    pair_frames, pairability = build_pairs(endpoints)
    if completion_candidate_mismatches or completion_cohort_mismatches:
        pairability["status"] = "FAIL"
    stats = statistical_results(pair_frames)
    summary = synthesis(root, stats)
    hash_checks = old_g2_hash_checks(root)
    old_g2_unchanged = all(check["match"] for check in hash_checks.values())

    expected_grids = {
        "GE10": [0, 1, 2, 3, 4, 5, 10],
        "GE20": [0, 1, 2, 3, 4, 5, 10, 15, 20],
    }
    final_grid = pd.read_csv(root / FINAL_GRID, dtype=str, keep_default_na=False)
    actual_grids = {
        "GE10": sorted(
            final_grid.loc[
                final_grid["row_type"].eq("metric") & final_grid["analysis_membership"].eq("PRIMARY_B1_GE10"),
                "history_length",
            ].astype(int).unique().tolist()
        ),
        "GE20": sorted(
            final_grid.loc[
                final_grid["row_type"].eq("metric")
                & final_grid["analysis_membership"].eq("SECONDARY_FULL_CURVE_GE20"),
                "history_length",
            ].astype(int).unique().tolist()
        ),
    }

    accounting_path = output_dir / "request_accounting.csv"
    pairability_path = output_dir / "pairability_validation.csv"
    stats_path = output_dir / "statistical_results.csv"
    summary_path = output_dir / "scientific_synthesis_summary.csv"
    manifest_path = output_dir / "freeze_manifest.json"
    accounting.to_csv(accounting_path, index=False)
    pairability.to_csv(pairability_path, index=False)
    stats.to_csv(stats_path, index=False)
    summary.to_csv(summary_path, index=False)

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    totals = {
        column: int(accounting[column].sum())
        for column in ("expected", "accounted", "valid", "invalid", "backend_incomplete", "missing")
    }
    validations = {
        "terminal_accounting_pass": accounting["status"].eq("PASS").all(),
        "completion_candidate_contract_pass": completion_candidate_mismatches == 0,
        "completion_cohort_contract_pass": completion_cohort_mismatches == 0,
        "pairability_all_pass": pairability["status"].eq("PASS").all(),
        "metric_grid_means_match": grid_ok,
        "statistical_rows_450": len(stats) == 450,
        "synthesis_rows_15": len(summary) == 15,
        "old_g2_unchanged": old_g2_unchanged,
        "history_grids_match": actual_grids == expected_grids,
        "no_p_values_invented": stats["p_value"].eq("").all(),
    }
    freeze_status = "FINAL_SCIENTIFIC_FREEZE" if all(validations.values()) else "BLOCKED"
    manifest = {
        "experiment_id": "RQ3_H3_H4_H15_COMPLETION_SCIENTIFIC_FREEZE",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "protocol": {
            "config": str(G2_CONFIG).replace("\\", "/"),
            "config_sha256": sha256_file(root / G2_CONFIG),
            "metric_contract": "canonical_G2_HR5_NDCG5_MRR_invalid_terminal_zero",
        },
        "source_terminal_evidence": {
            "directory": str(TERMINAL_DIR).replace("\\", "/"),
            "request_manifest": str(REQUEST_MANIFEST).replace("\\", "/"),
            "request_manifest_sha256": sha256_file(root / REQUEST_MANIFEST),
            "terminal_set_sha256": terminal_set_hash,
        },
        "metric_grid_artifact": {
            "path": str(FINAL_GRID).replace("\\", "/"),
            "sha256": sha256_file(root / FINAL_GRID),
        },
        "statistical_artifact": {
            "path": str(stats_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(stats_path),
            "rows": len(stats),
        },
        "outputs": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in (accounting_path, pairability_path, stats_path, summary_path)
        },
        "request_accounting": {
            "totals": totals,
            "by_history_length": accounting.to_dict("records"),
        },
        "history_lengths_added": [3, 4, 15],
        "cohort_semantics": {
            "GE10": "frozen PRIMARY_B1_GE10 cohort; H3/H4 use identical users and candidates",
            "GE20": "frozen SECONDARY_FULL_CURVE_GE20 cohort; H15 and H0/H1/H2/H3/H4/H5/H10/H20 use identical users and candidates",
        },
        "history_grids": actual_grids,
        "bootstrap": {
            "method": "paired_user_bootstrap_mean_delta",
            "scope": "within_dataset_model_prompt",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence_interval": "two_sided_95_percent_percentile_ci",
            "engine": "empirical_multinomial_equivalent_to_paired_user_resampling_for_mean_delta",
            "percentile_method": "numpy_quantile_linear",
            "p_values": "NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD",
        },
        "old_g2_unchanged": old_g2_unchanged,
        "old_g2_hash_checks": hash_checks,
        "base_commit": git_commit,
        "freeze_status": freeze_status,
        "validations": validations,
        "validation_failures": grid_failures,
        "limitations": [
            "RQ3 completion extension; canonical G2 experiment and artifacts remain unchanged.",
            "Confidence intervals are configuration-specific; no post-hoc pooled cross-configuration CI is introduced.",
            "Invalid terminal responses contribute zero utility under the frozen fail-strict policy.",
            "No p-values or Holm correction are computed because the frozen primary method defines no p-value calculation.",
            "Local cohort identities are frozen operationalizations and are not claimed to reproduce author sample identities.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_scalar) + "\n",
        encoding="utf-8",
    )
    print(stable_json({"freeze_status": freeze_status, "totals": totals, "history_grids": actual_grids}))
    return 0 if freeze_status == "FINAL_SCIENTIFIC_FREEZE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

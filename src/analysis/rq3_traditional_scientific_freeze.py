"""Create the corrected, provenance-preserving RQ3 traditional scientific freeze.

This is deliberately a post-processing-only operation: it reads the existing
per-seed metrics, repairs a known aggregation defect, and never loads models,
datasets, checkpoints, or recommendation-generation code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HISTORY_GRID = [1, 2, 3, 4, 5, 10, 15, 20]
DATASETS = ["beauty", "sports", "ml1m", "lastfm"]
MODELS = ["MostPop", "BM25-like/content", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]
TRAINABLE_MODELS = {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}
SEEDS = [42, 43, 44, 45, 46]
METRICS = ["hr_at_5", "ndcg_at_5", "mrr"]
REPAIRED_MODELS = {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}
KEY_COLUMNS = ["dataset", "h", "method", "metric"]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_float(value: float) -> str:
    """Match the established traditional runner's float serialization."""
    return format(float(value), ".17g")


def source_paths(root: Path) -> dict[str, Path]:
    """Return immutable historical inputs required by this repair."""
    relative = [
        "results/tables/rq3_traditional_metrics_per_seed.csv",
        "results/tables/rq3_traditional_metrics_aggregated.csv",
        "results/tables/rq3_traditional_validation.csv",
        "configs/rq3_traditional_history_run.yaml",
        "src/extension/rq3_traditional_history_run.py",
        "artifacts/rq3_traditional_run/run_manifest.json",
        "artifacts/rq3_traditional_run/dprime_manifest.csv",
        "artifacts/rq3_final_freeze/rq3_final_manifest.json",
        "report/RQ3_TRADITIONAL_RUN_FREEZE.md",
        "report/RQ3_FINAL_FREEZE.md",
    ]
    paths = {path: root / path for path in relative}
    missing = [path for path, absolute in paths.items() if not absolute.is_file()]
    if missing:
        raise FileNotFoundError(f"required historical sources missing: {missing}")
    return paths


def expected_repaired_keys() -> set[tuple[str, int, str, str]]:
    """Return precisely the 12 approved aggregation-repair keys."""
    return {("beauty", 5, model, metric) for model in REPAIRED_MODELS for metric in METRICS}


def normalized_key(row: pd.Series) -> tuple[str, int, str, str]:
    return (str(row["dataset"]), int(row["h"]), str(row["method"]), str(row["metric"]))


def audit_inputs(per_seed: pd.DataFrame, aggregate: pd.DataFrame) -> dict[str, object]:
    """Fail closed unless the historical anomaly is exactly the approved one."""
    duplicate_groups = [group for _, group in aggregate.groupby(KEY_COLUMNS, sort=False) if len(group) > 1]
    duplicate_keys = {normalized_key(group.iloc[0]) for group in duplicate_groups}
    expected = expected_repaired_keys()
    if duplicate_keys != expected or len(duplicate_groups) != 12:
        raise ValueError(f"unexpected duplicate pattern: {sorted(duplicate_keys)}")

    repaired_seed_checks: list[dict[str, object]] = []
    for key in sorted(expected):
        dataset, h_value, method, metric = key
        rows = per_seed.loc[
            (per_seed["dataset"].astype(str) == dataset)
            & (pd.to_numeric(per_seed["h"], errors="raise") == h_value)
            & (per_seed["method"].astype(str) == method)
            & (per_seed["metric"].astype(str) == metric)
        ].copy()
        if len(rows) != 5:
            raise ValueError(f"{key}: expected five per-seed rows, found {len(rows)}")
        if set(pd.to_numeric(rows["seed"], errors="raise").astype(int)) != set(SEEDS):
            raise ValueError(f"{key}: seed set is not {SEEDS}")
        if rows["seed"].duplicated().any():
            raise ValueError(f"{key}: duplicate seed")
        values = pd.to_numeric(rows["value"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{key}: non-finite metric value")
        if set(rows["status"].astype(str)) != {"COMPLETED"}:
            raise ValueError(f"{key}: non-completed source status")
        repaired_seed_checks.append({"dataset": dataset, "h": h_value, "method": method, "metric": metric, "seed_set": "|".join(map(str, SEEDS)), "status": "PASS"})
    return {"duplicate_keys": duplicate_keys, "seed_checks": repaired_seed_checks}


def corrected_rows(per_seed: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    """Replace only the approved duplicate rows using all five saved seed values."""
    expected = expected_repaired_keys()
    replacements: dict[tuple[str, int, str, str], dict[str, object]] = {}
    for dataset, h_value, method, metric in sorted(expected):
        rows = per_seed.loc[
            (per_seed["dataset"].astype(str) == dataset)
            & (pd.to_numeric(per_seed["h"], errors="raise") == h_value)
            & (per_seed["method"].astype(str) == method)
            & (per_seed["metric"].astype(str) == metric)
        ]
        values = pd.to_numeric(rows["value"], errors="raise").to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        se = std / math.sqrt(len(values))
        ci = 1.96 * se
        replacements[(dataset, h_value, method, metric)] = {
            "dataset": dataset, "h": h_value, "method": method, "metric": metric,
            "mean": format_float(mean), "std": format_float(std), "se": format_float(se),
            "ci95_low": format_float(mean - ci), "ci95_high": format_float(mean + ci),
            "n": len(values), "status": "COMPLETED", "reason": "mean over seeds",
        }
    # Keep every unaffected row and its historical order byte-for-byte at field
    # level; emit the sole corrected row at the first occurrence of each defect.
    rows: list[dict[str, object]] = []
    emitted: set[tuple[str, int, str, str]] = set()
    for _, row in aggregate.iterrows():
        key = normalized_key(row)
        if key not in expected:
            rows.append(row.to_dict())
        elif key not in emitted:
            rows.append(replacements[key])
            emitted.add(key)
    return pd.DataFrame(rows, columns=aggregate.columns)


def validation_rows(root: Path, sources: dict[str, Path], source_hashes: dict[str, str], per_seed: pd.DataFrame, aggregate: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    """Build validation evidence without changing historical artifacts."""
    duplicate_count = sum(1 for _, group in corrected.groupby(KEY_COLUMNS) if len(group) > 1)
    repaired = corrected.loc[(corrected["dataset"] == "beauty") & (corrected["h"].astype(int) == 5) & corrected["method"].isin(REPAIRED_MODELS)]
    repaired_n_ok = len(repaired) == 12 and set(pd.to_numeric(repaired["n"], errors="raise").astype(int)) == {5}
    na = corrected.loc[(corrected["h"].astype(int) == 1) & corrected["method"].isin(["GRU4Rec", "SASRec"])]
    na_ok = len(na) == 24 and set(na["status"].astype(str)) == {"NOT_APPLICABLE"}
    numeric = corrected.loc[corrected["status"].eq("COMPLETED"), ["mean", "std", "se", "ci95_low", "ci95_high"]]
    no_nan = np.isfinite(numeric.astype(float).to_numpy()).all()
    expected_cells = 184 * 3
    completed_ok = int(corrected["status"].eq("COMPLETED").sum()) == expected_cells
    raw_files = list((root / "results/raw_outputs/rq3_traditional").glob("*.csv"))
    expected = expected_repaired_keys()
    old_unaffected = aggregate.loc[[normalized_key(row) not in expected for _, row in aggregate.iterrows()]].reset_index(drop=True)
    new_unaffected = corrected.loc[[normalized_key(row) not in expected for _, row in corrected.iterrows()]].reset_index(drop=True)
    unaffected_unchanged = old_unaffected.equals(new_unaffected)
    dprime = pd.read_csv(sources["artifacts/rq3_traditional_run/dprime_manifest.csv"])
    dprime_ok = len(dprime) == 32 and int(pd.to_numeric(dprime["target_leakage_count"]).sum()) == 0 and int(pd.to_numeric(dprime["test_leakage_count"]).sum()) == 0
    rows = [
        ("aggregate_duplicate_keys", duplicate_count == 0, str(duplicate_count)),
        ("repaired_keys_n5", repaired_n_ok, str(len(repaired))),
        ("applicable_metric_rows_complete", completed_ok, f"{int(corrected['status'].eq('COMPLETED').sum())}/{expected_cells}"),
        ("seed_coverage_repaired_keys", True, "42|43|44|45|46 for each repaired key"),
        ("gru4rec_sasrec_h1_not_applicable", na_ok, str(len(na))),
        ("no_unexpected_nan", bool(no_nan), "completed aggregate statistics finite"),
        ("raw_per_seed_evidence_present", len(raw_files) == 664, str(len(raw_files))),
        ("unaffected_historical_aggregate_rows_identical", unaffected_unchanged, str(len(old_unaffected))),
        ("dprime_and_leakage_contract_unchanged", dprime_ok, "32 D-prime cells; zero target/test leakage"),
    ]
    for path, expected_hash in source_hashes.items():
        rows.append((f"historical_source_hash::{path}", sha256(sources[path]) == expected_hash, expected_hash))
    return pd.DataFrame(rows, columns=["check", "status_bool", "evidence"]).assign(status=lambda frame: frame["status_bool"].map({True: "PASS", False: "FAIL"})).drop(columns="status_bool")


def write_report(path: Path, corrected_hash: str, validation_hash: str) -> None:
    """Write the concise professor-facing freeze report."""
    path.write_text(f"""# RQ3 Traditional Scientific Freeze\n\n## Scope\n\nHistory-length sensitivity of traditional recommenders, retained as a local diagnostic extension. This is post-processing only; raw evidence already existed.\n\n## History grid\n\n`H = {{1, 2, 3, 4, 5, 10, 15, 20}}` across Beauty, Sports, ML-1M and LastFM.\n\n## Models and H semantics\n\n- **MostPop:** H-invariant train-split popularity control.\n- **BM25-like/content:** H truncates the evaluation query history.\n- **BPRMF:** retrained on `D'_H`.\n- **GRU4Rec:** sequential retraining on `D'_H`.\n- **SASRec:** sequential retraining on `D'_H`.\n- **LightGCN:** retrained on `D'_H`.\n\nGRU4Rec H1 and SASRec H1 are `NOT_APPLICABLE`: one interaction cannot form a next-item sequence example.\n\n## Repair\n\nTwelve historical Beauty/H5 aggregate keys (four trainable models × HR@5, NDCG@5 and MRR) had separate pilot `n=1` and remaining-seed `n=4` rows. Each was reconstructed deterministically from the existing seed set 42–46. No model execution, inference, recommendation generation or dataset reconstruction was performed.\n\nCorrected aggregate SHA-256: `{corrected_hash}`. Validation SHA-256: `{validation_hash}`.\n\n## Scientific status\n\nThis is `LOCAL_OPERATIONALIZATION_NOT_PAPER_EXACT`, not a paper-exact ReChorus reproduction. The traditional intervention is not semantically identical to the LLM GE10/GE20 history intervention: MostPop is invariant, BM25 changes query construction, and trainable models retrain on `D'_H`. It supersedes neither the older operational freeze nor its historical artifacts; it adds a separately provenance-tracked corrected post-processing freeze.\n""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create corrected RQ3 traditional scientific freeze without execution.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    sources = source_paths(root)
    source_hashes = {path: sha256(absolute) for path, absolute in sources.items()}
    per_seed = pd.read_csv(sources["results/tables/rq3_traditional_metrics_per_seed.csv"], dtype=str, keep_default_na=False)
    aggregate = pd.read_csv(sources["results/tables/rq3_traditional_metrics_aggregated.csv"], dtype=str, keep_default_na=False)
    audit = audit_inputs(per_seed, aggregate)
    corrected = corrected_rows(per_seed, aggregate)
    if any(len(group) > 1 for _, group in corrected.groupby(KEY_COLUMNS)):
        raise AssertionError("repair failed: duplicate aggregate keys remain")

    output_table = root / "results/tables/rq3_traditional_metrics_aggregated_corrected.csv"
    output_validation = root / "results/tables/rq3_traditional_validation_corrected.csv"
    freeze_dir = root / "results/rq3_traditional_scientific_freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_table, index=False, lineterminator="\n")
    validation = validation_rows(root, sources, source_hashes, per_seed, aggregate, corrected)
    validation.to_csv(output_validation, index=False, lineterminator="\n")
    repair_summary = pd.DataFrame(audit["seed_checks"])
    repair_summary["historical_duplicate_rows"] = 2
    repair_summary["corrected_n"] = 5
    repair_summary["statistics"] = "mean; sample_std_ddof1; se=std/sqrt(n); ci95=mean±1.96*se"
    repair_summary_path = freeze_dir / "repair_summary.csv"
    repair_summary.to_csv(repair_summary_path, index=False, lineterminator="\n")
    summary_path = freeze_dir / "validation_summary.csv"
    validation.to_csv(summary_path, index=False, lineterminator="\n")
    corrected_hash = sha256(output_table)
    validation_hash = sha256(output_validation)
    report_path = root / "report/RQ3_TRADITIONAL_SCIENTIFIC_FREEZE.md"
    write_report(report_path, corrected_hash, validation_hash)
    manifest = {
        "experiment_id": "RQ3_TRADITIONAL_HISTORY_AGGREGATION_REPAIR_SCIENTIFIC_FREEZE",
        "status": "VALID" if validation["status"].eq("PASS").all() else "INVALID",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_classification": "LOCAL_OPERATIONALIZATION_NOT_PAPER_EXACT",
        "datasets": DATASETS, "history_grid": HISTORY_GRID, "models": MODELS, "seeds": SEEDS, "metrics": METRICS,
        "not_applicable_contract": "GRU4Rec and SASRec H1: INSUFFICIENT_SEQUENCE_LENGTH_UNDER_FROZEN_DPRIME",
        "repair": {"corrected_duplicate_keys": 12, "scope": "beauty/H5; BPRMF, GRU4Rec, SASRec, LightGCN; HR@5, NDCG@5, MRR", "derived_only_from_existing_per_seed_evidence": True, "training_executed": False, "inference_executed": False, "statistics": "sample_std_ddof1; se=std/sqrt(n); ci95=mean±1.96*se"},
        "source_files": source_hashes,
        "outputs": {"corrected_aggregate": {"path": output_table.relative_to(root).as_posix(), "sha256": corrected_hash}, "corrected_validation": {"path": output_validation.relative_to(root).as_posix(), "sha256": validation_hash}, "repair_summary": {"path": repair_summary_path.relative_to(root).as_posix(), "sha256": sha256(repair_summary_path)}, "validation_summary": {"path": summary_path.relative_to(root).as_posix(), "sha256": sha256(summary_path)}, "report": {"path": report_path.relative_to(root).as_posix(), "sha256": sha256(report_path)}},
        "relationship_to_older_operational_freeze": "Preserves historical source artifacts unchanged and adds a corrected post-processing-only scientific freeze.",
        "limitations": ["Not paper-exact ReChorus reproduction.", "Traditional H semantics differ by model and are not identical to the LLM GE10/GE20 intervention.", "MostPop is H-invariant; GRU4Rec/SASRec H1 are structurally not applicable."],
    }
    (freeze_dir / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"FREEZE_STATUS={manifest['status']}")
    print("REPAIRED_KEYS=12")
    return 0 if manifest["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())

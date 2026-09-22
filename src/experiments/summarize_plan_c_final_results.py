"""Create final Plan C summary tables and overview plots.

Run from the repository root:

    python src/experiments/summarize_plan_c_final_results.py

The script reads existing Plan C reconstruction, metric, comparison, and
blocker CSVs. It does not run experiments, download data, call APIs, train
models, or create placeholder metric files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FINAL_RESULTS_COLUMNS = [
    "dataset",
    "reconstruction_status",
    "exact_original_artifact_recovered",
    "candidate_item_universe",
    "baseline",
    "run_scope",
    "diagnostic_subset",
    "max_users",
    "hr_at_5",
    "ndcg_at_5",
    "catalog_coverage_at_5",
    "evaluated_users",
    "status",
    "notes",
]

CLAIMS_COLUMNS = ["claim", "status", "evidence", "limitation", "how_to_explain"]
INVENTORY_COLUMNS = [
    "phase",
    "artifact/script/report",
    "purpose",
    "generated_outputs",
    "status",
    "commit_relevance",
    "notes",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final Plan C results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Project-relative results directory.",
    )
    return parser.parse_args(argv)


def summarize_plan_c_final_results(
    *,
    project_root: str | Path | None = None,
    results_dir: str | Path = Path("results"),
) -> dict[str, pd.DataFrame | list[Path]]:
    """Write final Plan C result, claim, inventory, and plot outputs."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    results_root = _resolve_path(results_dir, root)
    tables_dir = results_root / "tables"
    plots_dir = results_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    reconstruction = _read_csv(tables_dir / "plan_c_reconstructed_dataset_summary.csv")
    comparison = _read_csv(tables_dir / "plan_c_baseline_comparison_summary.csv")
    blockers = _read_csv(tables_dir / "plan_c_baseline_blockers.csv")

    overview = build_final_results_overview(reconstruction, comparison, blockers)
    claims = build_claims_and_limitations()
    inventory = build_final_experiment_inventory()
    plot_paths = write_final_plots(overview, plots_dir)

    _write_csv(overview, tables_dir / "plan_c_final_results_overview.csv")
    _write_csv(claims, tables_dir / "plan_c_claims_and_limitations.csv")
    _write_csv(inventory, tables_dir / "plan_c_final_experiment_inventory.csv")
    return {
        "overview": overview,
        "claims": claims,
        "inventory": inventory,
        "plot_paths": plot_paths,
    }


def build_final_results_overview(
    reconstruction: pd.DataFrame,
    comparison: pd.DataFrame,
    blockers: pd.DataFrame,
) -> pd.DataFrame:
    """Combine reconstructed status, real metric rows, and blocker rows."""

    if comparison.empty and reconstruction.empty and blockers.empty:
        return pd.DataFrame(
            [
                {
                    "dataset": "plan_c",
                    "reconstruction_status": "unknown",
                    "exact_original_artifact_recovered": "unknown",
                    "candidate_item_universe": "",
                    "baseline": "",
                    "run_scope": "",
                    "diagnostic_subset": False,
                    "max_users": pd.NA,
                    "hr_at_5": pd.NA,
                    "ndcg_at_5": pd.NA,
                    "catalog_coverage_at_5": pd.NA,
                    "evaluated_users": 0,
                    "status": "missing_source_tables",
                    "notes": "Required Plan C summary inputs were not found; no values invented.",
                }
            ],
            columns=FINAL_RESULTS_COLUMNS,
        )

    reconstruction_lookup = _reconstruction_lookup(reconstruction)
    rows: list[dict[str, Any]] = []
    if not comparison.empty:
        normalized = _normalize_columns(comparison)
        for _, row in normalized.iterrows():
            dataset = str(row.get("dataset", ""))
            rec = reconstruction_lookup.get(dataset, {})
            rows.append(
                {
                    "dataset": dataset,
                    "reconstruction_status": row.get(
                        "reconstruction_status",
                        rec.get("reconstruction_status", "unknown"),
                    ),
                    "exact_original_artifact_recovered": rec.get(
                        "exact_original_artifact_recovered", "no"
                    ),
                    "candidate_item_universe": rec.get("candidate_item_universe", ""),
                    "baseline": row.get("baseline", ""),
                    "run_scope": row.get("run_scope", ""),
                    "diagnostic_subset": _as_bool(row.get("diagnostic_subset", False)),
                    "max_users": _optional_int(row.get("max_users", pd.NA)),
                    "hr_at_5": row.get("hr_at_5", pd.NA),
                    "ndcg_at_5": row.get("ndcg_at_5", pd.NA),
                    "catalog_coverage_at_5": row.get("catalog_coverage_at_5", pd.NA),
                    "evaluated_users": _optional_int(row.get("evaluated_users", 0)) or 0,
                    "status": row.get("status", "available"),
                    "notes": row.get("notes", "Real Plan C reconstructed-dataset metric row."),
                }
            )

    if not blockers.empty:
        normalized_blockers = _normalize_columns(blockers)
        for _, row in normalized_blockers.iterrows():
            dataset = str(row.get("dataset", ""))
            rec = reconstruction_lookup.get(dataset, {})
            rows.append(
                {
                    "dataset": dataset,
                    "reconstruction_status": row.get(
                        "reconstruction_status",
                        rec.get("reconstruction_status", "unknown"),
                    ),
                    "exact_original_artifact_recovered": rec.get(
                        "exact_original_artifact_recovered", "no"
                    ),
                    "candidate_item_universe": rec.get("candidate_item_universe", ""),
                    "baseline": row.get("baseline", ""),
                    "run_scope": row.get("run_scope", ""),
                    "diagnostic_subset": _as_bool(row.get("diagnostic_subset", False)),
                    "max_users": _optional_int(row.get("max_users", pd.NA)),
                    "hr_at_5": pd.NA,
                    "ndcg_at_5": pd.NA,
                    "catalog_coverage_at_5": pd.NA,
                    "evaluated_users": 0,
                    "status": row.get("status", "blocked"),
                    "notes": row.get("reason", row.get("notes", "Blocked Plan C component.")),
                }
            )

    return pd.DataFrame(rows, columns=FINAL_RESULTS_COLUMNS).sort_values(
        ["dataset", "diagnostic_subset", "baseline", "run_scope"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_claims_and_limitations() -> pd.DataFrame:
    """Return the final Plan C claims and limitations table."""

    rows = [
        {
            "claim": "exact original paper artifact recovery",
            "status": "blocked",
            "evidence": "Artifact manifest, source ledger, and manual Dropbox checks.",
            "limitation": "Official Beauty/Sports processed artifacts remain unavailable.",
            "how_to_explain": "We audited the sources and documented the missing files instead of inventing them.",
        },
        {
            "claim": "Beauty/Sports exact reproduction",
            "status": "blocked_missing_processed_artifacts",
            "evidence": "Missing sequential_data.txt, negative_samples.txt, datamaps.json, and paper processed meta.json.",
            "limitation": "Reconstructed public-raw data cannot prove equivalence to author preprocessing.",
            "how_to_explain": "Exact reproduction requires the official processed artifacts or author-confirmed protocol.",
        },
        {
            "claim": "Beauty/Sports reconstructed evaluation",
            "status": "completed_reconstructed",
            "evidence": "Plan C reconstructed datasets and full Random/MostPop metric CSVs.",
            "limitation": "These are reconstructed-dataset metrics, not original paper results.",
            "how_to_explain": "They show a defensible reconstruction after artifact recovery failed.",
        },
        {
            "claim": "ML-1M reconstructed evaluation",
            "status": "completed_reconstructed",
            "evidence": "Full Random, MostPop, and ContentBM25 Plan C metrics.",
            "limitation": "Still a reconstruction protocol, not proof of exact paper preprocessing.",
            "how_to_explain": "ML-1M is stable and suitable for end-to-end evaluation evidence.",
        },
        {
            "claim": "LastFM utility evaluation",
            "status": "blocked_extension_only",
            "evidence": "C5 writes interactions/catalog only; no ground_truth/candidates.",
            "limitation": "HetRec user_artists.dat lacks listening timestamps for chronological split.",
            "how_to_explain": "Without a valid split and candidates, HR/NDCG would be fabricated.",
        },
        {
            "claim": "LLM/API reproduction",
            "status": "not_run",
            "evidence": "No --allow-api-calls workflow was executed in Plan C.",
            "limitation": "API access, costs, model drift, and unavailable exact versions remain issues.",
            "how_to_explain": "The project prioritized artifact validity and deterministic baselines first.",
        },
        {
            "claim": "deterministic baseline evaluation",
            "status": "completed",
            "evidence": "Random, MostPop, and ContentBM25 scripts/tests and metrics.",
            "limitation": "Advanced neural and LLM baselines are future work.",
            "how_to_explain": "Baselines validate the reconstructed candidate evaluation pipeline.",
        },
        {
            "claim": "candidate generation validity",
            "status": "improved_with_diagnostics",
            "evidence": "C5 summaries and C6 candidate diagnostics.",
            "limitation": "The original paper's exact candidate sampling protocol remains unrecovered.",
            "how_to_explain": "We corrected the default to interaction items and made the choice explicit.",
        },
        {
            "claim": "negative sampling correction",
            "status": "completed",
            "evidence": "negative_sampling.item_universe=interaction_items and regenerated metrics.",
            "limitation": "Catalog-wide mode is experimental only and can inflate MostPop.",
            "how_to_explain": "The correction reduced suspicious Amazon MostPop HR@5 values.",
        },
        {
            "claim": "publication-style extension potential",
            "status": "promising_not_final",
            "evidence": "Audits, manifests, reconstruction pipeline, metrics, and reports.",
            "limitation": "Needs author artifacts or additional experiments for publication-grade claims.",
            "how_to_explain": "It is a strong reproducibility engineering study, not a finished paper replication.",
        },
    ]
    return pd.DataFrame(rows, columns=CLAIMS_COLUMNS)


def build_final_experiment_inventory() -> pd.DataFrame:
    """Return the final Plan C phase inventory."""

    rows = [
        ("C0", "plan_c_full_replication_matrix.md; plan_c_*.csv", "Inventory the paper/repo replication surface.", "replication matrix and artifact/task/model inventory", "completed", "include", "Defines exact/reconstruction/blocker taxonomy."),
        ("C1", "validate_plan_c_artifacts.py; artifact manifests", "Validate local artifact availability.", "artifact validation details and summary", "completed", "include", "No downloads or mutations."),
        ("C2", "check_plan_c_source_urls.py; source reports", "Check source reachability and recovery routes.", "source URL checks and manual download checklist", "completed", "include", "Dropbox HTML reachability is not artifact recovery."),
        ("C3", "inspect_plan_c_downloads.py", "Inspect staged manual downloads without extracting by default.", "manual download inventory and suggested manifest updates", "completed", "include", "Raw public data classified separately from official processed artifacts."),
        ("C4", "validate_plan_c_raw_datasets.py", "Validate raw dataset schemas and reconstruction readiness.", "raw schema and readiness tables", "completed", "include", "LastFM marked extension-only for utility metrics."),
        ("C5", "reconstruct_plan_c_datasets.py", "Create reconstructed canonical datasets.", "data/reconstructed/plan_c and reconstructed summary tables", "completed", "include", "Negative sampling default corrected to interaction_items."),
        ("C6", "run_plan_c_baselines.py", "Evaluate deterministic baselines on reconstructed datasets.", "Plan C metrics, comparison tables, candidate diagnostics, plots", "completed_with_scope_limits", "include", "Amazon ContentBM25 full runs remain future work; diagnostic subsets are marked."),
        ("C7", "summarize_plan_c_final_results.py; final reports", "Consolidate claims, limitations, and exam narrative.", "final overview, claims, inventory, reports, oral guide", "completed", "include", "Final story separates exact reproduction blockers from reconstruction evidence."),
    ]
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def write_final_plots(overview: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Write simple final HR@5 and NDCG@5 overview plots from real rows."""

    written: list[Path] = []
    if overview.empty:
        return written
    for metric, filename, ylabel in (
        ("hr_at_5", "plan_c_final_hr_at_5_overview.png", "HR@5"),
        ("ndcg_at_5", "plan_c_final_ndcg_at_5_overview.png", "NDCG@5"),
    ):
        frame = overview.copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame.dropna(subset=[metric])
        if frame.empty:
            continue
        labels = [
            f"{row.dataset}\n{row.baseline}\n{row.run_scope}"
            for row in frame.itertuples(index=False)
        ]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.0), 5.2))
        ax.bar(range(len(labels)), frame[metric].astype(float).tolist())
        ax.set_title(f"Plan C final reconstructed {ylabel} overview")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Dataset, baseline, and run scope")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def _reconstruction_lookup(reconstruction: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if reconstruction.empty:
        return {}
    normalized = _normalize_columns(reconstruction)
    return {
        str(row["dataset"]): row.to_dict()
        for _, row in normalized.iterrows()
        if "dataset" in normalized.columns
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _optional_int(value: Any) -> int | Any:
    if pd.isna(value) or str(value).strip() == "":
        return pd.NA
    try:
        return int(float(str(value)))
    except ValueError:
        return pd.NA


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = summarize_plan_c_final_results(results_dir=args.results_dir)
    overview = outputs["overview"]
    print("Plan C final summary complete")
    print(f"overview_rows={len(overview)}")
    print("results=tables/plan_c_final_results_overview.csv")
    print("claims=tables/plan_c_claims_and_limitations.csv")
    print("inventory=tables/plan_c_final_experiment_inventory.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

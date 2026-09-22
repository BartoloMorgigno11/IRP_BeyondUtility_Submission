"""Summarize Plan D extended neural baselines across utility-evaluable data.

This module reads existing Plan D metric CSVs and writes derived summary,
coverage, comparison, and markdown report artifacts. It does not train models,
run inference, create recommendation outputs, download data, or call APIs.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILITY_DATASETS = ("ml1m", "amazon_beauty", "amazon_sports")
COVERAGE_DATASETS = ("ml1m", "amazon_beauty", "amazon_sports", "lastfm")
METHODS = ("bprmf", "lightgcn", "sasrec", "gru4rec")
SUMMARY_METRICS = ("hit_rate", "ndcg", "final_train_loss", "best_train_loss")
MAX_USERS = 1000
MAX_TRAIN_INTERACTIONS = 100000

SUMMARY_COLUMNS = [
    "plan",
    "dataset",
    "scope",
    "method",
    "hr_at_5",
    "ndcg_at_5",
    "final_train_loss",
    "best_train_loss",
    "num_users_evaluated",
    "num_recommendations_evaluated",
    "num_train_interactions",
    "num_items",
    "num_users",
    "runtime_seconds",
    "device",
    "epochs_completed",
    "best_epoch",
    "seed",
    "config_hash",
    "metrics_path",
    "recommendations_path",
    "dataset_method_hr_rank",
    "dataset_method_ndcg_rank",
    "global_hr_rank",
    "global_ndcg_rank",
    "notes",
]

COVERAGE_COLUMNS = [
    "dataset",
    "method",
    "diagnostic_subset_available",
    "extended_subset_available",
    "full_available",
    "extended_metrics_path",
    "diagnostic_metrics_path",
    "status",
    "notes",
]

COMPARISON_COLUMNS = [
    "dataset",
    "method",
    "metric",
    "diagnostic_value",
    "extended_value",
    "absolute_delta",
    "relative_delta",
    "diagnostic_num_users_evaluated",
    "extended_num_users_evaluated",
    "diagnostic_num_train_interactions",
    "extended_num_train_interactions",
    "diagnostic_epochs_completed",
    "extended_epochs_completed",
    "notes",
]


def load_metric_file(path: str | Path) -> pd.DataFrame:
    """Load a non-empty Plan D long-form metric CSV as strings."""

    metric_path = Path(path)
    if not metric_path.is_file():
        raise FileNotFoundError(f"Missing metric file: {metric_path}")
    frame = pd.read_csv(metric_path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"Metric file is empty: {metric_path}")
    required = {"plan", "dataset", "method", "scope", "metric", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metric file {metric_path} is missing required columns: {sorted(missing)}")
    return frame


def extract_dataset_method_summary(path: str | Path, *, project_root: str | Path | None = None) -> dict[str, str]:
    """Extract one extended-subset summary row from a Plan D metric CSV."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    metric_path = Path(path)
    frame = load_metric_file(metric_path)
    values = _metric_value_map(frame)
    missing_metrics = [metric for metric in SUMMARY_METRICS if metric not in values]
    if missing_metrics:
        raise ValueError(f"{metric_path} is missing metrics: {missing_metrics}")

    first = frame.iloc[0].to_dict()
    return {
        "plan": first.get("plan", ""),
        "dataset": first.get("dataset", ""),
        "scope": first.get("scope", ""),
        "method": first.get("method", ""),
        "hr_at_5": values["hit_rate"],
        "ndcg_at_5": values["ndcg"],
        "final_train_loss": values["final_train_loss"],
        "best_train_loss": values["best_train_loss"],
        "num_users_evaluated": first.get("num_users_evaluated", ""),
        "num_recommendations_evaluated": first.get("num_recommendations_evaluated", ""),
        "num_train_interactions": first.get("num_train_interactions", ""),
        "num_items": first.get("num_items", ""),
        "num_users": first.get("num_users", ""),
        "runtime_seconds": first.get("runtime_seconds", ""),
        "device": first.get("device", ""),
        "epochs_completed": first.get("epochs_completed", ""),
        "best_epoch": first.get("best_epoch", ""),
        "seed": first.get("seed", ""),
        "config_hash": first.get("config_hash", ""),
        "metrics_path": _relative_path(metric_path, root),
        "recommendations_path": first.get("recommendations_path", ""),
        "dataset_method_hr_rank": "",
        "dataset_method_ndcg_rank": "",
        "global_hr_rank": "",
        "global_ndcg_rank": "",
        "notes": _summary_notes(first),
    }


def build_extended_summary(*, project_root: str | Path | None = None) -> pd.DataFrame:
    """Build one-row-per-dataset-method summary for utility-evaluable data."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    rows = [
        extract_dataset_method_summary(_metric_path(root, dataset, method, "extended_subset"), project_root=root)
        for dataset in UTILITY_DATASETS
        for method in METHODS
    ]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    _assign_rank_within_dataset(summary, "hr_at_5", "dataset_method_hr_rank")
    _assign_rank_within_dataset(summary, "ndcg_at_5", "dataset_method_ndcg_rank")
    _assign_global_rank(summary, "hr_at_5", "global_hr_rank")
    _assign_global_rank(summary, "ndcg_at_5", "global_ndcg_rank")
    return summary[SUMMARY_COLUMNS]


def build_coverage_table(*, project_root: str | Path | None = None) -> pd.DataFrame:
    """Build explicit dataset/model availability coverage, including LastFM."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    rows: list[dict[str, str]] = []
    for dataset in COVERAGE_DATASETS:
        for method in METHODS:
            diagnostic_path = _metric_path(root, dataset, method, "diagnostic_subset")
            extended_path = _metric_path(root, dataset, method, "extended_subset")
            full_path = _metric_path(root, dataset, method, "full")
            diagnostic_available = diagnostic_path.is_file()
            extended_available = extended_path.is_file()
            full_available = full_path.is_file()

            if dataset == "lastfm" and not (extended_available or full_available):
                status = "extension_only_no_valid_utility_evaluation"
                notes = (
                    "LastFM is excluded from HR@5/NDCG@5 ranking because no valid comparable chronological "
                    "split/candidate/ground-truth setup is available in the reconstructed project; no fake "
                    "LastFM utility metrics were generated."
                )
            elif full_available:
                status = "full_available"
                notes = "Full-scope metric file exists for this dataset/method."
            elif extended_available:
                status = "extended_available"
                notes = "Extended-subset metric file exists for this dataset/method."
            elif diagnostic_available:
                status = "diagnostic_only"
                notes = "Diagnostic-subset metric file exists, but no extended/full metric file was found."
            else:
                status = "missing"
                notes = "No Plan D utility metric file was found for this dataset/method."

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "diagnostic_subset_available": _bool_string(diagnostic_available),
                    "extended_subset_available": _bool_string(extended_available),
                    "full_available": _bool_string(full_available),
                    "extended_metrics_path": _relative_path(extended_path, root) if extended_available else "",
                    "diagnostic_metrics_path": _relative_path(diagnostic_path, root) if diagnostic_available else "",
                    "status": status,
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_diagnostic_vs_extended(*, project_root: str | Path | None = None) -> pd.DataFrame:
    """Compare diagnostic-subset and extended-subset values when both exist."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    rows: list[dict[str, str]] = []
    for dataset in UTILITY_DATASETS:
        for method in METHODS:
            diagnostic_path = _metric_path(root, dataset, method, "diagnostic_subset")
            extended_path = _metric_path(root, dataset, method, "extended_subset")
            if not diagnostic_path.is_file() or not extended_path.is_file():
                continue

            diagnostic = load_metric_file(diagnostic_path)
            extended = load_metric_file(extended_path)
            diagnostic_values = _metric_value_map(diagnostic)
            extended_values = _metric_value_map(extended)
            diagnostic_first = diagnostic.iloc[0].to_dict()
            extended_first = extended.iloc[0].to_dict()

            for metric in SUMMARY_METRICS:
                if metric not in diagnostic_values or metric not in extended_values:
                    continue
                absolute_delta, relative_delta = _compute_deltas(diagnostic_values[metric], extended_values[metric])
                note = "Compared from existing diagnostic_subset and extended_subset metric CSVs."
                if relative_delta == "":
                    note = "Diagnostic value is zero or non-numeric; relative delta not computed."
                rows.append(
                    _comparison_row(
                        dataset=dataset,
                        method=method,
                        metric=metric,
                        diagnostic_value=diagnostic_values[metric],
                        extended_value=extended_values[metric],
                        diagnostic_first=diagnostic_first,
                        extended_first=extended_first,
                        note=note,
                        absolute_delta=absolute_delta,
                        relative_delta=relative_delta,
                    )
                )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def write_markdown_report(
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    comparison: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write a markdown report from generated global summary tables."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_summary = _sort_summary_for_report(summary)
    global_top = summary.sort_values(["global_hr_rank", "dataset", "method"], key=_rank_sort_key).head(5)
    missing_comparisons = _missing_comparison_notes(coverage)
    user_constraint_notes = _user_constraint_notes(summary)

    lines = [
        "# Plan D Global Extended Neural Baseline Summary",
        "",
        "## Purpose",
        "",
        "This report aggregates reconstructed Plan D extended_subset results across the utility-evaluable datasets:",
        "ML-1M, Amazon Beauty, and Amazon Sports.",
        "",
        "## Paper Dataset Coverage",
        "",
        "The original paper includes four datasets:",
        "",
        "- Amazon Beauty",
        "- Amazon Sports and Outdoors",
        "- MovieLens-1M",
        "- LastFM",
        "",
        "## LastFM Status and Exclusion Rationale",
        "",
        "LastFM is present in the broader project as extension_only/problematic, but it is excluded from HR@5/NDCG@5",
        "ranking because the reconstructed project does not have a valid comparable chronological split/candidate/",
        "ground-truth setup. No fake LastFM utility metrics were generated.",
        "",
        "## Scope",
        "",
        "This is not an exact paper reproduction because the official preprocessed artifacts and exact paper candidate",
        "pools are unavailable. The results are controlled Plan D extended_subset evaluations on reconstructed inputs.",
        "",
        "## Common Setup",
        "",
        "- Scope: extended_subset",
        "- k: 5",
        f"- max_users: {MAX_USERS}",
        f"- max_train_interactions: {MAX_TRAIN_INTERACTIONS} cap",
        "- epochs: 5",
        "- Evaluation: candidate-set ranking",
        "- Device: CUDA where available",
        "- Hyperparameters: controlled safe settings, not exhaustive tuning",
        "",
        "## Global Summary",
        "",
        _markdown_table(
            sorted_summary,
            [
                "dataset",
                "method",
                "hr_at_5",
                "ndcg_at_5",
                "final_train_loss",
                "best_train_loss",
                "num_users_evaluated",
                "num_train_interactions",
                "runtime_seconds",
                "device",
                "dataset_method_hr_rank",
                "dataset_method_ndcg_rank",
                "global_hr_rank",
                "global_ndcg_rank",
            ],
            numeric_columns={"hr_at_5", "ndcg_at_5", "final_train_loss", "best_train_loss", "runtime_seconds"},
        ),
        "",
        "## Dataset-Level Findings",
        "",
    ]
    for dataset in UTILITY_DATASETS:
        lines.append(f"- {_display_dataset(dataset)} HR@5 ranking: {_ranking_text(summary, dataset, 'dataset_method_hr_rank')}.")
        lines.append(
            f"- {_display_dataset(dataset)} NDCG@5 ranking: "
            f"{_ranking_text(summary, dataset, 'dataset_method_ndcg_rank')}."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Global HR@5 top 5: {_global_top_text(global_top)}.",
            f"- {_best_method_statement(summary, 'ml1m')}",
            f"- {_best_method_statement(summary, 'amazon_beauty')}",
            f"- {_best_method_statement(summary, 'amazon_sports')}",
            f"- {_lightgcn_vs_bprmf_statement(summary)}",
            f"- {_sasrec_amazon_statement(summary)}",
            f"- {_amazon_sparsity_statement(summary)}",
        ]
    )
    lines.extend(f"- {note}" for note in user_constraint_notes)
    lines.extend(
        [
            "",
            "## Diagnostic vs Extended",
            "",
            f"The diagnostic-vs-extended comparison table contains {len(comparison)} rows.",
        ]
    )
    if missing_comparisons:
        lines.append("Missing diagnostic comparisons were skipped for:")
        lines.extend(f"- {note}" for note in missing_comparisons)
    else:
        lines.append("All utility-evaluable dataset/method pairs with extended metrics also have diagnostic comparisons.")
    lines.extend(
        [
            "",
            "## Dataset/Model Coverage",
            "",
            f"The coverage table contains {len(coverage)} rows across four datasets and four methods.",
            "LastFM rows are included with extension_only_no_valid_utility_evaluation status unless real comparable",
            "metric files exist.",
            "",
            "## Limitations",
            "",
            "- The datasets are reconstructed, not official preprocessed paper artifacts.",
            "- These are extended_subset results, not full reproduction results.",
            "- LastFM is excluded from utility ranking for methodological reasons.",
            "- Candidate-set evaluation depends on reconstructed candidates.",
            "- Hyperparameters are controlled but not exhaustively tuned.",
            "- LLM ranking/reranking remains blocked until a backend/API is approved.",
            "",
            "## Recommended Next Steps",
            "",
            "1. Update the reproducibility matrix with completed extended baselines and the LastFM coverage note.",
            "2. Create the final Plan D baseline report section.",
            "3. Decide the LLM backend: Ollama, local transformers, or API.",
            "4. If an LLM backend is available, run LLM ranking diagnostic.",
            "5. Otherwise document the LLM blocker and proceed to the final report.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_outputs(*, project_root: str | Path | None = None) -> dict[str, Path]:
    """Write Plan D global extended neural summary tables and report."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    tables_dir = root / "results" / "tables"
    report_dir = root / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = build_extended_summary(project_root=root)
    coverage = build_coverage_table(project_root=root)
    comparison = build_diagnostic_vs_extended(project_root=root)

    summary_path = tables_dir / "plan_d_global_extended_neural_baselines_summary.csv"
    coverage_path = tables_dir / "plan_d_global_extended_dataset_model_coverage.csv"
    comparison_path = tables_dir / "plan_d_global_neural_diagnostic_vs_extended.csv"
    report_path = report_dir / "plan_d_global_extended_neural_baselines.md"

    summary.to_csv(summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    write_markdown_report(summary, coverage, comparison, report_path)
    return {
        "summary": summary_path,
        "coverage": coverage_path,
        "comparison": comparison_path,
        "report": report_path,
    }


def _metric_path(root: Path, dataset: str, method: str, scope: str) -> Path:
    if scope == "full":
        return root / "results" / "metrics" / f"plan_d_{dataset}_{method}_metrics.csv"
    return root / "results" / "metrics" / f"plan_d_{dataset}_{method}_{scope}_metrics.csv"


def _metric_value_map(frame: pd.DataFrame) -> dict[str, str]:
    values: dict[str, str] = {}
    for _, row in frame.iterrows():
        metric = str(row.get("metric", ""))
        if metric and metric not in values:
            values[metric] = str(row.get("value", ""))
    return values


def _summary_notes(first: dict[str, str]) -> str:
    notes = first.get("notes", "")
    users = first.get("num_users_evaluated", "")
    if users and _to_float(users) < MAX_USERS:
        extra = (
            f"Evaluated {users} users rather than {MAX_USERS}; this can occur when a model requires valid "
            "non-empty sequence/history constraints."
        )
        return f"{notes} {extra}".strip()
    return notes


def _assign_rank_within_dataset(frame: pd.DataFrame, value_column: str, rank_column: str) -> None:
    sortable = frame.assign(_metric_value=pd.to_numeric(frame[value_column], errors="coerce"))
    if sortable["_metric_value"].isna().any():
        missing = sortable.loc[sortable["_metric_value"].isna(), ["dataset", "method"]].to_dict("records")
        raise ValueError(f"Cannot rank non-numeric {value_column} values for rows: {missing}")
    for _, dataset_rows in sortable.groupby("dataset", sort=True):
        ordered = dataset_rows.sort_values(["_metric_value", "method"], ascending=[False, True])
        for rank, index in enumerate(ordered.index, start=1):
            frame.at[index, rank_column] = str(rank)


def _assign_global_rank(frame: pd.DataFrame, value_column: str, rank_column: str) -> None:
    sortable = frame.assign(_metric_value=pd.to_numeric(frame[value_column], errors="coerce"))
    if sortable["_metric_value"].isna().any():
        missing = sortable.loc[sortable["_metric_value"].isna(), ["dataset", "method"]].to_dict("records")
        raise ValueError(f"Cannot rank non-numeric {value_column} values for rows: {missing}")
    ordered = sortable.sort_values(["_metric_value", "dataset", "method"], ascending=[False, True, True])
    for rank, index in enumerate(ordered.index, start=1):
        frame.at[index, rank_column] = str(rank)


def _comparison_row(
    *,
    dataset: str,
    method: str,
    metric: str,
    diagnostic_value: str,
    extended_value: str,
    diagnostic_first: dict[str, str],
    extended_first: dict[str, str],
    note: str,
    absolute_delta: str = "",
    relative_delta: str = "",
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "method": method,
        "metric": metric,
        "diagnostic_value": diagnostic_value,
        "extended_value": extended_value,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "diagnostic_num_users_evaluated": diagnostic_first.get("num_users_evaluated", ""),
        "extended_num_users_evaluated": extended_first.get("num_users_evaluated", ""),
        "diagnostic_num_train_interactions": diagnostic_first.get("num_train_interactions", ""),
        "extended_num_train_interactions": extended_first.get("num_train_interactions", ""),
        "diagnostic_epochs_completed": diagnostic_first.get("epochs_completed", ""),
        "extended_epochs_completed": extended_first.get("epochs_completed", ""),
        "notes": note,
    }


def _compute_deltas(diagnostic_value: str, extended_value: str) -> tuple[str, str]:
    try:
        diagnostic = Decimal(diagnostic_value)
        extended = Decimal(extended_value)
    except (InvalidOperation, ValueError):
        return "", ""
    absolute_delta = extended - diagnostic
    if diagnostic == 0:
        return _decimal_to_string(absolute_delta), ""
    with localcontext() as context:
        context.prec = 28
        relative_delta = absolute_delta / diagnostic
    return _decimal_to_string(absolute_delta), _decimal_to_string(relative_delta)


def _decimal_to_string(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _bool_string(value: bool) -> str:
    return "true" if value else "false"


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    numeric_columns: set[str] | None = None,
) -> str:
    numeric_columns = numeric_columns or set()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = str(row.get(column, ""))
            if column in numeric_columns and value != "":
                value = _format_float(value)
            cells.append(value.replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _display_dataset(dataset: object) -> str:
    names = {
        "ml1m": "ML-1M",
        "amazon_beauty": "Amazon Beauty",
        "amazon_sports": "Amazon Sports",
        "lastfm": "LastFM",
    }
    return names.get(str(dataset), str(dataset))


def _display_method(method: object) -> str:
    names = {
        "bprmf": "BPRMF",
        "lightgcn": "LightGCN",
        "sasrec": "SASRec",
        "gru4rec": "GRU4Rec",
    }
    return names.get(str(method), str(method))


def _rank_sort_key(series: pd.Series) -> pd.Series:
    rank_columns = {"dataset_method_hr_rank", "dataset_method_ndcg_rank", "global_hr_rank", "global_ndcg_rank"}
    if series.name in rank_columns:
        return pd.to_numeric(series, errors="coerce")
    return series


def _sort_summary_for_report(summary: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {dataset: index for index, dataset in enumerate(UTILITY_DATASETS)}
    return (
        summary.assign(_dataset_order=summary["dataset"].map(dataset_order))
        .sort_values(["_dataset_order", "dataset_method_hr_rank", "method"], key=_rank_sort_key)
        .drop(columns=["_dataset_order"])
    )


def _ranking_text(summary: pd.DataFrame, dataset: str, rank_column: str) -> str:
    rows = summary.loc[summary["dataset"] == dataset].sort_values([rank_column, "method"], key=_rank_sort_key)
    return " > ".join(_display_method(method) for method in rows["method"].tolist())


def _global_top_text(rows: pd.DataFrame) -> str:
    return " > ".join(
        f"{_display_dataset(row['dataset'])} {_display_method(row['method'])}"
        for _, row in rows.sort_values(["global_hr_rank", "dataset", "method"], key=_rank_sort_key).iterrows()
    )


def _best_method_statement(summary: pd.DataFrame, dataset: str) -> str:
    rows = summary.loc[summary["dataset"] == dataset]
    hr_best = rows.sort_values(["dataset_method_hr_rank", "method"], key=_rank_sort_key).iloc[0]
    ndcg_best = rows.sort_values(["dataset_method_ndcg_rank", "method"], key=_rank_sort_key).iloc[0]
    if hr_best["method"] == ndcg_best["method"]:
        return (
            f"{_display_method(hr_best['method'])} is strongest on {_display_dataset(dataset)} extended_subset "
            f"by HR@5 and NDCG@5."
        )
    return (
        f"{_display_dataset(dataset)} has {_display_method(hr_best['method'])} as best HR@5 and "
        f"{_display_method(ndcg_best['method'])} as best NDCG@5."
    )


def _lightgcn_vs_bprmf_statement(summary: pd.DataFrame) -> str:
    wins = []
    for dataset in ("amazon_beauty", "amazon_sports"):
        rows = summary.loc[summary["dataset"] == dataset].set_index("method")
        wins.append(
            _to_float(rows.at["lightgcn", "hr_at_5"]) > _to_float(rows.at["bprmf", "hr_at_5"])
            and _to_float(rows.at["lightgcn", "ndcg_at_5"]) > _to_float(rows.at["bprmf", "ndcg_at_5"])
        )
    if all(wins):
        return "LightGCN consistently beats BPRMF on both Amazon datasets by HR@5 and NDCG@5."
    return "LightGCN and BPRMF ordering is mixed across the Amazon datasets."


def _sasrec_amazon_statement(summary: pd.DataFrame) -> str:
    amazon_rows = summary.loc[summary["dataset"].isin(["amazon_beauty", "amazon_sports"])]
    sasrec_ranks = pd.to_numeric(
        amazon_rows.loc[amazon_rows["method"] == "sasrec", "dataset_method_hr_rank"], errors="coerce"
    )
    if not sasrec_ranks.empty and (sasrec_ranks >= 3).all():
        return "SASRec performs poorly on the Amazon extended subsets despite strong ML-1M performance."
    return "SASRec Amazon performance is mixed relative to the other neural baselines."


def _amazon_sparsity_statement(summary: pd.DataFrame) -> str:
    amazon_rows = summary.loc[summary["dataset"].isin(["amazon_beauty", "amazon_sports"])]
    interactions = sorted({int(float(value)) for value in amazon_rows["num_train_interactions"] if value != ""})
    values = ", ".join(str(value) for value in interactions)
    return (
        f"The reconstructed Amazon extended subsets are sparse, with {values} train interactions for 1000 users, "
        "which likely affects sequential transformer performance."
    )


def _user_constraint_notes(summary: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    for _, row in summary.iterrows():
        users = _to_float(row.get("num_users_evaluated", ""))
        if users < MAX_USERS:
            notes.append(
                f"{_display_dataset(row['dataset'])} {_display_method(row['method'])} evaluated "
                f"{int(users)} users rather than {MAX_USERS} because of valid sequence/history constraints."
            )
    return notes


def _missing_comparison_notes(coverage: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    rows = coverage.loc[
        coverage["dataset"].isin(UTILITY_DATASETS)
        & coverage["extended_subset_available"].eq("true")
        & coverage["diagnostic_subset_available"].ne("true")
    ]
    for _, row in rows.iterrows():
        notes.append(f"{_display_dataset(row['dataset'])} {_display_method(row['method'])}")
    return notes

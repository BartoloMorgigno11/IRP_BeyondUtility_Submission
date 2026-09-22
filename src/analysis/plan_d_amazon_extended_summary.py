"""Summarize existing Plan D Amazon BPRMF/LightGCN extended metrics.

This module reads already-generated Plan D metric CSVs and writes derived
comparison tables plus a markdown report. It does not train models, score
candidates, modify raw recommendation outputs, download data, or call APIs.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_METHODS = (
    ("amazon_beauty", "bprmf"),
    ("amazon_beauty", "lightgcn"),
    ("amazon_sports", "bprmf"),
    ("amazon_sports", "lightgcn"),
)
SUMMARY_METRICS = ("hit_rate", "ndcg", "final_train_loss", "best_train_loss")

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
        "notes": first.get("notes", ""),
    }


def build_extended_summary(
    *,
    project_root: str | Path | None = None,
    dataset_methods: tuple[tuple[str, str], ...] = DATASET_METHODS,
) -> pd.DataFrame:
    """Build one-row-per-dataset-method summary for existing extended metrics."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    rows = [
        extract_dataset_method_summary(_metric_path(root, dataset, method, "extended_subset"), project_root=root)
        for dataset, method in dataset_methods
    ]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    _assign_rank_within_dataset(summary, "hr_at_5", "dataset_method_hr_rank")
    _assign_rank_within_dataset(summary, "ndcg_at_5", "dataset_method_ndcg_rank")
    return summary[SUMMARY_COLUMNS]


def build_diagnostic_vs_extended(
    *,
    project_root: str | Path | None = None,
    dataset_methods: tuple[tuple[str, str], ...] = DATASET_METHODS,
) -> pd.DataFrame:
    """Compare diagnostic-subset and extended-subset values when both exist."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    rows: list[dict[str, str]] = []
    for dataset, method in dataset_methods:
        extended_path = _metric_path(root, dataset, method, "extended_subset")
        extended = load_metric_file(extended_path)
        extended_values = _metric_value_map(extended)
        extended_first = extended.iloc[0].to_dict()
        diagnostic_path = _metric_path(root, dataset, method, "diagnostic_subset")

        if not diagnostic_path.is_file():
            for metric in SUMMARY_METRICS:
                rows.append(
                    _comparison_row(
                        dataset=dataset,
                        method=method,
                        metric=metric,
                        diagnostic_value="",
                        extended_value=extended_values.get(metric, ""),
                        diagnostic_first={},
                        extended_first=extended_first,
                        note=f"Diagnostic metric file missing: {_relative_path(diagnostic_path, root)}",
                    )
                )
            continue

        diagnostic = load_metric_file(diagnostic_path)
        diagnostic_values = _metric_value_map(diagnostic)
        diagnostic_first = diagnostic.iloc[0].to_dict()
        for metric in SUMMARY_METRICS:
            if metric not in diagnostic_values or metric not in extended_values:
                rows.append(
                    _comparison_row(
                        dataset=dataset,
                        method=method,
                        metric=metric,
                        diagnostic_value=diagnostic_values.get(metric, ""),
                        extended_value=extended_values.get(metric, ""),
                        diagnostic_first=diagnostic_first,
                        extended_first=extended_first,
                        note="Metric missing from one scope; no delta computed.",
                    )
                )
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


def write_markdown_report(summary: pd.DataFrame, comparison: pd.DataFrame, path: str | Path) -> Path:
    """Write a compact markdown report from generated Amazon summary tables."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    beauty = summary.loc[summary["dataset"] == "amazon_beauty"].sort_values(
        ["dataset_method_hr_rank", "method"], key=_rank_sort_key
    )
    sports = summary.loc[summary["dataset"] == "amazon_sports"].sort_values(
        ["dataset_method_hr_rank", "method"], key=_rank_sort_key
    )
    best_lines = _best_method_lines(summary)
    sparsity_text = _sparsity_text(summary)
    runtime_text = _runtime_text(summary)
    lightgcn_note = _lightgcn_note(summary)

    lines = [
        "# Plan D Amazon Extended BPRMF/LightGCN Summary",
        "",
        "## Purpose",
        "",
        "This report aggregates existing Plan D Amazon Beauty and Amazon Sports extended_subset metric CSVs for BPRMF",
        "and LightGCN. It is a reconstructed Plan D extended_subset evaluation, not an exact paper reproduction, because",
        "the official preprocessed Amazon artifacts were unavailable. The CSV summary preserves exact values read from",
        "the metric files; the markdown tables format values to 4 decimals for readability.",
        "",
        "## Common Setup",
        "",
        "- Datasets: amazon_beauty and amazon_sports",
        "- Scope: extended_subset",
        "- max_users: 1000",
        "- max_train_interactions: 100000 cap, with actual interactions lower because these Amazon subsets are sparse",
        "- epochs: 5",
        "- Evaluation: candidate-set ranking at k=5",
        "- Device: CUDA where available",
        "",
        "## Amazon Beauty Results",
        "",
        _markdown_table(
            beauty,
            [
                "method",
                "hr_at_5",
                "ndcg_at_5",
                "final_train_loss",
                "best_train_loss",
                "num_users_evaluated",
                "num_train_interactions",
                "runtime_seconds",
                "device",
                "epochs_completed",
                "dataset_method_hr_rank",
                "dataset_method_ndcg_rank",
            ],
            numeric_columns={"hr_at_5", "ndcg_at_5", "final_train_loss", "best_train_loss", "runtime_seconds"},
        ),
        "",
        "## Amazon Sports Results",
        "",
        _markdown_table(
            sports,
            [
                "method",
                "hr_at_5",
                "ndcg_at_5",
                "final_train_loss",
                "best_train_loss",
                "num_users_evaluated",
                "num_train_interactions",
                "runtime_seconds",
                "device",
                "epochs_completed",
                "dataset_method_hr_rank",
                "dataset_method_ndcg_rank",
            ],
            numeric_columns={"hr_at_5", "ndcg_at_5", "final_train_loss", "best_train_loss", "runtime_seconds"},
        ),
        "",
        "## Diagnostic vs Extended",
        "",
        _markdown_table(
            comparison,
            [
                "dataset",
                "method",
                "metric",
                "diagnostic_value",
                "extended_value",
                "absolute_delta",
                "relative_delta",
                "diagnostic_num_users_evaluated",
                "extended_num_users_evaluated",
            ],
            numeric_columns={"diagnostic_value", "extended_value", "absolute_delta", "relative_delta"},
        ),
        "",
        "## Interpretation",
        "",
    ]
    lines.extend(f"- {line}" for line in best_lines)
    lines.extend(
        [
            f"- {lightgcn_note}",
            f"- {sparsity_text}",
            f"- {runtime_text}",
            "- Runtime comparisons are descriptive only because the models use different graph or embedding workloads.",
            "",
            "## Limitations",
            "",
            "- The Amazon datasets are reconstructed, not official preprocessed paper artifacts.",
            "- This is an extended_subset evaluation, not a full reproduction.",
            "- Only BPRMF and LightGCN are included here; SASRec and GRU4Rec Amazon extended runs remain future work.",
            "- Hyperparameters are controlled and safe rather than exhaustively tuned.",
            "",
            "## Recommended Next Steps",
            "",
            "1. Run SASRec Amazon Beauty extended_subset.",
            "2. Run SASRec Amazon Sports extended_subset.",
            "3. Run GRU4Rec Amazon Beauty and Amazon Sports extended_subset if time permits.",
            "4. Create a global extended summary across all datasets and methods.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_outputs(*, project_root: str | Path | None = None) -> dict[str, Path]:
    """Write Plan D Amazon extended BPRMF/LightGCN summary tables and report."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    tables_dir = root / "results" / "tables"
    report_dir = root / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = build_extended_summary(project_root=root)
    comparison = build_diagnostic_vs_extended(project_root=root)

    summary_path = tables_dir / "plan_d_amazon_bprmf_lightgcn_extended_summary.csv"
    comparison_path = tables_dir / "plan_d_amazon_bprmf_lightgcn_diagnostic_vs_extended.csv"
    report_path = report_dir / "plan_d_amazon_bprmf_lightgcn_extended_summary.md"

    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    write_markdown_report(summary, comparison, report_path)
    return {
        "summary": summary_path,
        "comparison": comparison_path,
        "report": report_path,
    }


def _metric_path(root: Path, dataset: str, method: str, scope: str) -> Path:
    return root / "results" / "metrics" / f"plan_d_{dataset}_{method}_{scope}_metrics.csv"


def _metric_value_map(frame: pd.DataFrame) -> dict[str, str]:
    values: dict[str, str] = {}
    for _, row in frame.iterrows():
        metric = str(row.get("metric", ""))
        if metric and metric not in values:
            values[metric] = str(row.get("value", ""))
    return values


def _assign_rank_within_dataset(frame: pd.DataFrame, value_column: str, rank_column: str) -> None:
    sortable = frame.assign(_metric_value=pd.to_numeric(frame[value_column], errors="coerce"))
    if sortable["_metric_value"].isna().any():
        missing = sortable.loc[sortable["_metric_value"].isna(), ["dataset", "method"]].to_dict("records")
        raise ValueError(f"Cannot rank non-numeric {value_column} values for rows: {missing}")
    for dataset, dataset_rows in sortable.groupby("dataset", sort=True):
        ordered = dataset_rows.sort_values(["_metric_value", "method"], ascending=[False, True])
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


def _display_dataset(dataset: object) -> str:
    names = {
        "amazon_beauty": "Amazon Beauty",
        "amazon_sports": "Amazon Sports",
    }
    return names.get(str(dataset), str(dataset))


def _display_method(method: object) -> str:
    names = {
        "bprmf": "BPRMF",
        "lightgcn": "LightGCN",
    }
    return names.get(str(method), str(method))


def _rank_sort_key(series: pd.Series) -> pd.Series:
    if series.name in {"dataset_method_hr_rank", "dataset_method_ndcg_rank"}:
        return pd.to_numeric(series, errors="coerce")
    return series


def _best_method_lines(summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for dataset, rows in summary.groupby("dataset", sort=True):
        hr_best = rows.sort_values(["dataset_method_hr_rank", "method"], key=_rank_sort_key).iloc[0]
        ndcg_best = rows.sort_values(["dataset_method_ndcg_rank", "method"], key=_rank_sort_key).iloc[0]
        lines.append(
            f"{_display_dataset(dataset)} best HR@5: {_display_method(hr_best['method'])} "
            f"({_format_float(hr_best['hr_at_5'])})."
        )
        lines.append(
            f"{_display_dataset(dataset)} best NDCG@5: {_display_method(ndcg_best['method'])} "
            f"({_format_float(ndcg_best['ndcg_at_5'])})."
        )
    return lines


def _lightgcn_note(summary: pd.DataFrame) -> str:
    winners = []
    for dataset, rows in summary.groupby("dataset", sort=True):
        hr_best = rows.sort_values(["dataset_method_hr_rank", "method"], key=_rank_sort_key).iloc[0]["method"]
        ndcg_best = rows.sort_values(["dataset_method_ndcg_rank", "method"], key=_rank_sort_key).iloc[0]["method"]
        winners.append(hr_best == "lightgcn" and ndcg_best == "lightgcn")
    if winners and all(winners):
        return "LightGCN outperforms BPRMF on both Amazon datasets by HR@5 and NDCG@5 in these source metrics."
    return "Method ordering differs by dataset or metric; see the per-dataset rank columns for the exact comparison."


def _sparsity_text(summary: pd.DataFrame) -> str:
    interactions = sorted({int(float(value)) for value in summary["num_train_interactions"].tolist() if value != ""})
    if not interactions:
        return "The Amazon extended subsets are sparse relative to the 100000 interaction cap."
    if len(interactions) == 1:
        return f"The Amazon extended subsets use {interactions[0]} train interactions, well below the 100000 cap."
    values = ", ".join(str(value) for value in interactions)
    return f"The Amazon extended subsets use {values} train interactions, well below the 100000 cap."


def _runtime_text(summary: pd.DataFrame) -> str:
    runtimes = summary.assign(_runtime=pd.to_numeric(summary["runtime_seconds"], errors="coerce"))
    fastest = runtimes.sort_values(["_runtime", "dataset", "method"]).iloc[0]
    slowest = runtimes.sort_values(["_runtime", "dataset", "method"]).iloc[-1]
    return (
        f"Runtime ranges from {_format_float(fastest['runtime_seconds'])} seconds "
        f"({_display_dataset(fastest['dataset'])} {_display_method(fastest['method'])}) to "
        f"{_format_float(slowest['runtime_seconds'])} seconds "
        f"({_display_dataset(slowest['dataset'])} {_display_method(slowest['method'])})."
    )

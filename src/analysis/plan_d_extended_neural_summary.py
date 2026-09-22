"""Summarize existing Plan D ML-1M extended neural baseline metrics.

This module reads already-generated Plan D metric CSVs and writes comparison
tables plus a markdown report. It does not train models, score candidates, or
modify raw recommendation outputs.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = "ml1m"
METHODS = ("bprmf", "lightgcn", "sasrec", "gru4rec")
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
    "hr_rank",
    "ndcg_rank",
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


def extract_method_summary(path: str | Path, *, project_root: str | Path | None = None) -> dict[str, str]:
    """Extract one extended-subset summary row from a Plan D metric CSV."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    metric_path = Path(path)
    frame = load_metric_file(metric_path)
    values = _metric_value_map(frame)
    missing_metrics = [metric for metric in SUMMARY_METRICS if metric not in values]
    if missing_metrics:
        raise ValueError(f"{metric_path} is missing metrics: {missing_metrics}")

    first = frame.iloc[0].to_dict()
    row = {
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
        "hr_rank": "",
        "ndcg_rank": "",
        "notes": first.get("notes", ""),
    }
    return row


def build_extended_summary(
    *,
    project_root: str | Path | None = None,
    dataset: str = DATASET,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    """Build one-row-per-method summary for existing extended-subset metrics."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    rows = [
        extract_method_summary(_metric_path(root, dataset, method, "extended_subset"), project_root=root)
        for method in methods
    ]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    _assign_descending_rank(summary, "hr_at_5", "hr_rank")
    _assign_descending_rank(summary, "ndcg_at_5", "ndcg_rank")
    return summary[SUMMARY_COLUMNS]


def build_diagnostic_vs_extended(
    *,
    project_root: str | Path | None = None,
    dataset: str = DATASET,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    """Compare diagnostic-subset and extended-subset values when both exist."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    rows: list[dict[str, str]] = []
    for method in methods:
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
    """Write a compact markdown report from generated summary tables."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best_hr = summary.sort_values(["hr_rank", "method"], key=_rank_sort_key).iloc[0]
    best_ndcg = summary.sort_values(["ndcg_rank", "method"], key=_rank_sort_key).iloc[0]
    fastest = summary.assign(_runtime=pd.to_numeric(summary["runtime_seconds"], errors="coerce")).sort_values(
        ["_runtime", "method"]
    ).iloc[0]

    user_counts = pd.to_numeric(summary["num_users_evaluated"], errors="coerce")
    max_users = int(user_counts.max()) if user_counts.notna().any() else 0
    hr_ranking = " > ".join(
        _display_method(method)
        for method in summary.sort_values(["hr_rank", "method"], key=_rank_sort_key)["method"].tolist()
    )
    ndcg_ranking = " > ".join(
        _display_method(method)
        for method in summary.sort_values(["ndcg_rank", "method"], key=_rank_sort_key)["method"].tolist()
    )
    sequence_notes: list[str] = []
    for method in ("sasrec", "gru4rec"):
        method_rows = summary.loc[summary["method"] == method]
        if method_rows.empty:
            continue
        method_users = int(float(method_rows.iloc[0]["num_users_evaluated"]))
        if max_users and method_users < max_users:
            sequence_notes.append(
                f"{_display_method(method)} evaluated {method_users} users rather than {max_users}, because the sequential "
                "evaluation requires usable non-empty training histories and candidate rows."
            )

    lines = [
        "# Plan D ML-1M Extended Neural Baselines Summary",
        "",
        "## Purpose",
        "",
        "This report aggregates existing Plan D ML-1M extended_subset metric CSVs for BPRMF, LightGCN, SASRec,",
        "and GRU4Rec.",
        "It is not an exact reproduction of the paper because it uses reconstructed Plan C data rather than official",
        "preprocessed artifacts. The CSV summary preserves exact values read from the metric files; the markdown tables",
        "format values to 4 decimals for readability.",
        "",
        "## Common Setup",
        "",
        "- Dataset: ml1m",
        "- Scope: extended_subset",
        "- max_users: 1000",
        "- max_train_interactions: 100000",
        "- epochs: 5",
        "- Evaluation: candidate-set ranking at k=5",
        "- Device: CUDA where available",
        "",
        "## Extended Subset Results",
        "",
        _markdown_table(
            summary.sort_values(["hr_rank", "ndcg_rank", "method"]),
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
                "hr_rank",
                "ndcg_rank",
            ],
            numeric_columns={"hr_at_5", "ndcg_at_5", "final_train_loss", "best_train_loss", "runtime_seconds"},
        ),
        "",
        "## Diagnostic vs Extended",
        "",
        _markdown_table(
            comparison,
            [
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
        f"- Best HR@5: {_display_method(best_hr['method'])} ({_format_float(best_hr['hr_at_5'])}).",
        f"- Best NDCG@5: {_display_method(best_ndcg['method'])} ({_format_float(best_ndcg['ndcg_at_5'])}).",
        f"- HR@5 ranking: {hr_ranking}.",
        f"- NDCG@5 ranking: {ndcg_ranking}.",
        f"- Fastest runtime: {fastest['method']} ({_format_float(fastest['runtime_seconds'])} seconds).",
    ]
    lines.extend(f"- {note}" for note in sequence_notes)
    lines.extend(
        [
            "- Runtime comparisons are descriptive only because the architectures perform different amounts of work per epoch.",
            "",
            "## Limitations",
            "",
            "- The dataset is reconstructed, not the official preprocessed paper artifact.",
            "- This is an extended_subset run, not a full paper reproduction.",
            "- Hyperparameters are controlled and safe rather than exhaustively tuned.",
            "- SASRec required deterministic CUDA setup through CUBLAS workspace and attention-kernel configuration.",
            "",
            "## Recommended Next Steps",
            "",
            "1. Run Amazon Beauty and Amazon Sports extended_subset experiments for BPRMF, LightGCN, SASRec, and GRU4Rec.",
            "2. Attempt ML-1M full runs one model at a time if extended results remain stable.",
            "3. Decide the LLM backend before starting LLM ranking or reranking work.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_outputs(*, project_root: str | Path | None = None) -> dict[str, Path]:
    """Write Plan D ML-1M extended neural summary tables and report."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    tables_dir = root / "results" / "tables"
    report_dir = root / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = build_extended_summary(project_root=root)
    comparison = build_diagnostic_vs_extended(project_root=root)

    summary_path = tables_dir / "plan_d_ml1m_extended_neural_baselines_summary.csv"
    comparison_path = tables_dir / "plan_d_ml1m_neural_diagnostic_vs_extended.csv"
    report_path = report_dir / "plan_d_ml1m_extended_neural_baselines.md"

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


def _assign_descending_rank(frame: pd.DataFrame, value_column: str, rank_column: str) -> None:
    sortable = frame.assign(_metric_value=pd.to_numeric(frame[value_column], errors="coerce"))
    if sortable["_metric_value"].isna().any():
        missing = sortable.loc[sortable["_metric_value"].isna(), "method"].tolist()
        raise ValueError(f"Cannot rank non-numeric {value_column} values for methods: {missing}")
    ordered = sortable.sort_values(["_metric_value", "method"], ascending=[False, True])
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


def _display_method(method: object) -> str:
    names = {
        "bprmf": "BPRMF",
        "lightgcn": "LightGCN",
        "sasrec": "SASRec",
        "gru4rec": "GRU4Rec",
    }
    return names.get(str(method), str(method))


def _rank_sort_key(series: pd.Series) -> pd.Series:
    if series.name in {"hr_rank", "ndcg_rank"}:
        return pd.to_numeric(series, errors="coerce")
    return series

"""Run the Catalog Coverage@K extension when catalog inputs are ready.

Run from the repository root:

    python src/experiments/run_extension_coverage.py --config configs/beauty.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import read_table
from src.data.schemas import validate_recommendations
from src.evaluation.coverage import coverage_by_model
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("run_extension_coverage")

METRIC_COLUMNS = [
    "run_id",
    "dataset",
    "task",
    "model_name",
    "metric",
    "k",
    "value",
    "config_path",
    "timestamp",
    "source_file",
    "metric_context",
    "catalog_size",
    "unique_recommended_items",
]

SUMMARY_COLUMNS = [
    "dataset",
    "status",
    "check",
    "path",
    "message",
    "row_count",
    "timestamp",
]

RECOMMENDATION_SOURCES = (
    ("llm", "{dataset}_llm_outputs_normalized.csv"),
    ("traditional_baseline", "{dataset}_baseline_outputs_normalized.csv"),
    ("mostpop", "{dataset}_mostpop_recommendations.csv"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Catalog Coverage@5 when catalog and normalized recommendations are ready."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    return parser.parse_args(argv)


def run_coverage_extension(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, bool]:
    """Run Catalog Coverage@5 or write readiness diagnostics when blocked."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    task = str(config["experiment"]["task"])
    k = 5
    timestamp = datetime.now(timezone.utc).isoformat()
    config_label = str(config_path)

    metrics_path = _configured_output_dir(config, root, "metrics_dir", "results/metrics") / f"{dataset}_coverage_metrics.csv"
    summary_path = _configured_output_dir(config, root, "tables_dir", "results/tables") / f"{dataset}_coverage_summary.csv"
    analysis_path = resolve_project_path(
        config.get("output", {}).get("analysis_path", "report/results_analysis.md"),
        root,
    )
    if analysis_path is None:
        analysis_path = root / "report/results_analysis.md"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    catalog, catalog_ready = _load_catalog(
        config,
        root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    recommendations = _load_recommendation_sources(
        config,
        root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )

    if catalog_ready and catalog is not None:
        for source_name, frame in recommendations.items():
            metric_rows.extend(
                _coverage_metric_rows(
                    frame,
                    catalog,
                    dataset=dataset,
                    task=task,
                    source_name=source_name,
                    k=k,
                    config_path=config_label,
                    timestamp=timestamp,
                    summary_rows=summary_rows,
                )
            )

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    metrics_written = False
    if metrics.empty:
        _remove_empty_metrics_file(metrics_path)
    else:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
        metrics_written = True

    summary.to_csv(summary_path, index=False)
    _write_markdown_report(
        analysis_path,
        dataset=dataset,
        config_path=config_label,
        project_root=root,
        metrics_path=metrics_path,
        metrics_written=metrics_written,
        summary_path=summary_path,
        metrics=metrics,
        summary=summary,
        blocked=metrics.empty,
    )

    if metrics_written:
        LOGGER.info("Saved coverage metrics to %s", _display_path(metrics_path, root))
    else:
        LOGGER.info("No coverage metrics were written; see readiness summary.")
    LOGGER.info("Saved coverage summary to %s", _display_path(summary_path, root))
    LOGGER.info("Saved Markdown analysis to %s", _display_path(analysis_path, root))
    return metrics, summary, metrics_path, summary_path, analysis_path, metrics.empty


def _load_catalog(
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame | None, bool]:
    path_value = config.get("paths", {}).get("catalog_path") or config.get("dataset", {}).get("catalog_path")
    if path_value is None:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="not_configured",
                check="catalog_path",
                path=None,
                message="Catalog Coverage@5 is blocked because catalog_path is not configured.",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return None, False

    path = resolve_project_path(path_value, root)
    if path is None or not path.is_file():
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="missing",
                check="catalog_path",
                path=str(path_value),
                message=f"Catalog Coverage@5 is blocked because catalog file does not exist: {path_value}",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return None, False

    catalog = read_table(path)
    catalog.columns = [str(column).strip().lower() for column in catalog.columns]
    if "item_id" not in catalog.columns:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="catalog_path",
                path=str(path_value),
                message="Catalog Coverage@5 requires a catalog item_id column; text is not used to infer item IDs.",
                row_count=len(catalog),
                timestamp=timestamp,
            )
        )
        return None, False

    usable = catalog.loc[~_is_empty_series(catalog["item_id"])].copy()
    if usable.empty:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="catalog_path",
                path=str(path_value),
                message="Catalog Coverage@5 requires at least one non-empty catalog item_id.",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return None, False

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            status="ready",
            check="catalog_path",
            path=str(path_value),
            message="Loaded catalog for Catalog Coverage@5.",
            row_count=len(usable),
            timestamp=timestamp,
        )
    )
    return usable, True


def _load_recommendation_sources(
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    raw_outputs_dir = _configured_output_dir(config, root, "raw_outputs_dir", "results/raw_outputs")
    loaded: dict[str, pd.DataFrame] = {}
    for source_name, template in RECOMMENDATION_SOURCES:
        path = raw_outputs_dir / template.format(dataset=dataset)
        if not path.is_file():
            summary_rows.append(
                _summary_row(
                    dataset=dataset,
                    status="missing",
                    check=f"{source_name}_recommendations",
                    path=str(path.relative_to(root)),
                    message=f"Normalized {source_name} recommendation file is missing.",
                    row_count=0,
                    timestamp=timestamp,
                )
            )
            continue

        try:
            frame, skipped_rows = _read_usable_recommendations(path)
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            summary_rows.append(
                _summary_row(
                    dataset=dataset,
                    status="blocked",
                    check=f"{source_name}_recommendations",
                    path=str(path.relative_to(root)),
                    message=f"Normalized {source_name} recommendation file could not be loaded or validated: {exc}",
                    row_count=0,
                    timestamp=timestamp,
                )
            )
            continue

        status = "ready_with_warnings" if skipped_rows else "ready"
        message = f"Loaded normalized {source_name} recommendations."
        if skipped_rows:
            message = (
                f"{message} Skipped {skipped_rows} row(s) without a non-empty "
                "recommended_item_id or recommended_item_text."
            )
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status=status,
                check=f"{source_name}_recommendations",
                path=str(path.relative_to(root)),
                message=message,
                row_count=len(frame),
                timestamp=timestamp,
            )
        )
        loaded[source_name] = frame
    return loaded


def _coverage_metric_rows(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    dataset: str,
    task: str,
    source_name: str,
    k: int,
    config_path: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if "recommended_item_id" not in recommendations.columns or _is_empty_series(recommendations["recommended_item_id"]).all():
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check=f"{source_name}_coverage",
                path=None,
                message="Catalog Coverage@5 was not computed for this source because no non-empty recommended_item_id values are available.",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return []

    rows: list[dict[str, Any]] = []
    coverage_rows = coverage_by_model(
        recommendations,
        catalog,
        k,
        recommendation_col="recommended_item_id",
        catalog_col="item_id",
    )
    for _, row in coverage_rows.iterrows():
        rows.append(
            {
                "run_id": row["run_id"],
                "dataset": dataset,
                "task": task,
                "model_name": row["model_name"],
                "metric": "catalog_coverage",
                "k": k,
                "value": float(row["value"]),
                "config_path": config_path,
                "timestamp": timestamp,
                "source_file": _first_non_empty(recommendations.get("source_file")),
                "metric_context": source_name,
                "catalog_size": int(row["catalog_size"]),
                "unique_recommended_items": int(row["unique_recommended_items"]),
            }
        )
    return rows


def _read_usable_recommendations(path: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(path)
    normalized = raw.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    recommendation_columns = [
        column
        for column in ("recommended_item_id", "recommended_item_text")
        if column in normalized.columns
    ]
    if not recommendation_columns:
        raise ValueError("recommendations dataframe must contain recommended_item_id or recommended_item_text")

    has_recommendation = pd.Series(False, index=normalized.index)
    for column in recommendation_columns:
        has_recommendation = has_recommendation | ~_is_empty_series(normalized[column])

    skipped_rows = int((~has_recommendation).sum())
    if not bool(has_recommendation.any()):
        raise ValueError("recommendations dataframe has no non-empty recommendation values")

    return validate_recommendations(normalized.loc[has_recommendation].copy()), skipped_rows


def _configured_output_dir(
    config: dict[str, Any],
    root: Path,
    key: str,
    fallback: str,
) -> Path:
    resolved = resolve_project_path(config.get("output", {}).get(key, fallback), root)
    return resolved if resolved is not None else root / fallback


def _summary_row(
    *,
    dataset: str,
    status: str,
    check: str,
    path: str | None,
    message: str,
    row_count: int,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "status": status,
        "check": check,
        "path": path,
        "message": message,
        "row_count": int(row_count),
        "timestamp": timestamp,
    }


def _write_markdown_report(
    path: Path,
    *,
    dataset: str,
    config_path: str,
    project_root: Path,
    metrics_path: Path,
    metrics_written: bool,
    summary_path: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    blocked: bool,
) -> None:
    status = "blocked" if blocked else "complete"
    lines = [
        "# Results Analysis",
        "",
        "## Catalog Coverage@K Extension",
        "",
        "Catalog Coverage@K measures how much of the item catalog appears at least once in top-K recommendation lists. It can change the interpretation of utility metrics because two systems with similar HR@5 or NDCG@5 may expose very different portions of the catalog.",
        "",
        f"- Dataset: `{dataset}`",
        f"- Config: `{config_path}`",
        f"- Status: `{status}`",
        f"- Coverage metrics CSV: `{_display_path(metrics_path, project_root) if metrics_written else 'not written'}`",
        f"- Coverage summary CSV: `{_display_path(summary_path, project_root)}`",
        "",
        "Real Coverage@5 is computed only when `catalog_path` points to a readable catalog with `item_id` values and normalized recommendations contain comparable `recommended_item_id` values. With current Beauty and Sports configs, real Coverage@5 is blocked when `catalog_path` is not configured.",
        "",
        "## Readiness",
        "",
    ]
    lines.extend(_markdown_table(summary))
    lines.extend(["", "## Coverage Metrics", ""])
    if metrics.empty:
        lines.append("No real Coverage@5 values were computed.")
    else:
        lines.extend(_markdown_table(metrics[["model_name", "run_id", "metric", "k", "value", "catalog_size", "unique_recommended_items"]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) if not pd.isna(row[column]) else "" for column in columns]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return lines


def _remove_empty_metrics_file(metrics_path: Path) -> None:
    if not metrics_path.is_file():
        return
    try:
        existing = pd.read_csv(metrics_path)
    except pd.errors.EmptyDataError:
        metrics_path.unlink()
        return
    if existing.empty:
        metrics_path.unlink()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _first_non_empty(series: pd.Series | None) -> Any:
    if series is None:
        return pd.NA
    for value in series:
        if not pd.isna(value) and str(value).strip():
            return value
    return pd.NA


def _is_empty_series(series: pd.Series) -> pd.Series:
    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("")


def _ensure_api_calls_disabled(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls"):
        raise ConfigError("api.allow_api_calls must be false; this runner never calls APIs.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError(
            "experiment.allow_api_calls must be false; this runner never calls APIs."
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    try:
        root = find_project_root()
        metrics, summary, metrics_path, summary_path, analysis_path, blocked = run_coverage_extension(
            args.config,
            project_root=root,
        )
    except (ConfigError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(f"metrics={_display_path(metrics_path, root) if not metrics.empty else 'not_written'}")
    print(f"summary={_display_path(summary_path, root)}")
    print(f"analysis={_display_path(analysis_path, root)}")
    print(f"metric_rows={len(metrics)}")
    print(f"statuses={','.join(summary['status'].astype(str).tolist()) if not summary.empty else ''}")
    if blocked:
        LOGGER.info("Coverage metrics are blocked by missing prerequisites. See %s.", _display_path(summary_path, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())

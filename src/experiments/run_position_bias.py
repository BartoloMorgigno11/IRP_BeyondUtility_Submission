"""Run candidate position-bias analysis when required inputs are ready.

Run from the repository root:

    python src/experiments/run_position_bias.py --config configs/beauty.yaml
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
from src.data.schemas import validate_candidates, validate_recommendations
from src.evaluation.position_bias import position_bias_summary
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("run_position_bias")

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
    "candidate_position",
    "count",
    "matched_recommendations",
    "missing_recommendations",
    "duplicate_recommendations",
    "empty_recommendation_rows",
    "total_recommendation_rows",
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
        description="Compute candidate position-bias diagnostics when candidates and recommendations are ready."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    parser.add_argument(
        "--allow-api-calls",
        action="store_true",
        help="Accepted for CLI consistency; this runner still does not call APIs.",
    )
    return parser.parse_args(argv)


def run_position_bias(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, bool]:
    """Run position-bias metrics or write readiness diagnostics when blocked."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    task = str(config["experiment"]["task"])
    k = 5
    timestamp = datetime.now(timezone.utc).isoformat()
    config_label = str(config_path)

    metrics_path = _configured_output_dir(config, root, "metrics_dir", "results/metrics") / f"{dataset}_position_bias_metrics.csv"
    summary_path = _configured_output_dir(config, root, "tables_dir", "results/tables") / f"{dataset}_position_bias_summary.csv"
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

    candidates, candidates_ready = _load_candidates(
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

    if candidates_ready and candidates is not None:
        if recommendations:
            for source_name, frame in recommendations.items():
                metric_rows.extend(
                    _position_bias_metric_rows(
                        frame,
                        candidates,
                        dataset=dataset,
                        task=task,
                        source_name=source_name,
                        k=k,
                        config_path=config_label,
                        timestamp=timestamp,
                        summary_rows=summary_rows,
                    )
                )
        else:
            summary_rows.append(
                _summary_row(
                    dataset=dataset,
                    status="blocked",
                    check="recommendation_inputs",
                    path=None,
                    message="Position-bias analysis is blocked because no normalized recommendation outputs are available.",
                    row_count=0,
                    timestamp=timestamp,
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
        LOGGER.info("Saved position-bias metrics to %s", _display_path(metrics_path, root))
    else:
        LOGGER.info("No position-bias metrics were written; see readiness summary.")
    LOGGER.info("Saved position-bias summary to %s", _display_path(summary_path, root))
    LOGGER.info("Saved Markdown analysis to %s", _display_path(analysis_path, root))
    return metrics, summary, metrics_path, summary_path, analysis_path, metrics.empty


def _load_candidates(
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame | None, bool]:
    path_value = config.get("paths", {}).get("candidates_path") or config.get("dataset", {}).get("candidates_path")
    if path_value is None:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="not_configured",
                check="candidates_path",
                path=None,
                message="Position-bias analysis is blocked because candidates_path is not configured.",
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
                check="candidates_path",
                path=str(path_value),
                message=f"Position-bias analysis is blocked because candidate file does not exist: {path_value}",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return None, False

    try:
        candidates = validate_candidates(read_table(path))
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="candidates_path",
                path=str(path_value),
                message=f"Candidate file could not be loaded or validated: {exc}",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return None, False

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            status="ready",
            check="candidates_path",
            path=str(path_value),
            message="Loaded candidate set for position-bias analysis.",
            row_count=len(candidates),
            timestamp=timestamp,
        )
    )
    return candidates, True


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


def _position_bias_metric_rows(
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    dataset: str,
    task: str,
    source_name: str,
    k: int,
    config_path: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bias = position_bias_summary(recommendations, candidates, k=k)
    matched_total = int(bias["matched_recommendations"].max()) if not bias.empty else 0
    if bias.empty or matched_total == 0:
        missing_total = int(bias["missing_recommendations"].max()) if not bias.empty else 0
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check=f"{source_name}_position_bias",
                path=None,
                message=(
                    "Position-bias metrics were not computed for this source because "
                    f"no recommendations matched configured candidates. Missing recommendations: {missing_total}."
                ),
                row_count=0,
                timestamp=timestamp,
            )
        )
        return []

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            status="ready",
            check=f"{source_name}_position_bias",
            path=None,
            message="Computed selected candidate-position distribution for this source.",
            row_count=len(bias),
            timestamp=timestamp,
        )
    )

    rows: list[dict[str, Any]] = []
    for _, row in bias.iterrows():
        if pd.isna(row["candidate_position"]):
            continue
        rows.append(
            {
                "run_id": row["run_id"],
                "dataset": dataset,
                "task": task,
                "model_name": row["model_name"],
                "metric": row["metric"],
                "k": k,
                "value": float(row["fraction"]),
                "config_path": config_path,
                "timestamp": timestamp,
                "source_file": _first_non_empty(recommendations.get("source_file")),
                "metric_context": source_name,
                "candidate_position": int(row["candidate_position"]),
                "count": int(row["count"]),
                "matched_recommendations": int(row["matched_recommendations"]),
                "missing_recommendations": int(row["missing_recommendations"]),
                "duplicate_recommendations": int(row["duplicate_recommendations"]),
                "empty_recommendation_rows": int(row["empty_recommendation_rows"]),
                "total_recommendation_rows": int(row["total_recommendation_rows"]),
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
        "## Candidate Position Bias",
        "",
        "Candidate position bias measures whether recommended items are disproportionately selected from particular input candidate positions. This diagnostic is separate from HR@5 and NDCG@5 and requires real candidate lists plus normalized recommendation outputs.",
        "",
        f"- Dataset: `{dataset}`",
        f"- Config: `{config_path}`",
        f"- Status: `{status}`",
        f"- Position-bias metrics CSV: `{_display_path(metrics_path, project_root) if metrics_written else 'not written'}`",
        f"- Position-bias summary CSV: `{_display_path(summary_path, project_root)}`",
        "",
        "Real position-bias metrics are computed only when `candidates_path` points to a readable candidate file and normalized recommendations can be matched to candidate item IDs or titles. With the current Beauty and Sports configs, real position-bias metrics are blocked while `candidates_path` is not configured.",
        "",
        "## Readiness",
        "",
    ]
    lines.extend(_markdown_table(summary))
    lines.extend(["", "## Position-Bias Metrics", ""])
    if metrics.empty:
        lines.append("No real position-bias metric values were computed.")
    else:
        display_columns = [
            "model_name",
            "run_id",
            "metric",
            "k",
            "candidate_position",
            "value",
            "count",
            "matched_recommendations",
            "missing_recommendations",
        ]
        lines.extend(_markdown_table(metrics[display_columns]))
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
    if args.allow_api_calls:
        LOGGER.info("--allow-api-calls was provided, but this runner does not call APIs.")
    try:
        root = find_project_root()
        metrics, summary, metrics_path, summary_path, analysis_path, blocked = run_position_bias(
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
        LOGGER.info(
            "Position-bias metrics are blocked by missing prerequisites. See %s.",
            _display_path(summary_path, root),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

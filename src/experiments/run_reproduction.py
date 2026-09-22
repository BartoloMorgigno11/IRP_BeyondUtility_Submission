"""Run the local reproduction evaluation over normalized recommendation files.

Run from the repository root:

    python src/experiments/run_reproduction.py --config configs/beauty.yaml
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
from src.data.schemas import (
    validate_candidates,
    validate_ground_truth,
    validate_recommendations,
)
from src.evaluation.hallucination import (
    candidate_hallucination_rate,
    catalog_hallucination_rate,
)
from src.evaluation.utility_metrics import evaluate_utility_at_k
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("run_reproduction")

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
]

SUMMARY_COLUMNS = [
    "dataset",
    "status",
    "check",
    "path",
    "message",
    "upstream_phase",
    "row_count",
    "timestamp",
]

RECOMMENDATION_SOURCES = (
    ("llm", "{dataset}_llm_outputs_normalized.csv", "Prompt 09: run load_llm_outputs.py"),
    (
        "traditional_baseline",
        "{dataset}_baseline_outputs_normalized.csv",
        "Prompt 10: run load_baseline_outputs.py",
    ),
    ("mostpop", "{dataset}_mostpop_recommendations.csv", "MostPop phase: run mostpop.py after configuring interactions_path"),
)

HALLUCINATION_NUMERIC_METRICS = (
    "hallucination_rate",
    "strict_hallucination_rate",
    "total",
    "hallucinated",
    "item_id_matches",
    "exact_matches",
    "normalized_matches",
    "fuzzy_matches",
    "unmatched",
)


class ReproductionBlocked(RuntimeError):
    """Raised by the CLI when required reproduction inputs are missing."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute local HR@5, NDCG@5, and eligible LLM hallucination metrics."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    return parser.parse_args(argv)


def run_reproduction(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, bool]:
    """Run reproduction metrics when required normalized inputs are available."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    task = str(config["experiment"]["task"])
    k = 5
    timestamp = datetime.now(timezone.utc).isoformat()
    config_label = str(config_path)

    metrics_path = _output_path(config, root, "metrics_dir", "results/metrics", f"{dataset}_reproduction_metrics.csv")
    summary_path = _output_path(config, root, "tables_dir", "results/tables", f"{dataset}_reproduction_summary.csv")
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

    ground_truth, ground_truth_ready = _load_ground_truth(
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

    utility_blocked = not ground_truth_ready or not recommendations
    if not utility_blocked:
        for source_name, frame in recommendations.items():
            metric_rows.extend(
                _utility_metric_rows(
                    frame,
                    ground_truth,
                    dataset=dataset,
                    task=task,
                    k=k,
                    config_path=config_label,
                    timestamp=timestamp,
                    source_name=source_name,
                )
            )
    elif not recommendations:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="recommendation_inputs",
                path=None,
                message="No normalized LLM, traditional baseline, or MostPop recommendation files are available.",
                upstream_phase="Run Prompt 09 load_llm_outputs.py and/or Prompt 10 load_baseline_outputs.py; run MostPop only after interactions_path is configured.",
                row_count=0,
                timestamp=timestamp,
            )
        )

    llm_frame = recommendations.get("llm")
    if llm_frame is not None and not llm_frame.empty:
        metric_rows.extend(
            _llm_hallucination_metric_rows(
                llm_frame,
                config,
                root,
                dataset=dataset,
                task=task,
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
        LOGGER.info("Saved reproduction metrics to %s", _display_path(metrics_path, root))
    else:
        LOGGER.info("No reproduction metrics were written; see readiness summary.")
    LOGGER.info("Saved reproduction summary to %s", _display_path(summary_path, root))
    LOGGER.info("Saved Markdown analysis to %s", _display_path(analysis_path, root))
    return metrics, summary, metrics_path, summary_path, analysis_path, metrics.empty


def _load_ground_truth(
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, bool]:
    path_value = (
        config.get("paths", {}).get("ground_truth_path")
        or config.get("dataset", {}).get("ground_truth_path")
    )
    if path_value is None:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="ground_truth_path",
                path=None,
                message="Ground-truth path is not configured.",
                upstream_phase="Complete the dataset preparation/audit phase and set paths.ground_truth_path to a canonical CSV/TSV with user_id and ground_truth_item_id.",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return pd.DataFrame(), False

    path = resolve_project_path(path_value, root)
    if path is None or not path.is_file():
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="ground_truth_path",
                path=path_value,
                message=f"Configured ground-truth file does not exist: {path_value}",
                upstream_phase="Complete the dataset preparation/audit phase and regenerate or configure ground truth.",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return pd.DataFrame(), False

    ground_truth = validate_ground_truth(read_table(path))
    summary_rows.append(
        _summary_row(
            dataset=dataset,
            status="ready",
            check="ground_truth_path",
            path=path_value,
            message="Loaded canonical ground truth.",
            upstream_phase=None,
            row_count=len(ground_truth),
            timestamp=timestamp,
        )
    )
    return ground_truth, True


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
    for source_name, template, upstream_phase in RECOMMENDATION_SOURCES:
        path = raw_outputs_dir / template.format(dataset=dataset)
        if not path.is_file():
            summary_rows.append(
                _summary_row(
                    dataset=dataset,
                    status="missing",
                    check=f"{source_name}_recommendations",
                    path=str(path.relative_to(root)),
                    message=f"Normalized {source_name} recommendation file is missing.",
                    upstream_phase=upstream_phase,
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
                    upstream_phase=upstream_phase,
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
        loaded[source_name] = frame
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status=status,
                check=f"{source_name}_recommendations",
                path=str(path.relative_to(root)),
                message=message,
                upstream_phase=None,
                row_count=len(frame),
                timestamp=timestamp,
            )
        )
    return loaded


def _read_usable_recommendations(path: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(path)
    normalized = raw.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    duplicated = normalized.columns[normalized.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            "recommendations dataframe has duplicate column name(s) after "
            f"normalization: {', '.join(duplicated)}"
        )

    recommendation_columns = [
        column
        for column in ("recommended_item_id", "recommended_item_text")
        if column in normalized.columns
    ]
    if not recommendation_columns:
        raise ValueError(
            "recommendations dataframe must contain at least one of: "
            "recommended_item_id, recommended_item_text"
        )

    has_recommendation = pd.Series(False, index=normalized.index)
    for column in recommendation_columns:
        has_recommendation = has_recommendation | ~_is_empty_series(normalized[column])

    skipped_rows = int((~has_recommendation).sum())
    if not bool(has_recommendation.any()):
        raise ValueError(
            "recommendations dataframe has no rows with a non-empty "
            "recommended_item_id or recommended_item_text"
        )

    return validate_recommendations(normalized.loc[has_recommendation].copy()), skipped_rows


def _is_empty_series(series: pd.Series) -> pd.Series:
    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("")


def _utility_metric_rows(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    dataset: str,
    task: str,
    k: int,
    config_path: str,
    timestamp: str,
    source_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model_name, run_id), group in recommendations.groupby(["model_name", "run_id"], sort=False):
        values = evaluate_utility_at_k(group, ground_truth, k)
        source_file = _first_non_empty(group.get("source_file"))
        for metric_name in ("hit_rate", "ndcg"):
            rows.append(
                _metric_row(
                    run_id=run_id,
                    dataset=dataset,
                    task=task,
                    model_name=model_name,
                    metric=metric_name,
                    k=k,
                    value=float(values[metric_name]),
                    config_path=config_path,
                    timestamp=timestamp,
                    source_file=source_file,
                    metric_context=f"utility:{source_name}",
                )
            )
    return rows


def _llm_hallucination_metric_rows(
    llm_outputs: pd.DataFrame,
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    task: str,
    k: int,
    config_path: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates, candidates_path, candidates_status = _load_optional_table(
        config,
        root,
        "candidates_path",
        validator=validate_candidates,
    )
    catalog, catalog_path, catalog_status = _load_optional_table(config, root, "catalog_path")

    if candidates is not None:
        metric_source = "candidate_hallucination"
        calculator = lambda frame: candidate_hallucination_rate(frame, candidates, k=k)
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="ready",
                check="hallucination_reference",
                path=candidates_path,
                message="Using configured candidates for LLM hallucination metrics.",
                upstream_phase=None,
                row_count=len(candidates),
                timestamp=timestamp,
            )
        )
    elif catalog is not None:
        metric_source = "catalog_hallucination"
        calculator = lambda frame: catalog_hallucination_rate(frame, catalog, k=k)
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="ready",
                check="hallucination_reference",
                path=catalog_path,
                message="Using configured catalog for LLM hallucination metrics.",
                upstream_phase=None,
                row_count=len(catalog),
                timestamp=timestamp,
            )
        )
    else:
        status = (
            "not_configured"
            if candidates_status == "not_configured" and catalog_status == "not_configured"
            else "blocked"
        )
        missing_parts = [
            f"candidates_path={candidates_path or candidates_status}",
            f"catalog_path={catalog_path or catalog_status}",
        ]
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status=status,
                check="hallucination_reference",
                path="; ".join(missing_parts),
                message="LLM hallucination metrics are blocked because neither candidates_path nor catalog_path is configured to a valid readable file.",
                upstream_phase="Complete dataset preparation/audit and configure paths.candidates_path or paths.catalog_path.",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return []

    rows: list[dict[str, Any]] = []
    for (model_name, run_id), group in llm_outputs.groupby(["model_name", "run_id"], sort=False):
        values = calculator(group)
        source_file = _first_non_empty(group.get("source_file"))
        for metric_name in HALLUCINATION_NUMERIC_METRICS:
            rows.append(
                _metric_row(
                    run_id=run_id,
                    dataset=dataset,
                    task=task,
                    model_name=model_name,
                    metric=metric_name,
                    k=k,
                    value=float(values[metric_name]),
                    config_path=config_path,
                    timestamp=timestamp,
                    source_file=source_file,
                    metric_context=metric_source,
                )
            )
    return rows


def _load_optional_table(
    config: dict[str, Any],
    root: Path,
    key: str,
    validator: Any | None = None,
) -> tuple[pd.DataFrame | None, str | None, str]:
    path_value = config.get("paths", {}).get(key) or config.get("dataset", {}).get(key)
    if path_value is None:
        return None, None, "not_configured"

    path = resolve_project_path(path_value, root)
    if path is None or not path.is_file():
        return None, str(path_value), "missing"

    table = read_table(path)
    if validator is not None:
        table = validator(table)
    return table, str(path_value), "ready"


def _output_path(
    config: dict[str, Any],
    root: Path,
    dir_key: str,
    fallback_dir: str,
    filename: str,
) -> Path:
    return _configured_output_dir(config, root, dir_key, fallback_dir) / filename


def _configured_output_dir(
    config: dict[str, Any],
    root: Path,
    key: str,
    fallback: str,
) -> Path:
    resolved = resolve_project_path(config.get("output", {}).get(key, fallback), root)
    return resolved if resolved is not None else root / fallback


def _metric_row(
    *,
    run_id: Any,
    dataset: str,
    task: str,
    model_name: Any,
    metric: str,
    k: int,
    value: float,
    config_path: str,
    timestamp: str,
    source_file: Any,
    metric_context: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset": dataset,
        "task": task,
        "model_name": model_name,
        "metric": metric,
        "k": k,
        "value": value,
        "config_path": config_path,
        "timestamp": timestamp,
        "source_file": source_file,
        "metric_context": metric_context,
    }


def _summary_row(
    *,
    dataset: str,
    status: str,
    check: str,
    path: str | None,
    message: str,
    upstream_phase: str | None,
    row_count: int,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "status": status,
        "check": check,
        "path": path,
        "message": message,
        "upstream_phase": upstream_phase,
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
        "# Reproduction Results Analysis",
        "",
        f"- Dataset: `{dataset}`",
        f"- Config: `{config_path}`",
        f"- Status: `{status}`",
        f"- Metrics CSV: `{_display_path(metrics_path, project_root) if metrics_written else 'not written'}`",
        f"- Summary CSV: `{_display_path(summary_path, project_root)}`",
        "",
        "## Diagnostics",
        "",
    ]

    if summary.empty:
        lines.append("No diagnostics were produced.")
    else:
        lines.extend(_markdown_table(summary[["status", "check", "message", "upstream_phase"]]))

    lines.extend(["", "## Metrics", ""])
    if metrics.empty:
        lines.append("No metric values were computed.")
    else:
        display = metrics[["model_name", "run_id", "metric", "k", "value", "metric_context"]]
        lines.extend(_markdown_table(display))

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
        metrics, summary, metrics_path, summary_path, analysis_path, blocked = run_reproduction(
            args.config,
            project_root=root,
        )
    except (ConfigError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(
        f"metrics={_display_path(metrics_path, root) if not metrics.empty else 'not_written'}"
    )
    print(f"summary={_display_path(summary_path, root)}")
    print(f"analysis={_display_path(analysis_path, root)}")
    print(f"metric_rows={len(metrics)}")
    print(f"statuses={','.join(summary['status'].astype(str).tolist()) if not summary.empty else ''}")
    if blocked:
        LOGGER.info(
            "Reproduction metrics are blocked by missing prerequisites. See %s.",
            _display_path(summary_path, root),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

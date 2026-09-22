"""Prepare history-length prompt-input variants when histories are available.

Run from the repository root:

    python src/experiments/run_history_length.py --config configs/beauty.yaml
"""

from __future__ import annotations

import argparse
import json
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
from src.data.schemas import validate_interactions
from src.prompts.history_length import DEFAULT_HISTORY_LENGTHS, create_history_length_variants
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("run_history_length")

SUMMARY_COLUMNS = [
    "dataset",
    "status",
    "check",
    "path",
    "message",
    "row_count",
    "timestamp",
]

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
    "history_length",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare history-length prompt-input variants from configured histories."
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


def run_history_length(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, Path, bool]:
    """Prepare history-length prompt inputs or write readiness diagnostics."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    task = str(config["experiment"]["task"])
    timestamp = datetime.now(timezone.utc).isoformat()
    config_label = str(config_path)

    raw_outputs_dir = _configured_output_dir(config, root, "raw_outputs_dir", "results/raw_outputs")
    metrics_path = _configured_output_dir(config, root, "metrics_dir", "results/metrics") / f"{dataset}_history_length_metrics.csv"
    summary_path = _configured_output_dir(config, root, "tables_dir", "results/tables") / f"{dataset}_history_length_summary.csv"
    prompts_path = raw_outputs_dir / f"{dataset}_history_length_prompts.csv"
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
    prompts = pd.DataFrame()

    histories, history_ready = _load_history_data(
        config,
        root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    if history_ready and histories is not None:
        prompts = _prepare_prompt_variants(histories, dataset=dataset)
        if prompts.empty:
            summary_rows.append(
                _summary_row(
                    dataset=dataset,
                    status="blocked",
                    check="history_length_prompts",
                    path=str(prompts_path.relative_to(root)),
                    message="History-length prompt variants were not written because no users were present in the history data.",
                    row_count=0,
                    timestamp=timestamp,
                )
            )
        else:
            prompts_path.parent.mkdir(parents=True, exist_ok=True)
            prompts.to_csv(prompts_path, index=False)
            summary_rows.append(
                _summary_row(
                    dataset=dataset,
                    status="ready",
                    check="history_length_prompts",
                    path=str(prompts_path.relative_to(root)),
                    message="Prepared history-length prompt-input variants without API calls.",
                    row_count=len(prompts),
                    timestamp=timestamp,
                )
            )

    if prompts.empty:
        _remove_empty_metrics_file(metrics_path)
    else:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="history_length_metrics",
                path=str(metrics_path.relative_to(root)),
                message=(
                    "History-length metrics were not computed because this phase "
                    "only prepares prompt inputs; valid model outputs and ground truth "
                    "must be supplied by a later phase."
                ),
                row_count=0,
                timestamp=timestamp,
            )
        )

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    metrics_written = False
    if not metrics.empty:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
        metrics_written = True

    summary.to_csv(summary_path, index=False)
    _write_markdown_report(
        analysis_path,
        dataset=dataset,
        config_path=config_label,
        project_root=root,
        prompts_path=prompts_path,
        prompts_written=not prompts.empty,
        metrics_path=metrics_path,
        metrics_written=metrics_written,
        summary_path=summary_path,
        summary=summary,
        blocked=metrics.empty,
    )

    if prompts.empty:
        LOGGER.info("No history-length prompt inputs were written; see readiness summary.")
    else:
        LOGGER.info("Saved history-length prompt inputs to %s", _display_path(prompts_path, root))
    LOGGER.info("Saved history-length summary to %s", _display_path(summary_path, root))
    LOGGER.info("Saved Markdown analysis to %s", _display_path(analysis_path, root))
    return metrics, summary, prompts_path, metrics_path, summary_path, analysis_path, metrics.empty


def _load_history_data(
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame | None, bool]:
    path_value = (
        config.get("paths", {}).get("history_path")
        or config.get("dataset", {}).get("history_path")
    )
    source = "history_path"
    if path_value is None:
        path_value = (
            config.get("paths", {}).get("interactions_path")
            or config.get("dataset", {}).get("interactions_path")
        )
        source = "interactions_path"

    if path_value is None:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="not_configured",
                check="history_data",
                path=None,
                message="History-length prompt preparation is blocked because neither history_path nor interactions_path is configured.",
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
                check="history_data",
                path=str(path_value),
                message=f"History-length prompt preparation is blocked because configured history data does not exist: {path_value}",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return None, False

    try:
        raw = read_table(path)
        histories = _normalize_history_frame(raw, source=source)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="history_data",
                path=str(path_value),
                message=f"Configured history data could not be loaded or validated: {exc}",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return None, False

    if histories.empty:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                status="blocked",
                check="history_data",
                path=str(path_value),
                message="Configured history data contains no usable history rows.",
                row_count=0,
                timestamp=timestamp,
            )
        )
        return None, False

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            status="ready",
            check="history_data",
            path=str(path_value),
            message=f"Loaded user history data from {source}.",
            row_count=len(histories),
            timestamp=timestamp,
        )
    )
    return histories, True


def _normalize_history_frame(raw: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if source == "interactions_path":
        interactions = validate_interactions(raw)
        train = interactions.loc[
            interactions["split"].astype("string").str.lower().str.strip().eq("train")
        ].copy()
        if train.empty:
            raise ValueError("interactions_path does not contain any split == 'train' rows")
        return train

    histories = raw.copy()
    histories.columns = [str(column).strip().lower() for column in histories.columns]
    missing = [column for column in ("user_id", "item_id") if column not in histories.columns]
    if missing:
        raise ValueError(
            "history_path dataframe is missing required column(s): "
            f"{', '.join(missing)}"
        )
    has_values = (
        ~_is_empty_series(histories["user_id"])
        & ~_is_empty_series(histories["item_id"])
    )
    return histories.loc[has_values].copy()


def _prepare_prompt_variants(histories: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    variants = create_history_length_variants(histories, lengths=DEFAULT_HISTORY_LENGTHS)
    if variants.empty:
        return variants

    output = variants.copy()
    output.insert(0, "dataset", dataset)
    output["history_item_ids"] = output["history_item_ids"].map(_json_list)
    return output


def _json_list(values: Any) -> str:
    if isinstance(values, list):
        return json.dumps([str(value) for value in values])
    if pd.isna(values):
        return "[]"
    return json.dumps([str(values)])


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
    prompts_path: Path,
    prompts_written: bool,
    metrics_path: Path,
    metrics_written: bool,
    summary_path: Path,
    summary: pd.DataFrame,
    blocked: bool,
) -> None:
    status = "blocked" if blocked else "complete"
    lines = [
        "# Results Analysis",
        "",
        "## History Length Sensitivity",
        "",
        "History length sensitivity evaluates how changing the amount of user history in a prompt could affect recommendation behavior. This phase only prepares deterministic prompt-input variants and does not call APIs or generate LLM outputs.",
        "",
        f"- Dataset: `{dataset}`",
        f"- Config: `{config_path}`",
        f"- Status: `{status}`",
        f"- Prompt-input CSV: `{_display_path(prompts_path, project_root) if prompts_written else 'not written'}`",
        f"- History-length metrics CSV: `{_display_path(metrics_path, project_root) if metrics_written else 'not written'}`",
        f"- History-length summary CSV: `{_display_path(summary_path, project_root)}`",
        "",
        "Real history-length metrics require prepared prompt inputs, model outputs for each history length, and valid ground truth. With the current Beauty and Sports configs, prompt-input preparation is blocked while `interactions_path` or `history_path` is not configured.",
        "",
        "## Readiness",
        "",
    ]
    lines.extend(_markdown_table(summary))
    lines.extend(["", "## History-Length Metrics", ""])
    lines.append("No real history-length metric values were computed.")
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
        metrics, summary, prompts_path, metrics_path, summary_path, analysis_path, blocked = run_history_length(
            args.config,
            project_root=root,
        )
    except (ConfigError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(f"prompts={_display_path(prompts_path, root) if prompts_path.is_file() else 'not_written'}")
    print(f"metrics={_display_path(metrics_path, root) if not metrics.empty else 'not_written'}")
    print(f"summary={_display_path(summary_path, root)}")
    print(f"analysis={_display_path(analysis_path, root)}")
    print(f"metric_rows={len(metrics)}")
    print(f"statuses={','.join(summary['status'].astype(str).tolist()) if not summary.empty else ''}")
    if blocked:
        LOGGER.info(
            "History-length metrics are blocked by missing prerequisites. See %s.",
            _display_path(summary_path, root),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

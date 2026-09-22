"""Final safe pipeline orchestrator for completed reproducibility phases.

Run from the repository root:

    python src/experiments/run_pipeline.py --config configs/beauty.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.load_baseline_outputs import load_baseline_outputs
from src.baselines.mostpop import MostPopError, run_mostpop
from src.data.audit_datasets import audit_dataset, save_audit_outputs
from src.experiments.exp00_smoke_test import build_checks
from src.experiments.load_llm_outputs import load_llm_outputs
from src.experiments.run_reproduction import run_reproduction
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path

try:  # Optional extension runner.
    from src.experiments.run_extension_coverage import run_coverage_extension
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests.
    run_coverage_extension = None

try:  # Optional extension runner.
    from src.experiments.run_position_bias import run_position_bias
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests.
    run_position_bias = None

try:  # Optional extension runner.
    from src.experiments.run_history_length import run_history_length
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests.
    run_history_length = None

try:  # Cost utilities are availability-checked only in this phase.
    from src.evaluation import cost_metrics as cost_metrics_module
except ImportError:  # pragma: no cover - defensive only.
    cost_metrics_module = None


LOGGER = logging.getLogger("run_pipeline")

SUMMARY_COLUMNS = [
    "dataset",
    "phase",
    "step",
    "status",
    "message",
    "path",
    "row_count",
    "timestamp",
]

KNOWN_METRIC_COLUMNS = [
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

RECOMMENDATION_VALUE_COLUMNS = ("recommended_item_id", "recommended_item_text")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the safe reproducibility pipeline and record readiness."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    return parser.parse_args(argv)


def run_pipeline(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path, bool]:
    """Run safe completed phases and write final summary/metrics if available."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    timestamp = datetime.now(timezone.utc).isoformat()
    config_label = str(config_path)

    tables_dir = _configured_output_dir(config, root, "tables_dir", "results/tables")
    metrics_dir = _configured_output_dir(config, root, "metrics_dir", "results/metrics")
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    final_summary_path = tables_dir / f"{dataset}_final_summary.csv"
    final_metrics_path = metrics_dir / f"{dataset}_final_results.csv"
    analysis_path = resolve_project_path(
        config.get("output", {}).get("analysis_path", "report/results_analysis.md"),
        root,
    )
    if analysis_path is None:
        analysis_path = root / "report/results_analysis.md"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []

    _run_smoke_checks(root, dataset=dataset, timestamp=timestamp, summary_rows=summary_rows)
    _run_or_verify_dataset_audit(
        config_path,
        config,
        root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    _run_or_verify_llm_outputs(
        config_path,
        config,
        root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    _run_or_verify_baseline_outputs(
        config_path,
        config,
        root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    _run_mostpop_if_ready(
        config_path,
        config,
        root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )

    reproduction_metrics = _run_metric_phase(
        phase_name="reproduction",
        runner=run_reproduction,
        config_path=config_path,
        root=root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    _append_metrics(metric_frames, reproduction_metrics, "reproduction")

    coverage_metrics = _run_optional_metric_phase(
        phase_name="coverage",
        runner=run_coverage_extension,
        missing_message="Catalog Coverage@K extension runner is not implemented.",
        config_path=config_path,
        root=root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    _append_metrics(metric_frames, coverage_metrics, "coverage")

    position_metrics = _run_optional_metric_phase(
        phase_name="position_bias",
        runner=run_position_bias,
        missing_message="Candidate position-bias runner is not implemented.",
        config_path=config_path,
        root=root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    _append_metrics(metric_frames, position_metrics, "position_bias")

    history_metrics = _run_optional_metric_phase(
        phase_name="history_length",
        runner=run_history_length,
        missing_message="History length runner is not implemented.",
        config_path=config_path,
        root=root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    _append_metrics(metric_frames, history_metrics, "history_length")

    _record_cost_availability(
        config,
        root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )

    final_metrics = _combine_metric_frames(metric_frames, dataset=dataset)
    if final_metrics.empty:
        _remove_file(final_metrics_path)
        final_metrics_written = False
    else:
        final_metrics.to_csv(final_metrics_path, index=False)
        final_metrics_written = True

    final_summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    final_summary.to_csv(final_summary_path, index=False)
    _write_markdown_report(
        analysis_path,
        dataset=dataset,
        config_path=config_label,
        project_root=root,
        final_summary_path=final_summary_path,
        final_metrics_path=final_metrics_path,
        final_metrics_written=final_metrics_written,
        summary=final_summary,
        metrics=final_metrics,
    )

    LOGGER.info("Saved final summary to %s", _display_path(final_summary_path, root))
    if final_metrics_written:
        LOGGER.info("Saved final metrics to %s", _display_path(final_metrics_path, root))
    else:
        LOGGER.info("No final metrics were written because no real metric rows were computed.")
    LOGGER.info("Saved Markdown analysis to %s", _display_path(analysis_path, root))
    return (
        final_metrics,
        final_summary,
        final_metrics_path,
        final_summary_path,
        analysis_path,
        final_metrics.empty,
    )


def _run_smoke_checks(
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    with _pushd(root):
        checks = build_checks(root, config_path=None)
    failed = [check for check in checks if check.critical and not check.passed]
    status = "completed" if not failed else "blocked"
    message = "Smoke checks passed." if not failed else f"{len(failed)} critical smoke check(s) failed."
    summary_rows.append(
        _summary_row(
            dataset=dataset,
            phase="smoke",
            step="smoke_checks",
            status=status,
            message=message,
            path=None,
            row_count=len(checks),
            timestamp=timestamp,
        )
    )


def _run_or_verify_dataset_audit(
    config_path: str | Path,
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    audit_path = resolve_project_path(
        config.get("output", {}).get("dataset_audit_path"),
        root,
    )
    if audit_path is None:
        audit_path = root / "results" / "tables" / f"{dataset}_dataset_audit.csv"

    if audit_path.is_file():
        row_count = _csv_row_count(audit_path)
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="dataset_audit",
                step="dataset_audit",
                status="ready",
                message="Verified existing dataset audit output.",
                path=_display_path(audit_path, root),
                row_count=row_count,
                timestamp=timestamp,
            )
        )
        return

    try:
        with _pushd(root):
            audit = audit_dataset(_config_path_for_runner(config_path, root))
            written_path, _ = save_audit_outputs(audit, config, root)
    except Exception as exc:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="dataset_audit",
                step="dataset_audit",
                status="blocked",
                message=f"Dataset audit could not be completed: {exc}",
                path=_display_path(audit_path, root),
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            phase="dataset_audit",
            step="dataset_audit",
            status="completed",
            message="Ran dataset audit and wrote output.",
            path=_display_path(written_path, root),
            row_count=len(audit),
            timestamp=timestamp,
        )
    )


def _run_or_verify_llm_outputs(
    config_path: str | Path,
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    raw_outputs_dir = _configured_output_dir(config, root, "raw_outputs_dir", "results/raw_outputs")
    normalized_path = raw_outputs_dir / f"{dataset}_llm_outputs_normalized.csv"
    if normalized_path.is_file():
        status, message, row_count = _recommendation_file_status(normalized_path)
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="llm_outputs",
                step="llm_outputs",
                status=status,
                message=message,
                path=_display_path(normalized_path, root),
                row_count=row_count,
                timestamp=timestamp,
            )
        )
        return

    entries = config.get("models", {}).get("llm_outputs", [])
    if not entries:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="llm_outputs",
                step="llm_outputs",
                status="not_configured",
                message="No models.llm_outputs entries are configured.",
                path=_display_path(normalized_path, root),
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    try:
        outputs, _, output_path, _ = load_llm_outputs(
            _config_path_for_runner(config_path, root),
            project_root=root,
        )
    except Exception as exc:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="llm_outputs",
                step="llm_outputs",
                status="blocked",
                message=f"LLM output loading failed: {exc}",
                path=_display_path(normalized_path, root),
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    status = "completed" if not outputs.empty else "blocked"
    message = "Loaded normalized LLM outputs." if not outputs.empty else "No usable LLM output rows were normalized."
    summary_rows.append(
        _summary_row(
            dataset=dataset,
            phase="llm_outputs",
            step="llm_outputs",
            status=status,
            message=message,
            path=_display_path(output_path, root),
            row_count=len(outputs),
            timestamp=timestamp,
        )
    )


def _run_or_verify_baseline_outputs(
    config_path: str | Path,
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    raw_outputs_dir = _configured_output_dir(config, root, "raw_outputs_dir", "results/raw_outputs")
    normalized_path = raw_outputs_dir / f"{dataset}_baseline_outputs_normalized.csv"
    if normalized_path.is_file():
        row_count = _csv_row_count(normalized_path)
        status = "ready" if row_count else "blocked"
        message = "Verified existing normalized baseline outputs." if row_count else "Baseline output file exists but has no rows."
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="baseline_outputs",
                step="baseline_outputs",
                status=status,
                message=message,
                path=_display_path(normalized_path, root),
                row_count=row_count,
                timestamp=timestamp,
            )
        )
        return

    configured = bool(config.get("models", {}).get("baseline_outputs")) or bool(
        config.get("paths", {}).get("baseline_outputs_path")
        or config.get("dataset", {}).get("baseline_outputs_path")
    )
    if not configured:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="baseline_outputs",
                step="baseline_outputs",
                status="not_configured",
                message="No baseline output source is configured.",
                path=_display_path(normalized_path, root),
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    try:
        outputs, _, output_path, _ = load_baseline_outputs(
            _config_path_for_runner(config_path, root),
            project_root=root,
        )
    except Exception as exc:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="baseline_outputs",
                step="baseline_outputs",
                status="blocked",
                message=f"Baseline output loading failed: {exc}",
                path=_display_path(normalized_path, root),
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    status = "completed" if not outputs.empty else "missing"
    message = "Loaded normalized baseline outputs." if not outputs.empty else "No usable baseline output rows were found."
    summary_rows.append(
        _summary_row(
            dataset=dataset,
            phase="baseline_outputs",
            step="baseline_outputs",
            status=status,
            message=message,
            path=_display_path(output_path, root),
            row_count=len(outputs),
            timestamp=timestamp,
        )
    )


def _run_mostpop_if_ready(
    config_path: str | Path,
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    configured = config.get("paths", {}).get("interactions_path") or config.get("dataset", {}).get("interactions_path")
    if configured is None:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="mostpop",
                step="mostpop",
                status="not_configured",
                message="MostPop is blocked because interactions_path is not configured.",
                path=None,
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    interactions_path = resolve_project_path(configured, root)
    if interactions_path is None or not interactions_path.is_file():
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="mostpop",
                step="mostpop",
                status="missing",
                message=f"MostPop is blocked because interactions_path is missing: {configured}",
                path=str(configured),
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    try:
        recommendations, _, output_path, _ = run_mostpop(
            _config_path_for_runner(config_path, root),
            project_root=root,
        )
    except (MostPopError, ConfigError, OSError, ValueError) as exc:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="mostpop",
                step="mostpop",
                status="blocked",
                message=f"MostPop could not run: {exc}",
                path=str(configured),
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            phase="mostpop",
            step="mostpop",
            status="completed",
            message="Generated MostPop recommendations from configured interactions.",
            path=_display_path(output_path, root),
            row_count=len(recommendations),
            timestamp=timestamp,
        )
    )


def _run_metric_phase(
    *,
    phase_name: str,
    runner: Callable[..., tuple],
    config_path: str | Path,
    root: Path,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    try:
        result = runner(_config_path_for_runner(config_path, root), project_root=root)
    except Exception as exc:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase=phase_name,
                step=phase_name,
                status="blocked",
                message=f"{phase_name} phase failed: {exc}",
                path=None,
                row_count=0,
                timestamp=timestamp,
            )
        )
        return pd.DataFrame()

    metrics = result[0] if len(result) > 0 and isinstance(result[0], pd.DataFrame) else pd.DataFrame()
    phase_summary = result[1] if len(result) > 1 and isinstance(result[1], pd.DataFrame) else pd.DataFrame()
    primary_path = _phase_summary_path(result)
    status = "completed" if not metrics.empty else "blocked"
    message = (
        f"{phase_name} computed {len(metrics)} metric row(s)."
        if not metrics.empty
        else f"{phase_name} produced diagnostics but no real metric rows."
    )
    summary_rows.append(
        _summary_row(
            dataset=dataset,
            phase=phase_name,
            step=phase_name,
            status=status,
            message=message,
            path=_display_path(primary_path, root) if primary_path is not None else None,
            row_count=len(metrics),
            timestamp=timestamp,
        )
    )
    _append_phase_summary_rows(
        phase_summary,
        phase_name=phase_name,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    return metrics


def _phase_summary_path(result: tuple) -> Path | None:
    paths = [item for item in result if isinstance(item, Path)]
    for path in paths:
        if path.name.endswith("_summary.csv"):
            return path
    return paths[0] if paths else None


def _run_optional_metric_phase(
    *,
    phase_name: str,
    runner: Callable[..., tuple] | None,
    missing_message: str,
    config_path: str | Path,
    root: Path,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    if runner is None:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase=phase_name,
                step=phase_name,
                status="skipped",
                message=missing_message,
                path=None,
                row_count=0,
                timestamp=timestamp,
            )
        )
        return pd.DataFrame()
    return _run_metric_phase(
        phase_name=phase_name,
        runner=runner,
        config_path=config_path,
        root=root,
        dataset=dataset,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )


def _append_phase_summary_rows(
    phase_summary: pd.DataFrame,
    *,
    phase_name: str,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    if phase_summary.empty:
        return
    normalized = phase_summary.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    for _, row in normalized.iterrows():
        check = row.get("check", phase_name)
        status = _normalize_status(row.get("status", "ready"))
        message = row.get("message", "")
        path = row.get("path")
        row_count = row.get("row_count", 0)
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase=phase_name,
                step=str(check),
                status=status,
                message=str(message) if not pd.isna(message) else "",
                path=str(path) if not pd.isna(path) else None,
                row_count=_safe_int(row_count),
                timestamp=timestamp,
            )
        )


def _record_cost_availability(
    config: dict[str, Any],
    root: Path,
    *,
    dataset: str,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> None:
    if cost_metrics_module is None:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="cost",
                step="cost_metrics_module",
                status="missing",
                message="Cost metric utilities are not importable.",
                path=None,
                row_count=0,
                timestamp=timestamp,
            )
        )
        return

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            phase="cost",
            step="cost_metrics_module",
            status="ready",
            message="Cost/token utility functions are importable; no API calls are made.",
            path="src/evaluation/cost_metrics.py",
            row_count=0,
            timestamp=timestamp,
        )
    )

    token_path_value = (
        config.get("paths", {}).get("token_usage_path")
        or config.get("paths", {}).get("cost_records_path")
        or config.get("dataset", {}).get("token_usage_path")
        or config.get("dataset", {}).get("cost_records_path")
    )
    if token_path_value is None:
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="cost",
                step="token_latency_inputs",
                status="not_configured",
                message="No token/latency record path is configured; cost estimates were not computed.",
                path=None,
                row_count=0,
                timestamp=timestamp,
            )
        )
    else:
        token_path = resolve_project_path(token_path_value, root)
        status = "ready" if token_path is not None and token_path.is_file() else "missing"
        message = (
            "Token/latency record path exists, but cost computation is deferred until a dedicated cost runner is requested."
            if status == "ready"
            else f"Configured token/latency record path is missing: {token_path_value}"
        )
        summary_rows.append(
            _summary_row(
                dataset=dataset,
                phase="cost",
                step="token_latency_inputs",
                status=status,
                message=message,
                path=str(token_path_value),
                row_count=0,
                timestamp=timestamp,
            )
        )

    cost_config = root / "configs" / "cost.yaml"
    pricing_status, pricing_message = _cost_pricing_status(cost_config)
    summary_rows.append(
        _summary_row(
            dataset=dataset,
            phase="cost",
            step="pricing_assumptions",
            status=pricing_status,
            message=pricing_message,
            path=_display_path(cost_config, root),
            row_count=0,
            timestamp=timestamp,
        )
    )


def _cost_pricing_status(cost_config: Path) -> tuple[str, str]:
    if not cost_config.is_file():
        return "not_configured", "configs/cost.yaml is not present; cost estimates were not computed."
    try:
        with cost_config.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as exc:
        return "blocked", f"Could not read configs/cost.yaml: {exc}"

    price_values: list[Any] = []
    models = loaded.get("pricing", {}).get("models", {}) if isinstance(loaded, dict) else {}
    if isinstance(models, dict):
        for model_prices in models.values():
            if isinstance(model_prices, dict):
                for key in ("prompt_per_1k_tokens", "completion_per_1k_tokens", "total_per_1k_tokens"):
                    price_values.append(model_prices.get(key))
    has_price = any(_has_numeric_value(value) for value in price_values)
    if has_price:
        return "ready", "At least one explicit pricing assumption is configured."
    return "not_configured", "configs/cost.yaml is present, but prices are null; cost estimates were not computed."


def _append_metrics(metric_frames: list[pd.DataFrame], metrics: pd.DataFrame, phase_name: str) -> None:
    if metrics.empty:
        return
    frame = metrics.copy()
    frame["final_source_phase"] = phase_name
    metric_frames.append(frame)


def _combine_metric_frames(metric_frames: list[pd.DataFrame], *, dataset: str) -> pd.DataFrame:
    if not metric_frames:
        return pd.DataFrame()
    combined = pd.concat(metric_frames, ignore_index=True, sort=False)
    if "dataset" in combined.columns:
        combined = combined.loc[combined["dataset"].astype("string") == dataset].copy()
    if combined.empty:
        return combined
    preferred = [column for column in [*KNOWN_METRIC_COLUMNS, "final_source_phase"] if column in combined.columns]
    extras = [column for column in combined.columns if column not in preferred]
    return combined.loc[:, [*preferred, *extras]]


def _recommendation_file_status(path: Path) -> tuple[str, str, int]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        return "blocked", f"Recommendation file could not be read: {exc}", 0
    if frame.empty:
        return "blocked", "Recommendation file exists but has no rows.", 0

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    available = [column for column in RECOMMENDATION_VALUE_COLUMNS if column in normalized.columns]
    if not available:
        return "blocked", "Recommendation file lacks recommended_item_id/recommended_item_text columns.", len(frame)
    has_value = pd.Series(False, index=normalized.index)
    for column in available:
        has_value = has_value | ~_is_empty_series(normalized[column])
    usable = int(has_value.sum())
    empty = int((~has_value).sum())
    if usable == 0:
        return "blocked", "Recommendation file has no non-empty recommendation values.", len(frame)
    if empty:
        return "ready", f"Verified recommendation file with {usable} usable row(s) and {empty} empty row(s).", usable
    return "ready", "Verified recommendation file with usable recommendation rows.", usable


def _configured_output_dir(
    config: dict[str, Any],
    root: Path,
    key: str,
    fallback: str,
) -> Path:
    resolved = resolve_project_path(config.get("output", {}).get(key, fallback), root)
    return resolved if resolved is not None else root / fallback


def _config_path_for_runner(config_path: str | Path, root: Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return root / path


def _summary_row(
    *,
    dataset: str,
    phase: str,
    step: str,
    status: str,
    message: str,
    path: str | None,
    row_count: int,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "phase": phase,
        "step": step,
        "status": status,
        "message": message,
        "path": path,
        "row_count": int(row_count),
        "timestamp": timestamp,
    }


def _write_markdown_report(
    path: Path,
    *,
    dataset: str,
    config_path: str,
    project_root: Path,
    final_summary_path: Path,
    final_metrics_path: Path,
    final_metrics_written: bool,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    status_counts = summary["status"].value_counts().to_dict() if not summary.empty else {}
    lines = [
        "# Results Analysis",
        "",
        "## Final Safe Pipeline",
        "",
        "The final safe pipeline orchestrates completed local phases only. It does not call APIs, download data, train models, modify `original_repo/`, or invent missing inputs or metric values.",
        "",
        f"- Dataset: `{dataset}`",
        f"- Config: `{config_path}`",
        f"- Final summary CSV: `{_display_path(final_summary_path, project_root)}`",
        f"- Final metrics CSV: `{_display_path(final_metrics_path, project_root) if final_metrics_written else 'not written'}`",
        f"- Metric rows: `{len(metrics)}`",
        "",
        "## Status Counts",
        "",
    ]
    if status_counts:
        status_frame = pd.DataFrame(
            [{"status": status, "count": count} for status, count in status_counts.items()]
        )
        lines.extend(_markdown_table(status_frame))
    else:
        lines.append("No pipeline summary rows were written.")

    lines.extend(["", "## Pipeline Summary", ""])
    display_columns = ["phase", "step", "status", "message", "path", "row_count"]
    lines.extend(_markdown_table(summary[display_columns] if not summary.empty else summary))
    lines.extend(["", "## Final Metrics", ""])
    if metrics.empty:
        lines.append("No final metric values were written because no underlying phase computed real metrics.")
    else:
        metric_columns = [column for column in ("model_name", "metric", "k", "value", "final_source_phase") if column in metrics.columns]
        lines.extend(_markdown_table(metrics[metric_columns]))
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


def _normalize_status(status: Any) -> str:
    value = str(status).strip().lower()
    if value in {"completed", "ready", "skipped", "blocked", "missing", "not_configured"}:
        return value
    if value == "ready_with_warnings":
        return "ready"
    if value in {"found", "success", "pass"}:
        return "ready"
    return "blocked"


def _csv_row_count(path: Path) -> int:
    try:
        return len(pd.read_csv(path))
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return 0


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
    except TypeError:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _has_numeric_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        return False
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def _remove_file(path: Path) -> None:
    if path.is_file():
        path.unlink()


def _display_path(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _is_empty_series(series: pd.Series) -> pd.Series:
    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("")


def _ensure_api_calls_disabled(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls"):
        raise ConfigError("api.allow_api_calls must be false; pipeline never calls APIs.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError(
            "experiment.allow_api_calls must be false; pipeline never calls APIs."
        )


@contextmanager
def _pushd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    try:
        root = find_project_root()
        metrics, summary, metrics_path, summary_path, analysis_path, _ = run_pipeline(
            args.config,
            project_root=root,
        )
    except (ConfigError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(f"summary={_display_path(summary_path, root)}")
    print(f"metrics={_display_path(metrics_path, root) if not metrics.empty else 'not_written'}")
    print(f"analysis={_display_path(analysis_path, root)}")
    print(f"summary_rows={len(summary)}")
    print(f"metric_rows={len(metrics)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

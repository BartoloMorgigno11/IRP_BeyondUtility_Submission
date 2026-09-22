"""Generate plots from existing result CSV files only.

Run from the repository root:

    python src/experiments/make_plots.py --config configs/beauty.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("make_plots")

PLOT_SUMMARY_COLUMNS = [
    "dataset",
    "plot_name",
    "status",
    "reason",
    "output_path",
    "source_files",
    "timestamp",
]

HR_ALIASES = {
    "hr",
    "hr_at_5",
    "hit_rate",
    "hit_rate_at_5",
    "hit_rate_at_k",
}
NDCG_ALIASES = {
    "ndcg",
    "ndcg_at_5",
    "ndcg_at_k",
}
COVERAGE_ALIASES = {
    "coverage",
    "coverage_at_5",
    "catalog_coverage",
    "catalog_coverage_at_5",
    "catalog_coverage_at_k",
}
HALLUCINATION_ALIASES = {
    "hallucination_rate",
    "strict_hallucination_rate",
    "candidate_hallucination_rate",
    "catalog_hallucination_rate",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV-backed plots without computing new metrics."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    return parser.parse_args(argv)


def run_make_plots(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Generate available plots and write a deterministic plot summary."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    timestamp = datetime.now(timezone.utc).isoformat()

    metrics_dir = _configured_output_dir(config, root, "metrics_dir", "results/metrics")
    tables_dir = _configured_output_dir(config, root, "tables_dir", "results/tables")
    plots_dir = _configured_output_dir(config, root, "plots_dir", "results/plots")
    summary_path = tables_dir / f"{dataset}_plot_summary.csv"
    analysis_path = resolve_project_path(
        config.get("output", {}).get("analysis_path", "report/results_analysis.md"),
        root,
    )
    if analysis_path is None:
        analysis_path = root / "report/results_analysis.md"

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    metric_rows = _load_metric_rows(metrics_dir, dataset=dataset, project_root=root)
    summary_rows: list[dict[str, Any]] = []

    generated_metric_plots = 0
    generated_metric_plots += _plot_metric_bar_if_available(
        metric_rows,
        dataset=dataset,
        plot_name="hr_at_5",
        metric_aliases=HR_ALIASES,
        ylabel="HR@5",
        output_path=plots_dir / f"{dataset}_hr_at_5.png",
        project_root=root,
        timestamp=timestamp,
        summary_rows=summary_rows,
        k=5,
    )
    generated_metric_plots += _plot_metric_bar_if_available(
        metric_rows,
        dataset=dataset,
        plot_name="ndcg_at_5",
        metric_aliases=NDCG_ALIASES,
        ylabel="NDCG@5",
        output_path=plots_dir / f"{dataset}_ndcg_at_5.png",
        project_root=root,
        timestamp=timestamp,
        summary_rows=summary_rows,
        k=5,
    )
    generated_metric_plots += _plot_metric_bar_if_available(
        metric_rows,
        dataset=dataset,
        plot_name="catalog_coverage_at_5",
        metric_aliases=COVERAGE_ALIASES,
        ylabel="Catalog Coverage@5",
        output_path=plots_dir / f"{dataset}_catalog_coverage_at_5.png",
        project_root=root,
        timestamp=timestamp,
        summary_rows=summary_rows,
        k=5,
    )
    generated_metric_plots += _plot_metric_bar_if_available(
        metric_rows,
        dataset=dataset,
        plot_name="hallucination_rate",
        metric_aliases=HALLUCINATION_ALIASES,
        ylabel="Hallucination Rate",
        output_path=plots_dir / f"{dataset}_hallucination_rate.png",
        project_root=root,
        timestamp=timestamp,
        summary_rows=summary_rows,
        contains="hallucination",
    )
    generated_metric_plots += _plot_scatter_if_available(
        metric_rows,
        dataset=dataset,
        output_path=plots_dir / f"{dataset}_ndcg_at_5_vs_catalog_coverage_at_5.png",
        project_root=root,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    generated_metric_plots += _plot_position_bias_if_available(
        metric_rows,
        dataset=dataset,
        output_path=plots_dir / f"{dataset}_position_bias.png",
        project_root=root,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )
    generated_metric_plots += _plot_history_length_if_available(
        metric_rows,
        dataset=dataset,
        output_path=plots_dir / f"{dataset}_history_length.png",
        project_root=root,
        timestamp=timestamp,
        summary_rows=summary_rows,
    )

    _plot_readiness_if_needed(
        tables_dir,
        dataset=dataset,
        output_path=plots_dir / f"{dataset}_readiness_status_counts.png",
        project_root=root,
        timestamp=timestamp,
        summary_rows=summary_rows,
        generated_metric_plots=generated_metric_plots,
    )

    plot_summary = pd.DataFrame(summary_rows, columns=PLOT_SUMMARY_COLUMNS)
    plot_summary.to_csv(summary_path, index=False)
    _write_markdown_section(
        analysis_path,
        dataset=dataset,
        config_path=str(config_path),
        project_root=root,
        summary_path=summary_path,
        plot_summary=plot_summary,
    )

    generated = int((plot_summary["status"] == "generated").sum()) if not plot_summary.empty else 0
    skipped = int((plot_summary["status"] == "skipped").sum()) if not plot_summary.empty else 0
    LOGGER.info("Saved plot summary to %s", _display_path(summary_path, root))
    LOGGER.info("Generated %d plot(s), skipped %d plot(s).", generated, skipped)
    return plot_summary, summary_path, analysis_path


def _load_metric_rows(metrics_dir: Path, *, dataset: str, project_root: Path) -> pd.DataFrame:
    """Load existing metric CSV rows without computing or inferring metrics."""

    frames: list[pd.DataFrame] = []
    if not metrics_dir.is_dir():
        return pd.DataFrame()

    for path in sorted(metrics_dir.glob(f"{dataset}_*.csv")):
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            LOGGER.info("Skipping empty metrics CSV: %s", _display_path(path, project_root))
            continue
        except OSError as exc:
            LOGGER.warning("Skipping unreadable metrics CSV %s: %s", path, exc)
            continue

        if not {"metric", "value"}.issubset(frame.columns):
            LOGGER.info(
                "Skipping non-metric CSV without metric/value columns: %s",
                _display_path(path, project_root),
            )
            continue
        if "dataset" in frame.columns:
            frame = frame.loc[frame["dataset"].astype(str) == dataset].copy()
        if frame.empty:
            continue
        frame["_csv_source_file"] = _display_path(path, project_root)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _plot_metric_bar_if_available(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    plot_name: str,
    metric_aliases: set[str],
    ylabel: str,
    output_path: Path,
    project_root: Path,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
    k: int | None = None,
    contains: str | None = None,
) -> int:
    rows = _filter_metric_rows(metrics, metric_aliases, k=k, contains=contains)
    rows = _with_numeric_values(rows)
    if rows.empty:
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name=plot_name,
            status="skipped",
            reason=f"No existing CSV metric rows were available for {plot_name}.",
            output_path=None,
            source_files=[],
            timestamp=timestamp,
        )
        LOGGER.info("Skipped %s: unavailable metric rows.", plot_name)
        return 0

    labels = _row_labels(rows)
    fig, ax = plt.subplots()
    ax.bar(labels, rows["value"].tolist())
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset} {ylabel} by model")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    _append_summary(
        summary_rows,
        dataset=dataset,
        plot_name=plot_name,
        status="generated",
        reason=f"Generated from {len(rows)} existing metric row(s).",
        output_path=_display_path(output_path, project_root),
        source_files=_source_files(rows),
        timestamp=timestamp,
    )
    return 1


def _plot_scatter_if_available(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    output_path: Path,
    project_root: Path,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> int:
    ndcg = _with_numeric_values(_filter_metric_rows(metrics, NDCG_ALIASES, k=5))
    coverage = _with_numeric_values(_filter_metric_rows(metrics, COVERAGE_ALIASES, k=5))
    required_columns = {"model_name", "value"}
    if ndcg.empty or coverage.empty or not required_columns.issubset(ndcg.columns) or not required_columns.issubset(coverage.columns):
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name="ndcg_at_5_vs_catalog_coverage_at_5",
            status="skipped",
            reason="NDCG@5 and Catalog Coverage@5 metric rows were not both available.",
            output_path=None,
            source_files=_source_files(pd.concat([ndcg, coverage], ignore_index=True, sort=False)),
            timestamp=timestamp,
        )
        LOGGER.info("Skipped NDCG@5 vs Coverage@5 scatter: missing paired metrics.")
        return 0

    ndcg = ndcg.rename(columns={"value": "ndcg_value"})
    coverage = coverage.rename(columns={"value": "coverage_value"})
    paired = pd.merge(
        ndcg[["model_name", "ndcg_value", "_csv_source_file"]],
        coverage[["model_name", "coverage_value", "_csv_source_file"]],
        on="model_name",
        how="inner",
        suffixes=("_ndcg", "_coverage"),
    )
    if paired.empty:
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name="ndcg_at_5_vs_catalog_coverage_at_5",
            status="skipped",
            reason="NDCG@5 and Catalog Coverage@5 rows were present but had no shared model_name.",
            output_path=None,
            source_files=_source_files(pd.concat([ndcg, coverage], ignore_index=True, sort=False)),
            timestamp=timestamp,
        )
        return 0

    fig, ax = plt.subplots()
    ax.scatter(paired["ndcg_value"], paired["coverage_value"])
    for _, row in paired.iterrows():
        ax.annotate(str(row["model_name"]), (row["ndcg_value"], row["coverage_value"]))
    ax.set_xlabel("NDCG@5")
    ax.set_ylabel("Catalog Coverage@5")
    ax.set_title(f"{dataset} NDCG@5 vs Catalog Coverage@5")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    source_files = sorted(
        {
            str(value)
            for column in ("_csv_source_file_ndcg", "_csv_source_file_coverage")
            for value in paired[column].dropna().tolist()
            if str(value)
        }
    )
    _append_summary(
        summary_rows,
        dataset=dataset,
        plot_name="ndcg_at_5_vs_catalog_coverage_at_5",
        status="generated",
        reason=f"Generated from {len(paired)} model pair(s) with both metrics.",
        output_path=_display_path(output_path, project_root),
        source_files=source_files,
        timestamp=timestamp,
    )
    return 1


def _plot_position_bias_if_available(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    output_path: Path,
    project_root: Path,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> int:
    required = {"candidate_position", "value"}
    rows = _with_numeric_values(metrics.copy())
    if rows.empty or not required.issubset(rows.columns):
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name="position_bias",
            status="skipped",
            reason="No existing position-bias metric rows with candidate_position and value were available.",
            output_path=None,
            source_files=[],
            timestamp=timestamp,
        )
        LOGGER.info("Skipped position-bias plot: metric rows unavailable.")
        return 0

    position_values = pd.to_numeric(rows["candidate_position"], errors="coerce")
    rows = rows.loc[position_values.notna()].copy()
    rows["candidate_position"] = position_values.loc[rows.index].astype(int)
    if rows.empty:
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name="position_bias",
            status="skipped",
            reason="Position-bias rows existed but candidate_position values were not numeric.",
            output_path=None,
            source_files=_source_files(metrics),
            timestamp=timestamp,
        )
        return 0

    fig, ax = plt.subplots()
    if "model_name" in rows.columns:
        for model_name, model_rows in rows.sort_values("candidate_position").groupby("model_name", sort=True):
            ax.plot(
                model_rows["candidate_position"].tolist(),
                model_rows["value"].tolist(),
                marker="o",
                label=str(model_name),
            )
        ax.legend()
    else:
        rows = rows.sort_values("candidate_position")
        ax.plot(rows["candidate_position"].tolist(), rows["value"].tolist(), marker="o")
    ax.set_xlabel("Candidate Position")
    ax.set_ylabel("Selection Fraction")
    ax.set_title(f"{dataset} candidate position bias")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    _append_summary(
        summary_rows,
        dataset=dataset,
        plot_name="position_bias",
        status="generated",
        reason=f"Generated from {len(rows)} existing position-bias metric row(s).",
        output_path=_display_path(output_path, project_root),
        source_files=_source_files(rows),
        timestamp=timestamp,
    )
    return 1


def _plot_history_length_if_available(
    metrics: pd.DataFrame,
    *,
    dataset: str,
    output_path: Path,
    project_root: Path,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
) -> int:
    required = {"history_length", "value"}
    rows = _with_numeric_values(metrics.copy())
    if rows.empty or not required.issubset(rows.columns):
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name="history_length",
            status="skipped",
            reason="No existing history-length metric rows with history_length and value were available.",
            output_path=None,
            source_files=[],
            timestamp=timestamp,
        )
        LOGGER.info("Skipped history-length plot: metric rows unavailable.")
        return 0

    rows = rows.sort_values(["model_name", "history_length"] if "model_name" in rows.columns else ["history_length"])
    fig, ax = plt.subplots()
    if "model_name" in rows.columns:
        for model_name, model_rows in rows.groupby("model_name", sort=True):
            ax.plot(
                model_rows["history_length"].astype(str).tolist(),
                model_rows["value"].tolist(),
                marker="o",
                label=str(model_name),
            )
        ax.legend()
    else:
        ax.plot(rows["history_length"].astype(str).tolist(), rows["value"].tolist(), marker="o")
    ax.set_xlabel("History Length")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"{dataset} history length sensitivity")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    _append_summary(
        summary_rows,
        dataset=dataset,
        plot_name="history_length",
        status="generated",
        reason=f"Generated from {len(rows)} existing history-length metric row(s).",
        output_path=_display_path(output_path, project_root),
        source_files=_source_files(rows),
        timestamp=timestamp,
    )
    return 1


def _plot_readiness_if_needed(
    tables_dir: Path,
    *,
    dataset: str,
    output_path: Path,
    project_root: Path,
    timestamp: str,
    summary_rows: list[dict[str, Any]],
    generated_metric_plots: int,
) -> int:
    if generated_metric_plots:
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name="readiness_status_counts",
            status="skipped",
            reason="Real metric plot(s) were generated, so readiness fallback plot was not needed.",
            output_path=None,
            source_files=[],
            timestamp=timestamp,
        )
        return 0

    readiness, source_files = _load_readiness_rows(tables_dir, dataset=dataset, project_root=project_root)
    if readiness.empty or "status" not in readiness.columns:
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name="readiness_status_counts",
            status="skipped",
            reason="No existing readiness summary CSV with a status column was available.",
            output_path=None,
            source_files=source_files,
            timestamp=timestamp,
        )
        LOGGER.info("Skipped readiness plot: no status summary rows available.")
        return 0

    statuses = readiness["status"].fillna("unknown").astype(str).str.strip()
    statuses = statuses.mask(statuses == "", "unknown")
    counts = statuses.value_counts().sort_index()
    if counts.empty:
        _append_summary(
            summary_rows,
            dataset=dataset,
            plot_name="readiness_status_counts",
            status="skipped",
            reason="Readiness summary rows existed but status values were empty.",
            output_path=None,
            source_files=source_files,
            timestamp=timestamp,
        )
        return 0

    fig, ax = plt.subplots()
    ax.bar(counts.index.tolist(), counts.values.tolist())
    ax.set_xlabel("Status")
    ax.set_ylabel("Row Count")
    ax.set_title(f"{dataset} readiness and blocker status counts")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    _append_summary(
        summary_rows,
        dataset=dataset,
        plot_name="readiness_status_counts",
        status="generated",
        reason="Generated from existing readiness/status summary rows; no metric values were inferred.",
        output_path=_display_path(output_path, project_root),
        source_files=source_files,
        timestamp=timestamp,
    )
    return 1


def _filter_metric_rows(
    metrics: pd.DataFrame,
    aliases: set[str],
    *,
    k: int | None = None,
    contains: str | None = None,
) -> pd.DataFrame:
    if metrics.empty or "metric" not in metrics.columns:
        return pd.DataFrame()

    metric_names = metrics["metric"].fillna("").astype(str)
    normalized = metric_names.map(_normalize_metric_name)
    alias_set = {_normalize_metric_name(alias) for alias in aliases}
    mask = normalized.isin(alias_set)
    if contains is not None:
        mask = mask | normalized.str.contains(_normalize_metric_name(contains), regex=False)

    filtered = metrics.loc[mask].copy()
    if filtered.empty:
        return filtered

    if k is not None:
        if "k" in filtered.columns:
            k_values = pd.to_numeric(filtered["k"], errors="coerce")
            filtered = filtered.loc[k_values == k].copy()
        else:
            metric_mentions_k = filtered["metric"].fillna("").astype(str).str.contains(
                rf"(^|[^0-9]){re.escape(str(k))}([^0-9]|$)",
                regex=True,
            )
            filtered = filtered.loc[metric_mentions_k].copy()
    return filtered


def _with_numeric_values(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "value" not in rows.columns:
        return pd.DataFrame()
    rows = rows.copy()
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    return rows.loc[rows["value"].notna()].copy()


def _load_readiness_rows(
    tables_dir: Path,
    *,
    dataset: str,
    project_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    if not tables_dir.is_dir():
        return pd.DataFrame(), []

    preferred = tables_dir / f"{dataset}_final_summary.csv"
    candidate_files = [preferred] if preferred.is_file() else []
    if not candidate_files:
        candidate_files = [
            path
            for path in sorted(tables_dir.glob(f"{dataset}_*summary.csv"))
            if path.is_file() and not path.name.endswith("_plot_summary.csv")
        ]

    frames: list[pd.DataFrame] = []
    source_files: list[str] = []
    for path in candidate_files:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "status" not in frame.columns:
            continue
        frames.append(frame)
        source_files.append(_display_path(path, project_root))

    if not frames:
        return pd.DataFrame(), source_files
    return pd.concat(frames, ignore_index=True, sort=False), source_files


def _row_labels(rows: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    used: dict[str, int] = {}
    for _, row in rows.iterrows():
        label = str(row.get("model_name", "unknown_model"))
        context = str(row.get("metric_context", "") or "")
        run_id = str(row.get("run_id", "") or "")
        if context and context.lower() != "nan":
            label = f"{label} ({context})"
        elif run_id and run_id.lower() != "nan":
            label = f"{label} ({run_id})"

        count = used.get(label, 0)
        used[label] = count + 1
        if count:
            label = f"{label} #{count + 1}"
        labels.append(label)
    return labels


def _source_files(rows: pd.DataFrame) -> list[str]:
    if rows.empty or "_csv_source_file" not in rows.columns:
        return []
    return sorted({str(value) for value in rows["_csv_source_file"].dropna().tolist() if str(value)})


def _configured_output_dir(config: dict[str, Any], root: Path, key: str, default: str) -> Path:
    value = config.get("output", {}).get(key, default)
    resolved = resolve_project_path(value, root)
    return resolved if resolved is not None else root / default


def _ensure_api_calls_disabled(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls") or config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError("Plot generation never calls APIs; allow_api_calls must be false.")


def _append_summary(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    plot_name: str,
    status: str,
    reason: str,
    output_path: str | None,
    source_files: Iterable[str],
    timestamp: str,
) -> None:
    rows.append(
        {
            "dataset": dataset,
            "plot_name": plot_name,
            "status": status,
            "reason": reason,
            "output_path": output_path or "",
            "source_files": ";".join(sorted(str(source) for source in source_files if str(source))),
            "timestamp": timestamp,
        }
    )


def _write_markdown_section(
    path: Path,
    *,
    dataset: str,
    config_path: str,
    project_root: Path,
    summary_path: Path,
    plot_summary: pd.DataFrame,
) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else "# Results Analysis\n"
    marker = "## Plot Generation"
    intro = (
        "Plots are generated only from existing CSV files. Metric plots require rows "
        "with explicit `metric` and `value` columns; readiness summaries are plotted "
        "separately and are not treated as metric values."
    )
    generated = plot_summary.loc[plot_summary["status"] == "generated"] if not plot_summary.empty else pd.DataFrame()
    skipped = plot_summary.loc[plot_summary["status"] == "skipped"] if not plot_summary.empty else pd.DataFrame()
    dataset_block_lines = [
        f"### Dataset: `{dataset}`",
        "",
        f"- Config: `{config_path}`",
        f"- Plot summary CSV: `{_display_path(summary_path, project_root)}`",
        f"- Generated plots: `{len(generated)}`",
        f"- Skipped plots: `{len(skipped)}`",
        "",
    ]

    if generated.empty:
        dataset_block_lines.append("No metric plots were generated from real metric CSV rows.")
    else:
        dataset_block_lines.extend(["Generated plot files:", ""])
        for _, row in generated.iterrows():
            dataset_block_lines.append(f"- `{row['plot_name']}` -> `{row['output_path']}`")

    if not skipped.empty:
        dataset_block_lines.extend(["", "Skipped plot requests:", ""])
        for _, row in skipped.iterrows():
            dataset_block_lines.append(f"- `{row['plot_name']}`: {row['reason']}")

    dataset_block = "\n".join(dataset_block_lines).rstrip()
    if marker in existing:
        prefix, section = existing.split(marker, 1)
        prefix = prefix.rstrip()
        blocks = _existing_plot_dataset_blocks(section)
    else:
        prefix = existing.rstrip()
        blocks = []

    dataset_heading = f"### Dataset: `{dataset}`"
    blocks = [block for block in blocks if not block.startswith(dataset_heading)]
    blocks.append(dataset_block)

    lines = [prefix, "", marker, "", intro, "", "\n\n".join(blocks)]
    path.write_text("\n".join(part for part in lines if part is not None).rstrip() + "\n", encoding="utf-8")


def _existing_plot_dataset_blocks(section: str) -> list[str]:
    blocks: list[str] = []
    for block in re.split(r"\n(?=### Dataset: `)", section.strip()):
        block = block.strip()
        if block.startswith("### Dataset: `"):
            blocks.append(block)
    return blocks


def _normalize_metric_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("@", "_at_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    try:
        root = find_project_root()
        plot_summary, summary_path, analysis_path = run_make_plots(args.config, project_root=root)
    except (ConfigError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    generated = int((plot_summary["status"] == "generated").sum()) if not plot_summary.empty else 0
    skipped = int((plot_summary["status"] == "skipped").sum()) if not plot_summary.empty else 0
    print(f"summary={_display_path(summary_path, root)}")
    print(f"analysis={_display_path(analysis_path, root)}")
    print(f"generated_plots={generated}")
    print(f"skipped_plots={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

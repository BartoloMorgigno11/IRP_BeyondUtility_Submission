"""Generate and validate the final RQ1 utility figures from frozen G1 evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.final_figures.common import (
    FigureValidationError,
    git_head,
    repository_path,
    sha256_file,
    validate_figure_manifest,
    write_dataframe,
    write_json,
)

SOURCE_UTILITY = "results/tables/g1_primary_ranking_utility.csv"
SOURCE_CONTRACT = "results/manifests/g1_metric_contract_and_provenance.json"
SOURCE_HASHES = {
    SOURCE_UTILITY: "369db87fb19686525135aa1f5cd9d1b1d6d790e4696aad0b70613cb235b33308",
    SOURCE_CONTRACT: "ca7fc85bd346720c8529698ecfc592da557c06dab1494e2f5b60f1a8c5d7abf6",
}
GENERATOR = "src/analysis/final_figures/rq1.py"
OUTPUT_ROOT = "results/final_figures/rq1"
MANIFEST_ROOT = "results/manifests/final_figures"

METRIC_IDS = ["hr_at_5", "ndcg_at_5", "mrr"]
METRIC_LABELS = {"hr_at_5": "HR@5", "ndcg_at_5": "NDCG@5", "mrr": "MRR"}
DATASET_ORDER = ["beauty", "sports", "ml1m", "lastfm", "foodcom"]
DATASET_LABELS = {
    "beauty": "Beauty",
    "sports": "Sports",
    "ml1m": "ML-1M",
    "lastfm": "LastFM",
    "foodcom": "Food.com",
}
MODEL_ORDER = ["qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b"]
MODEL_LABELS = {
    "qwen2:7b-instruct": "Qwen2",
    "mistral:7b-instruct": "Mistral",
    "gemma2:9b": "Gemma2",
}
PROMPT_ORDER = ["BASE", "LLMRANK"]
FIGURE_IDS = ["fig_rq1_utility_landscape_main", "fig_rq1_utility_heatmap_appendix"]
FIXED_TIME = datetime(2026, 9, 20, tzinfo=timezone.utc)


def validate_sources(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Validate canonical hashes and the frozen MRR presentation contract."""

    root = root.resolve()
    for relative, expected in SOURCE_HASHES.items():
        path = repository_path(root, relative)
        if not path.is_file():
            raise FigureValidationError(f"canonical RQ1 source missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise FigureValidationError(
                f"canonical RQ1 source hash mismatch: {relative}: {actual}"
            )

    contract = json.loads(repository_path(root, SOURCE_CONTRACT).read_text(encoding="utf-8"))
    mrr = contract.get("mrr_contract", {})
    if mrr.get("canonical_name") != "mrr":
        raise FigureValidationError("MRR canonical name is not mrr")
    if mrr.get("display_name") != "MRR":
        raise FigureValidationError("MRR display label is not MRR")
    if mrr.get("inspected_ranking_depth") != 10:
        raise FigureValidationError("MRR inspected ranking depth is not 10")
    return dict(SOURCE_HASHES)


def build_figure_values(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Return one exact utility row for each frozen dataset/model/prompt cell."""

    validate_sources(root)
    source = pd.read_csv(
        repository_path(root, SOURCE_UTILITY),
        dtype=str,
        keep_default_na=False,
    )
    required = {"dataset", "model_name", "prompt_variant", "metric", "value", "status"}
    if not required.issubset(source.columns):
        raise FigureValidationError(f"canonical utility columns missing: {sorted(required-set(source.columns))}")
    if set(source["metric"]) != set(METRIC_IDS):
        raise FigureValidationError("canonical utility metrics are not exactly HR@5, NDCG@5, and MRR")
    if not source["status"].eq("COMPUTED").all():
        raise FigureValidationError("canonical utility contains a non-computed row")

    keys = ["dataset", "model_name", "prompt_variant", "metric"]
    if source.duplicated(keys).any():
        raise FigureValidationError("duplicate dataset/model/prompt/metric rows")
    expected_cells = {
        (dataset, model, prompt)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for prompt in PROMPT_ORDER
    }
    actual_cells = set(
        source[["dataset", "model_name", "prompt_variant"]].itertuples(index=False, name=None)
    )
    if actual_cells != expected_cells or len(actual_cells) != 30:
        raise FigureValidationError("RQ1 utility does not contain the exact 30-cell Cartesian grid")
    if len(source) != 90:
        raise FigureValidationError("RQ1 utility must contain exactly 90 metric rows")

    numeric = pd.to_numeric(source["value"], errors="raise")
    if not np.isfinite(numeric).all() or not numeric.between(0.0, 1.0).all():
        raise FigureValidationError("RQ1 utility contains a non-finite or out-of-range value")

    wide = (
        source.pivot(
            index=["dataset", "model_name", "prompt_variant"],
            columns="metric",
            values="value",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    wide["_dataset_order"] = wide["dataset"].map({v: i for i, v in enumerate(DATASET_ORDER)})
    wide["_model_order"] = wide["model_name"].map({v: i for i, v in enumerate(MODEL_ORDER)})
    wide["_prompt_order"] = wide["prompt_variant"].map({v: i for i, v in enumerate(PROMPT_ORDER)})
    wide = wide.sort_values(
        ["_dataset_order", "_model_order", "_prompt_order"], kind="stable"
    ).drop(columns=["_dataset_order", "_model_order", "_prompt_order"])
    wide = wide[["dataset", "model_name", "prompt_variant", *METRIC_IDS]].reset_index(drop=True)
    wide["mrr_display_label"] = "MRR"
    wide["mrr_inspected_depth"] = "10"
    return wide


def validate_values_against_source(values: pd.DataFrame, root: Path = PROJECT_ROOT) -> None:
    """Require figure values to be byte-equivalent at the cell/value level."""

    expected = build_figure_values(root)
    actual = values.astype(str).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
            "svg.hashsalt": "irp-final-figures-v1",
            "pdf.compression": 9,
        }
    )


def _main_figure(values: pd.DataFrame) -> plt.Figure:
    data = values.copy()
    for metric in METRIC_IDS:
        data[metric] = pd.to_numeric(data[metric], errors="raise")

    colors = {
        "qwen2:7b-instruct": "#0072B2",
        "mistral:7b-instruct": "#D55E00",
        "gemma2:9b": "#009E73",
    }
    prompt_styles = {
        "BASE": {"linestyle": "-", "marker": "o"},
        "LLMRANK": {"linestyle": "--", "marker": "s"},
    }
    x = np.arange(len(DATASET_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.6), sharex=True)

    for axis, metric in zip(axes, METRIC_IDS, strict=True):
        for model in MODEL_ORDER:
            for prompt in PROMPT_ORDER:
                subset = data.loc[
                    data["model_name"].eq(model) & data["prompt_variant"].eq(prompt)
                ].set_index("dataset").loc[DATASET_ORDER]
                axis.plot(
                    x,
                    subset[metric].to_numpy(),
                    color=colors[model],
                    linewidth=1.8,
                    markersize=5,
                    label=f"{MODEL_LABELS[model]} / {prompt}",
                    **prompt_styles[prompt],
                )
        metric_max = float(data[metric].max())
        upper = max(0.1, math.ceil(metric_max * 10.0) / 10.0 + 0.05)
        axis.set_ylim(0.0, min(1.0, upper))
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(x, [DATASET_LABELS[item] for item in DATASET_ORDER], rotation=25, ha="right")
        axis.set_ylabel("Cell mean")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("RQ1 utility varies across domains, models, and prompt strategies", y=0.99)
    fig.text(
        0.5,
        0.015,
        "Frozen G1 fail-strict cell means. MRR inspects ranks 1-10. Local adapted evidence; no paper-exact identity claim.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.18, top=0.76, wspace=0.28)
    return fig


def _heatmap_figure(values: pd.DataFrame) -> plt.Figure:
    data = values.copy()
    for metric in METRIC_IDS:
        data[metric] = pd.to_numeric(data[metric], errors="raise")

    row_keys = [(dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER]
    row_labels = [
        f"{DATASET_LABELS[dataset]} / {MODEL_LABELS[model]}" for dataset, model in row_keys
    ]
    fig = plt.figure(figsize=(12.4, 9.0))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        left=0.19,
        right=0.965,
        bottom=0.07,
        top=0.92,
        wspace=0.34,
    )
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(fig.add_subplot(grid[0, index], sharey=axes[0]) for index in [1, 2])
    colorbar_axis = fig.add_subplot(grid[0, 3])
    last_image = None
    for axis, metric in zip(axes, METRIC_IDS, strict=True):
        matrix = np.empty((len(row_keys), len(PROMPT_ORDER)), dtype=float)
        for row_index, (dataset, model) in enumerate(row_keys):
            for column_index, prompt in enumerate(PROMPT_ORDER):
                selected = data.loc[
                    data["dataset"].eq(dataset)
                    & data["model_name"].eq(model)
                    & data["prompt_variant"].eq(prompt),
                    metric,
                ]
                if len(selected) != 1:
                    raise FigureValidationError("heatmap cell does not resolve to one canonical value")
                matrix[row_index, column_index] = float(selected.iloc[0])
        last_image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(range(len(PROMPT_ORDER)), PROMPT_ORDER)
        axis.set_yticks(range(len(row_labels)), row_labels)
        axis.tick_params(axis="y", labelsize=7, labelleft=axis is axes[0])
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.55 else "black",
                )
        for dataset_index in range(1, len(DATASET_ORDER)):
            axis.axhline(dataset_index * len(MODEL_ORDER) - 0.5, color="white", linewidth=1.2)

    assert last_image is not None
    colorbar = fig.colorbar(last_image, cax=colorbar_axis)
    colorbar.set_label("Metric value (absolute 0-1 scale)")
    fig.suptitle("RQ1 complete utility matrix: 30 dataset x model x prompt cells", y=0.985)
    fig.text(
        0.5,
        0.015,
        "Annotated values are unnormalized canonical cell means. MRR inspects ranks 1-10.",
        ha="center",
        fontsize=8,
    )

    return fig


def _save_figure(fig: plt.Figure, artifact_root: Path, figure_id: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for extension in ["png", "pdf", "svg"]:
        relative = f"{OUTPUT_ROOT}/figures/{figure_id}.{extension}"
        path = repository_path(artifact_root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if extension == "png":
            metadata = {"Software": "IRP deterministic final-figure pipeline"}
        elif extension == "pdf":
            metadata = {
                "Creator": "IRP deterministic final-figure pipeline",
                "Producer": "Matplotlib",
                "CreationDate": FIXED_TIME,
                "ModDate": FIXED_TIME,
            }
        else:
            metadata = {
                "Creator": "IRP deterministic final-figure pipeline",
                "Date": "2026-09-20",
            }
        fig.savefig(path, dpi=180, metadata=metadata, facecolor="white")
        outputs[extension] = relative
        outputs[f"{extension}_sha256"] = sha256_file(path)
    plt.close(fig)
    return outputs


def _manifest(
    *,
    root: Path,
    artifact_root: Path,
    figure_id: str,
    purpose: str,
    interpretation: str,
    limitations: list[str],
    values_path: str,
    outputs: dict[str, str],
    created_from_commit: str,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "figure_id": figure_id,
        "rq": "RQ1",
        "scientific_purpose": purpose,
        "canonical_source_artifacts": list(SOURCE_HASHES),
        "canonical_source_sha256": dict(SOURCE_HASHES),
        "figure_values_csv": values_path,
        "figure_values_sha256": sha256_file(repository_path(artifact_root, values_path)),
        "generator": GENERATOR,
        "generator_sha256": sha256_file(repository_path(root, GENERATOR)),
        "output_png": outputs["png"],
        "output_png_sha256": outputs["png_sha256"],
        "output_pdf": outputs["pdf"],
        "output_pdf_sha256": outputs["pdf_sha256"],
        "output_svg": outputs["svg"],
        "output_svg_sha256": outputs["svg_sha256"],
        "scientific_interpretation": interpretation,
        "limitations": limitations,
        "created_from_commit": created_from_commit,
        "metric_contract": {
            "HR@5": "hr_at_5",
            "NDCG@5": "ndcg_at_5",
            "MRR": "mrr",
            "MRR_INSPECTED_DEPTH": 10,
        },
        "cell_contract": {
            "configuration_cells": 30,
            "datasets": DATASET_ORDER,
            "models": MODEL_ORDER,
            "prompt_strategies": PROMPT_ORDER,
        },
    }
    return manifest


def generate_all(
    root: Path = PROJECT_ROOT,
    *,
    artifact_root: Path | None = None,
    created_from_commit: str | None = None,
) -> list[dict[str, object]]:
    """Generate both RQ1 figures and their complete provenance manifests."""

    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    validate_sources(root)
    values = build_figure_values(root)
    created_from_commit = created_from_commit or git_head(root)
    if not created_from_commit or len(created_from_commit) != 40:
        raise FigureValidationError("created_from_commit must be a full commit hash")
    _configure_matplotlib()

    specifications: list[
        tuple[str, Callable[[pd.DataFrame], plt.Figure], str, str, list[str]]
    ] = [
        (
            "fig_rq1_utility_landscape_main",
            _main_figure,
            "Summarize utility variation across datasets, models, and prompt strategies for HR@5, NDCG@5, and MRR.",
            "The frozen local G1 evidence shows domain-, model-, and prompt-dependent utility without defining an overall universal winner.",
            [
                "MRR is the canonical label and inspects ranking depth 10.",
                "Metrics are shown separately and are not combined into a synthetic score.",
                "This is local adapted evidence, not paper-exact numerical reproduction.",
            ],
        ),
        (
            "fig_rq1_utility_heatmap_appendix",
            _heatmap_figure,
            "Expose all 30 frozen dataset-by-model-by-prompt utility cells for external inspection.",
            "The appendix preserves the full canonical G1 utility matrix using absolute, annotated metric values.",
            [
                "Color is mapped to the absolute 0-1 metric scale; annotated values retain canonical precision in the values CSV.",
                "MRR is the canonical label and inspects ranking depth 10.",
                "This is local adapted evidence, not paper-exact numerical reproduction.",
            ],
        ),
    ]

    manifests: list[dict[str, object]] = []
    for figure_id, builder, purpose, interpretation, limitations in specifications:
        values_relative = f"{OUTPUT_ROOT}/values/{figure_id}_values.csv"
        values_path = repository_path(artifact_root, values_relative)
        write_dataframe(values_path, values)
        validate_values_against_source(
            pd.read_csv(values_path, dtype=str, keep_default_na=False), root
        )
        outputs = _save_figure(builder(values), artifact_root, figure_id)
        manifest = _manifest(
            root=root,
            artifact_root=artifact_root,
            figure_id=figure_id,
            purpose=purpose,
            interpretation=interpretation,
            limitations=limitations,
            values_path=values_relative,
            outputs=outputs,
            created_from_commit=created_from_commit,
        )
        manifest_relative = f"{MANIFEST_ROOT}/{figure_id}_manifest.json"
        manifest_path = repository_path(artifact_root, manifest_relative)
        write_json(manifest_path, manifest)
        validate_figure_manifest(root, manifest_path, artifact_root=artifact_root)
        manifests.append(manifest)
    return manifests


def validate_all(
    root: Path = PROJECT_ROOT,
    *,
    artifact_root: Path | None = None,
) -> list[dict[str, object]]:
    """Validate both RQ1 manifests and exact values against frozen G1 evidence."""

    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    validate_sources(root)
    validated: list[dict[str, object]] = []
    for figure_id in FIGURE_IDS:
        path = repository_path(
            artifact_root, f"{MANIFEST_ROOT}/{figure_id}_manifest.json"
        )
        manifest = validate_figure_manifest(root, path, artifact_root=artifact_root)
        values = pd.read_csv(
            repository_path(artifact_root, manifest["figure_values_csv"]),
            dtype=str,
            keep_default_na=False,
        )
        validate_values_against_source(values, root)
        validated.append(manifest)
    return validated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generate", "validate"], required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Optional physical staging root; manifest paths remain repository-relative.",
    )
    parser.add_argument("--created-from-commit")
    args = parser.parse_args(argv)
    if args.mode == "generate":
        result = generate_all(
            PROJECT_ROOT,
            artifact_root=args.artifact_root,
            created_from_commit=args.created_from_commit,
        )
    else:
        result = validate_all(PROJECT_ROOT, artifact_root=args.artifact_root)
    print(json.dumps({"status": "PASS", "figures": [item["figure_id"] for item in result]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

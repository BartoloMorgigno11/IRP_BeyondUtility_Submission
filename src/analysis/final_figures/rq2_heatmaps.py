"""Generate and validate the four final RQ2 appendix heatmaps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

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
from src.analysis.final_figures.rq2 import (
    DATASET_LABELS,
    DATASET_ORDER,
    METRIC_LABELS,
    MODEL_LABELS,
    MODEL_ORDER,
    PRE_FIX_HASH,
    PROMPT_ORDER,
    SOURCE_BEYOND,
    SOURCE_COLUMNS,
    SOURCE_HASHES,
    load_and_validate_canonical,
    validate_sources,
)

GENERATOR = "src/analysis/final_figures/rq2_heatmaps.py"
CONTRACT_DEPENDENCY = "src/analysis/final_figures/rq2.py"
OUTPUT_ROOT = "results/final_figures/rq2"
MANIFEST_ROOT = "results/manifests/final_figures"
FIXED_TIME = datetime(2026, 9, 20, tzinfo=timezone.utc)

COLUMN_ORDER = [(model, prompt) for model in MODEL_ORDER for prompt in PROMPT_ORDER]
COLUMN_LABELS = [
    f"{MODEL_LABELS[model]}\n{prompt}" for model, prompt in COLUMN_ORDER
]

HEATMAP_SPECS: dict[str, dict[str, Any]] = {
    "fig_rq2_llm_longtail_novelty_heatmap": {
        "metrics": ["aplt_at_5", "arp_at_5", "popreo_at_5", "self_information_at_5"],
        "ncols": 2,
        "title": "RQ2 long-tail, popularity, and novelty: complete configuration heatmaps",
        "purpose": "Expose all 30 canonical dataset-model-prompt cells for APLT@5, ARP@5, PopREO@5, and self-information@5.",
        "interpretation": "The panels preserve raw values for four distinct constructs; their independent color scales support within-metric inspection only.",
        "limitations": [
            "APLT@5 is canonical; calACLT is only an implementation alias and ACLT is not a separate result metric.",
            "Food.com PopREO@5 has six DEGENERATE_PROTOCOL_INPUT cells shown as N/C and never as zero.",
            "Each metric has an independent linear color normalization and raw value annotations.",
            "No cross-metric aggregate or direction reversal is applied.",
        ],
        "footer": "Raw values are printed in every computed cell. Color normalization is independent per metric; N/C means noncomputed.",
    },
    "fig_rq2_llm_serendipity_coverage_heatmap": {
        "metrics": ["serendipity_at_5", "item_coverage_at_5", "catalog_coverage_at_5"],
        "ncols": 3,
        "title": "RQ2 corrected serendipity and distinct coverage: complete configuration heatmaps",
        "purpose": "Expose all 30 canonical cells for corrected serendipity@5 and the two distinct coverage constructs.",
        "interpretation": "Corrected serendipity, item coverage, and catalog coverage remain separate metrics with distinct meanings and denominators.",
        "limitations": [
            "Only the corrected canonical serendipity source is used; historical pre-fix evidence is excluded.",
            "Item coverage uses the prompt-compatible candidate union; catalog coverage uses the catalog universe.",
            "Each metric has an independent linear color normalization and raw value annotations.",
            "Coverage metrics are not collapsed or averaged.",
        ],
        "footer": "Corrected canonical serendipity only. Item and catalog coverage remain separate; colors are normalized per metric.",
    },
    "fig_rq2_llm_diversity_distribution_heatmap": {
        "metrics": [
            "ild_at_5",
            "shannon_category_entropy_at_5",
            "gini_at_5",
            "dpd_at_5",
            "jain_index_at_5",
            "oic_at_5",
        ],
        "ncols": 3,
        "title": "RQ2 diversity and distribution: complete configuration heatmaps",
        "purpose": "Expose all 30 canonical cells for ILD@5, category entropy@5, Gini@5, DPD@5, Jain index@5, and OIC@5.",
        "interpretation": "Six distinct distributional metrics are retained on their original scales with no synthetic aggregate.",
        "limitations": [
            "Every panel uses raw canonical values and an independent linear color normalization.",
            "Color intensity is comparable within a panel only, never across metrics.",
            "No metric direction is reversed and no averaging across incompatible constructs is performed.",
        ],
        "footer": "Raw values are annotated; independent per-metric color scales are display-only. No direction reversal or aggregate.",
    },
    "fig_rq2_llm_calibration_heatmap": {
        "metrics": ["calibration_at_5"],
        "ncols": 1,
        "title": "RQ2 Calibration@5: complete eligible-configuration heatmap",
        "purpose": "Expose all 30 canonical Calibration@5 cells with eligible and undefined user counts.",
        "interpretation": "Calibration@5 is the canonical eligible-user mean; every cell retains its eligible and undefined-user accounting.",
        "limitations": [
            "Calibration is averaged only over users with both history and recommendation feature distributions.",
            "Cell annotations report the raw value plus eligible (e) and undefined (u) user counts.",
            "N/C is reserved for a noncomputed configuration and is visually distinct from numeric zero; no value is imputed.",
            "All 30 frozen configurations are computed, although user-level eligibility is partial where u is nonzero.",
        ],
        "footer": "Each cell shows raw Calibration@5, eligible users (e), and undefined users (u). N/C would denote noncomputed, never zero.",
    },
}
HEATMAP_IDS = list(HEATMAP_SPECS)


def validate_heatmap_sources(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Validate the corrected RQ2 source and the shared frozen metric contract."""

    validated = validate_sources(root)
    if validated != SOURCE_HASHES:
        raise FigureValidationError("RQ2 source validation returned an unexpected identity")
    dependency = repository_path(root, CONTRACT_DEPENDENCY)
    if not dependency.is_file():
        raise FigureValidationError("frozen RQ2 contract implementation is missing")
    return validated


def _format_numeric(value: float, maximum: float) -> str:
    magnitude = abs(maximum)
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    if magnitude < 0.1:
        return f"{value:.4f}"
    return f"{value:.3f}"


def build_heatmap_values(figure_id: str, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Build a deterministic long-form heatmap table from exact canonical rows."""

    if figure_id not in HEATMAP_SPECS:
        raise FigureValidationError(f"unknown RQ2 heatmap: {figure_id}")
    source = load_and_validate_canonical(root)
    metrics = HEATMAP_SPECS[figure_id]["metrics"]
    frame = source.loc[source["metric"].isin(metrics), SOURCE_COLUMNS].copy()
    frame = frame.rename(columns={"metric": "metric_id"})
    frame["metric_display_label"] = frame["metric_id"].map(METRIC_LABELS)
    frame["heatmap_row"] = frame["dataset"].map(
        {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    )
    column_map = {pair: index for index, pair in enumerate(COLUMN_ORDER)}
    frame["heatmap_column"] = [
        column_map[(model, prompt)]
        for model, prompt in zip(frame["model_name"], frame["prompt_variant"], strict=True)
    ]
    frame["configuration_label"] = [
        COLUMN_LABELS[index] for index in frame["heatmap_column"].astype(int)
    ]
    frame["display_transform"] = "PER_METRIC_LINEAR_COLOR_NORMALIZATION_RAW_ANNOTATIONS"
    frame["eligible_user_count"] = ""
    frame["cell_annotation"] = ""

    for metric in metrics:
        metric_mask = frame["metric_id"].eq(metric)
        computed = frame.loc[metric_mask & frame["status"].eq("COMPUTED"), "value"]
        numeric = pd.to_numeric(computed, errors="raise")
        maximum = float(numeric.abs().max()) if len(numeric) else 0.0
        for index in frame.index[metric_mask]:
            status = frame.at[index, "status"]
            if status != "COMPUTED":
                frame.at[index, "cell_annotation"] = "N/C"
                continue
            value = float(frame.at[index, "value"])
            label = _format_numeric(value, maximum)
            if metric == "calibration_at_5":
                payload = json.loads(frame.at[index, "value_json"])
                eligible = int(payload["defined_user_count"])
                undefined = int(frame.at[index, "undefined_user_count"])
                frame.at[index, "eligible_user_count"] = str(eligible)
                label = f"{label}\ne={eligible}\nu={undefined}"
            frame.at[index, "cell_annotation"] = label

    frame["_metric_order"] = frame["metric_id"].map(
        {metric: index for index, metric in enumerate(metrics)}
    )
    frame = frame.sort_values(
        ["_metric_order", "heatmap_row", "heatmap_column"], kind="stable"
    ).drop(columns=["_metric_order"])
    return frame.reset_index(drop=True)


def validate_heatmap_values(
    figure_id: str, values: pd.DataFrame, root: Path = PROJECT_ROOT
) -> None:
    """Require every stored heatmap row and value to equal the canonical derivation."""

    expected = build_heatmap_values(figure_id, root).astype(str).reset_index(drop=True)
    actual = values.astype(str).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    metrics = HEATMAP_SPECS[figure_id]["metrics"]
    if len(actual) != 30 * len(metrics):
        raise FigureValidationError("heatmap does not contain exactly 30 rows per metric")
    if actual.duplicated(["metric_id", "heatmap_row", "heatmap_column"]).any():
        raise FigureValidationError("heatmap contains duplicate cells")


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 11,
            "figure.titlesize": 14,
            "svg.hashsalt": "irp-final-figures-v1",
            "pdf.compression": 9,
        }
    )


def _metric_matrix(metric_rows: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
    matrix = np.full((len(DATASET_ORDER), len(COLUMN_ORDER)), np.nan, dtype=float)
    annotations = [["" for _ in COLUMN_ORDER] for _ in DATASET_ORDER]
    for row in metric_rows.to_dict("records"):
        row_index = int(row["heatmap_row"])
        column_index = int(row["heatmap_column"])
        annotations[row_index][column_index] = str(row["cell_annotation"])
        if row["status"] == "COMPUTED":
            matrix[row_index, column_index] = float(row["value"])
    return matrix, annotations


def _text_color(cmap: Any, norm: Normalize, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.52 else "white"


def _heatmap_figure(figure_id: str, values: pd.DataFrame) -> plt.Figure:
    spec = HEATMAP_SPECS[figure_id]
    metrics = spec["metrics"]
    ncols = int(spec["ncols"])
    nrows = math.ceil(len(metrics) / ncols)
    if ncols == 1:
        figsize = (12.8, 7.2)
    elif nrows == 1:
        figsize = (17.0, 6.1)
    elif ncols == 2:
        figsize = (14.8, 10.2)
    else:
        figsize = (17.0, 10.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_list = list(axes.flat)

    for axis, metric in zip(axes_list, metrics, strict=False):
        metric_rows = values.loc[values["metric_id"].eq(metric)]
        matrix, annotations = _metric_matrix(metric_rows)
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            raise FigureValidationError(f"heatmap has no computed values for {metric}")
        minimum = float(finite.min())
        maximum = float(finite.max())
        if math.isclose(minimum, maximum):
            padding = max(abs(maximum) * 0.05, 0.01)
            minimum -= padding
            maximum += padding
        norm = Normalize(vmin=minimum, vmax=maximum)
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad(color="0.82")
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")

        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                text_color = "black" if not np.isfinite(value) else _text_color(cmap, norm, value)
                axis.text(
                    column_index,
                    row_index,
                    annotations[row_index][column_index],
                    ha="center",
                    va="center",
                    fontsize=6.7 if metric == "calibration_at_5" else 7.2,
                    color=text_color,
                    fontweight="bold" if not np.isfinite(value) else "normal",
                    linespacing=1.05,
                )

        axis.set_title(METRIC_LABELS[metric], pad=9)
        axis.set_xticks(range(len(COLUMN_ORDER)), COLUMN_LABELS)
        axis.set_yticks(range(len(DATASET_ORDER)), [DATASET_LABELS[item] for item in DATASET_ORDER])
        axis.tick_params(axis="x", labelrotation=35)
        for label in axis.get_xticklabels():
            label.set_ha("right")
        axis.set_xlabel("Model / prompt")
        axis.set_ylabel("Dataset")
        axis.set_xticks(np.arange(-0.5, len(COLUMN_ORDER), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
        colorbar.set_label("Raw value", fontsize=8)
        colorbar.ax.tick_params(labelsize=7.5)

    for axis in axes_list[len(metrics):]:
        axis.set_visible(False)
    fig.suptitle(spec["title"], y=0.988)
    fig.text(0.5, 0.018, spec["footer"], ha="center", fontsize=8)
    top = 0.90 if nrows == 1 else 0.92
    bottom = 0.20 if nrows == 1 else 0.13
    fig.subplots_adjust(left=0.075, right=0.965, bottom=bottom, top=top, wspace=0.40, hspace=0.47)
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
            metadata = {"Creator": "IRP deterministic final-figure pipeline", "Date": "2026-09-20"}
        fig.savefig(path, dpi=180, metadata=metadata, facecolor="white")
        if extension == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            ) + "\n"
            path.write_text(normalized, encoding="utf-8", newline="\n")
        outputs[extension] = relative
        outputs[f"{extension}_sha256"] = sha256_file(path)
    plt.close(fig)
    return outputs


def _status_counts(values: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for metric, group in values.groupby("metric_id", sort=False):
        result[str(metric)] = {
            str(status): int(count)
            for status, count in group["status"].value_counts(sort=False).items()
        }
    return result


def _eligibility_summary(values: pd.DataFrame) -> dict[str, Any] | None:
    calibration = values.loc[values["metric_id"].eq("calibration_at_5")]
    if calibration.empty:
        return None
    eligible = calibration["eligible_user_count"].astype(int)
    undefined = calibration["undefined_user_count"].astype(int)
    return {
        "configuration_cells": int(len(calibration)),
        "computed_cells": int(calibration["status"].eq("COMPUTED").sum()),
        "eligible_user_count_min": int(eligible.min()),
        "eligible_user_count_max": int(eligible.max()),
        "undefined_user_count_min": int(undefined.min()),
        "undefined_user_count_max": int(undefined.max()),
        "cell_annotation_contract": "RAW_VALUE_PLUS_ELIGIBLE_E_AND_UNDEFINED_U_COUNTS",
    }


def generate_all(
    root: Path = PROJECT_ROOT,
    *,
    artifact_root: Path | None = None,
    created_from_commit: str | None = None,
) -> list[dict[str, Any]]:
    """Generate four RQ2 appendix heatmaps and their complete hash manifests."""

    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    validate_heatmap_sources(root)
    created_from_commit = created_from_commit or git_head(root)
    if not created_from_commit or len(created_from_commit) != 40:
        raise FigureValidationError("created_from_commit must be a full commit hash")
    _configure_matplotlib()

    manifests: list[dict[str, Any]] = []
    for figure_id, spec in HEATMAP_SPECS.items():
        values = build_heatmap_values(figure_id, root)
        values_relative = f"{OUTPUT_ROOT}/values/{figure_id}_values.csv"
        values_path = repository_path(artifact_root, values_relative)
        write_dataframe(values_path, values)
        validate_heatmap_values(
            figure_id,
            pd.read_csv(values_path, dtype=str, keep_default_na=False),
            root,
        )
        outputs = _save_figure(_heatmap_figure(figure_id, values), artifact_root, figure_id)
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "figure_id": figure_id,
            "rq": "RQ2",
            "scientific_purpose": spec["purpose"],
            "canonical_source_artifacts": list(SOURCE_HASHES),
            "canonical_source_sha256": dict(SOURCE_HASHES),
            "figure_values_csv": values_relative,
            "figure_values_sha256": sha256_file(values_path),
            "generator": GENERATOR,
            "generator_sha256": sha256_file(repository_path(root, GENERATOR)),
            "generator_dependencies": {
                CONTRACT_DEPENDENCY: sha256_file(repository_path(root, CONTRACT_DEPENDENCY))
            },
            "output_png": outputs["png"],
            "output_png_sha256": outputs["png_sha256"],
            "output_pdf": outputs["pdf"],
            "output_pdf_sha256": outputs["pdf_sha256"],
            "output_svg": outputs["svg"],
            "output_svg_sha256": outputs["svg_sha256"],
            "scientific_interpretation": spec["interpretation"],
            "limitations": spec["limitations"],
            "created_from_commit": created_from_commit,
            "configuration_contract": {
                "configuration_cells_per_metric": 30,
                "datasets": DATASET_ORDER,
                "columns": [
                    {"model_name": model, "prompt_variant": prompt}
                    for model, prompt in COLUMN_ORDER
                ],
            },
            "metric_ids": spec["metrics"],
            "display_contract": {
                "panel_color_transform": "PER_METRIC_LINEAR_NORMALIZATION",
                "cell_annotations": "RAW_CANONICAL_VALUES",
                "cross_metric_color_comparison": "NOT_VALID",
                "cross_metric_aggregate": False,
                "direction_reversal": False,
                "noncomputed_label": "N/C",
                "numeric_zero_label": "NUMERIC_ZERO_FORMATTED_AS_A_NUMBER",
                "imputation": False,
            },
            "source_status_counts": _status_counts(values),
            "calibration_eligibility": _eligibility_summary(values),
        }
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
) -> list[dict[str, Any]]:
    """Validate all four RQ2 heatmap source-value-image-manifest chains."""

    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    validate_heatmap_sources(root)
    dependency_hash = sha256_file(repository_path(root, CONTRACT_DEPENDENCY))
    validated: list[dict[str, Any]] = []
    for figure_id in HEATMAP_IDS:
        manifest_path = repository_path(
            artifact_root, f"{MANIFEST_ROOT}/{figure_id}_manifest.json"
        )
        manifest_text = manifest_path.read_text(encoding="utf-8")
        if PRE_FIX_HASH in manifest_text:
            raise FigureValidationError("pre-fix source appears in a foreground RQ2 heatmap manifest")
        manifest = validate_figure_manifest(root, manifest_path, artifact_root=artifact_root)
        if manifest.get("generator_dependencies", {}).get(CONTRACT_DEPENDENCY) != dependency_hash:
            raise FigureValidationError("RQ2 heatmap contract dependency hash mismatch")
        if manifest.get("metric_ids") != HEATMAP_SPECS[figure_id]["metrics"]:
            raise FigureValidationError("RQ2 heatmap metric contract mismatch")
        display = manifest.get("display_contract", {})
        if display.get("direction_reversal") is not False or display.get("cross_metric_aggregate") is not False:
            raise FigureValidationError("RQ2 heatmap display contract changed")
        values = pd.read_csv(
            repository_path(artifact_root, manifest["figure_values_csv"]),
            dtype=str,
            keep_default_na=False,
        )
        validate_heatmap_values(figure_id, values, root)
        validated.append(manifest)
    return validated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generate", "validate"], required=True)
    parser.add_argument("--artifact-root", type=Path)
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

"""Generate and validate the four final RQ2 beyond-utility landscape figures."""

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

SOURCE_BEYOND = "results/tables/g1_primary_ranking_beyond_utility.csv"
SOURCE_CONTRACT = "results/manifests/g1_metric_contract_and_provenance.json"
SOURCE_HASHES = {
    SOURCE_BEYOND: "2e5c774f552cb47f16622f2a989ce033ee05036ad814558926b40148e8857c5b",
    SOURCE_CONTRACT: "ca7fc85bd346720c8529698ecfc592da557c06dab1494e2f5b60f1a8c5d7abf6",
}
PRE_FIX_HASH = "15ca9025aada73feba7513e06756b6c3ede42542befd82c1a76dbc757b018ae1"
GENERATOR = "src/analysis/final_figures/rq2.py"
OUTPUT_ROOT = "results/final_figures/rq2"
MANIFEST_ROOT = "results/manifests/final_figures"
FIXED_TIME = datetime(2026, 9, 20, tzinfo=timezone.utc)

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

METRIC_LABELS = {
    "aplt_at_5": "APLT@5",
    "arp_at_5": "ARP@5",
    "popreo_at_5": "PopREO@5",
    "self_information_at_5": "Self-information@5",
    "serendipity_at_5": "Serendipity@5 (corrected)",
    "item_coverage_at_5": "Item coverage@5",
    "catalog_coverage_at_5": "Catalog coverage@5",
    "ild_at_5": "ILD@5",
    "shannon_category_entropy_at_5": "Category entropy@5",
    "gini_at_5": "Gini@5",
    "dpd_at_5": "DPD@5",
    "jain_index_at_5": "Jain index@5",
    "oic_at_5": "OIC@5",
    "calibration_at_5": "Calibration@5",
}
ALL_METRICS = list(METRIC_LABELS)

FIGURE_SPECS: dict[str, dict[str, Any]] = {
    "fig_rq2_llm_longtail_novelty_landscape": {
        "metrics": ["aplt_at_5", "arp_at_5", "popreo_at_5", "self_information_at_5"],
        "ncols": 2,
        "title": "RQ2 long-tail, popularity, and novelty landscape",
        "purpose": "Summarize canonical APLT@5, ARP@5, PopREO@5, and self-information@5 across datasets, models, and prompts.",
        "interpretation": "Long-tail, popularity, and novelty behavior is multidimensional and configuration-dependent; no synthetic aggregate or universal winner is defined.",
        "limitations": [
            "APLT@5 is the canonical metric; calACLT is only an implementation alias and ACLT is not a separate result metric.",
            "Food.com PopREO@5 has six DEGENERATE_PROTOCOL_INPUT cells and is displayed as noncomputed, never as zero.",
            "Each metric uses its own raw-value axis; cross-panel heights are not directly comparable.",
            "This is local adapted evidence, not paper-exact numerical reproduction.",
        ],
        "footer": "Raw canonical values; independent y-axis per metric. Food.com PopREO is N/C for all six configurations.",
    },
    "fig_rq2_llm_serendipity_coverage_landscape": {
        "metrics": ["serendipity_at_5", "item_coverage_at_5", "catalog_coverage_at_5"],
        "ncols": 3,
        "title": "RQ2 corrected serendipity and distinct coverage constructs",
        "purpose": "Summarize corrected serendipity@5, item coverage@5, and catalog coverage@5 across datasets, models, and prompts.",
        "interpretation": "Corrected serendipity and the two coverage constructs vary by configuration and retain distinct denominators and meanings.",
        "limitations": [
            "Serendipity uses only the corrected canonical G1 beyond-utility artifact.",
            "Item coverage uses the prompt-compatible candidate union; catalog coverage uses the catalog universe.",
            "Coverage metrics are not collapsed into a generic coverage value.",
            "Each metric uses its own raw-value axis; cross-panel heights are not directly comparable.",
        ],
        "footer": "Corrected canonical serendipity. Item coverage and catalog coverage retain distinct denominators.",
    },
    "fig_rq2_llm_diversity_distribution_landscape": {
        "metrics": [
            "ild_at_5",
            "shannon_category_entropy_at_5",
            "gini_at_5",
            "dpd_at_5",
            "jain_index_at_5",
            "oic_at_5",
        ],
        "ncols": 3,
        "title": "RQ2 diversity and distribution landscape",
        "purpose": "Summarize canonical ILD@5, category entropy@5, Gini@5, DPD@5, Jain index@5, and OIC@5.",
        "interpretation": "The frozen distributional metrics retain their original numerical definitions and reveal configuration-dependent behavior on different scales.",
        "limitations": [
            "No cross-metric normalization or direction reversal is applied.",
            "Each panel has an independent raw-value y-axis and cannot be compared by visual height across panels.",
            "No synthetic diversity or fairness score is constructed.",
        ],
        "footer": "No normalization or direction reversal; every panel shows its original raw metric scale.",
    },
    "fig_rq2_llm_calibration_landscape": {
        "metrics": ["calibration_at_5"],
        "ncols": 1,
        "title": "RQ2 Calibration@5 across eligible G1 configurations",
        "purpose": "Summarize canonical Calibration@5 while preserving per-cell user eligibility and undefined-user accounting.",
        "interpretation": "Calibration@5 is shown only where the canonical artifact reports a computed eligible-user mean; no missing value is imputed.",
        "limitations": [
            "Calibration is averaged over users with both history and recommendation feature distributions.",
            "Eligible and undefined-user counts are preserved in the figure-values CSV.",
            "All 30 canonical cells are computed, but eligibility is partial in cells with undefined users; Food.com has 260-273 eligible users per cell.",
            "A noncomputed cell would be shown as N/C, never as zero.",
        ],
        "footer": "Computed over eligible users only; exact eligible and undefined-user counts are preserved in the values CSV.",
    },
}
FIGURE_IDS = list(FIGURE_SPECS)

SOURCE_COLUMNS = [
    "result_id",
    "dataset",
    "model_name",
    "prompt_variant",
    "metric",
    "value",
    "value_json",
    "status",
    "planned_users",
    "undefined_user_count",
    "source_recommendation_hash",
    "reason",
]


def validate_sources(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Validate corrected source identity and the APLT/serendipity authority contract."""

    root = root.resolve()
    for relative, expected in SOURCE_HASHES.items():
        path = repository_path(root, relative)
        if not path.is_file():
            raise FigureValidationError(f"canonical RQ2 source missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise FigureValidationError(f"canonical RQ2 source hash mismatch: {relative}: {actual}")

    contract = json.loads(repository_path(root, SOURCE_CONTRACT).read_text(encoding="utf-8"))
    authority = contract.get("canonical_artifact_authority", {})
    if authority.get("canonical_beyond_utility_sha256") != SOURCE_HASHES[SOURCE_BEYOND]:
        raise FigureValidationError("contract does not authorize the corrected beyond-utility source")
    if authority.get("historical_pre_fix_sha256") != PRE_FIX_HASH:
        raise FigureValidationError("historical pre-fix identity is not frozen by the contract")
    if authority.get("canonical_beyond_utility_sha256") == PRE_FIX_HASH:
        raise FigureValidationError("historical pre-fix source selected as canonical")
    long_tail = contract.get("long_tail_contract", {})
    expected_alias = {
        "canonical_metric": "APLT@5",
        "canonical_metric_id": "aplt_at_5",
        "implementation_alias": "calACLT",
        "aclt_is_separate_metric": False,
    }
    for key, expected in expected_alias.items():
        if long_tail.get(key) != expected:
            raise FigureValidationError(f"APLT contract mismatch: {key}")
    return dict(SOURCE_HASHES)


def load_and_validate_canonical(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load the exact 30-cell x 14-metric corrected G1 evidence universe."""

    validate_sources(root)
    source = pd.read_csv(
        repository_path(root, SOURCE_BEYOND), dtype=str, keep_default_na=False
    )
    missing_columns = set(SOURCE_COLUMNS) - set(source.columns)
    if missing_columns:
        raise FigureValidationError(f"canonical RQ2 columns missing: {sorted(missing_columns)}")
    if set(source["metric"]) != set(ALL_METRICS) or len(source) != 420:
        raise FigureValidationError("canonical RQ2 source is not the exact 14-metric universe")

    keys = ["dataset", "model_name", "prompt_variant", "metric"]
    if source.duplicated(keys).any():
        raise FigureValidationError("duplicate RQ2 configuration/metric rows")
    expected_cells = {
        (dataset, model, prompt)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for prompt in PROMPT_ORDER
    }
    actual_cells = set(source[keys[:3]].itertuples(index=False, name=None))
    if actual_cells != expected_cells or len(actual_cells) != 30:
        raise FigureValidationError("RQ2 source is not the exact 30-cell configuration universe")
    if not source.groupby("metric", sort=False).size().eq(30).all():
        raise FigureValidationError("each RQ2 metric must contain exactly 30 cells")

    computed = source["status"].eq("COMPUTED")
    if source.loc[computed, "value"].eq("").any():
        raise FigureValidationError("computed RQ2 cell has an empty value")
    if source.loc[~computed, "value"].ne("").any():
        raise FigureValidationError("noncomputed RQ2 cell contains an imputed value")
    numeric = pd.to_numeric(source.loc[computed, "value"], errors="raise")
    if not np.isfinite(numeric).all():
        raise FigureValidationError("computed RQ2 cell is non-finite")

    popreo = source.loc[source["metric"].eq("popreo_at_5")]
    noncomputed = popreo.loc[~popreo["status"].eq("COMPUTED")]
    if len(noncomputed) != 6 or set(noncomputed["dataset"]) != {"foodcom"}:
        raise FigureValidationError("Food.com PopREO must contain exactly six noncomputed cells")
    if not noncomputed["status"].eq("DEGENERATE_PROTOCOL_INPUT").all():
        raise FigureValidationError("Food.com PopREO noncomputed status changed")
    if not noncomputed["value"].eq("").all():
        raise FigureValidationError("Food.com PopREO was imputed")

    calibration = source.loc[source["metric"].eq("calibration_at_5")]
    for row in calibration.to_dict("records"):
        payload = json.loads(row["value_json"])
        defined = int(payload.get("defined_user_count", 0))
        planned = int(row["planned_users"])
        undefined = int(row["undefined_user_count"])
        if defined + undefined != planned:
            raise FigureValidationError("calibration eligibility accounting changed")
        if row["status"] == "COMPUTED" and defined <= 0:
            raise FigureValidationError("computed calibration cell has no eligible users")
    return source


def build_figure_values(figure_id: str, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Extract exact canonical rows plus explicit display and eligibility metadata."""

    if figure_id not in FIGURE_SPECS:
        raise FigureValidationError(f"unknown RQ2 figure: {figure_id}")
    source = load_and_validate_canonical(root)
    metrics = FIGURE_SPECS[figure_id]["metrics"]
    frame = source.loc[source["metric"].isin(metrics), SOURCE_COLUMNS].copy()
    frame = frame.rename(columns={"metric": "metric_id"})
    frame["metric_display_label"] = frame["metric_id"].map(METRIC_LABELS)
    frame["display_transform"] = "NONE_RAW_CANONICAL_VALUE"
    frame["eligible_user_count"] = ""
    frame["eligibility_status"] = frame["status"].map(
        lambda status: "ELIGIBLE_COMPUTED" if status == "COMPUTED" else "NONCOMPUTED"
    )

    calibration_mask = frame["metric_id"].eq("calibration_at_5")
    for index in frame.index[calibration_mask]:
        payload = json.loads(frame.at[index, "value_json"])
        defined = str(int(payload["defined_user_count"]))
        frame.at[index, "eligible_user_count"] = defined
        frame.at[index, "eligibility_status"] = (
            "FULL_USER_ELIGIBILITY"
            if frame.at[index, "undefined_user_count"] == "0"
            else "PARTIAL_USER_ELIGIBILITY"
        )

    frame["_dataset_order"] = frame["dataset"].map({v: i for i, v in enumerate(DATASET_ORDER)})
    frame["_model_order"] = frame["model_name"].map({v: i for i, v in enumerate(MODEL_ORDER)})
    frame["_prompt_order"] = frame["prompt_variant"].map({v: i for i, v in enumerate(PROMPT_ORDER)})
    frame["_metric_order"] = frame["metric_id"].map({v: i for i, v in enumerate(metrics)})
    frame = frame.sort_values(
        ["_metric_order", "_dataset_order", "_model_order", "_prompt_order"],
        kind="stable",
    ).drop(columns=["_metric_order", "_dataset_order", "_model_order", "_prompt_order"])
    return frame.reset_index(drop=True)


def validate_values_against_source(
    figure_id: str, values: pd.DataFrame, root: Path = PROJECT_ROOT
) -> None:
    """Require figure values and statuses to equal the canonical source exactly."""

    expected = build_figure_values(figure_id, root).astype(str).reset_index(drop=True)
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


def _axis_upper(values: np.ndarray) -> float:
    maximum = float(np.nanmax(values))
    if maximum <= 1.0:
        return min(1.0, max(0.1, math.ceil(maximum * 10.0) / 10.0 + 0.05))
    return maximum * 1.10


def _landscape_figure(figure_id: str, values: pd.DataFrame) -> plt.Figure:
    spec = FIGURE_SPECS[figure_id]
    metrics = spec["metrics"]
    ncols = int(spec["ncols"])
    nrows = math.ceil(len(metrics) / ncols)
    width = 10.0 if ncols == 1 else 14.2
    height = 5.7 if nrows == 1 else 9.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    axes_list = list(axes.flat)
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

    for axis, metric in zip(axes_list, metrics, strict=False):
        metric_rows = values.loc[values["metric_id"].eq(metric)]
        plotted: list[float] = []
        for model in MODEL_ORDER:
            for prompt in PROMPT_ORDER:
                subset = (
                    metric_rows.loc[
                        metric_rows["model_name"].eq(model)
                        & metric_rows["prompt_variant"].eq(prompt)
                    ]
                    .set_index("dataset")
                    .loc[DATASET_ORDER]
                )
                y = np.array(
                    [float(value) if status == "COMPUTED" else np.nan for value, status in zip(subset["value"], subset["status"], strict=True)],
                    dtype=float,
                )
                plotted.extend(y[np.isfinite(y)].tolist())
                axis.plot(
                    x,
                    y,
                    color=colors[model],
                    linewidth=1.7,
                    markersize=4.8,
                    label=f"{MODEL_LABELS[model]} / {prompt}",
                    **prompt_styles[prompt],
                )
        if not plotted:
            raise FigureValidationError(f"no computed values for {metric}")
        upper = _axis_upper(np.asarray(plotted, dtype=float))
        axis.set_ylim(0.0, upper)
        for dataset_index, dataset in enumerate(DATASET_ORDER):
            rows = metric_rows.loc[metric_rows["dataset"].eq(dataset)]
            if rows["status"].ne("COMPUTED").any():
                axis.axvspan(dataset_index - 0.32, dataset_index + 0.32, color="#BDBDBD", alpha=0.28)
                axis.text(
                    dataset_index,
                    upper * 0.94,
                    "N/C",
                    ha="center",
                    va="top",
                    color="#555555",
                    fontsize=8,
                    fontweight="bold",
                )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_ylabel("Raw canonical value")
        axis.set_xticks(x, [DATASET_LABELS[item] for item in DATASET_ORDER], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes_list[len(metrics):]:
        axis.set_visible(False)
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.suptitle(spec["title"], y=0.99)
    fig.text(0.5, 0.015, spec["footer"], ha="center", fontsize=8)
    top = 0.79 if nrows == 1 else 0.82
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14 if nrows == 1 else 0.10, top=top, wspace=0.30, hspace=0.43)
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


def _metric_contract(figure_id: str) -> dict[str, Any]:
    if figure_id == "fig_rq2_llm_longtail_novelty_landscape":
        return {
            "APLT@5": "aplt_at_5",
            "implementation_alias": "calACLT",
            "ACLT_is_separate_metric": False,
            "PopREO_noncomputed_policy": "PRESERVE_STATUS_AND_BLANK_VALUE",
        }
    if figure_id == "fig_rq2_llm_serendipity_coverage_landscape":
        return {
            "serendipity_source": "CORRECTED_CANONICAL_G1",
            "item_coverage": "prompt-compatible candidate-union denominator",
            "catalog_coverage": "catalog-universe denominator",
            "coverage_metrics_collapsed": False,
        }
    if figure_id == "fig_rq2_llm_diversity_distribution_landscape":
        return {"display_transform": "NONE", "direction_reversal": False, "synthetic_aggregate": False}
    return {
        "metric": "calibration_at_5",
        "eligibility_fields": ["eligible_user_count", "undefined_user_count", "eligibility_status"],
        "noncomputed_policy": "PRESERVE_STATUS_AND_BLANK_VALUE",
        "imputation": False,
    }


def _status_counts(values: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for metric, group in values.groupby("metric_id", sort=False):
        counts[str(metric)] = {str(key): int(value) for key, value in group["status"].value_counts(sort=False).items()}
    return counts


def generate_all(
    root: Path = PROJECT_ROOT,
    *,
    artifact_root: Path | None = None,
    created_from_commit: str | None = None,
) -> list[dict[str, Any]]:
    """Generate four RQ2 landscapes and complete hash manifests."""

    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    load_and_validate_canonical(root)
    created_from_commit = created_from_commit or git_head(root)
    if not created_from_commit or len(created_from_commit) != 40:
        raise FigureValidationError("created_from_commit must be a full commit hash")
    _configure_matplotlib()

    manifests: list[dict[str, Any]] = []
    for figure_id, spec in FIGURE_SPECS.items():
        values = build_figure_values(figure_id, root)
        values_relative = f"{OUTPUT_ROOT}/values/{figure_id}_values.csv"
        values_path = repository_path(artifact_root, values_relative)
        write_dataframe(values_path, values)
        validate_values_against_source(
            figure_id,
            pd.read_csv(values_path, dtype=str, keep_default_na=False),
            root,
        )
        outputs = _save_figure(_landscape_figure(figure_id, values), artifact_root, figure_id)
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
                "configuration_cells": 30,
                "datasets": DATASET_ORDER,
                "models": MODEL_ORDER,
                "prompt_strategies": PROMPT_ORDER,
            },
            "metric_ids": spec["metrics"],
            "metric_contract": _metric_contract(figure_id),
            "display_contract": {
                "transform": "NONE_RAW_CANONICAL_VALUE",
                "independent_y_axis_per_metric": True,
                "cross_panel_height_comparison": "NOT_VALID",
                "noncomputed_visual_label": "N/C",
            },
            "source_status_counts": _status_counts(values),
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
    """Validate all four frozen RQ2 figure chains and exact source equality."""

    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    load_and_validate_canonical(root)
    validated: list[dict[str, Any]] = []
    for figure_id in FIGURE_IDS:
        manifest_path = repository_path(
            artifact_root, f"{MANIFEST_ROOT}/{figure_id}_manifest.json"
        )
        manifest = validate_figure_manifest(root, manifest_path, artifact_root=artifact_root)
        if PRE_FIX_HASH in manifest_path.read_text(encoding="utf-8"):
            raise FigureValidationError("pre-fix serendipity hash appears in a foreground figure manifest")
        values = pd.read_csv(
            repository_path(artifact_root, manifest["figure_values_csv"]),
            dtype=str,
            keep_default_na=False,
        )
        validate_values_against_source(figure_id, values, root)
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

"""Render three final-presentation RQ5 figures from frozen scientific evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_figures.common import (  # noqa: E402
    FigureValidationError,
    git_head,
    repository_path,
    sha256_file,
    validate_figure_manifest,
    write_dataframe,
    write_json,
)

BASE = "results/rq5_final_scientific_freeze"
INDEX = f"{BASE}/figures/figure_index.json"
FROZEN = f"{BASE}/freeze_manifest.json"
METRICS = f"{BASE}/corrected/metrics.csv"
BOOTSTRAP = f"{BASE}/corrected/bootstrap_results.csv"
HISTORY = f"{BASE}/history_v2/metrics.csv"
MIGRATION = f"{BASE}/history_v2/migration_manifest.json"
FIG1 = "fig_rq5_profile_conditions_main"
FIG2 = "fig_rq5_profile_deltas_heatmap"
FIG5 = "fig_rq5_history_length_profile_conditions"
IDS = (FIG1, FIG2, FIG5)
OLD_VALUES = {figure_id: f"{BASE}/figures/{figure_id}_values.csv" for figure_id in IDS}
SOURCE_HASHES = {
    METRICS: "98c7782211ec143c46205591a1cebc2dc4e5798926aef23badd34287cdd3f312",
    BOOTSTRAP: "f81d55d54468fe363b898635040c0299fa4303a9d94ce0dcdfcb10e5a8e712dc",
    HISTORY: "7ace71bc8354515d2fb099883c7f6cba908d91bd49be01c7fb04c86ac66386fa",
    MIGRATION: "d07254a556ef01b720fa7094eb63249ce7f3ec5b5fea5de6401cb381f5439fce",
}
OLD_VALUE_HASHES = {
    FIG1: "8fcc680a8fd7b6cc5718fdeed951c45c2b679f983f76575ce57084afe9f12451",
    FIG2: "b6189c79d9ce7c15c47760f7d0972563bf6683bfc7b07adbcfd8f11e2ca3a50f",
    FIG5: "387f617f5652b76361190c7f9c5575498e5855941ff217000d3f1de697136789",
}
SOURCES = {
    FIG1: [METRICS, OLD_VALUES[FIG1], INDEX, FROZEN],
    FIG2: [BOOTSTRAP, OLD_VALUES[FIG2], INDEX, FROZEN],
    FIG5: [HISTORY, MIGRATION, OLD_VALUES[FIG5], INDEX, FROZEN],
}
OUT = "results/final_figures/rq5"
MANIFESTS = "results/manifests/final_figures"
GENERATOR = "src/analysis/final_figures/rq5.py"
FIXED_TIME = datetime(2026, 9, 21, tzinfo=timezone.utc)
CONDITIONS = ("history_only", "profile_only", "profile_plus_history")
CONDITION_LABELS = {
    "history_only": "History-only",
    "profile_only": "Profile-only",
    "profile_plus_history": "Profile + History",
}
COLORS = {"history_only": "#4D4D4D", "profile_only": "#0072B2", "profile_plus_history": "#D55E00"}
CONTRASTS = ("profile_only_vs_history_only", "profile_plus_history_vs_history_only")
METRIC_LABELS = {"hr_at_5": "HR@5", "ndcg_at_5": "NDCG@5", "aplt_at_5": "APLT@5", "serendipity_at_5": "Serendipity@5"}
DECISIONS = {
    "CI_SUPPORTED_POSITIVE": "+CI",
    "CI_SUPPORTED_NEGATIVE": "-CI",
    "NON_SIGNIFICANT_CI_INCLUDES_ZERO": "",
}


def _read(root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(repository_path(root, relative), dtype=str, keep_default_na=False)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "svg.hashsalt": "irp-final-rq5-v1", "pdf.compression": 9,
    })


def validate_sources(root: Path = ROOT) -> dict[str, str]:
    """Check immutable RQ5 tables, old figure-values, index and surviving plots."""
    root = root.resolve()
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(repository_path(root, relative)) != expected:
            raise FigureValidationError(f"RQ5 source hash mismatch: {relative}")
    index = json.loads(repository_path(root, INDEX).read_text(encoding="utf-8"))
    by_id = {entry["figure_id"]: entry for entry in index}
    if len(by_id) != 5 or not set(IDS).issubset(by_id):
        raise FigureValidationError("RQ5 frozen figure index changed")
    for figure_id in IDS:
        relative = OLD_VALUES[figure_id]
        if sha256_file(repository_path(root, relative)) != OLD_VALUE_HASHES[figure_id]:
            raise FigureValidationError(f"frozen figure values changed: {figure_id}")
        if by_id[figure_id]["source_csv_sha256"] != OLD_VALUE_HASHES[figure_id]:
            raise FigureValidationError(f"figure-index value hash mismatch: {figure_id}")
        for source, expected in by_id[figure_id]["source_artifacts"].items():
            if SOURCE_HASHES.get(source) != expected:
                raise FigureValidationError(f"figure-index source hash mismatch: {source}")
    for figure_id in ("fig_rq5_profile_plus_history_extension", "fig_rq5_profile_example_qualitative"):
        for relative, expected in by_id[figure_id]["figure_sha256"].items():
            if sha256_file(repository_path(root, relative)) != expected:
                raise FigureValidationError(f"unchanged frozen figure differs: {relative}")
    migration = json.loads(repository_path(root, MIGRATION).read_text(encoding="utf-8"))
    if migration["version"] != "V2" or migration["new_metrics_sha256"] != SOURCE_HASHES[HISTORY]:
        raise FigureValidationError("history slice is not canonical V2")
    if migration["tie_break"] != "descending_frequency_then_ascending_exact_canonical_item_id_unicode_string":
        raise FigureValidationError("MostPop V2 tie break changed")
    return {relative: sha256_file(repository_path(root, relative)) for paths in SOURCES.values() for relative in paths}


def _figure1_values(root: Path) -> pd.DataFrame:
    source = _read(root, METRICS)
    old = _read(root, OLD_VALUES[FIG1])
    keys = ["dataset", "recommender", "profile_generator", "condition"]
    if len(source) != 20 or source.duplicated(keys).any() or set(source.condition) != set(CONDITIONS):
        raise FigureValidationError("corrected matched condition universe changed")
    if not source.requests.eq("200").all() or not source.denominator_policy.eq("FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO").all():
        raise FigureValidationError("corrected full denominator changed")
    # The frozen long-form figure values must be a direct transpose of the canonical table.
    long = source.melt(id_vars=keys, value_vars=["hr_at_5", "ndcg_at_5"], var_name="metric", value_name="value")
    merged = old.merge(long, on=keys + ["metric"], suffixes=("_old", "_source"), validate="one_to_one")
    if len(merged) != 40 or not np.allclose(merged.value_old.astype(float), merged.value_source.astype(float), atol=1e-15, rtol=0):
        raise FigureValidationError("old figure values differ from corrected metrics")
    rows: list[dict[str, str]] = []
    for dataset in sorted(source.dataset.unique()):
        for recommender in sorted(source.recommender.unique()):
            group = source.loc[source.dataset.eq(dataset) & source.recommender.eq(recommender)]
            baseline = group.loc[group.condition.eq("history_only")]
            if len(baseline) != 1:
                raise FigureValidationError("missing or duplicate shared history comparator")
            for generator in sorted(group.loc[group.condition.eq("profile_only"), "profile_generator"]):
                for condition in CONDITIONS:
                    row = baseline.iloc[0] if condition == "history_only" else group.loc[group.condition.eq(condition) & group.profile_generator.eq(generator)].squeeze()
                    if not isinstance(row, pd.Series):
                        raise FigureValidationError("missing matched profile cell")
                    for metric in ("hr_at_5", "ndcg_at_5"):
                        rows.append({
                            "dataset": dataset, "recommender": recommender,
                            "display_profile_generator": generator,
                            "source_profile_generator": str(row.profile_generator),
                            "condition": condition, "metric": metric, "value": str(row[metric]),
                            "requests": str(row.requests), "strict_valid": str(row.strict_valid),
                            "denominator_policy": str(row.denominator_policy),
                            "shared_history_comparator_repeated_for_display": str(condition == "history_only").lower(),
                        })
    result = pd.DataFrame(rows)
    if len(result) != 48 or result.duplicated(["dataset", "recommender", "display_profile_generator", "condition", "metric"]).any():
        raise FigureValidationError("matched categorical display cells incomplete")
    return result


def _figure2_values(root: Path) -> pd.DataFrame:
    source = _read(root, BOOTSTRAP)
    old = _read(root, OLD_VALUES[FIG2])
    keys = ["dataset", "recommender", "profile_generator", "comparison", "metric"]
    if len(source) != 32 or source.duplicated(keys).any() or set(source.comparison) != set(CONTRASTS):
        raise FigureValidationError("corrected paired contrast universe changed")
    merged = old.merge(source, on=keys, suffixes=("_old", "_source"), validate="one_to_one")
    if len(merged) != 32:
        raise FigureValidationError("frozen delta figure values incomplete")
    for field in ("mean_delta", "ci_lower", "ci_upper"):
        if not np.allclose(merged[f"{field}_old"].astype(float), merged[f"{field}_source"].astype(float), atol=1e-15, rtol=0):
            raise FigureValidationError(f"frozen {field} differs")
    if not merged.decision_old.eq(merged.decision_source).all():
        raise FigureValidationError("frozen CI decisions differ")
    if not source.paired_n.eq("200").all() or not source.bootstrap_resamples.eq("10000").all():
        raise FigureValidationError("paired bootstrap denominator/resamples changed")
    if not source.p_value.eq("").all() or not source.multiplicity_correction.eq("not_computed").all():
        raise FigureValidationError("unexpected p-value or multiplicity correction")
    lower = source.ci_lower.astype(float)
    upper = source.ci_upper.astype(float)
    derived = np.where(lower > 0, "CI_SUPPORTED_POSITIVE", np.where(upper < 0, "CI_SUPPORTED_NEGATIVE", "NON_SIGNIFICANT_CI_INCLUDES_ZERO"))
    if not np.array_equal(derived, source.decision.to_numpy()):
        raise FigureValidationError("CI decision differs from its frozen bounds")
    result = source.copy()
    result["ci_marker"] = result.decision.map(DECISIONS)
    return result


def _figure5_values(root: Path) -> pd.DataFrame:
    source = _read(root, HISTORY)
    old = _read(root, OLD_VALUES[FIG5])
    cols = ["history_length", "condition", "requests", "strict_valid", "denominator_policy", *METRIC_LABELS]
    if len(source) != 24 or set(source.condition) != set(CONDITIONS):
        raise FigureValidationError("history V2 universe changed")
    if source.history_length.astype(int).sort_values().unique().tolist() != [1, 2, 3, 4, 5, 10, 15, 20]:
        raise FigureValidationError("history V2 length grid changed")
    if not source.requests.eq("200").all() or not source.denominator_policy.eq("FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO").all():
        raise FigureValidationError("history V2 denominator changed")
    if not source[cols].equals(old[cols]):
        raise FigureValidationError("history V2 values differ from frozen figure values")
    return source


def values_for(figure_id: str, root: Path = ROOT) -> pd.DataFrame:
    """Return presentation rows copied from, or linked exactly to, frozen data."""
    if figure_id == FIG1:
        return _figure1_values(root)
    if figure_id == FIG2:
        return _figure2_values(root)
    if figure_id == FIG5:
        return _figure5_values(root)
    raise FigureValidationError(f"unauthorized figure: {figure_id}")


def _short(name: str) -> str:
    return "Qwen2" if name.startswith("qwen2:") else "Mistral" if name.startswith("mistral:") else name


def _config_label(dataset: str, recommender: str, generator: str) -> str:
    return f"{dataset.title()} | {_short(recommender)} rec | {_short(generator)} profile"


def _plot_conditions(values: pd.DataFrame) -> plt.Figure:
    configs = values[["dataset", "recommender", "display_profile_generator"]].drop_duplicates().reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.4), sharey=True)
    offsets = {"history_only": -0.23, "profile_only": 0.0, "profile_plus_history": 0.23}
    labels = [_config_label(*row) for row in configs.itertuples(index=False, name=None)]
    for ax, metric in zip(axes, ("hr_at_5", "ndcg_at_5"), strict=True):
        for i, config in enumerate(configs.itertuples(index=False, name=None)):
            dataset, rec, gen = config
            block = values.loc[values.dataset.eq(dataset) & values.recommender.eq(rec) & values.display_profile_generator.eq(gen) & values.metric.eq(metric)]
            for condition in CONDITIONS:
                row = block.loc[block.condition.eq(condition)].iloc[0]
                ax.scatter(float(row.value), i + offsets[condition], s=52, color=COLORS[condition], marker="o", label=CONDITION_LABELS[condition] if i == 0 else None, zorder=3)
        ax.set_title(METRIC_LABELS[metric]); ax.set_xlabel("Absolute utility (full matched denominator n=200)")
        ax.set_xlim(left=0); ax.set_ylim(len(configs) - .35, -.65)
        ax.set_yticks(range(len(configs)), labels); ax.grid(axis="x", alpha=.22)
        ax.tick_params(axis="y", labelleft=(ax is axes[0]), labelsize=9)
    fig.suptitle("RQ5 corrected matched conditions: absolute utility", y=.97, fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .135), ncol=3, frameon=False)
    fig.text(.5, .075, "History-only is the same matched comparator within each dataset/recommender and is repeated for display across profile generators.", ha="center", fontsize=8.5)
    fig.text(.5, .045, "Absolute matched-condition utility; inferential treatment effects are shown separately in the delta figure. Invalid outputs count as zero.", ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.29, right=.985, top=.88, bottom=.25, wspace=.18)
    return fig


def _plot_deltas(values: pd.DataFrame) -> plt.Figure:
    configs = values[["dataset", "recommender", "profile_generator"]].drop_duplicates().reset_index(drop=True)
    columns = [(comparison, metric) for comparison in CONTRASTS for metric in ("hr_at_5", "ndcg_at_5")]
    matrix = np.zeros((len(configs), len(columns)))
    markers = [["" for _ in columns] for _ in range(len(configs))]
    for i, (dataset, rec, gen) in enumerate(configs.itertuples(index=False, name=None)):
        for j, (contrast, metric) in enumerate(columns):
            row = values.loc[values.dataset.eq(dataset) & values.recommender.eq(rec) & values.profile_generator.eq(gen) & values.comparison.eq(contrast) & values.metric.eq(metric)].iloc[0]
            matrix[i, j] = float(row.mean_delta)
            markers[i][j] = row.ci_marker
    limit = max(abs(matrix.min()), abs(matrix.max()))
    fig, ax = plt.subplots(figsize=(12.8, 8.3))
    image = ax.imshow(matrix, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:+.3f}\n{markers[i][j]}", ha="center", va="center", fontsize=10, color="black")
    ax.set_xticks(range(4), ["P-H\nHR@5", "P-H\nNDCG@5", "H+P-H\nHR@5", "H+P-H\nNDCG@5"])
    ax.set_yticks(range(len(configs)), [_config_label(*row) for row in configs.itertuples(index=False, name=None)])
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, len(configs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=ax, shrink=.78, label="Matched mean delta")
    fig.suptitle("RQ5 corrected matched profile effects with paired bootstrap decisions", y=.98, fontsize=14)
    fig.text(.5, .15, "P-H = Profile-only minus History-only = profile substitution effect", ha="center", fontsize=9.5)
    fig.text(.5, .12, "H+P-H = Profile + History minus History-only = profile addition effect", ha="center", fontsize=9.5)
    fig.text(.5, .09, "+CI: 95% bootstrap CI entirely above zero    -CI: entirely below zero    No marker: CI includes zero (inconclusive)", ha="center", fontsize=8.7)
    fig.text(.5, .06, "Local exploratory paired comparisons (n=200 each); CI decisions are unadjusted for multiplicity. No general profile-benefit claim.", ha="center", fontsize=8.3)
    fig.subplots_adjust(left=.31, right=.91, top=.88, bottom=.23)
    return fig


def _plot_history(values: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True, sharey=False)
    lengths = sorted(values.history_length.astype(int).unique())
    for ax, metric in zip(axes.flat, METRIC_LABELS, strict=True):
        for condition in CONDITIONS:
            block = values.loc[values.condition.eq(condition)].copy().sort_values("history_length", key=lambda col: col.astype(int))
            ax.plot(block.history_length.astype(int), block[metric].astype(float), marker="o", linewidth=1.5, markersize=4,
                    label=CONDITION_LABELS[condition], color=COLORS[condition])
        ax.set_title(METRIC_LABELS[metric]); ax.set_xticks(lengths)
        ax.set_ylabel(f"{METRIC_LABELS[metric]} (raw)"); ax.set_xlabel("History length")
        ax.grid(alpha=.2)
        # Each metric retains its own data-space scale; no normalisation or shared y axis.
        ymax = float(values[metric].astype(float).max())
        ax.set_ylim(0, ymax * 1.12 if ymax else 1)
    fig.suptitle("RQ5 Beauty / Qwen only: exploratory adapted history-profile slice", y=.98, fontsize=14)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False)
    fig.text(.5, .035, "Frozen V2 values: deterministic MostPop tie-break, corrected APLT and serendipity. Distinct raw metric scales; no pooled effect inference.", ha="center", fontsize=8.3)
    fig.subplots_adjust(left=.09, right=.97, top=.91, bottom=.17, hspace=.34, wspace=.22)
    return fig


def _save(fig: plt.Figure, target: Path, figure_id: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for ext in ("png", "pdf", "svg"):
        relative = f"{OUT}/figures/{figure_id}.{ext}"
        path = repository_path(target, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = ({"Software": "IRP deterministic final-figure pipeline"} if ext == "png" else
                    {"Creator": "IRP deterministic final-figure pipeline", "Producer": "Matplotlib", "CreationDate": FIXED_TIME, "ModDate": FIXED_TIME} if ext == "pdf" else
                    {"Creator": "IRP deterministic final-figure pipeline", "Date": "2026-09-21"})
        fig.savefig(path, dpi=170, metadata=metadata, facecolor="white")
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
        output[ext] = relative
        output[f"{ext}_sha256"] = sha256_file(path)
    plt.close(fig)
    return output


def generate_all(root: Path = ROOT, *, artifact_root: Path | None = None, created_from_commit: str | None = None) -> list[dict]:
    """Write only three RQ5 final-presentation chains, never frozen evidence."""
    root = root.resolve()
    target = (artifact_root or root).resolve()
    hashes = validate_sources(root)
    _style()
    commit = created_from_commit or git_head(root)
    if len(commit) != 40:
        raise FigureValidationError("full source commit required")
    output: list[dict] = []
    for figure_id in IDS:
        values = values_for(figure_id, root)
        relative = f"{OUT}/values/{figure_id}_values.csv"
        value_path = repository_path(target, relative)
        write_dataframe(value_path, values)
        figure = (_plot_conditions(values) if figure_id == FIG1 else
                  _plot_deltas(values) if figure_id == FIG2 else _plot_history(values))
        images = _save(figure, target, figure_id)
        manifest = {
            "schema_version": "1.0", "figure_id": figure_id, "rq": "RQ5",
            "scientific_purpose": {FIG1: "Corrected matched absolute utility", FIG2: "Corrected matched substitution and addition effects", FIG5: "Exploratory Beauty/Qwen history-profile slice"}[figure_id],
            "canonical_source_artifacts": SOURCES[figure_id],
            "canonical_source_sha256": {key: hashes[key] for key in SOURCES[figure_id]},
            "figure_values_csv": relative, "figure_values_sha256": sha256_file(value_path),
            "generator": GENERATOR, "generator_sha256": sha256_file(repository_path(root, GENERATOR)),
            "output_png": images["png"], "output_png_sha256": images["png_sha256"],
            "output_pdf": images["pdf"], "output_pdf_sha256": images["pdf_sha256"],
            "output_svg": images["svg"], "output_svg_sha256": images["svg_sha256"],
            "scientific_interpretation": {FIG1: "Absolute matched utility varies across configurations; treatment effects require separate inference.", FIG2: "Local corrected substitution/addition effects include positive, negative and inconclusive CI decisions; no general benefit is established.", FIG5: "Beauty/Qwen adapted history-profile slice is exploratory and uses corrected deterministic V2 metrics."}[figure_id],
            "limitations": {FIG1: ["History comparator repeated for visual alignment only.", "Full matched 200-request denominator per configuration; invalid outputs count as zero."], FIG2: ["Paired 95% bootstrap CIs are unadjusted for multiplicity.", "Local exploratory effects do not support a general profile-benefit claim."], FIG5: ["Beauty/Qwen only; exploratory adapted slice.", "Metrics have independent raw y-axis scales; no cross-metric normalization."]}[figure_id],
            "created_from_commit": commit,
            "evidence_role": "MAIN_TEXT" if figure_id in (FIG1, FIG2) else "APPENDIX_METHODOLOGY",
            "presentation_only": True, "frozen_scientific_results_unchanged": True,
            "row_count": len(values),
            "display_contract": ({"categorical_unconnected_points": True, "explicit_treatment_labels": list(CONDITION_LABELS.values()), "shared_history_comparator_repeated": True} if figure_id == FIG1 else
                                 {"substitution": "Profile-only minus History-only", "addition": "Profile + History minus History-only", "ci_marker_map": DECISIONS, "multiplicity_adjusted": False} if figure_id == FIG2 else
                                 {"scope": "Beauty / Qwen only", "panels": list(METRIC_LABELS), "independent_raw_y_axes": True, "mostpop_version": "V2"}),
        }
        manifest_path = repository_path(target, f"{MANIFESTS}/{figure_id}_manifest.json")
        write_json(manifest_path, manifest)
        validate_figure_manifest(root, manifest_path, artifact_root=target)
        output.append(manifest)
    return output


def validate_all(root: Path = ROOT, *, artifact_root: Path | None = None) -> list[dict]:
    """Validate all three saved RQ5 figure chains against frozen inputs."""
    root = root.resolve()
    target = (artifact_root or root).resolve()
    validate_sources(root)
    result = []
    for figure_id in IDS:
        manifest_path = repository_path(target, f"{MANIFESTS}/{figure_id}_manifest.json")
        manifest = validate_figure_manifest(root, manifest_path, artifact_root=target)
        expected = values_for(figure_id, root).astype(str).reset_index(drop=True)
        actual = _read(target, manifest["figure_values_csv"]).astype(str).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
        if manifest["evidence_role"] != ("MAIN_TEXT" if figure_id in (FIG1, FIG2) else "APPENDIX_METHODOLOGY"):
            raise FigureValidationError("RQ5 figure role changed")
        result.append(manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "validate"), required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--created-from-commit")
    args = parser.parse_args(argv)
    result = (generate_all(ROOT, artifact_root=args.artifact_root, created_from_commit=args.created_from_commit)
              if args.mode == "generate" else validate_all(ROOT, artifact_root=args.artifact_root))
    print(json.dumps({"status": "PASS", "figures": [entry["figure_id"] for entry in result]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

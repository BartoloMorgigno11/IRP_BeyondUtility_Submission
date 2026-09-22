"""Render-only title and label corrections for two frozen final figures."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_figures import rq2_heatmaps, rq4  # noqa: E402
from src.analysis.final_figures.common import (  # noqa: E402
    FigureValidationError, repository_path, sha256_file, validate_figure_manifest,
    write_json,
)

GENERATOR = "src/analysis/final_figures/render_polish.py"
CALIBRATION = "fig_rq2_llm_calibration_heatmap"
BOOTSTRAP = "fig_rq4_g3_first_random_bootstrap_deltas"
CALIBRATION_MANIFEST = f"results/manifests/final_figures/{CALIBRATION}_manifest.json"
BOOTSTRAP_MANIFEST = f"results/manifests/final_figures/{BOOTSTRAP}_manifest.json"
BASE_HEAD = "0b2fbaa3d91e46d0daf85c9b981b57375b323ae0"
CALIBRATION_VALUES_SHA256 = "e732f83d363c337fbe2e7890585664409b9038b8b6305a12fb0ceacdbca925fd"
BOOTSTRAP_VALUES_SHA256 = "708b7cd6160a3140bd91b02a3d06d8ebf178743d5610720ff8c9e8773e08eeaa"
CALIBRATION_TITLE = "RQ2 Calibration@5 — local extension"
METRIC_TITLES = {"hr_at_5": "HR@5", "ndcg_at_5": "NDCG@5", "mrr": "MRR"}


def _stored_values(root: Path, manifest: dict, expected_hash: str) -> pd.DataFrame:
    """Read frozen values without rewriting them."""
    relative = manifest["figure_values_csv"]
    path = repository_path(root, relative)
    if manifest["figure_values_sha256"] != expected_hash or sha256_file(path) != expected_hash:
        raise FigureValidationError(f"frozen figure values changed: {relative}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _calibration_values(root: Path, manifest: dict) -> pd.DataFrame:
    values = _stored_values(root, manifest, CALIBRATION_VALUES_SHA256)
    rq2_heatmaps.validate_heatmap_sources(root)
    rq2_heatmaps.validate_heatmap_values(CALIBRATION, values, root)
    if len(values) != 30 or not values.metric_id.eq("calibration_at_5").all():
        raise FigureValidationError("calibration cell universe changed")
    if not values.status.eq("COMPUTED").all():
        raise FigureValidationError("calibration computed-cell status changed")
    if not values.cell_annotation.str.contains(r"\ne=\d+\nu=\d+", regex=True).all():
        raise FigureValidationError("calibration eligible/undefined annotations changed")
    if not (values.eligible_user_count.astype(int) + values.undefined_user_count.astype(int)
            == values.planned_users.astype(int)).all():
        raise FigureValidationError("calibration eligibility accounting changed")
    if manifest["calibration_eligibility"] != rq2_heatmaps._eligibility_summary(values):
        raise FigureValidationError("calibration eligibility summary changed")
    return values


def _bootstrap_values(root: Path, manifest: dict) -> pd.DataFrame:
    values = _stored_values(root, manifest, BOOTSTRAP_VALUES_SHA256)
    rq4.validate_sources(root)
    expected = rq4.bootstrap_values(root)
    pd.testing.assert_frame_equal(values.astype(str).reset_index(drop=True),
                                  expected.astype(str).reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    counts = {decision: int(values.decision.eq(decision).sum())
              for decision in ("POSITIVE", "NEGATIVE", "INCONCLUSIVE")}
    if counts != manifest["decision_counts"] or sum(counts.values()) != 90:
        raise FigureValidationError("bootstrap decision counts changed")
    if not values.orientation.eq("FIRST_MINUS_RANDOM").all():
        raise FigureValidationError("bootstrap contrast orientation changed")
    return values


def _labels(values: pd.DataFrame) -> dict[str, str]:
    """Compute explicit decision-count titles from frozen values."""
    labels = {}
    for metric, display in METRIC_TITLES.items():
        group = values.loc[values.metric_id.eq(metric)]
        if len(group) != 30:
            raise FigureValidationError(f"bootstrap metric has an incomplete grid: {metric}")
        counts = group.decision.value_counts()
        labels[metric] = (f"{display} (pos={counts.get('POSITIVE', 0)} / "
                          f"neg={counts.get('NEGATIVE', 0)} / "
                          f"inc={counts.get('INCONCLUSIVE', 0)})")
    return labels


def _copy_values_to_staging(root: Path, artifact_root: Path, manifest: dict) -> None:
    """Provide unchanged values bytes in a temporary regeneration tree."""
    if root == artifact_root:
        return
    relative = manifest["figure_values_csv"]
    destination = repository_path(artifact_root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository_path(root, relative), destination)


def _updated_manifest(root: Path, original: dict, outputs: dict[str, str],
                      figure_id: str, labels: dict[str, str] | None = None) -> dict:
    """Change renderer/output provenance while retaining scientific interpretation."""
    manifest = dict(original)
    previous = original.get("render_only_remediation", {}).get("previous_output_sha256")
    if previous is None:
        previous = {extension: original[f"output_{extension}_sha256"]
                    for extension in ("png", "pdf", "svg")}
    manifest["generator"] = GENERATOR
    manifest["generator_sha256"] = sha256_file(repository_path(root, GENERATOR))
    for extension in ("png", "pdf", "svg"):
        if original[f"output_{extension}"] != outputs[extension]:
            raise FigureValidationError("render-only polish changed an output path")
        manifest[f"output_{extension}_sha256"] = outputs[f"{extension}_sha256"]
    dependencies = dict(original.get("generator_dependencies", {}))
    upstream = (rq2_heatmaps.GENERATOR if figure_id == CALIBRATION else rq4.GENERATOR)
    dependencies[upstream] = sha256_file(repository_path(root, upstream))
    manifest["generator_dependencies"] = dependencies
    manifest["render_only_remediation"] = {
        "base_figure_head": BASE_HEAD,
        "scope": "TYPOGRAPHY_ONLY_NO_SCIENTIFIC_VALUE_CHANGE",
        "figure_values_sha256_unchanged": original["figure_values_sha256"],
        "previous_output_sha256": previous,
        "change": ("Calibration@5 title explicitly identifies local extension"
                   if figure_id == CALIBRATION else
                   "HR@5/NDCG@5/MRR labels and explicit pos/neg/inc bootstrap counts"),
    }
    if figure_id == CALIBRATION:
        manifest["render_only_remediation"]["visible_title"] = CALIBRATION_TITLE
        manifest["render_only_remediation"]["not_target_paper_metric"] = True
    else:
        assert labels is not None
        manifest["render_only_remediation"]["panel_titles"] = labels
        manifest["render_only_remediation"]["decision_counts_from_frozen_values"] = True
    return manifest


def generate(root: Path = ROOT, *, artifact_root: Path | None = None) -> list[dict]:
    """Regenerate only the requested RQ2 and RQ4 renderings and manifests."""
    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    generated = []
    for figure_id, manifest_relative in ((CALIBRATION, CALIBRATION_MANIFEST),
                                          (BOOTSTRAP, BOOTSTRAP_MANIFEST)):
        original = json.loads(repository_path(root, manifest_relative).read_text(encoding="utf-8"))
        if original["figure_id"] != figure_id:
            raise FigureValidationError("original figure manifest identity changed")
        _copy_values_to_staging(root, artifact_root, original)
        if figure_id == CALIBRATION:
            values = _calibration_values(root, original)
            rq2_heatmaps._configure_matplotlib()
            fig = rq2_heatmaps._heatmap_figure(figure_id, values)
            if fig._suptitle is None:
                raise FigureValidationError("calibration title is absent")
            fig._suptitle.set_text(CALIBRATION_TITLE)
            outputs = rq2_heatmaps._save_figure(fig, artifact_root, figure_id)
            labels = None
        else:
            values = _bootstrap_values(root, original)
            labels = _labels(values)
            rq4._style()
            fig = rq4._forest_plot(values)
            if len(fig.axes) != 4:
                raise FigureValidationError("bootstrap forest panel layout changed")
            for axis, metric in zip(fig.axes[:3], rq4.STAT_METRICS, strict=True):
                axis.set_title(labels[metric])
            outputs = rq4._save(fig, artifact_root, figure_id)
        manifest = _updated_manifest(root, original, outputs, figure_id, labels)
        target = repository_path(artifact_root, manifest_relative)
        write_json(target, manifest)
        validate_figure_manifest(root, target, artifact_root=artifact_root)
        generated.append(manifest)
    return generated


def validate(root: Path = ROOT, *, artifact_root: Path | None = None) -> list[dict]:
    """Verify unchanged values/sources and the two corrected render contracts."""
    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    validated = []
    for figure_id, relative in ((CALIBRATION, CALIBRATION_MANIFEST),
                                (BOOTSTRAP, BOOTSTRAP_MANIFEST)):
        manifest = validate_figure_manifest(root, repository_path(artifact_root, relative),
                                            artifact_root=artifact_root)
        if manifest["generator"] != GENERATOR or manifest["figure_id"] != figure_id:
            raise FigureValidationError("render-only manifest identity changed")
        if figure_id == CALIBRATION:
            values = _calibration_values(root, manifest)
            if manifest["render_only_remediation"]["visible_title"] != CALIBRATION_TITLE:
                raise FigureValidationError("calibration local-extension title changed")
        else:
            values = _bootstrap_values(root, manifest)
            if manifest["render_only_remediation"]["panel_titles"] != _labels(values):
                raise FigureValidationError("bootstrap decision labels changed")
        svg = repository_path(artifact_root, manifest["output_svg"]).read_text(encoding="utf-8")
        for caption in ([CALIBRATION_TITLE] if figure_id == CALIBRATION else list(_labels(values).values())):
            if caption not in svg:
                raise FigureValidationError(f"rendered SVG caption missing: {caption}")
        validated.append(manifest)
    return validated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "validate"), required=True)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args(argv)
    result = generate(ROOT, artifact_root=args.artifact_root) if args.mode == "generate" else validate(ROOT, artifact_root=args.artifact_root)
    print(json.dumps({"status": "PASS", "figures": [item["figure_id"] for item in result]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

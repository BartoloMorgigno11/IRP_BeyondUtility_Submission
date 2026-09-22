"""Freeze two RQ2 traditional-versus-LLM beyond-utility comparisons."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_figures.common import (  # noqa: E402
    FigureValidationError, git_head, repository_path, sha256_file,
    validate_figure_manifest, write_dataframe, write_json,
)

GENERATOR = "src/analysis/final_figures/rq2_traditional_llm.py"
OUTPUT_ROOT = "results/final_figures/rq2"
MANIFEST_ROOT = "results/manifests/final_figures"
B = "results/tables/f1bx_baseline_beyond_utility_results.csv"
F = "results/tables/f1bz_aggregate_beyond_utility_results.csv"
S = "results/tables/f1bz_per_seed_beyond_utility_results.csv"
G = "results/tables/g1_primary_ranking_beyond_utility.csv"
G_CONTRACT = "results/manifests/g1_metric_contract_and_provenance.json"
G_FIX = "results/manifests/g1_serendipity_correction.json"
B_MANIFEST = "results/manifests/f1bx_execution_manifest.json"
F_MANIFEST = "results/manifests/f1bz_final_traditional_execution.json"
SOURCES = {
    B: "28b4719d3cb4c7c2a2e79aaae74aade1925f764104bc071c1eefc53c6f52da15",
    F: "cb6a666078fc41fd63404dda264123a1e6890ae044450f3918b3cab15b8c3ba1",
    S: "c56be8f5034f4457eca43e44e5feb188f6d81b9c619ad6e3e52d23fdc2a9510d",
    G: "2e5c774f552cb47f16622f2a989ce033ee05036ad814558926b40148e8857c5b",
    G_CONTRACT: "ca7fc85bd346720c8529698ecfc592da557c06dab1494e2f5b60f1a8c5d7abf6",
    G_FIX: "c33cca450081a00eb7937b524ffe58fff309e0b6583d9fc2ec35ac47edfce6f1",
    B_MANIFEST: "b6251567466574ba397e1f0d5014c0ea357f9973b4ba0f20cad2bbb5334c7ac9",
    F_MANIFEST: "99acd7fc2f1ee967cb158b9f9d27912d7d92441048a9cbc47fda3c344f4b7ed8",
}
FIGURES = {
    "fig_rq2_traditional_vs_llm_longtail_popularity_novelty": (
        ("aplt_at_5", "arp_at_5", "self_information_at_5"), "MAIN_TEXT"),
    "fig_rq2_traditional_vs_llm_serendipity_diversity_coverage": (
        ("serendipity_at_5", "ild_at_5", "item_coverage_at_5"), "SUPPORTING_APPENDIX"),
}
METRIC_LABELS = {
    "aplt_at_5": "APLT@5 (fraction)",
    "arp_at_5": "ARP@5 (train interactions/item)",
    "self_information_at_5": "Self-information@5 (bits)",
    "serendipity_at_5": "Serendipity@5 (fraction)",
    "ild_at_5": "ILD@5 (Jaccard distance)",
    "item_coverage_at_5": "Item Coverage@5 (candidate fraction)",
}
CLASSES = {metric: ("DESCRIPTIVE_ONLY" if metric == "item_coverage_at_5" else "ADAPTED_NUMERICAL_COMPARISON")
           for metric in METRIC_LABELS}
DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
DATASET_LABELS = ("Beauty", "Sports", "ML-1M", "LastFM", "Food.com")
N_USERS = dict(zip(DATASETS, (990, 892, 1000, 1000, 1000), strict=True))
BASELINES = ("Random", "MostPop", "BM25-like/content")
TRAINABLE = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
LLMS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
LLM_LABELS = {LLMS[0]: "Qwen2", LLMS[1]: "Mistral", LLMS[2]: "Gemma2"}
PROMPTS = ("BASE", "LLMRANK")
SYSTEM_ORDER = (*BASELINES, *TRAINABLE, *(f"{LLM_LABELS[m]} / {p}" for m in LLMS for p in PROMPTS))
FIXED_TIME = datetime(2026, 9, 21, tzinfo=timezone.utc)
CALIBRATION_FIGURES = (
    "fig_rq2_llm_calibration_landscape", "fig_rq2_llm_calibration_heatmap",
)


def _csv(root: Path, path: str) -> pd.DataFrame:
    return pd.read_csv(repository_path(root, path), dtype=str, keep_default_na=False)


def _json(root: Path, path: str) -> dict:
    return json.loads(repository_path(root, path).read_text(encoding="utf-8"))


def validate_sources(root: Path = ROOT) -> dict[str, str]:
    """Check frozen identities, corrected serendipity, and metric semantics."""
    for relative, expected in SOURCES.items():
        path = repository_path(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise FigureValidationError(f"RQ2 source absent or changed: {relative}")
    for source, manifest_path in ((B, B_MANIFEST), (F, F_MANIFEST), (S, F_MANIFEST)):
        if _json(root, manifest_path)["artifact_hashes"][source] != SOURCES[source]:
            raise FigureValidationError(f"source/freeze manifest mismatch: {source}")
    contract = _json(root, G_CONTRACT)
    authority = contract["canonical_artifact_authority"]
    fix = _json(root, G_FIX)
    if (authority["canonical_beyond_utility_sha256"] != SOURCES[G]
            or authority["historical_pre_fix_sha256"] != "15ca9025aada73feba7513e06756b6c3ede42542befd82c1a76dbc757b018ae1"
            or authority["serendipity_correction_manifest_sha256"] != SOURCES[G_FIX]
            or fix["post_fix_artifact_hash"] != SOURCES[G]
            or fix["affected_cells"] != 30):
        raise FigureValidationError("G1 corrected serendipity authority changed")
    tail = contract["long_tail_contract"]
    if (tail["canonical_metric_id"] != "aplt_at_5" or tail["implementation_alias"] != "calACLT"
            or tail["aclt_is_separate_metric"] is not False
            or "0.8 quantile" not in tail["formula"]):
        raise FigureValidationError("APLT canonical definition changed")
    for figure_id in CALIBRATION_FIGURES:
        path = repository_path(root, f"{MANIFEST_ROOT}/{figure_id}_manifest.json")
        if not path.is_file():
            raise FigureValidationError(f"frozen LLM calibration figure absent: {figure_id}")
        validate_figure_manifest(root, path)
    b, f, g = (_csv(root, path) for path in (B, F, G))
    selected = set(METRIC_LABELS)
    if set(b.metric) != set(g.metric) or set(f.metric) != set(g.metric) or not selected.issubset(set(g.metric)):
        raise FigureValidationError("traditional/G1 beyond-utility metric catalog differs")
    if len(b) != 210 or len(f) != 280 or len(g) != 420:
        raise FigureValidationError("beyond-utility cell grid incomplete")
    g_food = g.loc[g.metric.eq("popreo_at_5") & g.dataset.eq("foodcom")]
    b_food = b.loc[b.metric.eq("popreo_at_5") & b.dataset.eq("foodcom")]
    f_food = f.loc[f.metric.eq("popreo_at_5") & f.dataset.eq("foodcom")]
    if len(g_food) != 6 or not g_food.status.eq("DEGENERATE_PROTOCOL_INPUT").all() or not g_food.value.eq("").all():
        raise FigureValidationError("Food.com G1 PopREO undefined policy changed")
    if len(b_food) != 3 or not b_food.status.eq("DEGENERATE_PROTOCOL_INPUT").all() or not b_food.value.eq("").all():
        raise FigureValidationError("Food.com baseline PopREO undefined policy changed")
    if len(f_food) != 4 or not f_food.status.eq("DEGENERATE_PROTOCOL_INPUT").all() or not f_food.mean_value.eq("").all():
        raise FigureValidationError("Food.com trainable PopREO undefined policy changed")
    for frame, models, model_column in ((b, BASELINES, "method"), (f, TRAINABLE, "method"), (g, LLMS, "model_name")):
        if set(frame.dataset) != set(DATASETS) or set(frame[model_column]) != set(models):
            raise FigureValidationError("dataset/system coverage changed")
    seeds = _csv(root, S)
    for dataset in DATASETS:
        threshold_rows = [
            *b.loc[b.dataset.eq(dataset) & b.metric.eq("aplt_at_5"), "value_json"],
            *g.loc[g.dataset.eq(dataset) & g.metric.eq("aplt_at_5"), "value_json"],
            *seeds.loc[seeds.dataset.eq(dataset) & seeds.metric.eq("aplt_at_5"), "value_json"],
        ]
        thresholds = {float(json.loads(raw)["long_tail_threshold"]) for raw in threshold_rows}
        if len(threshold_rows) != 29 or len(thresholds) != 1:
            raise FigureValidationError(f"APLT train-count threshold differs: {dataset}")
    for frame in (b, g, seeds):
        ser = frame.loc[frame.metric.eq("serendipity_at_5")]
        if not ser.reason.str.contains("aligned MostPop primitive", regex=False).all():
            raise FigureValidationError("serendipity MostPop reference changed")
        ild = frame.loc[frame.metric.eq("ild_at_5")]
        if not ild.reason.str.contains("Jaccard ILD", regex=False).all():
            raise FigureValidationError("ILD feature/Jaccard definition changed")
        coverage = frame.loc[frame.metric.eq("item_coverage_at_5")]
        if not coverage.reason.str.contains("candidate union", regex=False).all():
            raise FigureValidationError("item coverage denominator changed")
    return dict(SOURCES)


def build_values(root: Path, figure_id: str) -> pd.DataFrame:
    """Map six selected metric dimensions to raw canonical values only."""
    validate_sources(root)
    metrics, _ = FIGURES[figure_id]
    b, f, s, g = (_csv(root, path) for path in (B, F, S, G))
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for metric in metrics:
            for model in (*BASELINES, *TRAINABLE):
                if model in BASELINES:
                    subset = b.loc[b.dataset.eq(dataset) & b.method.eq(model) & b.metric.eq(metric)]
                    if len(subset) != 1:
                        raise FigureValidationError("baseline metric cell missing")
                    src = subset.iloc[0]
                    value, std, seed_mean, source = src.value, "", "false", B
                    undefined_note = f"{src.undefined_user_count} undefined of {src.evaluated_users} full-cohort users"
                else:
                    subset = f.loc[f.dataset.eq(dataset) & f.method.eq(model) & f.metric.eq(metric)]
                    seed_rows = s.loc[s.dataset.eq(dataset) & s.method.eq(model) & s.metric.eq(metric)]
                    if len(subset) != 1 or len(seed_rows) != 5 or set(seed_rows.seed) != {"42", "43", "44", "45", "46"}:
                        raise FigureValidationError("trainable metric/seed cell incomplete")
                    src = subset.iloc[0]
                    if (src.seed_count != "5" or src.computed_seed_count != "5" or not seed_rows.status.eq("COMPUTED").all()
                            or not seed_rows.evaluated_users.eq(str(N_USERS[dataset])).all()):
                        raise FigureValidationError("trainable seed status/cohort changed")
                    seed_values = pd.to_numeric(seed_rows.value, errors="raise")
                    if not math.isclose(float(src.mean_value), float(seed_values.mean()), rel_tol=0, abs_tol=1e-14):
                        raise FigureValidationError("trainable seed mean mismatch")
                    if not math.isclose(float(src.std_value), float(seed_values.std(ddof=1)), rel_tol=0, abs_tol=1e-14):
                        raise FigureValidationError("trainable seed SD mismatch")
                    value, std, seed_mean, source = src.mean_value, src.std_value, "true", F
                    counts = ",".join(seed_rows.sort_values("seed").undefined_user_count)
                    undefined_note = f"five full-cohort seed metrics; undefined users by seed 42-46: {counts}"
                if src.status != "COMPUTED" or (model in BASELINES and src.evaluated_users != str(N_USERS[dataset])):
                    raise FigureValidationError("traditional metric cell not computed/full-cohort")
                if model in BASELINES and metric == "aplt_at_5" and not json.loads(src.value_json)["long_tail_threshold"] >= 0:
                    raise FigureValidationError("APLT threshold absent")
                rows.append({
                    "dataset": dataset, "system_family": "TRADITIONAL", "model": model,
                    "prompt": "", "metric": metric, "value": value,
                    "seed_mean_flag": seed_mean, "seed_std_if_available": std,
                    "n_users": str(N_USERS[dataset]),
                    "traditional_population_contract": "full_evaluation_cohort",
                    "llm_population_contract": "strict_valid_plus_metric_defined_users",
                    "comparability_class": CLASSES[metric], "defined_user_note": undefined_note,
                    "source_artifact": source, "source_result_id": src.result_id,
                })
            for model in LLMS:
                for prompt in PROMPTS:
                    subset = g.loc[g.dataset.eq(dataset) & g.model_name.eq(model)
                                   & g.prompt_variant.eq(prompt) & g.metric.eq(metric)]
                    if len(subset) != 1:
                        raise FigureValidationError("G1 metric configuration missing")
                    src = subset.iloc[0]
                    if (src.status != "COMPUTED" or src.evidence_view != "BEYOND_UTILITY_STRICT_VALID"
                            or src.planned_users != str(N_USERS[dataset]) or src.k != "5"):
                        raise FigureValidationError("G1 strict-valid beyond-utility contract changed")
                    if metric == "aplt_at_5" and not json.loads(src.value_json)["long_tail_threshold"] >= 0:
                        raise FigureValidationError("G1 APLT threshold absent")
                    rows.append({
                        "dataset": dataset, "system_family": "LLM", "model": model,
                        "prompt": prompt, "metric": metric, "value": src.value,
                        "seed_mean_flag": "false", "seed_std_if_available": "",
                        "n_users": src.planned_users,
                        "traditional_population_contract": "full_evaluation_cohort",
                        "llm_population_contract": "strict_valid_plus_metric_defined_users",
                        "comparability_class": CLASSES[metric],
                        "defined_user_note": f"{src.undefined_user_count} undefined/excluded of {src.planned_users} planned users",
                        "source_artifact": G, "source_result_id": src.result_id,
                    })
    values = pd.DataFrame(rows)
    if len(values) != 195 or values.duplicated(["dataset", "system_family", "model", "prompt", "metric"]).any():
        raise FigureValidationError("selected figure must have 5 x 3 x 13 unique cells")
    if not pd.to_numeric(values.value, errors="raise").map(math.isfinite).all():
        raise FigureValidationError("non-finite selected metric value")
    return values


def validate_values(values: pd.DataFrame, root: Path, figure_id: str) -> None:
    """Require every saved value and metadata field to equal canonical mapping."""
    pd.testing.assert_frame_equal(values.reset_index(drop=True), build_values(root, figure_id),
                                  check_dtype=False, check_exact=True)


def _plot(values: pd.DataFrame, figure_id: str) -> plt.Figure:
    metrics, role = FIGURES[figure_id]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.hashsalt": "irp-final-figures-v1", "pdf.compression": 9})
    fig, axes = plt.subplots(5, 3, figsize=(20.5, 18.7))
    positions = list(reversed(range(len(SYSTEM_ORDER))))
    for row_index, dataset in enumerate(DATASETS):
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            selected = values.loc[values.dataset.eq(dataset) & values.metric.eq(metric)]
            maximum = max(float(v) + (float(s) if s else 0)
                          for v, s in zip(selected.value, selected.seed_std_if_available, strict=True))
            for position, label in zip(positions, SYSTEM_ORDER, strict=True):
                if label in (*BASELINES, *TRAINABLE):
                    cell = selected.loc[selected.model.eq(label) & selected.system_family.eq("TRADITIONAL")]
                    color, marker = "#0072B2", "o"
                else:
                    model_label, prompt = label.split(" / ")
                    model = next(m for m, short in LLM_LABELS.items() if short == model_label)
                    cell = selected.loc[selected.model.eq(model) & selected.prompt.eq(prompt)]
                    color, marker = "#D55E00", "s" if prompt == "BASE" else "D"
                if len(cell) != 1:
                    raise FigureValidationError("plot cell not unique")
                point = cell.iloc[0]
                deviation = float(point.seed_std_if_available) if point.seed_std_if_available else None
                ax.errorbar(float(point.value), position, xerr=deviation, fmt=marker,
                            color=color, markersize=5.2, capsize=2.4 if deviation is not None else 0,
                            elinewidth=1.0, linewidth=0, zorder=3)
            ax.axhline(5.5, color="0.45", linewidth=1.0)
            ax.set_xlim(0, maximum * 1.08 if maximum > 0 else 1)
            ax.set_ylim(-0.7, 12.7)
            ax.set_yticks(positions, SYSTEM_ORDER if col_index == 0 else [""] * len(SYSTEM_ORDER))
            ax.grid(axis="x", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            label = f"{DATASET_LABELS[row_index]} (n={N_USERS[dataset]})  |  {METRIC_LABELS[metric]}"
            if metric == "item_coverage_at_5":
                label += "  [descriptive]"
            ax.set_title(label, loc="left", fontsize=10)
            if row_index == len(DATASETS) - 1:
                ax.set_xlabel(METRIC_LABELS[metric])
    title = ("RQ2  |  Long-tail, popularity and novelty" if role == "MAIN_TEXT"
             else "RQ2  |  Serendipity, diversity and item coverage")
    fig.suptitle(title, fontsize=15, y=0.994)
    fig.text(0.013, 0.962, "Traditional recommenders", color="#0072B2", weight="bold")
    fig.text(0.013, 0.948, "LLM rankers: BASE / LLMRANK", color="#D55E00", weight="bold")
    footer = ("Traditional: full cohort; LLM: strict-valid and metric-defined users. BPRMF/GRU4Rec/SASRec/LightGCN bars = across-seed SD, not CI.\n"
              "Original metric units; adapted numerical comparisons, not matched-user causal effects or paper-exact reproduction.")
    if role != "MAIN_TEXT":
        footer += " Item Coverage is descriptive only: its numerator depends on the retained population."
    fig.text(0.5, 0.014, footer, ha="center", fontsize=8)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.934, bottom=0.067, hspace=0.45, wspace=0.25)
    return fig


def _generate_one(root: Path, artifact_root: Path, figure_id: str, commit: str) -> dict[str, object]:
    metrics, role = FIGURES[figure_id]
    values = build_values(root, figure_id)
    values_relative = f"{OUTPUT_ROOT}/values/{figure_id}_values.csv"
    values_path = repository_path(artifact_root, values_relative)
    write_dataframe(values_path, values)
    validate_values(_csv(artifact_root, values_relative), root, figure_id)
    fig = _plot(values, figure_id)
    outputs: dict[str, str] = {}
    for extension in ("png", "pdf", "svg"):
        relative = f"{OUTPUT_ROOT}/figures/{figure_id}.{extension}"
        path = repository_path(artifact_root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = ({"Software": "IRP deterministic final-figure pipeline"} if extension == "png"
                    else {"Creator": "IRP deterministic final-figure pipeline", "Producer": "Matplotlib",
                          "CreationDate": FIXED_TIME, "ModDate": FIXED_TIME} if extension == "pdf"
                    else {"Creator": "IRP deterministic final-figure pipeline", "Date": "2026-09-21"})
        fig.savefig(path, dpi=180, metadata=metadata, facecolor="white")
        outputs[extension] = relative
        outputs[f"{extension}_sha256"] = sha256_file(path)
    plt.close(fig)
    manifest: dict[str, object] = {
        "schema_version": "1.0", "figure_id": figure_id, "rq": "RQ2", "role": role,
        "scientific_purpose": ("Compare principal long-tail, popularity and novelty dimensions by dataset and system."
                               if role == "MAIN_TEXT" else
                               "Compare corrected serendipity, feature diversity and descriptive item coverage."),
        "canonical_source_artifacts": list(SOURCES), "canonical_source_sha256": dict(SOURCES),
        "figure_values_csv": values_relative, "figure_values_sha256": sha256_file(values_path),
        "generator": GENERATOR, "generator_sha256": sha256_file(repository_path(root, GENERATOR)),
        "output_png": outputs["png"], "output_png_sha256": outputs["png_sha256"],
        "output_pdf": outputs["pdf"], "output_pdf_sha256": outputs["pdf_sha256"],
        "output_svg": outputs["svg"], "output_svg_sha256": outputs["svg_sha256"],
        "created_from_commit": commit,
        "scientific_interpretation": "Dataset-specific, metric-specific adapted numerical comparisons; item coverage is descriptive only where shown.",
        "limitations": [
            "Traditional metrics use full evaluation cohorts; G1 beyond-utility uses strict-valid outputs and metric-defined users.",
            "These are not matched-user causal effects or paper-exact numeric reproductions.",
            "F1BZ error bars are five-seed sample standard deviations (ddof=1), not confidence intervals; F1BX and G1 are single frozen values.",
            "Metric panels retain original units and must not be combined into a synthetic beyond-utility score.",
            "Food.com PopREO is undefined for all traditional and G1 cells; it is omitted, never set to zero.",
            "Item coverage uses the candidate-union denominator; its unique-item numerator depends on retained users." if role != "MAIN_TEXT" else
            "APLT uses positive train count below the train-count 0.8 quantile; calACLT is an alias, not another metric.",
        ],
        "metric_contract": {
            "metrics": list(metrics), "comparability_class": {m: CLASSES[m] for m in metrics},
            "top_k": 5, "raw_units_only": True,
            "aplt": "mean_fraction_of_top5_with_positive_train_count_strictly_below_train_count_0.8_quantile",
            "aplt_alias": "calACLT", "aclt_separate_metric": False,
            "arp": "mean_train_interaction_count_of_top5_items",
            "self_information": "mean_minus_log2(train_count/total_train_interactions)_for_defined_users",
            "serendipity": "HR_at_5_times_(1_minus_aligned_MostPop_HR_at_5)_corrected_G1",
            "ild": "mean_pairwise_Jaccard_distance_of_feature_backed_top5_items",
            "item_coverage": "unique_recommended_top5_items_divided_by_candidate_union_size",
        },
        "population_contract": {"traditional": "full_evaluation_cohort", "llm": "strict_valid_plus_metric_defined_users"},
        "system_contract": {"datasets": list(DATASETS), "n_users": N_USERS,
                            "traditional_models": list((*BASELINES, *TRAINABLE)),
                            "llm_models": list(LLMS), "prompts": list(PROMPTS)},
        "calibration_design_decision": {
            "status": "PRESERVED_AS_LOCAL_EXTENSION_NOT_IN_COMPARATIVE_FIGURES",
            "traditional_counterpart_exists": True,
            "reason": "feature-eligible defined-user populations differ materially; retain dedicated frozen LLM calibration figures",
            "existing_llm_figures": list(CALIBRATION_FIGURES),
        },
        "other_omitted_metrics": ["popreo_at_5", "catalog_coverage_at_5", "shannon_category_entropy_at_5",
                                  "gini_at_5", "dpd_at_5", "jain_index_at_5", "oic_at_5", "calibration_at_5"],
    }
    path = repository_path(artifact_root, f"{MANIFEST_ROOT}/{figure_id}_manifest.json")
    write_json(path, manifest)
    validate_figure_manifest(root, path, artifact_root=artifact_root)
    return manifest


def generate_all(root: Path = ROOT, *, artifact_root: Path | None = None,
                 created_from_commit: str | None = None) -> list[dict[str, object]]:
    """Generate exactly two RQ2 comparative figures and provenance chains."""
    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    commit = created_from_commit or git_head(root)
    if len(commit) != 40:
        raise FigureValidationError("created_from_commit must be a full hash")
    validate_sources(root)
    return [_generate_one(root, artifact_root, figure_id, commit) for figure_id in FIGURES]


def validate_all(root: Path = ROOT, *, artifact_root: Path | None = None) -> list[dict[str, object]]:
    """Validate the two saved manifests, source hashes and complete value maps."""
    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    validate_sources(root)
    manifests = []
    for figure_id, (metrics, role) in FIGURES.items():
        path = repository_path(artifact_root, f"{MANIFEST_ROOT}/{figure_id}_manifest.json")
        manifest = validate_figure_manifest(root, path, artifact_root=artifact_root)
        if (manifest["figure_id"] != figure_id or manifest["role"] != role
                or manifest["metric_contract"]["metrics"] != list(metrics)):
            raise FigureValidationError("RQ2 comparative figure contract changed")
        values = _csv(artifact_root, manifest["figure_values_csv"])
        validate_values(values, root, figure_id)
        manifests.append(manifest)
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "validate"), required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--created-from-commit")
    args = parser.parse_args(argv)
    result = (generate_all(ROOT, artifact_root=args.artifact_root,
                           created_from_commit=args.created_from_commit)
              if args.mode == "generate" else validate_all(ROOT, artifact_root=args.artifact_root))
    print(json.dumps({"status": "PASS", "figures": [item["figure_id"] for item in result]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

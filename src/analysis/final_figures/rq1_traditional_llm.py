"""Freeze the RQ1 traditional-versus-LLM utility comparison."""

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

FIGURE_ID = "fig_rq1_traditional_vs_llm_utility"
GENERATOR = "src/analysis/final_figures/rq1_traditional_llm.py"
OUTPUT_ROOT = "results/final_figures/rq1"
VALUES = f"{OUTPUT_ROOT}/values/{FIGURE_ID}_values.csv"
MANIFEST = f"results/manifests/final_figures/{FIGURE_ID}_manifest.json"
SOURCES = {
    "results/tables/f1bx_baseline_utility_results.csv": "2570b536621a05b5ccec860839e6bd6e7dc945ff9010c9dbf6c400b96e55ccbf",
    "results/tables/f1bz_aggregate_utility_results.csv": "1870cc13e8cbed1f34c5bd89d85028ce0e4fff87ce9fa48089de1a59a178ee30",
    "results/tables/f1bz_per_seed_utility_results.csv": "3cca2df12acb45f67e1aba28166073d6633242b14f95392963b36314143f5352",
    "results/tables/g1_primary_ranking_utility.csv": "369db87fb19686525135aa1f5cd9d1b1d6d790e4696aad0b70613cb235b33308",
    "results/manifests/f1bx_execution_manifest.json": "b6251567466574ba397e1f0d5014c0ea357f9973b4ba0f20cad2bbb5334c7ac9",
    "results/manifests/f1bz_final_traditional_execution.json": "99acd7fc2f1ee967cb158b9f9d27912d7d92441048a9cbc47fda3c344f4b7ed8",
    "results/manifests/g1_metric_contract_and_provenance.json": "ca7fc85bd346720c8529698ecfc592da557c06dab1494e2f5b60f1a8c5d7abf6",
    "results/tables/f1bx_baseline_execution.csv": "b1e82279107ad943fb4874e5f84317f9191e1d4b98d134273e68c6f3eb03edf9",
    "results/tables/f1bz_final_run_manifest.csv": "7ecdfa5e2c3d156422c37c0325ec3942d36ac0681de927611965b0f08fb3e0c0",
    "results/raw_outputs/f1bx_baseline_recommendations.csv": "6f7f6c53751038bfa7320898f0aa4f06a049a4bff888fdba369073ebf57cefb6",
    "results/tables/g1_primary_ranking_request_universe.csv.gz": "b4dcf0d80f3c4af0d2353ddec744c0856491d3a958b314ddbd8800811aa347d9",
}
F1BX = "results/tables/f1bx_baseline_utility_results.csv"
F1BZ = "results/tables/f1bz_aggregate_utility_results.csv"
PER_SEED = "results/tables/f1bz_per_seed_utility_results.csv"
G1 = "results/tables/g1_primary_ranking_utility.csv"
F1BX_EXECUTION = "results/tables/f1bx_baseline_execution.csv"
F1BZ_RUNS = "results/tables/f1bz_final_run_manifest.csv"
DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
DATASET_LABELS = ("Beauty", "Sports", "ML-1M", "LastFM", "Food.com")
N_USERS = dict(zip(DATASETS, (990, 892, 1000, 1000, 1000), strict=True))
BASELINES = ("Random", "MostPop", "BM25-like/content")
TRAINABLE = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
LLMS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
LLM_LABELS = {LLMS[0]: "Qwen2", LLMS[1]: "Mistral", LLMS[2]: "Gemma2"}
PROMPTS = ("BASE", "LLMRANK")
METRICS = ("hr_at_5", "ndcg_at_5")
METRIC_LABELS = {"hr_at_5": "HR@5", "ndcg_at_5": "NDCG@5"}
ORDER = (*BASELINES, *TRAINABLE, *(f"{LLM_LABELS[m]} / {p}" for m in LLMS for p in PROMPTS))
FIXED_TIME = datetime(2026, 9, 21, tzinfo=timezone.utc)


def _csv(root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(repository_path(root, relative), dtype=str, keep_default_na=False)


def validate_sources(root: Path = ROOT) -> dict[str, str]:
    """Check frozen source identities and local cohort/candidate contracts."""
    for relative, expected in SOURCES.items():
        path = repository_path(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise FigureValidationError(f"canonical source absent or changed: {relative}")
    baseline_manifest = json.loads(repository_path(root, "results/manifests/f1bx_execution_manifest.json").read_text(encoding="utf-8"))
    train_manifest = json.loads(repository_path(root, "results/manifests/f1bz_final_traditional_execution.json").read_text(encoding="utf-8"))
    g1_contract = json.loads(repository_path(root, "results/manifests/g1_metric_contract_and_provenance.json").read_text(encoding="utf-8"))
    for relative in (F1BX, F1BX_EXECUTION):
        if baseline_manifest["artifact_hashes"][relative] != SOURCES[relative]:
            raise FigureValidationError(f"F1BX freeze mismatch: {relative}")
    for relative in (F1BZ, PER_SEED, F1BZ_RUNS):
        if train_manifest["artifact_hashes"][relative] != SOURCES[relative]:
            raise FigureValidationError(f"F1BZ freeze mismatch: {relative}")
    if g1_contract["canonical_artifact_authority"]["canonical_utility_sha256"] != SOURCES[G1]:
        raise FigureValidationError("G1 utility contract mismatch")
    if g1_contract["mrr_contract"]["inspected_ranking_depth"] != 10:
        raise FigureValidationError("G1 MRR depth changed")
    baseline = _csv(root, F1BX_EXECUTION)
    runs = _csv(root, F1BZ_RUNS)
    if len(baseline) != 15 or len(runs) != 100:
        raise FigureValidationError("traditional run inventory incomplete")
    if set(baseline.dataset) != set(DATASETS) or set(baseline.method) != set(BASELINES):
        raise FigureValidationError("F1BX dataset/method grid changed")
    if set(runs.dataset) != set(DATASETS) or set(runs.method) != set(TRAINABLE):
        raise FigureValidationError("F1BZ dataset/method grid changed")
    if set(runs.seed) != {"42", "43", "44", "45", "46"}:
        raise FigureValidationError("F1BZ seed grid changed")
    if not runs.training_status.eq("COMPLETED").all():
        raise FigureValidationError("F1BZ contains incomplete runs")
    if not baseline.status.eq("COMPUTED").all():
        raise FigureValidationError("F1BX contains incomplete baseline rows")
    baseline_by_dataset = baseline.groupby("dataset", sort=False)
    for dataset in DATASETS:
        group = baseline_by_dataset.get_group(dataset)
        if not group.cohort_N.eq(str(N_USERS[dataset])).all() or group.candidate_set_identity.nunique() != 1:
            raise FigureValidationError(f"F1BX cohort/candidate contract changed: {dataset}")
        matching = runs.loc[runs.dataset.eq(dataset)]
        if not matching.evaluation_users.eq(str(N_USERS[dataset])).all():
            raise FigureValidationError(f"F1BZ cohort size changed: {dataset}")
        if not matching.cohort_id.eq(group.cohort_id.iloc[0]).all():
            raise FigureValidationError(f"F1BZ cohort identity differs: {dataset}")
        if not matching.candidate_contract_hash.eq(group.candidate_set_identity.iloc[0]).all():
            raise FigureValidationError(f"F1BZ candidate contract differs: {dataset}")
    # F1BX and G1 use different hash encodings, so verify actual item sets and
    # positive targets for every user rather than comparing their hash strings.
    baseline_items = _csv(root, "results/raw_outputs/f1bx_baseline_recommendations.csv")
    baseline_items = baseline_items.loc[baseline_items.method.eq("MostPop")]
    requests = _csv(root, "results/tables/g1_primary_ranking_request_universe.csv.gz")
    keys = ["dataset", "user_id"]
    if len(requests) != 29292 or not requests.groupby(keys, sort=False).size().eq(6).all():
        raise FigureValidationError("G1 request universe does not reuse each user six times")
    if not requests.candidate_count.eq("20").all() or not requests.metric_k.eq("5").all():
        raise FigureValidationError("G1 candidate size/top-5 protocol changed")
    if not requests.groupby(keys, sort=False).candidate_items.nunique().eq(1).all():
        raise FigureValidationError("G1 candidate items vary by model/prompt")
    if not requests.groupby(keys, sort=False).positive_item_id.nunique().eq(1).all():
        raise FigureValidationError("G1 target varies by model/prompt")
    if len(baseline_items) != 97640 or not baseline_items.groupby(keys, sort=False).size().eq(20).all():
        raise FigureValidationError("F1BX MostPop candidate universe incomplete")
    grouped = baseline_items.groupby(keys, sort=False)
    candidates = grouped.recommended_item_id.agg(lambda items: tuple(sorted(items.astype(str))))
    targets = grouped.positive_item_id.first()
    g1_rows = requests.drop_duplicates(keys).set_index(keys)
    if len(g1_rows) != 4882 or set(g1_rows.index) != set(candidates.index):
        raise FigureValidationError("G1 and traditional user cohorts differ")
    for key, row in g1_rows.iterrows():
        items = json.loads(row.candidate_items)
        if len(items) != 20 or len(set(items)) != 20 or row.positive_item_id not in items:
            raise FigureValidationError("G1 candidate pool/forced positive invalid")
        if tuple(sorted(map(str, items))) != candidates.loc[key] or row.positive_item_id != targets.loc[key]:
            raise FigureValidationError(f"G1/F1BX candidate or target mismatch: {key}")
    return dict(SOURCES)


def build_values(root: Path = ROOT) -> pd.DataFrame:
    """Map every plotted value to the frozen metric row and seed summary."""
    validate_sources(root)
    b = _csv(root, F1BX)
    f = _csv(root, F1BZ)
    s = _csv(root, PER_SEED)
    g = _csv(root, G1)
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for metric in METRICS:
            for model in (*BASELINES, *TRAINABLE):
                if model in BASELINES:
                    subset = b.loc[b.dataset.eq(dataset) & b.method.eq(model) & b.metric.eq(metric)]
                    if len(subset) != 1 or subset.iloc[0].status != "COMPUTED" or subset.iloc[0].evaluated_users != str(N_USERS[dataset]):
                        raise FigureValidationError(f"F1BX metric cell invalid: {dataset}/{model}/{metric}")
                    src = subset.iloc[0]
                    value, std, seeds, kind, source = src.value, "", "1", "false", F1BX
                else:
                    subset = f.loc[f.dataset.eq(dataset) & f.method.eq(model) & f.metric.eq(metric)]
                    seed_rows = s.loc[s.dataset.eq(dataset) & s.method.eq(model) & s.metric.eq(metric)]
                    if len(subset) != 1 or len(seed_rows) != 5 or set(seed_rows.seed) != {"42", "43", "44", "45", "46"}:
                        raise FigureValidationError(f"F1BZ metric/seed cell incomplete: {dataset}/{model}/{metric}")
                    src = subset.iloc[0]
                    if src.status != "COMPUTED" or src.seed_count != "5" or src.computed_seed_count != "5":
                        raise FigureValidationError("F1BZ aggregate status/seed count changed")
                    if not seed_rows.status.eq("COMPUTED").all() or not seed_rows.evaluated_users.eq(str(N_USERS[dataset])).all():
                        raise FigureValidationError("F1BZ per-seed status/cohort changed")
                    numeric = pd.to_numeric(seed_rows.value, errors="raise")
                    if not math.isclose(float(src.mean_value), float(numeric.mean()), rel_tol=0, abs_tol=1e-14):
                        raise FigureValidationError("F1BZ mean differs from seed values")
                    if not math.isclose(float(src.std_value), float(numeric.std(ddof=1)), rel_tol=0, abs_tol=1e-14):
                        raise FigureValidationError("F1BZ std differs from seed values")
                    value, std, seeds, kind, source = src.mean_value, src.std_value, "5", "true", F1BZ
                rows.append({
                    "dataset": dataset, "system_family": "TRADITIONAL", "model": model,
                    "prompt": "", "metric": metric, "value": value,
                    "seed_mean_flag": kind, "seed_std_if_available": std,
                    "seed_count": seeds, "n_users": str(N_USERS[dataset]),
                    "comparability_class": "DIRECT_NUMERICAL_COMPARISON",
                    "source_artifact": source, "source_result_id": src.result_id,
                })
            for model in LLMS:
                for prompt in PROMPTS:
                    subset = g.loc[g.dataset.eq(dataset) & g.model_name.eq(model) & g.prompt_variant.eq(prompt) & g.metric.eq(metric)]
                    if len(subset) != 1:
                        raise FigureValidationError("G1 configuration cell missing or duplicated")
                    src = subset.iloc[0]
                    if src.status != "COMPUTED" or src.evidence_view != "PRIMARY_FAIL_STRICT" or src.planned_users != str(N_USERS[dataset]) or src.k != "5":
                        raise FigureValidationError("G1 utility contract changed")
                    rows.append({
                        "dataset": dataset, "system_family": "LLM", "model": model,
                        "prompt": prompt, "metric": metric, "value": src.value,
                        "seed_mean_flag": "false", "seed_std_if_available": "",
                        "seed_count": "1", "n_users": src.planned_users,
                        "comparability_class": "DIRECT_NUMERICAL_COMPARISON",
                        "source_artifact": G1, "source_result_id": src.result_id,
                    })
    values = pd.DataFrame(rows)
    if len(values) != 130 or set(values.metric) != set(METRICS):
        raise FigureValidationError("figure must have 5 datasets x 13 systems x 2 metrics")
    numbers = pd.to_numeric(values.value, errors="raise")
    if not numbers.between(0, 1).all():
        raise FigureValidationError("utility value outside [0, 1]")
    return values


def validate_values(values: pd.DataFrame, root: Path = ROOT) -> None:
    """Require the saved values table to match the canonical source mapping."""
    expected = build_values(root)
    pd.testing.assert_frame_equal(values.reset_index(drop=True), expected, check_dtype=False, check_exact=True)


def _plot(values: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "svg.hashsalt": "irp-final-figures-v1", "pdf.compression": 9,
    })
    fig, axes = plt.subplots(5, 2, figsize=(15.6, 19.0), sharex=True)
    positions = list(reversed(range(len(ORDER))))
    for row_index, dataset in enumerate(DATASETS):
        for col_index, metric in enumerate(METRICS):
            ax = axes[row_index, col_index]
            cells = values.loc[values.dataset.eq(dataset) & values.metric.eq(metric)]
            for position, label in zip(positions, ORDER, strict=True):
                if label in (*BASELINES, *TRAINABLE):
                    selected = cells.loc[cells.model.eq(label) & cells.system_family.eq("TRADITIONAL")]
                    color, marker = "#0072B2", "o"
                else:
                    model_label, prompt = label.split(" / ")
                    model = next(m for m, short in LLM_LABELS.items() if short == model_label)
                    selected = cells.loc[cells.model.eq(model) & cells.prompt.eq(prompt)]
                    color, marker = "#D55E00", "s" if prompt == "BASE" else "D"
                if len(selected) != 1:
                    raise FigureValidationError("plot cell not unique")
                cell = selected.iloc[0]
                std = float(cell.seed_std_if_available) if cell.seed_std_if_available else None
                ax.errorbar(float(cell.value), position, xerr=std, fmt=marker,
                            color=color, markersize=5.4, capsize=2.6 if std is not None else 0,
                            elinewidth=1.1, linewidth=0, zorder=3)
            ax.axhline(5.5, color="0.45", linewidth=1.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.7, 12.7)
            ax.set_yticks(positions, ORDER if col_index == 0 else [""] * len(ORDER))
            ax.grid(axis="x", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{DATASET_LABELS[row_index]} (n={N_USERS[dataset]})  |  {METRIC_LABELS[metric]}", loc="left", fontsize=11)
            if row_index == 4:
                ax.set_xlabel("Mean top-5 utility (0–1)")
    fig.suptitle("RQ1  |  Traditional recommenders and LLM candidate-set rankers", fontsize=15, y=0.993)
    fig.text(0.02, 0.96, "Traditional recommenders", color="#0072B2", weight="bold", fontsize=10)
    fig.text(0.02, 0.948, "LLM rankers: BASE / LLMRANK", color="#D55E00", weight="bold", fontsize=10)
    fig.text(0.5, 0.013,
             "Shared local positive-forced 20-candidate protocol. Bars on BPRMF/GRU4Rec/SASRec/LightGCN = across-seed SD (5 seeds), not confidence intervals.\n"
             "G1 invalid outputs contribute zero; traditional rankings are complete. MRR excluded (depth 10 vs 20). No paper-exact identity claim.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.934, bottom=0.065, hspace=0.45, wspace=0.12)
    return fig


def generate(root: Path = ROOT, *, artifact_root: Path | None = None,
             created_from_commit: str | None = None) -> dict[str, object]:
    """Create the values, three renderings, and a complete hash manifest."""
    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    values = build_values(root)
    commit = created_from_commit or git_head(root)
    if len(commit) != 40:
        raise FigureValidationError("created_from_commit must be a full hash")
    values_path = repository_path(artifact_root, VALUES)
    write_dataframe(values_path, values)
    validate_values(_csv(artifact_root, VALUES), root)
    fig = _plot(values)
    outputs: dict[str, str] = {}
    for extension in ("png", "pdf", "svg"):
        relative = f"{OUTPUT_ROOT}/figures/{FIGURE_ID}.{extension}"
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
        "schema_version": "1.0", "figure_id": FIGURE_ID, "rq": "RQ1",
        "scientific_purpose": "Compare local traditional and LLM candidate-set ranking utility by dataset without pooling domains.",
        "canonical_source_artifacts": list(SOURCES), "canonical_source_sha256": dict(SOURCES),
        "figure_values_csv": VALUES, "figure_values_sha256": sha256_file(values_path),
        "generator": GENERATOR, "generator_sha256": sha256_file(repository_path(root, GENERATOR)),
        "output_png": outputs["png"], "output_png_sha256": outputs["png_sha256"],
        "output_pdf": outputs["pdf"], "output_pdf_sha256": outputs["pdf_sha256"],
        "output_svg": outputs["svg"], "output_svg_sha256": outputs["svg_sha256"],
        "created_from_commit": commit,
        "scientific_interpretation": "HR@5 and NDCG@5 are directly numerically comparable within the shared local adapted 20-candidate protocol; differences are dataset- and configuration-specific.",
        "limitations": [
            "Not historically identical to the target paper protocol; no cross-dataset pooled score or overall winner.",
            "G1 terminal-invalid outputs contribute zero over all planned users; deterministic traditional rankings are complete.",
            "Traditional systems rank 20 candidates; G1 returns 10. Only top-5 utility is compared; traditional depth-20 diagnostic MRR and G1 depth-10 MRR are excluded.",
            "BPRMF, GRU4Rec, SASRec and LightGCN points are five-seed means; error bars are across-seed standard deviations, not confidence intervals.",
            "Random, MostPop and BM25-like/content have single deterministic values; G1 has one frozen value per model/prompt condition.",
        ],
        "comparison_contract": {
            "classification": "DIRECT_NUMERICAL_COMPARISON_WITHIN_LOCAL_ADAPTED_PROTOCOL",
            "metrics": list(METRICS), "excluded_metric": "mrr",
            "datasets": list(DATASETS), "n_users": N_USERS,
            "traditional_models": list((*BASELINES, *TRAINABLE)),
            "llm_models": list(LLMS), "prompts": list(PROMPTS),
            "candidate_pool_size": 20, "positive_forced": True, "top_k": 5,
            "f1bz_seed_count": 5, "f1bz_seeds": [42, 43, 44, 45, 46],
            "f1bz_variability": "sample_standard_deviation_ddof_1_across_five_seed_level_metric_values",
            "g1_invalid_terminal": "zero_in_planned_cohort_denominator",
        },
    }
    manifest_path = repository_path(artifact_root, MANIFEST)
    write_json(manifest_path, manifest)
    validate_figure_manifest(root, manifest_path, artifact_root=artifact_root)
    return manifest


def validate(root: Path = ROOT, *, artifact_root: Path | None = None) -> dict[str, object]:
    """Validate all hashes and every saved figure value against canonical cells."""
    root = root.resolve()
    artifact_root = (artifact_root or root).resolve()
    validate_sources(root)
    manifest = validate_figure_manifest(root, repository_path(artifact_root, MANIFEST), artifact_root=artifact_root)
    if manifest["figure_id"] != FIGURE_ID or manifest["comparison_contract"]["metrics"] != list(METRICS):
        raise FigureValidationError("figure contract changed")
    validate_values(_csv(artifact_root, VALUES), root)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "validate"), required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--created-from-commit")
    args = parser.parse_args(argv)
    result = (generate(ROOT, artifact_root=args.artifact_root, created_from_commit=args.created_from_commit)
              if args.mode == "generate" else validate(ROOT, artifact_root=args.artifact_root))
    print(json.dumps({"status": "PASS", "figure_id": result["figure_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

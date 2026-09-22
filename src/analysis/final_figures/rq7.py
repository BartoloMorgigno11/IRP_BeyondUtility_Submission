"""Final RQ7 figure: upstream candidate recall and conditional re-ranking utility."""

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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_figures.common import (  # noqa: E402
    FigureValidationError, git_head, repository_path, sha256_file,
    validate_figure_manifest, write_dataframe, write_json,
)

FIGURE_ID = "fig_rq7_candidate_recall_and_reranking_utility"
PREFIX = "results/tables/g4_f1bzq_"
RECALL = PREFIX + "candidate_recall_at_20.csv"
METRICS = PREFIX + "metric_results.csv"
CONDITIONAL = PREFIX + "candidate_conditional_metrics.csv"
DENOMINATORS = PREFIX + "metric_denominators.csv"
DENOM_CONTRACT = PREFIX + "metric_denominator_contract.csv"
RELIABILITY = PREFIX + "reliability.csv"
STATISTICS = PREFIX + "statistical_results.csv"
HASHES_TABLE = PREFIX + "candidate_hashes.csv"
REQUESTS = PREFIX + "reranking_request_universe.csv"
PROVENANCE = "results/manifests/g4_f1bzq_canonical_provenance.json"
REPRO = "results/manifests/g4_f1bzq_reproducibility_contract.json"
PROTOCOL = "results/manifests/g4_f1bzq_reranking_protocol_freeze.json"
METRIC_FREEZE = "results/manifests/g4_f1bzq_metrics_freeze.json"
STAT_FREEZE = "results/manifests/g4_f1bzq_statistical_analysis_manifest.json"
POOL_FREEZE = "results/manifests/f1bzq_authoritative_pool_manifest.json"
POOL_CONFIG = "configs/f1bzq_authoritative_reranking_pool_materialization.yaml"
SOURCES = [RECALL, METRICS, CONDITIONAL, DENOMINATORS, DENOM_CONTRACT,
           RELIABILITY, STATISTICS, HASHES_TABLE, REQUESTS,
           PROVENANCE, REPRO, PROTOCOL, METRIC_FREEZE, STAT_FREEZE,
           POOL_FREEZE, POOL_CONFIG]
OUT = "results/final_figures/rq7"
MANIFEST = f"results/manifests/final_figures/{FIGURE_ID}_manifest.json"
GENERATOR = "src/analysis/final_figures/rq7.py"
FIXED_TIME = datetime(2026, 9, 21, tzinfo=timezone.utc)
DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
DATASET_LABELS = {"beauty": "Beauty", "sports": "Sports", "ml1m": "ML-1M", "lastfm": "LastFM", "foodcom": "Food.com"}
MODELS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
MODEL_LABELS = {MODELS[0]: "Qwen2", MODELS[1]: "Mistral", MODELS[2]: "Gemma2"}
PROMPTS = ("BASE", "LLMRANK")
COLORS = {MODELS[0]: "#0072B2", MODELS[1]: "#D55E00", MODELS[2]: "#009E73"}
EXPECTED_RECALL = {"beauty": (49, 990), "sports": (28, 892), "ml1m": (193, 1000), "lastfm": (469, 1000), "foodcom": (4, 1000)}
METRIC_MAP = {
    "hr_at_5": "unconditional_hr_at_5",
    "conditional_hr_at_5": "conditional_hr_at_5",
    "ndcg_at_5": "unconditional_ndcg_at_5",
    "conditional_ndcg_at_5": "conditional_ndcg_at_5",
    "mrr": "unconditional_mrr",
}


def _json(root: Path, relative: str) -> dict:
    return json.loads(repository_path(root, relative).read_text(encoding="utf-8"))


def _csv(root: Path, relative: str, *, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(repository_path(root, relative), dtype=str, keep_default_na=False, usecols=usecols)


def validate_sources(root: Path = ROOT) -> dict[str, str]:
    """Validate the canonical G4/F1BZQ provenance and all plotted source hashes."""
    root = root.resolve()
    provenance = _json(root, PROVENANCE)
    registry = {item["path"]: item["sha256"] for item in provenance["canonical_artifacts"] if item["authority"] == "CANONICAL"}
    if provenance["g1_g4_comparison"] != {"causal_ranking_vs_reranking_claim": "NOT_SUPPORTED", "classification": "DESCRIPTIVE_ONLY"}:
        raise FigureValidationError("G1/G4 comparison classification changed")
    hashes = {}
    for relative in SOURCES:
        path = repository_path(root, relative)
        if not path.is_file():
            raise FigureValidationError(f"G4 source missing: {relative}")
        digest = sha256_file(path)
        if relative in registry and digest != registry[relative]:
            raise FigureValidationError(f"G4 canonical provenance hash mismatch: {relative}")
        hashes[relative] = digest
    for relative in (RECALL, METRICS, CONDITIONAL, DENOMINATORS, DENOM_CONTRACT,
                     RELIABILITY, STATISTICS, HASHES_TABLE, PROTOCOL,
                     METRIC_FREEZE, STAT_FREEZE, POOL_FREEZE, POOL_CONFIG):
        if relative not in registry:
            raise FigureValidationError(f"required G4 source absent from canonical registry: {relative}")
    protocol = _json(root, PROTOCOL)
    if protocol["artifact_hashes"].get(REQUESTS) != hashes[REQUESTS]:
        raise FigureValidationError("G4 request-universe hash differs from protocol freeze")
    if protocol["candidate_pool_algorithm"] != "BALANCED_RANK_ROUND_ROBIN_UNIQUE_20" or protocol["candidate_source"] != "F1BZQ_ONLY":
        raise FigureValidationError("G4 protocol source/algorithm changed")
    pool = _json(root, POOL_FREEZE)
    if pool["authoritative_pool_users"] != 4882 or len(pool["candidate_recall"]) != 5:
        raise FigureValidationError("F1BZQ pool identity changed")
    if pool["artifact_hashes"][POOL_CONFIG] != hashes[POOL_CONFIG]:
        raise FigureValidationError("F1BZQ pool config hash mismatch")
    metrics_freeze = _json(root, METRIC_FREEZE)
    for relative in (METRICS, CONDITIONAL, DENOMINATORS, RELIABILITY):
        if metrics_freeze["output_hashes"][relative] != hashes[relative]:
            raise FigureValidationError(f"G4 metric freeze mismatch: {relative}")
    stats_freeze = _json(root, STAT_FREEZE)
    if stats_freeze["output_hashes"][STATISTICS] != hashes[STATISTICS] or stats_freeze["summary"]["inferential_result_rows"] != 225:
        raise FigureValidationError("G4 statistical freeze mismatch")
    repro = _json(root, REPRO)
    if repro["canonical_provenance_manifest"] != PROVENANCE or repro["tiers"]["TIER_1_FROZEN_REPRODUCTION"]["requires_llm_inference"] is not False:
        raise FigureValidationError("G4 reproducibility contract changed")
    return hashes


def _pool_contract(root: Path) -> dict[str, tuple[int, int]]:
    """Check 20-item, unforced, identical user pools across six LLM conditions."""
    hashes = _csv(root, HASHES_TABLE)
    keys = ["dataset", "user_id"]
    if len(hashes) != 4882 or hashes.duplicated(keys).any():
        raise FigureValidationError("candidate hash table lacks one row per user")
    if not hashes.positive_present.isin(["true", "false"]).all():
        raise FigureValidationError("candidate target-presence values changed")
    if not hashes.candidate_items.map(lambda raw: len(json.loads(raw)) == 20).all():
        raise FigureValidationError("F1BZQ pool is not size 20")
    requests = _csv(root, REQUESTS, usecols=["dataset", "user_id", "model", "prompt_strategy",
                                          "pool_hash", "candidate_hash", "candidate_count",
                                          "candidate_items", "force_positive", "positive_present",
                                          "candidate_order_immutable_for_llm", "source_recommenders"])
    if len(requests) != 29292 or requests.duplicated(keys + ["model", "prompt_strategy"]).any():
        raise FigureValidationError("G4 six-condition request universe changed")
    if set(requests.model) != set(MODELS) or set(requests.prompt_strategy) != set(PROMPTS):
        raise FigureValidationError("G4 model/prompt universe changed")
    if not requests.candidate_count.eq("20").all() or not requests.force_positive.eq("false").all():
        raise FigureValidationError("G4 size-20/no-positive-forcing contract changed")
    if not requests.candidate_order_immutable_for_llm.eq("true").all():
        raise FigureValidationError("G4 candidate ordering not fixed")
    expected_sources = '["BPRMF","GRU4Rec","SASRec","LightGCN"]'
    if not requests.source_recommenders.eq(expected_sources).all():
        raise FigureValidationError("G4 first-stage source models changed")
    grouped = requests.groupby(keys, sort=False)
    if not grouped.size().eq(6).all():
        raise FigureValidationError("not all user pools reused six times")
    for field in ("pool_hash", "candidate_hash", "candidate_items", "positive_present"):
        if not grouped[field].nunique().eq(1).all():
            raise FigureValidationError(f"pool field differs across LLM/prompt: {field}")
    one = requests.drop_duplicates(keys).merge(hashes[keys + ["pool_hash", "candidate_hash", "candidate_items", "positive_present"]], on=keys, suffixes=("_request", "_pool"), validate="one_to_one")
    if len(one) != len(hashes):
        raise FigureValidationError("request pools do not match candidate hash table")
    for field in ("pool_hash", "candidate_hash", "candidate_items", "positive_present"):
        if not one[f"{field}_request"].eq(one[f"{field}_pool"]).all():
            raise FigureValidationError(f"request/pool identity mismatch: {field}")
    result = {dataset: (int(group.positive_present.eq("true").sum()), len(group)) for dataset, group in hashes.groupby("dataset")}
    if result != EXPECTED_RECALL:
        raise FigureValidationError("F1BZQ target-present counts changed")
    return result


def _contract_rows(root: Path) -> pd.DataFrame:
    rows = _csv(root, DENOM_CONTRACT)
    by_id = rows.set_index("metric_id")
    for metric in ("hr_at_5", "ndcg_at_5", "mrr"):
        if by_id.loc[metric, "denominator"] != "planned_applicable_users_invalid_terminal_zero":
            raise FigureValidationError(f"unconditional denominator changed: {metric}")
    for metric in ("conditional_hr_at_5", "conditional_ndcg_at_5"):
        if by_id.loc[metric, "denominator"] != "positive_present_users_only":
            raise FigureValidationError(f"conditional denominator changed: {metric}")
    if "conditional_mrr" in set(rows.metric_id):
        raise FigureValidationError("unexpected conditional MRR in canonical contract")
    return rows


def figure_values(root: Path = ROOT) -> pd.DataFrame:
    """Join frozen G4 metric cells to the shared candidate-stage recall evidence."""
    validate_sources(root)
    pool_recall = _pool_contract(root)
    _contract_rows(root)
    recall = _csv(root, RECALL)
    if len(recall) != 5 or set(recall.dataset) != set(DATASETS):
        raise FigureValidationError("G4 recall table incomplete")
    for row in recall.itertuples(index=False):
        present, total = pool_recall[row.dataset]
        if (int(row.positive_present_N), int(row.evaluation_N)) != (present, total):
            raise FigureValidationError("recall table differs from candidate hashes")
        if row.force_positive != "false" or row.target_identity_used_for_ranking != "false":
            raise FigureValidationError("positive forcing/target-aware ranking detected")
        if not np.isclose(float(row.candidate_recall_at_20), present / total, atol=1e-15, rtol=0):
            raise FigureValidationError("candidate recall proportion changed")
    metrics = _csv(root, METRICS)
    conditional = _csv(root, CONDITIONAL)
    denominators = _csv(root, DENOMINATORS)
    reliability = _csv(root, RELIABILITY)
    keys = ["dataset", "model", "prompt_strategy"]
    subset = metrics.loc[metrics.metric_id.isin(METRIC_MAP)].copy()
    if len(subset) != 150 or subset.duplicated(keys + ["metric_id"]).any():
        raise FigureValidationError("G4 utility cells incomplete")
    if set(subset.dataset) != set(DATASETS) or set(subset.model) != set(MODELS) or set(subset.prompt_strategy) != set(PROMPTS):
        raise FigureValidationError("G4 utility configuration universe changed")
    if not subset.loc[subset.metric_id.isin(["hr_at_5", "ndcg_at_5", "mrr"]), "denominator_scope"].eq("all planned users").all():
        raise FigureValidationError("unconditional population changed")
    if not subset.loc[subset.metric_id.str.startswith("conditional_"), "denominator_scope"].eq("positive-present users only").all():
        raise FigureValidationError("conditional population changed")
    if not subset.loc[subset.metric_id.isin(["hr_at_5", "ndcg_at_5", "mrr"]), "invalid_terminal_policy"].eq("invalid terminal response contributes zero; positive-absent users contribute zero").all():
        raise FigureValidationError("unconditional invalid/absent policy changed")
    if not subset.loc[subset.metric_id.str.startswith("conditional_"), "invalid_terminal_policy"].eq("invalid terminal response contributes zero within positive-present subset").all():
        raise FigureValidationError("conditional invalid policy changed")
    cond = conditional.loc[conditional.metric_id.isin(["conditional_hr_at_5", "conditional_ndcg_at_5", "candidate_recall_at_20"])]
    joined = subset.loc[subset.metric_id.str.startswith("conditional_")].merge(cond, on=keys + ["metric_id"], suffixes=("_main", "_conditional"), validate="one_to_one")
    if len(joined) != 60 or not np.allclose(joined.value_main.astype(float), joined.value_conditional.astype(float), atol=1e-15, rtol=0):
        raise FigureValidationError("conditional metric tables disagree")
    den = denominators.loc[denominators.metric_id.isin(METRIC_MAP)]
    matched = subset.merge(den, on=keys + ["metric_id"], suffixes=("_metric", "_den"), validate="one_to_one")
    if len(matched) != 150 or not matched.planned_users_metric.eq(matched.planned_users_den).all() or not matched.positive_present_users_metric.eq(matched.positive_present_users_den).all():
        raise FigureValidationError("G4 denominator metadata differs")
    for row in matched.itertuples(index=False):
        expected_den = int(row.positive_present_users_metric) if row.metric_id.startswith("conditional_") else int(row.planned_users_metric)
        if int(row.denominator) != expected_den or int(row.denominator_value) != expected_den:
            raise FigureValidationError("G4 utility denominator differs from contract")
    if len(reliability) != 30 or reliability.duplicated(keys).any():
        raise FigureValidationError("G4 reliability cell universe changed")
    rows = []
    for dataset in DATASETS:
        rec = recall.loc[recall.dataset.eq(dataset)].iloc[0]
        present, planned = pool_recall[dataset]
        for model in MODELS:
            for prompt in PROMPTS:
                cell = subset.loc[subset.dataset.eq(dataset) & subset.model.eq(model) & subset.prompt_strategy.eq(prompt)]
                rel = reliability.loc[reliability.dataset.eq(dataset) & reliability.model.eq(model) & reliability.prompt_strategy.eq(prompt)]
                if len(cell) != 5 or len(rel) != 1:
                    raise FigureValidationError("G4 model/prompt cell incomplete")
                rel = rel.iloc[0]
                if (int(rel.planned_requests), int(rel.strict_valid) + int(rel.structural_invalid) + int(rel.backend_errors)) != (planned, planned):
                    raise FigureValidationError("G4 reliability denominator differs")
                if any(int(item.planned_users) != planned or int(item.positive_present_users) != present for item in cell.itertuples(index=False)):
                    raise FigureValidationError("G4 candidate-present denominator differs")
                values = {METRIC_MAP[item.metric_id]: float(item.value) for item in cell.itertuples(index=False)}
                hr_expected = (present / planned) * values["conditional_hr_at_5"]
                ndcg_expected = (present / planned) * values["conditional_ndcg_at_5"]
                rows.append({
                    "dataset": dataset, "model": model, "prompt": prompt,
                    "planned_n": planned, "candidate_present_n": present,
                    "candidate_recall_at_20": float(rec.candidate_recall_at_20),
                    **values, "conditional_population_n": present,
                    "foodcom_diagnostic_flag": dataset == "foodcom",
                    "hr_expected_from_decomposition": hr_expected,
                    "hr_observed": values["unconditional_hr_at_5"],
                    "hr_decomposition_error": abs(hr_expected - values["unconditional_hr_at_5"]),
                    "ndcg_expected_from_decomposition": ndcg_expected,
                    "ndcg_observed": values["unconditional_ndcg_at_5"],
                    "ndcg_decomposition_error": abs(ndcg_expected - values["unconditional_ndcg_at_5"]),
                    "unconditional_denominator_contract": "all_planned_users_invalid_terminal_zero_positive_absent_zero",
                    "conditional_denominator_contract": "positive_present_users_only_invalid_terminal_zero",
                    "terminal_invalid_policy": "zero_utility_in_both_populations",
                })
    result = pd.DataFrame(rows)
    if len(result) != 30 or result.hr_decomposition_error.max() > 1e-14 or result.ndcg_decomposition_error.max() > 1e-14:
        raise FigureValidationError("G4 single-positive decomposition failed")
    stats = _csv(root, STATISTICS)
    inferential = stats.loc[stats.inferential_test_allowed.eq("true")]
    counts = inferential.ci_supported_decision.value_counts().to_dict()
    if len(inferential) != 225 or counts != {"NON_SIGNIFICANT_CI_INCLUDES_ZERO": 172, "CI_SUPPORTED_POSITIVE": 29, "CI_SUPPORTED_NEGATIVE": 24}:
        raise FigureValidationError("internal G4 bootstrap decision accounting changed")
    return result


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.hashsalt": "irp-final-rq7-v1", "pdf.compression": 9})


def _paired_axis(ax: plt.Axes, values: pd.DataFrame, metric: str, dataset: str) -> None:
    """Show six paired conditional/end-to-end dots, never cross-user lines."""
    cell = values.loc[values.dataset.eq(dataset)]
    cond_key = f"conditional_{metric}"
    unconditional_key = f"unconditional_{metric}"
    for i, (model, prompt) in enumerate((m, p) for m in MODELS for p in PROMPTS):
        row = cell.loc[cell.model.eq(model) & cell.prompt.eq(prompt)].iloc[0]
        a, b = float(row[unconditional_key]), float(row[cond_key])
        ax.plot([a, b], [i, i], color="#AAAAAA", lw=1, zorder=1)
        ax.scatter(a, i, s=26, color=COLORS[model], marker="o", zorder=3)
        ax.scatter(b, i, s=39, facecolors="none" if prompt == "BASE" else COLORS[model],
                   edgecolors=COLORS[model], marker="X" if dataset == "foodcom" else "D", zorder=4)
    ax.set_yticks(range(6), [f"{MODEL_LABELS[m]} {p}" for m in MODELS for p in PROMPTS])
    ax.set_ylim(5.55, -.55); ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    ax.set_title(f"{DATASET_LABELS[dataset]}" + ("  •  target-present n=4, diagnostic only" if dataset == "foodcom" else ""), loc="left", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Utility (0–1)")


def render(values: pd.DataFrame) -> plt.Figure:
    """Show candidate availability, paired G4 utility and denominator identity."""
    fig = plt.figure(figsize=(17.4, 17.8))
    fig.suptitle("RQ7 | Two-stage candidate-aware re-ranking: availability and conditional utility", y=.99, fontsize=16)
    fig.text(.5, .961, "BPRMF / GRU4Rec / SASRec / LightGCN  →  shared top-20 pool  →  LLM re-ranking  →  final ranking", ha="center", fontsize=10.5)
    gs = fig.add_gridspec(6, 2, left=.14, right=.97, bottom=.075, top=.94,
                          height_ratios=[1.45, 1, 1, 1, 1, 1], hspace=.95, wspace=.23)
    ax_a = fig.add_subplot(gs[0, 0]); ax_d = fig.add_subplot(gs[0, 1])
    recall = values.drop_duplicates("dataset").set_index("dataset").loc[list(DATASETS)]
    y = np.arange(5)
    ax_a.barh(y, recall.candidate_recall_at_20.astype(float)*100, color="#4C78A8")
    ax_a.set_yticks(y, [DATASET_LABELS[d] for d in DATASETS]); ax_a.invert_yaxis()
    ax_a.set_xlim(0, 58); ax_a.set_xlabel("Candidate Recall@20 (%)")
    ax_a.set_title("A  |  FIRST-STAGE / UPSTREAM candidate availability", loc="left", fontsize=11)
    ax_a.grid(axis="x", alpha=.2); ax_a.set_axisbelow(True)
    for i, row in enumerate(recall.itertuples(index=False)):
        value = float(row.candidate_recall_at_20)*100
        ax_a.text(value+.8, i, f"{int(row.candidate_present_n)}/{int(row.planned_n)}  ({value:.2f}%)", va="center", fontsize=9)

    ax_d.set_axis_off()
    ax_d.set_title("D  |  Implemented single-positive denominator identity", loc="left", fontsize=11)
    for spine in ax_d.spines.values():
        spine.set_visible(True); spine.set_color("#AAAAAA")
    ax_d.text(.03, .85, "First-stage top-20 recall  ×  LLM conditional utility\n= all-user end-to-end utility", transform=ax_d.transAxes, va="top", fontsize=11)
    ax_d.text(.03, .54, "HR@5 e2e = Recall@20 × HR@5 | target present\nNDCG@5 e2e = Recall@20 × NDCG@5 | target present", transform=ax_d.transAxes, va="top", fontsize=10)
    ax_d.text(.03, .25, f"Verified in all 30 G4 cells. Max |error|: HR {values.hr_decomposition_error.max():.2e}; NDCG {values.ndcg_decomposition_error.max():.2e}.", transform=ax_d.transAxes, va="top", fontsize=8.7)
    ax_d.text(.03, .08, "Recall belongs upstream, not to the LLM. Absent target and invalid output get zero credit.", transform=ax_d.transAxes, fontsize=8.5)

    panels = []
    for index, dataset in enumerate(DATASETS, start=1):
        hr_ax = fig.add_subplot(gs[index, 0]); ndcg_ax = fig.add_subplot(gs[index, 1])
        _paired_axis(hr_ax, values, "hr_at_5", dataset)
        _paired_axis(ndcg_ax, values, "ndcg_at_5", dataset)
        ndcg_ax.tick_params(axis="y", labelleft=False)
        panels.append((hr_ax, ndcg_ax))
    panels[0][0].text(0, 1.56, "B  |  HR@5: end-to-end ● to conditional ◆", transform=panels[0][0].transAxes, fontsize=11, fontweight="bold")
    panels[0][1].text(0, 1.56, "C  |  NDCG@5: end-to-end ● to conditional ◆", transform=panels[0][1].transAxes, fontsize=11, fontweight="bold")
    fig.text(.5, .045, "Pairing joins the two utility populations for the same dataset/model/prompt. Color = model; open conditional marker = BASE; filled = LLMRANK; × = Food.com diagnostic (n=4).", ha="center", fontsize=8.5)
    fig.text(.5, .026, "G1 vs G4 is descriptive only (different candidate sources). No same-pool traditional-ordering superiority test; internal G4 bootstrap: 29 positive / 24 negative / 172 inconclusive.", ha="center", fontsize=8.2)
    return fig


def _save(fig: plt.Figure, target: Path) -> dict[str, str]:
    output = {}
    for ext in ("png", "pdf", "svg"):
        relative = f"{OUT}/figures/{FIGURE_ID}.{ext}"
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


def generate(root: Path = ROOT, *, artifact_root: Path | None = None, created_from_commit: str | None = None) -> dict:
    """Generate only the final G4/RQ7 figure and provenance chain."""
    root = root.resolve(); target = (artifact_root or root).resolve()
    hashes = validate_sources(root)
    values = figure_values(root)
    _style()
    commit = created_from_commit or git_head(root)
    if len(commit) != 40:
        raise FigureValidationError("full source commit required")
    value_relative = f"{OUT}/values/{FIGURE_ID}_values.csv"
    value_path = repository_path(target, value_relative)
    write_dataframe(value_path, values)
    images = _save(render(values), target)
    manifest = {
        "schema_version": "1.0", "figure_id": FIGURE_ID, "rq": "RQ7",
        "scientific_purpose": "Separate first-stage candidate availability from conditional LLM re-ranking and all-user utility.",
        "canonical_source_artifacts": SOURCES, "canonical_source_sha256": hashes,
        "figure_values_csv": value_relative, "figure_values_sha256": sha256_file(value_path),
        "generator": GENERATOR, "generator_sha256": sha256_file(repository_path(root, GENERATOR)),
        "output_png": images["png"], "output_png_sha256": images["png_sha256"],
        "output_pdf": images["pdf"], "output_pdf_sha256": images["pdf_sha256"],
        "output_svg": images["svg"], "output_svg_sha256": images["svg_sha256"],
        "scientific_interpretation": "End-to-end utility depends jointly on upstream target availability and LLM conditional re-ranking; candidate recall varies sharply across datasets.",
        "limitations": ["Food.com conditional utility is diagnostic only: four candidate-present users.", "G1 and G4 use structurally different candidate sources; their comparison is descriptive only.", "No same-pool baseline compares original first-stage ordering with the LLM-reordered final pooled candidates.", "Internal G4 bootstrap decisions do not establish superiority over traditional recommenders.", "The product identity is specific to the implemented single-positive and zero-credit denominator contract."],
        "created_from_commit": commit,
        "pool_size": 20, "positive_forcing": False, "same_pool_all_models_prompts": True,
        "candidate_recall_numerators_denominators": {d: list(EXPECTED_RECALL[d]) for d in DATASETS},
        "max_hr_decomposition_error": float(values.hr_decomposition_error.max()),
        "max_ndcg_decomposition_error": float(values.ndcg_decomposition_error.max()),
        "decomposition_cells": 30, "foodcom_conditional_n": 4,
        "g1_g4_comparability": "DESCRIPTIVE_ONLY", "same_pool_superiority_test": False,
        "universal_llmrank_claim": False, "internal_g4_bootstrap_counts": {"positive": 29, "negative": 24, "inconclusive": 172},
        "row_count": len(values),
    }
    manifest_path = repository_path(target, MANIFEST)
    write_json(manifest_path, manifest)
    validate_figure_manifest(root, manifest_path, artifact_root=target)
    return manifest


def validate(root: Path = ROOT, *, artifact_root: Path | None = None) -> dict:
    """Check saved RQ7 values, source and figure hashes, and claim boundaries."""
    root = root.resolve(); target = (artifact_root or root).resolve()
    expected = figure_values(root).astype(str).reset_index(drop=True)
    manifest = validate_figure_manifest(root, repository_path(target, MANIFEST), artifact_root=target)
    actual = _csv(target, manifest["figure_values_csv"]).astype(str).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    if manifest["canonical_source_artifacts"] != SOURCES or manifest["pool_size"] != 20 or manifest["positive_forcing"] is not False or manifest["same_pool_all_models_prompts"] is not True:
        raise FigureValidationError("RQ7 pool provenance changed")
    if manifest["g1_g4_comparability"] != "DESCRIPTIVE_ONLY" or manifest["same_pool_superiority_test"] is not False or manifest["universal_llmrank_claim"] is not False:
        raise FigureValidationError("RQ7 claim boundary changed")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "validate"), required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--created-from-commit")
    args = parser.parse_args(argv)
    result = (generate(ROOT, artifact_root=args.artifact_root, created_from_commit=args.created_from_commit)
              if args.mode == "generate" else validate(ROOT, artifact_root=args.artifact_root))
    print(json.dumps({"status": "PASS", "figure": result["figure_id"],
                      "max_hr_error": result["max_hr_decomposition_error"],
                      "max_ndcg_error": result["max_ndcg_decomposition_error"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

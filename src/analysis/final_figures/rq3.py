"""Deterministic final RQ3 descriptive and paired-bootstrap figures."""
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
from src.analysis.final_figures.common import (FigureValidationError, git_head,
    repository_path, sha256_file, validate_figure_manifest, write_dataframe, write_json)

GRID = "results/tables/rq3_llm_history_metric_grid_final.csv"
FREEZE = "results/rq3_completion_scientific_freeze/freeze_manifest.json"
STATS = "results/rq3_completion_scientific_freeze/statistical_results.csv"
SYNTHESIS = "results/rq3_completion_scientific_freeze/scientific_synthesis_summary.csv"
HASHES = {
    GRID: "96ab1971f5717c04faba7c0b61e9e1bfc92620ac2ee56eab33d622ec312a349d",
    FREEZE: "c23d83a1249e700f11ca6284fa9819db70009b4607d09c779432b3ac9a09f295",
    STATS: "976257e38d7cff90cc135bc126f27348020a1f9097f3260372bcd59c70a09f95",
    SYNTHESIS: "189ab285a873a74ef2c25f95f1f784b1b45edd000b21fb00f2fa265ad7fa824f",
}
GE10 = [0, 1, 2, 3, 4, 5, 10]
GE20 = [0, 1, 2, 3, 4, 5, 10, 15, 20]
METRICS = ["hr_at_5", "ndcg_at_5", "mrr"]
LABELS = {"hr_at_5": "HR@5", "ndcg_at_5": "NDCG@5", "mrr": "MRR"}
DATASETS = ["beauty", "sports", "ml1m", "lastfm", "foodcom"]
MODELS = ["qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b"]
MODEL_LABELS = {"qwen2:7b-instruct": "Qwen2", "mistral:7b-instruct": "Mistral", "gemma2:9b": "Gemma2"}
PROMPTS = ["BASE", "LLMRANK"]
CONTRASTS = ["H2_H3", "H3_H4", "H4_H5", "H10_H15", "H15_H20"]
DECISIONS = ["POSITIVE", "NEGATIVE", "INCONCLUSIVE"]
HISTORY_ID = "fig_rq3_history_metric_grid_final"
BOOTSTRAP_ID = "fig_rq3_paired_bootstrap_decisions"
GENERATOR = "src/analysis/final_figures/rq3.py"
OUTPUT = "results/final_figures/rq3"
MANIFESTS = "results/manifests/final_figures"
FIXED_TIME = datetime(2026, 9, 21, tzinfo=timezone.utc)


def validate_sources(root: Path = ROOT) -> dict[str, str]:
    """Require exact frozen source hashes and RQ3 protocol identity."""
    for rel, expected in HASHES.items():
        path = repository_path(root, rel)
        if not path.is_file() or sha256_file(path) != expected:
            raise FigureValidationError(f"RQ3 canonical source hash mismatch: {rel}")
    freeze = json.loads(repository_path(root, FREEZE).read_text(encoding="utf-8"))
    if freeze["history_grids"] != {"GE10": GE10, "GE20": GE20}:
        raise FigureValidationError("completed RQ3 history grids changed")
    if freeze["freeze_status"] != "FINAL_SCIENTIFIC_FREEZE":
        raise FigureValidationError("RQ3 scientific freeze status changed")
    if freeze["bootstrap"]["resamples"] != 10000 or freeze["bootstrap"]["confidence_interval"] != "two_sided_95_percent_percentile_ci":
        raise FigureValidationError("RQ3 bootstrap contract changed")
    if freeze["bootstrap"]["p_values"] != "NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD":
        raise FigureValidationError("RQ3 p-value policy changed")
    return dict(HASHES)


def _read_grid(root: Path) -> pd.DataFrame:
    validate_sources(root)
    frame = pd.read_csv(repository_path(root, GRID), dtype=str, keep_default_na=False)
    metrics = frame.loc[frame.row_type.eq("metric")].copy()
    if len(metrics) != 1440 or set(metrics.metric_id) != set(METRICS) or not metrics.status.eq("COMPUTED").all():
        raise FigureValidationError("RQ3 metric grid universe changed")
    for cohort, grid, membership in [("GE10", GE10, "PRIMARY_B1_GE10"), ("GE20", GE20, "SECONDARY_FULL_CURVE_GE20")]:
        part = metrics.loc[metrics.cohort.eq(cohort)]
        if set(part.history_length.astype(int)) != set(grid) or not part.analysis_membership.eq(membership).all():
            raise FigureValidationError(f"RQ3 {cohort} population or H grid changed")
        if len(part) != len(grid) * 30 * 3 or part.duplicated(["history_length", "dataset", "model", "prompt_strategy", "metric_id"]).any():
            raise FigureValidationError(f"RQ3 {cohort} configuration grid incomplete")
        if set(part.dataset) != set(DATASETS) or set(part.model) != set(MODELS) or set(part.prompt_strategy) != set(PROMPTS):
            raise FigureValidationError(f"RQ3 {cohort} configurations changed")
    return metrics


def history_values(root: Path = ROOT) -> pd.DataFrame:
    """Retain all 810 GE20 raw observations and their deterministic dataset means."""
    frame = _read_grid(root)
    frame = frame.loc[frame.cohort.eq("GE20"), ["cohort", "analysis_membership", "history_length", "dataset", "model", "prompt_strategy", "metric_id", "value", "status", "planned_applicable_users", "strict_valid_users", "invalid_users", "evidence_source"]].copy()
    frame = frame.rename(columns={"value": "raw_canonical_value"})
    frame["history_length"] = frame.history_length.astype(int)
    frame["displayed_value"] = ""
    frame["aggregation_method"] = "UNWEIGHTED_ARITHMETIC_MEAN_ACROSS_FIVE_DATASET_CONFIGURATION_VALUES"
    keys = ["history_length", "model", "prompt_strategy", "metric_id"]
    for _, indices in frame.groupby(keys, sort=False).groups.items():
        raw = pd.to_numeric(frame.loc[indices, "raw_canonical_value"], errors="raise")
        if len(raw) != 5 or not np.isfinite(raw).all():
            raise FigureValidationError("GE20 dataset aggregation incomplete")
        frame.loc[indices, "displayed_value"] = format(float(raw.mean()), ".17g")
    order = {name: index for index, name in enumerate(DATASETS)}
    frame["_dataset"] = frame.dataset.map(order)
    frame = frame.sort_values(["metric_id", "model", "prompt_strategy", "history_length", "_dataset"], kind="stable").drop(columns="_dataset")
    return frame.reset_index(drop=True)


def _read_stats(root: Path) -> pd.DataFrame:
    validate_sources(root)
    frame = pd.read_csv(repository_path(root, STATS), dtype=str, keep_default_na=False)
    if len(frame) != 450 or set(frame.contrast) != set(CONTRASTS) or set(frame.metric) != set(METRICS):
        raise FigureValidationError("RQ3 statistical grid changed")
    if set(frame.decision) != set(DECISIONS) or not frame.bootstrap_resamples.eq("10000").all():
        raise FigureValidationError("RQ3 bootstrap decisions/resamples changed")
    if not frame.p_value.eq("").all() or not frame.p_value_status.eq("NOT_COMPUTED_NO_FROZEN_PRIMARY_P_VALUE_METHOD").all():
        raise FigureValidationError("RQ3 p-values must remain absent")
    expected = {"H2_H3": (2,3,"PRIMARY_B1_GE10"), "H3_H4": (3,4,"PRIMARY_B1_GE10"), "H4_H5": (4,5,"PRIMARY_B1_GE10"), "H10_H15": (10,15,"SECONDARY_FULL_CURVE_GE20"), "H15_H20": (15,20,"SECONDARY_FULL_CURVE_GE20")}
    for contrast, (a,b,membership) in expected.items():
        rows = frame.loc[frame.contrast.eq(contrast)]
        if len(rows) != 90 or not rows.history_length_a.eq(str(a)).all() or not rows.history_length_b.eq(str(b)).all() or not rows.analysis_membership.eq(membership).all():
            raise FigureValidationError(f"RQ3 contrast orientation/population changed: {contrast}")
    if frame.duplicated(["contrast", "dataset", "model", "prompt_strategy", "metric"]).any():
        raise FigureValidationError("duplicate RQ3 inferential configuration")
    for row in frame.itertuples(index=False):
        lo, hi, delta = float(row.ci_low), float(row.ci_high), float(row.mean_delta)
        if not (np.isfinite([lo,hi,delta]).all() and lo <= delta <= hi):
            raise FigureValidationError("invalid RQ3 confidence interval")
        derived = "POSITIVE" if lo > 0 else "NEGATIVE" if hi < 0 else "INCONCLUSIVE"
        if row.decision != derived:
            raise FigureValidationError("RQ3 bootstrap decision differs from CI")
    summary = pd.read_csv(repository_path(root, SYNTHESIS), dtype=str, keep_default_na=False)
    counts = frame.groupby(["contrast", "metric", "decision"]).size()
    for row in summary.itertuples(index=False):
        if int(row.bootstrap_result_rows) != 30 or any(int(getattr(row, f"ci_{decision.lower()}_configurations")) != int(counts.get((row.contrast,row.metric,decision),0)) for decision in DECISIONS):
            raise FigureValidationError("RQ3 synthesis decision counts disagree with statistical CSV")
    return frame


def bootstrap_values(root: Path = ROOT) -> pd.DataFrame:
    """Copy all configuration-level frozen CIs, preserving a-to-b orientation."""
    source = _read_stats(root)
    names = ["result_id", "contrast", "analysis_membership", "dataset", "model", "prompt_strategy", "metric", "history_length_a", "history_length_b", "n_pairs", "mean_delta", "ci_low", "ci_high", "decision", "bootstrap_resamples", "bootstrap_engine", "bootstrap_percentile_method", "confidence_interval", "p_value_status"]
    frame = source[names].copy().rename(columns={"analysis_membership": "eligibility_population", "model": "model_name", "prompt_strategy": "prompt_variant", "metric": "metric_id"})
    frame["orientation"] = frame.history_length_a + "_TO_" + frame.history_length_b + "_B_MINUS_A"
    frame["display_transform"] = "NONE_RAW_CANONICAL_DELTA_AND_CI"
    frame["_contrast"] = frame.contrast.map({name:i for i,name in enumerate(CONTRASTS)})
    frame["_metric"] = frame.metric_id.map({name:i for i,name in enumerate(METRICS)})
    frame["_dataset"] = frame.dataset.map({name:i for i,name in enumerate(DATASETS)})
    frame["_model"] = frame.model_name.map({name:i for i,name in enumerate(MODELS)})
    frame["_prompt"] = frame.prompt_variant.map({name:i for i,name in enumerate(PROMPTS)})
    frame = frame.sort_values(["_contrast","_metric","_dataset","_model","_prompt"],kind="stable").drop(columns=["_contrast","_metric","_dataset","_model","_prompt"])
    return frame.reset_index(drop=True)


def _style() -> None:
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "svg.hashsalt":"irp-final-figures-v1", "pdf.compression":9})


def _history_plot(values: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1,3,figsize=(17,6),sharex=True)
    colors = {MODELS[0]:"#0072B2",MODELS[1]:"#D55E00",MODELS[2]:"#009E73"}
    for ax, metric in zip(axes,METRICS,strict=True):
        subset = values.loc[values.metric_id.eq(metric)].drop_duplicates(["history_length","model","prompt_strategy"])
        for model in MODELS:
            for prompt in PROMPTS:
                line = subset.loc[subset.model.eq(model) & subset.prompt_strategy.eq(prompt)].sort_values("history_length")
                ax.plot(line.history_length.astype(int),line.displayed_value.astype(float),color=colors[model],linestyle="-" if prompt=="BASE" else "--",marker="o" if prompt=="BASE" else "s",markersize=4,label=f"{MODEL_LABELS[model]} / {prompt}")
        ax.set_title(LABELS[metric]); ax.set_xticks(GE20); ax.set_xlim(-0.5,20.5); ax.set_xlabel("History length H (interactions)"); ax.set_ylabel("Mean across five datasets"); ax.grid(alpha=.25)
        ax.spines[["top","right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.5,.90))
    fig.suptitle("RQ3 GE20 history trajectories: same eligible population at every H",y=.99,fontsize=14)
    fig.text(.5,.025,"Descriptive, unweighted dataset means. H=0: controlled no-history condition on warm eligible users; not true cold-start.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.065,right=.985,bottom=.19,top=.77,wspace=.30)
    return fig


def _forest_plot(values: pd.DataFrame) -> plt.Figure:
    # Fifteen separate forest panels keep every one of the 450 frozen paired CIs visible.
    fig, axes = plt.subplots(5,3,figsize=(20,25),squeeze=False)
    colors = {"POSITIVE":"#0072B2","NEGATIVE":"#D55E00","INCONCLUSIVE":"#6B6B6B"}
    labels = [f"{d[:3]} {MODEL_LABELS[m]} {p[0]}" for d in DATASETS for m in MODELS for p in PROMPTS]
    for ri, contrast in enumerate(CONTRASTS):
        for ci, metric in enumerate(METRICS):
            ax = axes[ri,ci]
            rows = values.loc[values.contrast.eq(contrast) & values.metric_id.eq(metric)].copy()
            rows["_dataset"] = rows.dataset.map({v:i for i,v in enumerate(DATASETS)})
            rows["_model"] = rows.model_name.map({v:i for i,v in enumerate(MODELS)})
            rows["_prompt"] = rows.prompt_variant.map({v:i for i,v in enumerate(PROMPTS)})
            rows = rows.sort_values(["_dataset","_model","_prompt"])
            y = np.arange(30)
            low, high, delta = [rows[c].astype(float).to_numpy() for c in ["ci_low","ci_high","mean_delta"]]
            for decision in DECISIONS:
                mask = rows.decision.eq(decision).to_numpy()
                ax.errorbar(delta[mask],y[mask],xerr=np.vstack((delta[mask]-low[mask],high[mask]-delta[mask])),fmt="o",markersize=2.8,elinewidth=.75,capsize=0,color=colors[decision],label=f"{decision} ({int(mask.sum())})")
            ax.axvline(0,color="black",linewidth=.8)
            ax.set_ylim(29.7,-.7); ax.set_yticks(y)
            if ci == 0:
                ax.set_yticklabels(labels,fontsize=5.5)
            else:
                ax.set_yticklabels([])
            counts = rows.decision.value_counts()
            ax.set_title(f"{contrast.replace('_',' to ')} | {LABELS[metric]}  (+{counts.get('POSITIVE',0)} / -{counts.get('NEGATIVE',0)} / ?{counts.get('INCONCLUSIVE',0)})",fontsize=9)
            ax.set_xlabel("Paired mean delta (B - A), 95% CI",fontsize=7)
            ax.tick_params(axis="x",labelsize=6)
            ax.grid(axis="x",alpha=.15)
            # Decision counts in title; no in-panel legend obscures CIs.
            for boundary in [5.5,11.5,17.5,23.5]:
                ax.axhline(boundary,color="0.65",linewidth=.55)
    fig.suptitle("RQ3 paired-user bootstrap: all 450 configuration-level intervals (+ positive / - negative / ? inconclusive)",y=.997,fontsize=15)
    fig.text(.5,.005,"Blue=positive, orange=negative, gray=inconclusive. Each panel: 5 datasets x 3 models x 2 prompts. GE10: first three; GE20: last two. 10,000 resamples; no p-values or pooled CI.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.135,right=.985,bottom=.025,top=.974,wspace=.13,hspace=.34)
    return fig


def _save(fig: plt.Figure, target: Path, figure_id: str) -> dict[str,str]:
    result={}
    for ext in ("png","pdf","svg"):
        rel=f"{OUTPUT}/figures/{figure_id}.{ext}"; path=repository_path(target,rel); path.parent.mkdir(parents=True,exist_ok=True)
        metadata=({"Software":"IRP deterministic final-figure pipeline"} if ext=="png" else {"Creator":"IRP deterministic final-figure pipeline","Producer":"Matplotlib","CreationDate":FIXED_TIME,"ModDate":FIXED_TIME} if ext=="pdf" else {"Creator":"IRP deterministic final-figure pipeline","Date":"2026-09-21"})
        fig.savefig(path,dpi=160,metadata=metadata,facecolor="white")
        if ext=="svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
        result[ext]=rel; result[f"{ext}_sha256"]=sha256_file(path)
    plt.close(fig)
    return result


def _validate_values(figure_id: str, stored: pd.DataFrame, root: Path) -> None:
    expected=history_values(root) if figure_id==HISTORY_ID else bootstrap_values(root)
    pd.testing.assert_frame_equal(stored.astype(str).reset_index(drop=True),expected.astype(str).reset_index(drop=True),check_dtype=False,check_exact=True)


def generate_all(root: Path = ROOT, *, artifact_root: Path | None = None, created_from_commit: str | None = None) -> list[dict]:
    """Write only the two final RQ3 figures, values CSVs, and hash manifests."""
    root=root.resolve(); target=(artifact_root or root).resolve(); validate_sources(root); _style()
    commit=created_from_commit or git_head(root)
    if len(commit)!=40: raise FigureValidationError("created_from_commit must be full SHA")
    out=[]
    for figure_id in (HISTORY_ID,BOOTSTRAP_ID):
        values=history_values(root) if figure_id==HISTORY_ID else bootstrap_values(root)
        rel=f"{OUTPUT}/values/{figure_id}_values.csv"; path=repository_path(target,rel); write_dataframe(path,values)
        _validate_values(figure_id,pd.read_csv(path,dtype=str,keep_default_na=False),root)
        images=_save(_history_plot(values) if figure_id==HISTORY_ID else _forest_plot(values),target,figure_id)
        descriptive=figure_id==HISTORY_ID
        manifest={
            "schema_version":"1.0","figure_id":figure_id,"rq":"RQ3",
            "scientific_purpose":"GE20 complete-history descriptive trajectories" if descriptive else "Configuration-level paired-bootstrap contrast evidence",
            "canonical_source_artifacts":list(HASHES),"canonical_source_sha256":dict(HASHES),
            "figure_values_csv":rel,"figure_values_sha256":sha256_file(path),
            "generator":GENERATOR,"generator_sha256":sha256_file(repository_path(root,GENERATOR)),
            "output_png":images["png"],"output_png_sha256":images["png_sha256"],"output_pdf":images["pdf"],"output_pdf_sha256":images["pdf_sha256"],"output_svg":images["svg"],"output_svg_sha256":images["svg_sha256"],
            "scientific_interpretation":"History effects are configuration-dependent and nonlinear; descriptive curves do not define a universal optimum." if descriptive else "Frozen paired CIs show positive, negative, and inconclusive effects by configuration; later additions are not uniformly beneficial.",
            "limitations":["H=0 is a controlled no-history condition on warm eligible users, not true cold-start.","No monotonicity or universal optimal history-length claim.","Dataset means are descriptive, unweighted, and do not define a pooled user-level result."] if descriptive else ["GE10 and GE20 contrasts use different frozen eligible populations and are not joined as one trajectory.","No p-values, multiplicity adjustment, or pooled cross-configuration confidence interval.","H0 to H10 and H10 to H20 inferential contrasts are absent from the frozen statistical CSV and are not invented."],
            "created_from_commit":commit,
            "display_contract":({"cohort":"GE20","history_grid":GE20,"metrics":METRICS,"lines":"six model-prompt configurations per metric","aggregation":"arithmetic mean of five dataset configuration values, equal dataset weighting at every H","raw_rows":810,"numeric_x_axis":True,"population_mixing":False} if descriptive else {"contrasts":CONTRASTS,"rows":450,"panels":15,"interval":"two-sided 95% percentile CI","paired_user_bootstrap_resamples":10000,"orientation":"value_b_minus_value_a as frozen","decision_colors":{k:v for k,v in {"POSITIVE":"blue","NEGATIVE":"orange","INCONCLUSIVE":"gray"}.items()},"p_values":False}),
        }
        mpath=repository_path(target,f"{MANIFESTS}/{figure_id}_manifest.json"); write_json(mpath,manifest)
        validate_figure_manifest(root,mpath,artifact_root=target); out.append(manifest)
    return out


def validate_all(root: Path = ROOT, *, artifact_root: Path | None = None) -> list[dict]:
    """Verify source, values, images, and every manifest digest."""
    root=root.resolve(); target=(artifact_root or root).resolve(); validate_sources(root); _read_grid(root); _read_stats(root)
    out=[]
    for figure_id in (HISTORY_ID,BOOTSTRAP_ID):
        path=repository_path(target,f"{MANIFESTS}/{figure_id}_manifest.json")
        manifest=validate_figure_manifest(root,path,artifact_root=target)
        stored=pd.read_csv(repository_path(target,manifest["figure_values_csv"]),dtype=str,keep_default_na=False)
        _validate_values(figure_id,stored,root); out.append(manifest)
    return out


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode",choices=["generate","validate"],required=True); parser.add_argument("--artifact-root",type=Path); parser.add_argument("--created-from-commit")
    args=parser.parse_args(argv)
    result=generate_all(ROOT,artifact_root=args.artifact_root,created_from_commit=args.created_from_commit) if args.mode=="generate" else validate_all(ROOT,artifact_root=args.artifact_root)
    print(json.dumps({"status":"PASS","figures":[x["figure_id"] for x in result]})); return 0

if __name__=="__main__": raise SystemExit(main())

"""Generate the final seven-question scientific evidence map."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_figures.common import (  # noqa: E402
    FigureValidationError, git_head, repository_path, sha256_file,
    validate_figure_manifest, write_dataframe, write_json,
)

FIGURE_ID = "fig_cross_rq_evidence_overview"
GENERATOR = "src/analysis/final_figures/cross_rq.py"
VALUES = f"results/final_figures/cross_rq/values/{FIGURE_ID}_values.csv"
MANIFEST = f"results/manifests/final_figures/{FIGURE_ID}_manifest.json"
OUT = "results/final_figures/cross_rq/figures"
FIXED_TIME = datetime(2026, 9, 21, tzinfo=timezone.utc)
G2_FOCUS = "results/tables/g2_history_length_scientific_synthesis_focus_reconciliation.csv"
G3_BREAKDOWN = "results/tables/g3_candidate_position_synthesis_decision_breakdowns.csv"
RQ6_VALUES = "results/final_figures/rq6/values/fig_rq6_textual_hallucination_vs_structural_reliability_values.csv"
G4_RECALL = "results/tables/g4_f1bzq_candidate_recall_at_20.csv"
RQ5_FREEZE = "results/rq5_final_scientific_freeze/freeze_manifest.json"
RQ3_FREEZE = "results/rq3_completion_scientific_freeze/freeze_manifest.json"
RQ2_BEYOND = "results/tables/g1_primary_ranking_beyond_utility.csv"
G1_CONTRACT = "results/manifests/g1_metric_contract_and_provenance.json"

FIGURE_REFS = {
    "RQ1": ["fig_rq1_utility_heatmap_appendix", "fig_rq1_traditional_vs_llm_utility"],
    "RQ2": ["fig_rq2_llm_longtail_novelty_heatmap", "fig_rq2_llm_serendipity_coverage_heatmap",
            "fig_rq2_llm_calibration_heatmap", "fig_rq2_traditional_vs_llm_longtail_popularity_novelty",
            "fig_rq2_traditional_vs_llm_serendipity_diversity_coverage"],
    "RQ3": ["fig_rq3_history_metric_grid_final", "fig_rq3_paired_bootstrap_decisions"],
    "RQ4": ["fig_rq4_g3_first_random_utility_pairs", "fig_rq4_g3_first_random_bootstrap_deltas"],
    "RQ5": ["fig_rq5_profile_conditions_main", "fig_rq5_profile_deltas_heatmap"],
    "RQ6": ["fig_rq6_textual_hallucination_vs_structural_reliability"],
    "RQ7": ["fig_rq7_candidate_recall_and_reranking_utility"],
}
PRIMARY_MANIFESTS = {
    rq: f"results/manifests/final_figures/{refs[0]}_manifest.json"
    for rq, refs in FIGURE_REFS.items()
}
SOURCE_PATHS = [
    *(f"results/manifests/final_figures/{figure_id}_manifest.json"
      for refs in FIGURE_REFS.values() for figure_id in refs),
    G2_FOCUS, G3_BREAKDOWN, RQ6_VALUES, G4_RECALL,
    RQ5_FREEZE, RQ3_FREEZE, RQ2_BEYOND, G1_CONTRACT,
]
SOURCE_PATHS = list(dict.fromkeys(SOURCE_PATHS))


def _json(root: Path, path: str) -> dict:
    return json.loads(repository_path(root, path).read_text(encoding="utf-8"))


def _csv(root: Path, path: str) -> pd.DataFrame:
    return pd.read_csv(repository_path(root, path), dtype=str, keep_default_na=False)


def validate_sources(root: Path = ROOT) -> dict[str, str]:
    """Validate frozen figure chains and each numerical summary source."""
    root = root.resolve()
    for path in SOURCE_PATHS:
        if not repository_path(root, path).is_file():
            raise FigureValidationError(f"cross-RQ canonical source missing: {path}")
    for rq, refs in FIGURE_REFS.items():
        for figure_id in refs:
            manifest_path = repository_path(root, f"results/manifests/final_figures/{figure_id}_manifest.json")
            manifest = validate_figure_manifest(root, manifest_path)
            if manifest["figure_id"] != figure_id or manifest["rq"] != rq:
                raise FigureValidationError(f"figure identity changed: {figure_id}")
    rq1 = _json(root, f"results/manifests/final_figures/{FIGURE_REFS['RQ1'][1]}_manifest.json")
    comparison = rq1["comparison_contract"]
    if (comparison["metrics"] != ["hr_at_5", "ndcg_at_5"] or comparison["excluded_metric"] != "mrr"
            or comparison["candidate_pool_size"] != 20 or comparison["positive_forced"] is not True):
        raise FigureValidationError("RQ1 local direct top-5 comparison changed")
    contract = _json(root, G1_CONTRACT)
    if contract["canonical_artifact_authority"]["canonical_beyond_utility_sha256"] != sha256_file(repository_path(root, RQ2_BEYOND)):
        raise FigureValidationError("RQ2 corrected beyond-utility authority changed")
    beyond = _csv(root, RQ2_BEYOND)
    food = beyond.loc[beyond.dataset.eq("foodcom") & beyond.metric.eq("popreo_at_5")]
    if len(food) != 6 or not food.status.eq("DEGENERATE_PROTOCOL_INPUT").all() or not food.value.eq("").all():
        raise FigureValidationError("Food.com G1 PopREO undefined status changed")
    calibration = _json(root, f"results/manifests/final_figures/{FIGURE_REFS['RQ2'][2]}_manifest.json")
    if calibration.get("render_only_remediation", {}).get("not_target_paper_metric") is not True:
        raise FigureValidationError("Calibration local-extension label is missing")
    rq3 = _json(root, PRIMARY_MANIFESTS["RQ3"])
    if rq3["display_contract"]["cohort"] != "GE20" or rq3["display_contract"]["history_grid"] != [0,1,2,3,4,5,10,15,20]:
        raise FigureValidationError("RQ3 completed GE20 grid changed")
    if not _json(root, RQ3_FREEZE):
        raise FigureValidationError("RQ3 scientific freeze unreadable")
    focus = _csv(root, G2_FOCUS).set_index("contrast")
    expected = {"H10_minus_H0": ("PRIMARY_B1_GE10", 57, 1, 32),
                "H20_minus_H10": ("LONG_HISTORY_B2_GE20", 3, 18, 69)}
    for contrast, (population, positive, negative, inconclusive) in expected.items():
        row = focus.loc[contrast]
        if (row.analysis_membership != population or int(row.significant_positive) != positive
                or int(row.significant_negative) != negative or int(row.non_significant) != inconclusive):
            raise FigureValidationError(f"RQ3 paired contrast count changed: {contrast}")
    rq4 = _json(root, f"results/manifests/final_figures/{FIGURE_REFS['RQ4'][1]}_manifest.json")
    breakdown = _csv(root, G3_BREAKDOWN)
    overall = breakdown.loc[breakdown.scope.eq("overall")].iloc[0]
    if (rq4["decision_counts"] != {"POSITIVE":85,"NEGATIVE":1,"INCONCLUSIVE":4}
            or (int(overall.ci_supported_positive), int(overall.ci_supported_negative),
                int(overall.non_significant_ci_includes_zero)) != (85,1,4)
            or rq4["display_contract"]["first_target_informed_oracle"] is not True):
        raise FigureValidationError("RQ4 G3 stress evidence changed")
    rq5 = _json(root, RQ5_FREEZE)
    if rq5.get("status") != "FINAL_SCIENTIFIC_FREEZE" or not any("No semantic preference-preservation validation" in str(item) for item in rq5.get("limitations", [])):
        # The frozen manifest may nest its limitation list; require status and
        # the exact source phrase somewhere in the canonical document.
        if rq5.get("status") != "FINAL_SCIENTIFIC_FREEZE" or "No semantic preference-preservation validation" not in repository_path(root, RQ5_FREEZE).read_text(encoding="utf-8"):
            raise FigureValidationError("RQ5 matched scientific freeze changed")
    rq6 = _json(root, PRIMARY_MANIFESTS["RQ6"])
    reliability = _csv(root, RQ6_VALUES)
    rates = reliability.strict_valid_rate.astype(float)
    if (len(reliability) != 4 or rates.min() < .985 or rates.max() > .991
            or rq6["direct_comparability"] is not False
            or rq6["panel_b_evidence_type"] != "STRUCTURAL_RELIABILITY"):
        raise FigureValidationError("RQ6 construct/rate evidence changed")
    rq7 = _json(root, PRIMARY_MANIFESTS["RQ7"])
    recall = _csv(root, G4_RECALL)
    expected_recall = {"beauty": (49,990), "sports": (28,892), "ml1m": (193,1000),
                       "lastfm": (469,1000), "foodcom": (4,1000)}
    actual_recall = {row.dataset: (int(row.positive_present_N), int(row.evaluation_N))
                     for row in recall.itertuples(index=False)}
    if (actual_recall != expected_recall or rq7["candidate_recall_numerators_denominators"] !=
            {dataset:list(pair) for dataset,pair in expected_recall.items()}
            or rq7["same_pool_superiority_test"] is not False
            or rq7["foodcom_conditional_n"] != 4
            or rq7["max_hr_decomposition_error"] > 1e-12
            or rq7["max_ndcg_decomposition_error"] > 1e-12):
        raise FigureValidationError("RQ7 recall/decomposition/limitation changed")
    return {path: sha256_file(repository_path(root, path)) for path in SOURCE_PATHS}


def build_values(root: Path = ROOT) -> pd.DataFrame:
    """Build seven concise evidence-map rows from validated frozen artifacts."""
    validate_sources(root)
    focus = _csv(root, G2_FOCUS).set_index("contrast")
    early, late = (focus.loc[contrast] for contrast in ("H10_minus_H0", "H20_minus_H10"))
    overall = _csv(root, G3_BREAKDOWN).loc[lambda x: x.scope.eq("overall")].iloc[0]
    rates = _csv(root, RQ6_VALUES).strict_valid_rate.astype(float)
    recall = _csv(root, G4_RECALL)
    recall_range = (recall.candidate_recall_at_20.astype(float).min()*100,
                    recall.candidate_recall_at_20.astype(float).max()*100)
    records = [
        ("RQ1", "Ranking utility", "G1 primary LLM ranking; local traditional comparison",
         "Utility varies by dataset, model and prompt. HR@5/NDCG@5 admit local numerical comparison.",
         "Adapted; locally comparable",
         "Shared positive-forced 20-item pool; no universal winner. Traditional-vs-LLM MRR depths differ (20 vs 10). Not paper-exact."),
        ("RQ2", "Beyond-utility dimensions", "Corrected G1 metrics; traditional counterparts where defined",
         "Long-tail, popularity, novelty, diversity and coverage vary independently by configuration.",
         "Adapted + local extensions",
         "Traditional: full cohort; G1: strict-valid/defined users. Calibration is a local extension; Food.com PopREO is undefined."),
        ("RQ3", "History-length sensitivity", "Completed GE20 trajectory; separate paired G2 contrasts",
         (f"Early GE10 H10−H0: {early.significant_positive} pos / {early.significant_negative} neg / {early.non_significant} inc. "
          f"Later GE20 H20−H10: {late.significant_positive} pos / {late.significant_negative} neg / {late.non_significant} inc."),
         "Controlled extension",
         "Contrast counts use different eligible cohorts. H0 is controlled no-history on warm users; no monotonic benefit or universal optimum."),
        ("RQ4", "Candidate-position sensitivity", "G3 paired FIRST vs RANDOM order stress test",
         (f"FIRST−RANDOM: {overall.ci_supported_positive} pos / {overall.ci_supported_negative} neg / "
          f"{overall.non_significant_ci_includes_zero} inc across {overall.comparison_rows} configuration×metric contrasts."),
         "Controlled stress test", "FIRST places the known target first: target-informed oracle condition, not a deployable ordering policy."),
        ("RQ5", "Generated user profiles", "Corrected matched History-only / Profile-only / Profile+History; systematic support",
         "Profile effects are configuration-dependent; some gains occur, but no general benefit is established.",
         "Matched local evidence", "No validated semantic-fidelity claim or universal causal profile-benefit claim."),
        ("RQ6", "Output reliability", "G1–G4 constrained candidate-index structural-validity accounting",
         f"Strict structural validity spans {rates.min()*100:.1f}–{rates.max()*100:.1f}% across G1–G4.",
         "Adapted construct", "Paper free-text textual/catalogue hallucination is not local structural invalidity; direct reproduction is absent."),
        ("RQ7", "Two-stage reranking", "Canonical F1BZQ candidate pools + G4 conditional/unconditional utility",
         (f"Candidate Recall@20 spans {recall_range[0]:.1f}–{recall_range[1]:.1f}%. "
          "End-to-end HR@5/NDCG@5 factor into recall × conditional utility."),
         "Pipeline decomposition", "Single-positive identity only. No same-pool superiority test; Food.com conditional n=4 is diagnostic."),
    ]
    frame = pd.DataFrame(records, columns=["rq", "phenomenon", "evidence", "main_result",
                                          "evidence_status", "critical_qualification"])
    frame["canonical_source_manifest"] = frame.rq.map(PRIMARY_MANIFESTS)
    frame["final_figure_refs"] = frame.rq.map(lambda rq: ";".join(FIGURE_REFS[rq]))
    if list(frame.rq) != [f"RQ{i}" for i in range(1,8)]:
        raise FigureValidationError("cross-RQ overview must have exactly seven ordered rows")
    return frame


def _plot(values: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.hashsalt": "irp-final-figures-v1", "pdf.compression": 9})
    fig = plt.figure(figsize=(19.2, 11.2), facecolor="white")
    ax = fig.add_axes([0,0,1,1]); ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.text(.035,.958,"Seven research questions, seven evidence contracts",fontsize=18,
             fontweight="bold",color="#23364B")
    fig.text(.035,.928,"A map of local evidence and its limits  |  No common effect scale or overall ranking",
             fontsize=10,color="#566575")
    columns = [("Phenomenon",.079,.155,23),("Local evidence",.243,.183,25),
               ("Main finding",.438,.228,32),("Evidence status",.680,.120,18),
               ("Critical qualification",.813,.154,23)]
    for label,x,_,_ in columns:
        ax.text(x,.876,label,ha="left",va="center",fontsize=10,fontweight="bold",color="#31465C")
    top, height, gap = .847,.104,.012
    for index,row in enumerate(values.itertuples(index=False)):
        y = top-index*(height+gap)-height
        color = "#F3F6F8" if index%2==0 else "#F8FAFB"
        card = FancyBboxPatch((.025,y),.95,height,boxstyle="round,pad=0.004,rounding_size=0.008",
                              linewidth=.55,edgecolor="#DCE3E8",facecolor=color)
        ax.add_patch(card)
        badge = FancyBboxPatch((.034,y+.031),.037,.041,boxstyle="round,pad=0.003,rounding_size=0.007",
                               linewidth=0,facecolor="#334B63")
        ax.add_patch(badge)
        ax.text(.0525,y+.051,row.rq,ha="center",va="center",fontsize=9.3,
                fontweight="bold",color="white")
        texts = [row.phenomenon,row.evidence,row.main_result,row.evidence_status,row.critical_qualification]
        for (heading,x,width,wrap),value in zip(columns,texts,strict=True):
            del heading,width
            wrapped = textwrap.fill(str(value),width=wrap,break_long_words=False,break_on_hyphens=False)
            ax.text(x,y+height-.013,wrapped,ha="left",va="top",fontsize=8.5,
                    linespacing=1.24,color="#293746",fontweight="medium" if x==.680 else "normal")
    ax.text(.035,.022,"Statuses describe study design and evidence scope; they are not quality grades.",
            fontsize=8.5,color="#566575",va="center")
    return fig


def generate(root: Path = ROOT, *, artifact_root: Path | None = None,
             created_from_commit: str | None = None) -> dict:
    """Generate the single cross-RQ figure and complete hash manifest."""
    root = root.resolve(); artifact_root = (artifact_root or root).resolve()
    sources = validate_sources(root)
    values = build_values(root)
    commit = created_from_commit or git_head(root)
    if len(commit) != 40:
        raise FigureValidationError("created_from_commit must be full SHA")
    values_path = repository_path(artifact_root, VALUES)
    write_dataframe(values_path, values)
    pd.testing.assert_frame_equal(_csv(artifact_root, VALUES), values, check_dtype=False, check_exact=True)
    fig = _plot(values)
    outputs = {}
    for extension in ("png","pdf","svg"):
        relative = f"{OUT}/{FIGURE_ID}.{extension}"
        path = repository_path(artifact_root, relative); path.parent.mkdir(parents=True,exist_ok=True)
        metadata = ({"Software":"IRP deterministic final-figure pipeline"} if extension=="png"
                    else {"Creator":"IRP deterministic final-figure pipeline","Producer":"Matplotlib",
                          "CreationDate":FIXED_TIME,"ModDate":FIXED_TIME} if extension=="pdf"
                    else {"Creator":"IRP deterministic final-figure pipeline","Date":"2026-09-21"})
        fig.savefig(path,dpi=180,metadata=metadata,facecolor="white")
        if extension=="svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",
                            encoding="utf-8",newline="\n")
        outputs[extension]=relative; outputs[f"{extension}_sha256"]=sha256_file(path)
    plt.close(fig)
    manifest = {
        "schema_version":"1.0", "figure_id":FIGURE_ID,"rq":"SYNTHESIS","role":"MAIN_TEXT",
        "scientific_purpose":"Map RQ1–RQ7 phenomena, local evidence, findings, evidence status and critical qualifications without an aggregate effect score.",
        "canonical_source_artifacts":list(sources),"canonical_source_sha256":sources,
        "figure_values_csv":VALUES,"figure_values_sha256":sha256_file(values_path),
        "generator":GENERATOR,"generator_sha256":sha256_file(repository_path(root,GENERATOR)),
        "output_png":outputs["png"],"output_png_sha256":outputs["png_sha256"],
        "output_pdf":outputs["pdf"],"output_pdf_sha256":outputs["pdf_sha256"],
        "output_svg":outputs["svg"],"output_svg_sha256":outputs["svg_sha256"],
        "created_from_commit":commit,
        "scientific_interpretation":"Local evidence is heterogeneous in construct, design and comparability; each row states its own scope and qualification.",
        "limitations":["Not a meta-analysis, common effect scale, evidence grade, RQ ranking or causal link between research questions.",
                       "RQ3 early GE10 and late GE20 contrast counts belong to distinct eligible populations.",
                       "RQ6 structural invalidity is not free-text hallucination; RQ7 has no same-pool superiority comparison."],
        "row_count":7,"status_language":"DESIGN_AND_SCOPE_NOT_QUALITY_GRADE",
    }
    path = repository_path(artifact_root,MANIFEST); write_json(path,manifest)
    validate_figure_manifest(root,path,artifact_root=artifact_root)
    return manifest


def validate(root: Path = ROOT, *, artifact_root: Path | None = None) -> dict:
    """Validate sources, exact evidence-map values and generated output hashes."""
    root = root.resolve(); artifact_root = (artifact_root or root).resolve()
    manifest = validate_figure_manifest(root,repository_path(artifact_root,MANIFEST),artifact_root=artifact_root)
    if manifest["figure_id"]!=FIGURE_ID or manifest["canonical_source_sha256"]!=validate_sources(root):
        raise FigureValidationError("cross-RQ figure authority changed")
    pd.testing.assert_frame_equal(_csv(artifact_root,VALUES),build_values(root),check_dtype=False,check_exact=True)
    if manifest["row_count"]!=7 or manifest["status_language"]!="DESIGN_AND_SCOPE_NOT_QUALITY_GRADE":
        raise FigureValidationError("cross-RQ evidence map contract changed")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("generate","validate"),required=True)
    parser.add_argument("--artifact-root",type=Path)
    parser.add_argument("--created-from-commit")
    args=parser.parse_args(argv)
    result=(generate(ROOT,artifact_root=args.artifact_root,created_from_commit=args.created_from_commit)
            if args.mode=="generate" else validate(ROOT,artifact_root=args.artifact_root))
    print(json.dumps({"status":"PASS","figure_id":result["figure_id"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

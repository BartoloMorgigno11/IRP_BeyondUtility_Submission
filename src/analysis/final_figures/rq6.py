"""Render the final RQ6 construct distinction and local structural reliability."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

FIGURE_ID = "fig_rq6_textual_hallucination_vs_structural_reliability"
AUTHORITY = "results/manifests/rq6_evidence_authority.json"
CONTRACT = "results/manifests/rq6_scientific_contract.json"
SPEC = "results/manifests/rq6_final_figure_spec.json"
G1 = "results/tables/g1_primary_ranking_reliability.csv"
G2 = "results/tables/g2_history_length_analysis_universe.csv"
G3 = "results/tables/g3_candidate_position_reliability.csv"
G4 = "results/tables/g4_f1bzq_reliability.csv"
SOURCES = [AUTHORITY, CONTRACT, SPEC, G1, G2, G3, G4]
AUTHORITY_HASHES = {
    AUTHORITY: "a22ceae0565d87f1298ed1d97e8af1b53e24b3febb69ff81bb098c5bb1f5176e",
    CONTRACT: "f2df8cfc127a586d18082a6dc710356ba058ad1b641f2d38821fe588a87d050d",
    SPEC: "5752191568107c46318a9665b3aaea9c4a785e6b0a2c8f740473974d1017f280",
}
TABLE_HASHES = {
    G1: "1f5051ee58df03118617c981d13ece3a1679df2440b08f59e52a7ed068ebd023",
    G2: "02d501b1fd250f2da1ab33905a36da2a80803c8e16b763b5a31f56de5bc7e336",
    G3: "b31de0ea8c3b788091eeb7893af7a795029ddb36059d87a13c88f1257da7d537",
    G4: "666f4af4d2fda42fbd145e0d34dd881f2655c5582e8c88b83251690b4b2d27cb",
}
HASHES = {**AUTHORITY_HASHES, **TABLE_HASHES}
OUT = "results/final_figures/rq6"
MANIFEST = f"results/manifests/final_figures/{FIGURE_ID}_manifest.json"
GENERATOR = "src/analysis/final_figures/rq6.py"
FIXED_TIME = datetime(2026, 9, 21, tzinfo=timezone.utc)
ORDER = ("G1", "G2", "G3", "G4")
COUNTS = ("invalid_length", "duplicate_index", "unknown_or_out_of_candidate", "parser_or_schema_failure", "backend_failure")
COLORS = {
    "invalid_length": "#D55E00", "duplicate_index": "#E69F00",
    "unknown_or_out_of_candidate": "#CC79A7", "parser_or_schema_failure": "#56B4E9",
    "backend_failure": "#777777",
}


def _load_json(root: Path, relative: str) -> dict:
    return json.loads(repository_path(root, relative).read_text(encoding="utf-8"))


def validate_sources(root: Path = ROOT) -> tuple[dict, dict, dict]:
    """Verify the frozen authority documents, their meaning, and table identity."""
    root = root.resolve()
    for relative, expected in HASHES.items():
        path = repository_path(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise FigureValidationError(f"RQ6 source hash mismatch: {relative}")
    authority, contract, spec = (_load_json(root, relative) for relative in (AUTHORITY, CONTRACT, SPEC))
    if authority["direct_textual_hallucination_reproduced"] is not False or authority["direct_comparability_with_paper"] is not False:
        raise FigureValidationError("RQ6 evidence authority changed")
    if authority["local_canonical_evidence"] != "STRUCTURAL_RELIABILITY":
        raise FigureValidationError("RQ6 local evidence type changed")
    if contract["direct_comparability"] is not False or contract["direct_textual_hallucination_reproduced"] is not False:
        raise FigureValidationError("RQ6 scientific comparability contract changed")
    if contract["local_canonical_measure"]["output_space"] != "candidate-index JSON" or contract["target_paper_measure"]["output_space"] != "generated item titles":
        raise FigureValidationError("RQ6 output spaces changed")
    if spec["figure_id"] != FIGURE_ID or [p["panel_id"] for p in spec["panels"]] != ["A", "B"]:
        raise FigureValidationError("RQ6 two-panel figure spec changed")
    if spec["panels"][0]["quantitative_foreground"] is not False or spec["panels"][1]["numeric_sources"] != [G1, G2, G3, G4]:
        raise FigureValidationError("RQ6 panel authority changed")
    canonical = {item["path"]: item["sha256"] for item in authority["authority_classes"]["CANONICAL"]}
    for relative, expected in TABLE_HASHES.items():
        if canonical.get(relative) != expected:
            raise FigureValidationError(f"RQ6 canonical registry mismatch: {relative}")
    forbidden = set(spec["forbidden_quantitative_sources"])
    if forbidden != set(authority["foreground_policy"]["forbidden_quantitative_foreground_sources"]):
        raise FigureValidationError("RQ6 forensic exclusion changed")
    if forbidden.intersection(SOURCES):
        raise FigureValidationError("forensic R3/R4 became numeric inputs")
    return authority, contract, spec


def _count_g1(root: Path) -> dict[str, int]:
    frame = pd.read_csv(repository_path(root, G1), dtype=str, keep_default_na=False)
    parser = Counter()
    for raw in frame.parse_status_counts:
        parser.update(json.loads(raw))
    if set(parser) != {"parsed_valid", "invalid_length", "duplicate_indices"}:
        raise FigureValidationError("G1 parser status taxonomy changed")
    return {
        "total_requests": int(frame.planned_N.astype(int).sum()),
        "strict_valid": int(parser["parsed_valid"]),
        "invalid_total": int(frame.structurally_invalid_N.astype(int).sum()),
        "invalid_length": int(parser["invalid_length"]),
        "duplicate_index": int(parser["duplicate_indices"]),
        "unknown_or_out_of_candidate": int(frame.hallucination_out_of_candidate_count.astype(int).sum()),
        "parser_or_schema_failure": 0,
        "backend_failure": int(frame.backend_failure_N.astype(int).sum()),
    }


def _count_g2(root: Path) -> dict[str, int]:
    frame = pd.read_csv(repository_path(root, G2), dtype=str, keep_default_na=False)
    counts = frame.terminal_state.value_counts()
    expected_states = {"STRICT_VALID", "INVALID_LENGTH", "INVALID_DUPLICATE", "HALLUCINATION_OR_OUT_OF_CANDIDATE"}
    if set(counts.index) != expected_states or not frame.backend_success.eq("true").all():
        raise FigureValidationError("G2 terminal taxonomy or backend state changed")
    if not frame.strict_parse_valid.eq(frame.terminal_state.map({"STRICT_VALID": "true", "INVALID_LENGTH": "false", "INVALID_DUPLICATE": "false", "HALLUCINATION_OR_OUT_OF_CANDIDATE": "false"})).all():
        raise FigureValidationError("G2 strict-valid classification differs")
    unknown_values = int(frame.hallucination_count.astype(int).sum())
    if unknown_values != 16:
        raise FigureValidationError("G2 unknown-index value count changed")
    return {
        "total_requests": len(frame), "strict_valid": int(counts["STRICT_VALID"]),
        "invalid_total": len(frame) - int(counts["STRICT_VALID"]),
        "invalid_length": int(counts["INVALID_LENGTH"]),
        "duplicate_index": int(counts["INVALID_DUPLICATE"]),
        "unknown_or_out_of_candidate": int(counts["HALLUCINATION_OR_OUT_OF_CANDIDATE"]),
        "parser_or_schema_failure": 0, "backend_failure": 0,
    }


def _count_g3(root: Path) -> dict[str, int]:
    frame = pd.read_csv(repository_path(root, G3), dtype=str, keep_default_na=False)
    overall = frame.loc[frame.scope.eq("overall")]
    if len(overall) != 1:
        raise FigureValidationError("G3 overall reliability row missing")
    row = overall.iloc[0]
    return {
        "total_requests": int(row.planned_requests), "strict_valid": int(row.strict_valid),
        "invalid_total": int(row.invalid), "invalid_length": int(row.invalid_length),
        "duplicate_index": int(row.invalid_duplicate),
        "unknown_or_out_of_candidate": int(row.hallucination_or_out_of_candidate),
        "parser_or_schema_failure": 0, "backend_failure": int(row.backend_errors),
    }


def _count_g4(root: Path) -> dict[str, int]:
    frame = pd.read_csv(repository_path(root, G4), dtype=str, keep_default_na=False)
    columns = {
        "total_requests": "planned_requests", "strict_valid": "strict_valid",
        "invalid_total": "structural_invalid", "invalid_length": "invalid_length",
        "duplicate_index": "invalid_duplicate",
        "unknown_or_out_of_candidate": "hallucination_or_unknown_index",
        "backend_failure": "backend_errors",
    }
    result = {field: int(frame[column].astype(int).sum()) for field, column in columns.items()}
    result["parser_or_schema_failure"] = 0
    return result


def figure_values(root: Path = ROOT) -> pd.DataFrame:
    """Derive G1-G4 structural failure counts without using forensic R3/R4."""
    _, contract, _ = validate_sources(root)
    counters = {"G1": _count_g1, "G2": _count_g2, "G3": _count_g3, "G4": _count_g4}
    rows = []
    for experiment in ORDER:
        counts = counters[experiment](root)
        frozen = contract["canonical_strict_valid_accounting"][experiment]
        expected = {
            "total_requests": frozen["planned"], "strict_valid": frozen["strict_valid"],
            "invalid_total": frozen["invalid"], "invalid_length": frozen["invalid_length_requests"],
            "duplicate_index": frozen["duplicate_index_requests"],
            "unknown_or_out_of_candidate": frozen.get("unknown_index_requests", frozen.get("out_of_candidate_requests")),
            "backend_failure": frozen["backend_failures"],
        }
        if any(counts[field] != value for field, value in expected.items()):
            raise FigureValidationError(f"{experiment} canonical counts differ from scientific contract")
        if counts["strict_valid"] + counts["invalid_total"] + counts["backend_failure"] != counts["total_requests"]:
            raise FigureValidationError(f"{experiment} request total does not reconcile")
        if sum(counts[field] for field in COUNTS) != counts["invalid_total"]:
            raise FigureValidationError(f"{experiment} invalid subtypes do not reconcile")
        rows.append({
            "experiment": experiment, **counts,
            "strict_valid_rate": counts["strict_valid"] / counts["total_requests"],
            "evidence_type": "STRUCTURAL_RELIABILITY",
            "comparability_to_paper": "NOT_DIRECTLY_COMPARABLE",
            "canonical_source_artifact": frozen["source"],
            "canonical_source_sha256": TABLE_HASHES[frozen["source"]],
        })
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "irp-final-rq6-v1", "pdf.compression": 9})


def render(values: pd.DataFrame) -> plt.Figure:
    """Show distinct constructs and two honest views of local reliability."""
    fig = plt.figure(figsize=(15, 9.3))
    fig.suptitle("RQ6 | Different output contracts reveal different failure modes", y=.985, fontsize=16)

    left = fig.add_axes([.06, .69, .4, .22])
    right = fig.add_axes([.54, .69, .4, .22])
    for ax, title, flow, outcome, face in (
        (left, "TARGET PAPER: free-text output", "Generated item titles  →  title/catalogue matching", "Unmatched or fabricated titles\n= textual/catalogue hallucination", "#F6F1E8"),
        (right, "LOCAL STUDY: constrained output", "Candidate-index JSON  →  parser + validator", "Strict validity and structural failures\n= output reliability taxonomy", "#EAF3F7"),
    ):
        ax.set_facecolor(face)
        for spine in ax.spines.values():
            spine.set_color("#888888")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(.04, .84, title, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        ax.text(.04, .5, flow, transform=ax.transAxes, fontsize=10, va="center")
        ax.text(.04, .22, outcome, transform=ax.transAxes, fontsize=10, va="center")
    fig.text(.5, .935, "PANEL A  |  What is measured?", ha="center", fontsize=11, fontweight="bold")
    fig.text(.5, .655, "TEXTUAL HALLUCINATION  ≠  STRUCTURAL INVALIDITY     •     NOT DIRECTLY COMPARABLE", ha="center", fontsize=11, fontweight="bold")
    fig.text(.5, .61, "PANEL B  |  Local candidate-index JSON: structural reliability only", ha="center", fontsize=11, fontweight="bold")

    valid_ax = fig.add_axes([.12, .29, .34, .25])
    tax_ax = fig.add_axes([.59, .29, .34, .25])
    y = np.arange(len(values))
    rates = values.strict_valid_rate.astype(float).to_numpy() * 100
    valid_ax.barh(y, rates, color="#2A7F62", label="Strict valid")
    valid_ax.barh(y, 100-rates, left=rates, color="#D6D6D6", label="Structural invalid")
    valid_ax.set_yticks(y, values.experiment); valid_ax.invert_yaxis()
    valid_ax.set_xlim(0, 100); valid_ax.set_xlabel("Share of planned requests (%)")
    valid_ax.set_title("Strict validity: full 0–100% scale", fontsize=11)
    valid_ax.grid(axis="x", alpha=.15); valid_ax.set_axisbelow(True)
    for i, row in enumerate(values.itertuples(index=False)):
        valid_ax.text(50, i, f"{int(row.strict_valid):,}/{int(row.total_requests):,}  ({float(row.strict_valid_rate)*100:.2f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    valid_ax.legend(loc="upper center", bbox_to_anchor=(.5, -.27), ncol=2, frameon=False, fontsize=9)

    lefts = np.zeros(len(values))
    labels = {"invalid_length": "Invalid length", "duplicate_index": "Duplicate index", "unknown_or_out_of_candidate": "Unknown / out-of-candidate", "parser_or_schema_failure": "Parser/schema failure", "backend_failure": "Backend failure"}
    for field in COUNTS:
        counts = values[field].astype(int).to_numpy()
        tax_ax.barh(y, counts, left=lefts, color=COLORS[field], label=labels[field])
        lefts += counts
    tax_ax.set_yticks(y, values.experiment); tax_ax.invert_yaxis()
    tax_ax.set_xlim(0, max(values.invalid_total.astype(int))*1.11)
    tax_ax.set_xlabel("Invalid requests (counts; separate scale)")
    tax_ax.set_title("Invalid composition: counts, not a zoomed rate", fontsize=11)
    tax_ax.grid(axis="x", alpha=.15); tax_ax.set_axisbelow(True)
    for i, count in enumerate(values.invalid_total.astype(int)):
        tax_ax.text(count + max(values.invalid_total.astype(int))*.014, i, f"{count:,}", va="center", fontsize=9)
    tax_ax.legend(loc="upper center", bbox_to_anchor=(.5, -.27), ncol=2, frameon=False, fontsize=8)
    fig.text(.5, .08, "G2 unknown/out-of-candidate: 10 requests (16 index values). G1 duplicate-request count uses parser statuses, not raw violation instances.", ha="center", fontsize=9)
    fig.text(.5, .045, "High local structural validity does not reproduce the paper's free-text hallucination measure; no causal reduction or cross-protocol rate comparison is inferred.", ha="center", fontsize=8.7)
    return fig


def _save(fig: plt.Figure, target: Path) -> dict[str, str]:
    output = {}
    for extension in ("png", "pdf", "svg"):
        relative = f"{OUT}/figures/{FIGURE_ID}.{extension}"
        path = repository_path(target, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = ({"Software": "IRP deterministic final-figure pipeline"} if extension == "png" else
                    {"Creator": "IRP deterministic final-figure pipeline", "Producer": "Matplotlib", "CreationDate": FIXED_TIME, "ModDate": FIXED_TIME} if extension == "pdf" else
                    {"Creator": "IRP deterministic final-figure pipeline", "Date": "2026-09-21"})
        fig.savefig(path, dpi=170, metadata=metadata, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
        output[extension] = relative
        output[f"{extension}_sha256"] = sha256_file(path)
    plt.close(fig)
    return output


def generate(root: Path = ROOT, *, artifact_root: Path | None = None, created_from_commit: str | None = None) -> dict:
    """Create only the requested RQ6 final figure chain."""
    root = root.resolve()
    target = (artifact_root or root).resolve()
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
        "schema_version": "1.0", "figure_id": FIGURE_ID, "rq": "RQ6",
        "scientific_purpose": "Separate target-paper free-text hallucination from local constrained-output structural reliability.",
        "canonical_source_artifacts": SOURCES,
        "canonical_source_sha256": HASHES,
        "figure_values_csv": value_relative, "figure_values_sha256": sha256_file(value_path),
        "generator": GENERATOR, "generator_sha256": sha256_file(repository_path(root, GENERATOR)),
        "output_png": images["png"], "output_png_sha256": images["png_sha256"],
        "output_pdf": images["pdf"], "output_pdf_sha256": images["pdf_sha256"],
        "output_svg": images["svg"], "output_svg_sha256": images["svg_sha256"],
        "scientific_interpretation": "Local constrained-output protocols have high structural validity; this is not a direct reproduction of free-text textual hallucination.",
        "limitations": ["Paper textual/catalogue hallucination and local structural invalidity are not directly comparable.", "No locally reproduced free-text hallucination rate.", "No causal claim that constrained output reduces hallucination.", "Overlapping RQ5 views omitted from the quantitative panel."],
        "created_from_commit": commit,
        "panel_a_construct": "PAPER_FREE_TEXT_TEXTUAL_CATALOGUE_HALLUCINATION_VS_LOCAL_CANDIDATE_INDEX_STRUCTURAL_RELIABILITY",
        "panel_b_evidence_type": "STRUCTURAL_RELIABILITY",
        "direct_comparability": False,
        "r3_r4_quantitative_dependency": False,
        "r1_quantitative_dependency": False,
        "rq5_views_summed": False,
        "row_count": len(values),
    }
    manifest_path = repository_path(target, MANIFEST)
    write_json(manifest_path, manifest)
    validate_figure_manifest(root, manifest_path, artifact_root=target)
    return manifest


def validate(root: Path = ROOT, *, artifact_root: Path | None = None) -> dict:
    """Validate saved values, source identity, outputs and construct separation."""
    root = root.resolve()
    target = (artifact_root or root).resolve()
    expected = figure_values(root).astype(str).reset_index(drop=True)
    manifest = validate_figure_manifest(root, repository_path(target, MANIFEST), artifact_root=target)
    actual = pd.read_csv(repository_path(target, manifest["figure_values_csv"]), dtype=str, keep_default_na=False).astype(str).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    if manifest["direct_comparability"] is not False or manifest["r3_r4_quantitative_dependency"] is not False or manifest["rq5_views_summed"] is not False:
        raise FigureValidationError("RQ6 authority/construct flags changed")
    if manifest["canonical_source_artifacts"] != SOURCES:
        raise FigureValidationError("RQ6 source list changed")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "validate"), required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--created-from-commit")
    args = parser.parse_args(argv)
    result = (generate(ROOT, artifact_root=args.artifact_root, created_from_commit=args.created_from_commit)
              if args.mode == "generate" else validate(ROOT, artifact_root=args.artifact_root))
    print(json.dumps({"status": "PASS", "figure": result["figure_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate and register the final editorial roles of frozen figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.final_figures.common import (  # noqa: E402
    FigureValidationError, git_head, repository_path, sha256_file,
    validate_figure_manifest, write_dataframe, write_json,
)
from src.analysis.final_figures.cross_rq import validate as validate_cross_rq  # noqa: E402

INDEX = "results/final_figures/global_figure_index.csv"
MANIFEST = "results/manifests/final_figures/global_final_figure_freeze.json"
R5D_MANIFEST = "results/rq5_final_scientific_freeze/figures/fig_rq5_profile_plus_history_extension_manifest.json"
EXCLUDED_MANIFEST = "results/rq5_final_scientific_freeze/figures/fig_rq5_profile_example_qualitative_manifest.json"

ROLES = {
    "MAIN_TEXT": [
        "fig_rq1_utility_heatmap_appendix",
        "fig_rq2_llm_longtail_novelty_heatmap",
        "fig_rq2_llm_serendipity_coverage_heatmap",
        "fig_rq2_llm_calibration_heatmap",
        "fig_rq3_history_metric_grid_final",
        "fig_rq4_g3_first_random_utility_pairs",
        "fig_rq4_g3_first_random_bootstrap_deltas",
        "fig_rq5_profile_conditions_main",
        "fig_rq5_profile_deltas_heatmap",
        "fig_rq6_textual_hallucination_vs_structural_reliability",
        "fig_rq7_candidate_recall_and_reranking_utility",
        "fig_cross_rq_evidence_overview",
    ],
    "APPENDIX": [
        "fig_rq1_traditional_vs_llm_utility",
        "fig_rq2_traditional_vs_llm_longtail_popularity_novelty",
        "fig_rq2_traditional_vs_llm_serendipity_diversity_coverage",
        "fig_rq2_llm_diversity_distribution_heatmap",
        "fig_rq3_paired_bootstrap_decisions",
        "fig_rq5_history_length_profile_conditions",
    ],
    "REPOSITORY_ONLY": [
        "fig_rq1_utility_landscape_main",
        "fig_rq2_llm_longtail_novelty_landscape",
        "fig_rq2_llm_serendipity_coverage_landscape",
        "fig_rq2_llm_diversity_distribution_landscape",
        "fig_rq2_llm_calibration_landscape",
        "fig_rq4_targeted_r5b_canddif_recovery",
        "fig_rq5_profile_plus_history_extension",
    ],
    "EXCLUDED": ["fig_rq5_profile_example_qualitative"],
}

ASSERTIONS = {
    "all_scientific_sources_frozen": True,
    "no_scientific_values_modified": True,
    "no_missing_main_text_figures": True,
    "no_missing_appendix_figures": True,
    "calibration_local_extension_preserved": True,
    "rq6_textual_hallucination_not_equated_with_structural_invalidity": True,
    "rq7_same_pool_superiority_not_claimed": True,
    "traditional_vs_llm_role_separate_from_rq7": True,
    "foodcom_popreo_undefined_preserved": True,
    "foodcom_rq7_conditional_n4_diagnostic_preserved": True,
}


def _load(root: Path, relative: str) -> dict:
    return json.loads(repository_path(root, relative).read_text(encoding="utf-8"))


def _rq(figure_id: str) -> str:
    return "SYNTHESIS" if figure_id.startswith("fig_cross_rq_") else figure_id.split("_")[1].upper()


def _role_map() -> dict[str, str]:
    mapping = {}
    for role, ids in ROLES.items():
        for figure_id in ids:
            if figure_id in mapping:
                raise FigureValidationError(f"duplicate editorial role: {figure_id}")
            mapping[figure_id] = role
    if tuple(len(ROLES[role]) for role in ROLES) != (12, 6, 7, 1):
        raise FigureValidationError("editorial role counts changed")
    return mapping


def _validate_r5d(root: Path) -> dict:
    """Validate the already-frozen R5D figure's older manifest schema."""
    data = _load(root, R5D_MANIFEST)
    if data["figure_id"] != "fig_rq5_profile_plus_history_extension":
        raise FigureValidationError("R5D figure identity changed")
    checks = {**data["figure_sha256"], **data["source_artifacts"],
              data["source_csv"]: data["source_csv_sha256"]}
    for relative, digest in checks.items():
        path = repository_path(root, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise FigureValidationError(f"R5D frozen source/output mismatch: {relative}")
    for relative in (data["generation_script"], data["config"]):
        if not repository_path(root, relative).is_file():
            raise FigureValidationError(f"R5D dependency missing: {relative}")
    return data


def _scientific_sources(manifest: dict) -> dict[str, str]:
    return manifest.get("canonical_source_sha256", {})


def build(root: Path = ROOT, *, created_from_commit: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Build the global index and manifest after validating every active chain."""
    root = root.resolve()
    validate_cross_rq(root)
    mapping = _role_map()
    expected_standard = set(mapping) - {
        "fig_rq5_profile_plus_history_extension", "fig_rq5_profile_example_qualitative"}
    found_standard = {p.name.removesuffix("_manifest.json") for p in
                      repository_path(root, "results/manifests/final_figures").glob("fig_*_manifest.json")}
    if found_standard != expected_standard:
        raise FigureValidationError(f"unregistered final-layer manifests: {sorted(found_standard ^ expected_standard)}")

    calibration = _load(root, "results/manifests/final_figures/fig_rq2_llm_calibration_heatmap_manifest.json")
    rq6 = _load(root, "results/manifests/final_figures/fig_rq6_textual_hallucination_vs_structural_reliability_manifest.json")
    rq7 = _load(root, "results/manifests/final_figures/fig_rq7_candidate_recall_and_reranking_utility_manifest.json")
    if calibration.get("render_only_remediation", {}).get("not_target_paper_metric") is not True:
        raise FigureValidationError("calibration local-extension caption missing")
    if rq6.get("direct_comparability") is not False or rq6.get("panel_b_evidence_type") != "STRUCTURAL_RELIABILITY":
        raise FigureValidationError("RQ6 construct distinction missing")
    if rq7.get("same_pool_superiority_test") is not False or rq7.get("foodcom_conditional_n") != 4:
        raise FigureValidationError("RQ7 limitation missing")

    r5d = _validate_r5d(root)
    if repository_path(root, EXCLUDED_MANIFEST).is_file():
        excluded = _load(root, EXCLUDED_MANIFEST)
        if excluded.get("figure_id") != "fig_rq5_profile_example_qualitative":
            raise FigureValidationError("excluded RQ5 archival identity changed")

    records, figures = [], []
    for role, ids in ROLES.items():
        for figure_id in ids:
            rq = _rq(figure_id)
            if role == "EXCLUDED":
                record = dict(figure_id=figure_id, rq=rq, role=role,
                              scientific_purpose="Archived qualitative profile example",
                              png_path="", pdf_path="", svg_path="",
                              figure_manifest=EXCLUDED_MANIFEST, source_values_artifact="",
                              source_manifest_or_contract="results/rq5_final_scientific_freeze/freeze_manifest.json",
                              status="EXCLUDED_EDITORIAL",
                              notes="Editorial exclusion; archival evidence is retained, not regenerated.")
                figures.append({**record, "output_sha256": {}, "renderer_sha256": None,
                                "scientific_source_sha256": {}, "deterministic_regeneration": "NOT_APPLICABLE"})
            elif figure_id == "fig_rq5_profile_plus_history_extension":
                outputs = {Path(path).suffix[1:]: path for path in r5d["figure_sha256"]}
                record = dict(figure_id=figure_id, rq=rq, role=role,
                              scientific_purpose=r5d["scientific_interpretation"],
                              png_path=outputs["png"], pdf_path=outputs["pdf"], svg_path=outputs["svg"],
                              figure_manifest=R5D_MANIFEST, source_values_artifact=r5d["source_csv"],
                              source_manifest_or_contract="results/rq5_final_scientific_freeze/freeze_manifest.json",
                              status="FROZEN_SUPPORTING",
                              notes="R5D feasibility evidence; no matched History-only counterfactual.")
                figures.append({**record, "output_sha256": r5d["figure_sha256"],
                                "renderer_sha256": sha256_file(repository_path(root, r5d["generation_script"])),
                                "scientific_source_sha256": r5d["source_artifacts"],
                                "deterministic_regeneration": "FROZEN_HASH_VALIDATED"})
            else:
                manifest_path = f"results/manifests/final_figures/{figure_id}_manifest.json"
                data = validate_figure_manifest(root, repository_path(root, manifest_path))
                if data["figure_id"] != figure_id or data["rq"] != rq:
                    raise FigureValidationError(f"figure identity mismatch: {figure_id}")
                contract = [path for path in data["canonical_source_artifacts"]
                            if path.endswith((".json", ".yaml", ".yml"))]
                if not contract:
                    contract = data["canonical_source_artifacts"][:1]
                notes = (
                    "Final main text despite historical appendix filename." if figure_id == "fig_rq1_utility_heatmap_appendix" else
                    "Local extension, not a target-paper metric." if figure_id == "fig_rq2_llm_calibration_heatmap" else
                    "Constrained-output structural reliability, not textual hallucination." if rq == "RQ6" else
                    "No original-order same-pool superiority baseline; Food.com conditional n=4." if rq == "RQ7" else
                    "Target-informed FIRST oracle, not deployable." if rq == "RQ4" else
                    "No manuscript role implied by filename; see role column."
                )
                record = dict(figure_id=figure_id, rq=rq, role=role,
                              scientific_purpose=data["scientific_purpose"],
                              png_path=data["output_png"], pdf_path=data["output_pdf"], svg_path=data["output_svg"],
                              figure_manifest=manifest_path, source_values_artifact=data["figure_values_csv"],
                              source_manifest_or_contract=";".join(contract),
                              status="FROZEN_VALIDATED", notes=notes)
                figures.append({**record,
                                "output_sha256": {data[f"output_{ext}"]: data[f"output_{ext}_sha256"]
                                                  for ext in ("png", "pdf", "svg")},
                                "renderer_sha256": data["generator_sha256"],
                                "renderer_dependencies_sha256": data.get("generator_dependencies", {}),
                                "scientific_source_sha256": _scientific_sources(data),
                                "deterministic_regeneration": "HASH_VALIDATED"})
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    assert len(frame) == 26 and frame.figure_id.is_unique
    freeze = {
        "freeze_version": "1.0", "created_from_commit": created_from_commit or git_head(root),
        "figure_branch": "freeze/final-figures", "figure_ids": list(frame.figure_id),
        "role_counts": {role: len(ids) for role, ids in ROLES.items()},
        "figures": figures, "global_figure_index": INDEX,
        "global_figure_index_sha256": None,
        "global_assertions": ASSERTIONS,
        "validation": "ALL_ACTIVE_CHAINS_AND_SOURCE_HASHES_VALIDATED",
    }
    return frame, freeze


def generate(root: Path = ROOT) -> dict:
    """Write the final role registry and freeze manifest after validation."""
    root = root.resolve()
    frame, freeze = build(root)
    write_dataframe(repository_path(root, INDEX), frame)
    freeze["global_figure_index_sha256"] = sha256_file(repository_path(root, INDEX))
    write_json(repository_path(root, MANIFEST), freeze)
    validate(root)
    return freeze


def validate(root: Path = ROOT) -> dict:
    """Rebuild expected metadata and verify all final output hashes."""
    root = root.resolve()
    actual_manifest = _load(root, MANIFEST)
    expected_index, expected_manifest = build(root, created_from_commit=actual_manifest["created_from_commit"])
    actual_index = pd.read_csv(repository_path(root, INDEX), dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(actual_index, expected_index, check_dtype=False, check_exact=True)
    expected_manifest["global_figure_index_sha256"] = sha256_file(repository_path(root, INDEX))
    if actual_manifest != expected_manifest:
        raise FigureValidationError("global final figure freeze metadata changed")
    return actual_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("generate", "validate"))
    args = parser.parse_args(argv)
    result = generate(ROOT) if args.mode == "generate" else validate(ROOT)
    print(json.dumps({"status": "PASS", "role_counts": result["role_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

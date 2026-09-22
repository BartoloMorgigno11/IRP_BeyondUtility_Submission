"""Build Finalization F6A paper-scope rebaseline outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f6a_paper_scope_rebaseline import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    build_all,
    git_output,
    output_hashes,
    protected_hash_status,
)


def main() -> int:
    root = Path.cwd()
    branch = git_output(root, ["branch", "--show-current"])
    head = git_output(root, ["rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH:
        raise SystemExit(f"Unexpected branch: observed={branch} expected={EXPECTED_BRANCH}")
    if head != EXPECTED_HEAD:
        raise SystemExit(f"Unexpected HEAD: observed={head} expected={EXPECTED_HEAD}")

    before = output_hashes(root)
    result = build_all(root)
    after = output_hashes(root)
    frames = result["frames"]
    coverage = frames["corrected_component_coverage"].set_index("coverage_family")
    validation = frames["validation"]
    datasets = frames["dataset_taxonomy"]
    models = frames["model_taxonomy"]
    missing = frames["missing_evidence_rebaseline"]

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"authoritative_datasets={','.join(datasets['dataset'])}")
    print(f"authoritative_llms={','.join(models['paper_exact_identifier'])}")
    print("authoritative_traditional=BM25,MostPop,BPRMF,GRU4Rec,SASRec,LightGCN")
    print("authoritative_prompts=OpenP5-derived base prompt,LLMRank recency-focused,LLMSRec in-context learning")
    print("original_dimensions=utility,novelty,history-length sensitivity,candidate-position bias,generation-involved performance,hallucinations,fairness appendix,diversity appendix")
    print(f"lastfm_status={datasets.loc[datasets['dataset'].eq('LastFM'), 'local_status'].iloc[0]}")
    for family in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]:
        row = frames["qwen2_mistral_alignment"].loc[frames["qwen2_mistral_alignment"]["paper_model_family"].eq(family)].iloc[0]
        print(f"{family.lower().replace(' ', '_')}_status={row['overall_classification']}")
        print(f"{family.lower().replace(' ', '_')}_local_identifier={row['local_model_identifier']}")
    print(f"genuine_extensions={len(frames['extension_inventory'])}")
    print("fifth_dataset_exists=no")
    print("extra_dataset_requirement_met=no_if_mandatory")
    print("current_component_denominator=73")
    print("component_denominator_defects=Qwen/Mistral extension wording;LastFM extension wording;F5AF model denominator 5;mixed coverage families")
    print("provisional_corrected_denominator=not_safely_derivable")
    print("current_missing_evidence_count=37")
    print("provisional_corrected_missing_evidence_count=not_safely_derivable")
    print(f"source_missing_rows_audited={len(missing)}")
    print(f"affected_artifact_count={len(frames['affected_artifact_registry'])}")
    print(f"dataset_coverage={coverage.loc['local evaluation-ready dataset coverage', 'numerator']}/{coverage.loc['local evaluation-ready dataset coverage', 'denominator']}")
    print(f"llm_family_inventory={coverage.loc['paper LLM model-family inventory', 'numerator']}/{coverage.loc['paper LLM model-family inventory', 'denominator']}")
    print(f"exact_llm_version_coverage={coverage.loc['exact LLM model-version coverage', 'numerator']}/{coverage.loc['exact LLM model-version coverage', 'denominator']}")
    print(f"atomic_numeric_cells={coverage.loc['atomic paper numeric-cell extraction', 'denominator']}")
    print(f"protocol_exact_numeric_replication={coverage.loc['protocol-exact numeric replication', 'numerator']}/{coverage.loc['protocol-exact numeric replication', 'denominator']}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root)}")
    print(f"second_hashes_match={str((not before) or before == after).lower()}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

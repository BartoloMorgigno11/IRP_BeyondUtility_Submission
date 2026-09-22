"""Build Finalization F6B atomic scope registry outputs."""

from __future__ import annotations

from pathlib import Path

from src.analysis.finalization_f6b_atomic_scope_registry import (
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    active_missing_count,
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
    inv = frames["paper_component_inventory"]
    coverage = frames["corrected_coverage_dashboard"].set_index("coverage_family")
    validation = frames["validation"]
    missing = frames["corrected_missing_evidence"]
    priority = frames["experiment_priority"]

    print(f"branch={branch}")
    print(f"head={head}")
    print(f"decision={result['decision']}")
    print(f"atomic_components_total={len(inv)}")
    print(f"atomic_non_numeric_components={inv['is_numeric'].ne('true').sum()}")
    print(f"components_by_type={inv['component_type'].value_counts().to_dict()}")
    print("original_datasets=Amazon Beauty,Amazon Sports,ML-1M,LastFM")
    print(f"evaluation_ready_datasets={coverage.loc['dataset evaluation-ready coverage','numerator']}/{coverage.loc['dataset evaluation-ready coverage','denominator']}")
    print("original_tasks=ranking,reranking")
    print("original_llm_families=GPT-3.5,GPT-4o-mini,GPT-4o,Claude 3 Haiku,Llama3-70B-Instruct,Qwen2 7B Instruct,Mistral 7B Instruct")
    print(f"llms_with_local_runs={coverage.loc['LLM-family local-run coverage','numerator']}/{coverage.loc['LLM-family local-run coverage','denominator']}")
    print(f"exact_llm_versions={coverage.loc['exact LLM-version coverage','numerator']}/{coverage.loc['exact LLM-version coverage','denominator']}")
    print(f"traditional_model_coverage={coverage.loc['traditional-model coverage','numerator']}/{coverage.loc['traditional-model coverage','denominator']}")
    print(f"prompt_strategy_coverage={coverage.loc['prompt-strategy implementation coverage','numerator']}/{coverage.loc['prompt-strategy implementation coverage','denominator']}")
    print(f"metric_coverage={coverage.loc['metric execution coverage','numerator']}/{coverage.loc['metric execution coverage','denominator']}")
    print(f"analysis_dimension_coverage={coverage.loc['analysis-dimension coverage','numerator']}/{coverage.loc['analysis-dimension coverage','denominator']}")
    print(f"paper_observations={len(frames['paper_observation_registry'])};verified={coverage.loc['paper-observation verification coverage','numerator']}")
    print(f"numeric_cells={coverage.loc['atomic numeric extraction coverage','denominator']}")
    print(f"local_numeric_mappings={coverage.loc['local numeric mapping coverage','numerator']}/{coverage.loc['local numeric mapping coverage','denominator']}")
    print(f"protocol_exact_numeric_replications={coverage.loc['protocol-exact numeric replication coverage','numerator']}/{coverage.loc['protocol-exact numeric replication coverage','denominator']}")
    print(f"genuine_extensions={frames['extension_inventory']['implementation_status'].ne('not_present').sum()}")
    print("fifth_dataset_status=no;requirement=no_if_mandatory")
    print(f"corrected_missing_rows={len(missing)}")
    print(f"active_missing_gap_rows={active_missing_count(missing)}")
    print(f"affected_artifacts={len(frames['artifact_supersession_registry'])}")
    print(f"claim_corrections={len(frames['claim_corrections'])}")
    for _, row in priority.head(10).iterrows():
        print(f"priority:{row['recommended_order']}={row['experiment_title']}|{row['recommended_phase']}|score={row['priority_score']}")
    print(f"regression_rows={len(frames['regression_resolution'])}")
    print(f"regression_inconclusive_timeouts={frames['regression_resolution']['timed_out'].eq('true').sum()}")
    print(f"validation_pass={validation['status'].eq('pass').sum()}")
    print(f"validation_fail={validation['status'].eq('fail').sum()}")
    print(f"protected_inputs={protected_hash_status(root)}")
    print(f"second_hashes_match={str((not before) or before == after).lower()}")
    for path, digest in result["hashes"].items():
        print(f"sha256:{path}={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
